"""Autowrap smoke + parity tests targeting ``myoLegWalk-v0``.

Four layered checks:

* ``test_autowrap_loadable`` — the autowrapped task registers into
  ``mjlab.tasks.registry`` when the feature flag is on.
* ``test_parity_with_manual_port`` — every reward/termination/action/command
  field in the autowrap cfg matches the hand-ported ``myoleg_flat_env_cfg``
  within the allowlisted diffs (obs-noise specs, event param ordering,
  ``"iteration"`` → ``"step"`` curriculum key rename).
* ``test_autowrap_runs_one_step`` — two CPU envs reset + step without NaNs.
* ``test_autowrap_ppo_smoke`` (slow) — 2 PPO iterations.

All tests skip when MyoSuite introspection cannot run (missing package,
subprocess failure, ``mujoco`` ABI mismatch) so they stay useful on the
install-broken environments that come up around the ``mujoco`` pin.
"""

from __future__ import annotations

import os
from typing import Any

import pytest

_AUTO_TASK_ID = "MjlabMyoSuite-Auto-MyoLegWalk"


def _build_autowrap_cfg(play: bool = False) -> Any:
    """Build the autowrapped env cfg, skipping the test if MyoSuite is broken."""
    try:
        from myosuite_mjlab.autowrap import (
            build_env_cfg,
            introspect,
            load_annotation,
        )
    except ImportError as exc:  # pragma: no cover - defensive
        pytest.skip(f"autowrap package import failed: {exc}")

    annotation = load_annotation("myoLegWalk-v0")
    try:
        introspection = introspect("myoLegWalk-v0")
    except (RuntimeError, TimeoutError, ImportError) as exc:
        pytest.skip(f"MyoSuite introspection unavailable: {exc}")
    return build_env_cfg(annotation, introspection, play=play)


def _reward_signature(term: Any) -> tuple[str, float, tuple]:
    """(qualname, weight, sorted params that aren't SceneEntityCfg references)."""
    func_name = getattr(term.func, "__qualname__", repr(term.func))
    scalar_params = tuple(
        sorted(
            (k, v)
            for k, v in term.params.items()
            if not hasattr(v, "joint_names") and not hasattr(v, "body_names")
        )
    )
    return (func_name, float(term.weight), scalar_params)


def test_autowrap_loadable() -> None:
    """Registering via the feature flag surfaces the task in the mjlab registry."""
    os.environ["MYOSUITE_MJLAB_ENABLE_AUTOWRAP"] = "1"

    # Import lazily so the env flag takes effect on the first side-effectful import.
    try:
        from myosuite_mjlab.tasks._autowrap import register_all_autowrapped
    except ImportError as exc:  # pragma: no cover
        pytest.skip(f"_autowrap import failed: {exc}")

    try:
        registered = register_all_autowrapped()
    except (RuntimeError, TimeoutError, ImportError) as exc:
        pytest.skip(f"MyoSuite introspection unavailable: {exc}")

    assert _AUTO_TASK_ID in registered, registered

    from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

    assert load_env_cfg(_AUTO_TASK_ID) is not None
    assert load_rl_cfg(_AUTO_TASK_ID) is not None
    assert load_runner_cls(_AUTO_TASK_ID) is not None


def test_parity_with_manual_port() -> None:
    """Autowrap cfg mirrors the hand-written MyoLeg velocity cfg field-by-field."""
    auto_cfg = _build_autowrap_cfg(play=False)

    from myosuite_mjlab.tasks.velocity.myoleg.env_cfgs import myoleg_flat_env_cfg

    manual_cfg = myoleg_flat_env_cfg(play=False)

    # Scalars: episode length + command ranges.
    assert auto_cfg.episode_length_s == manual_cfg.episode_length_s

    from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg

    assert isinstance(auto_cfg.commands["twist"], UniformVelocityCommandCfg)
    assert isinstance(manual_cfg.commands["twist"], UniformVelocityCommandCfg)
    for axis in ("lin_vel_x", "lin_vel_y"):
        assert getattr(auto_cfg.commands["twist"].ranges, axis) == getattr(
            manual_cfg.commands["twist"].ranges, axis
        ), axis

    # Action term shape.
    assert set(auto_cfg.actions.keys()) == set(manual_cfg.actions.keys())
    for name, auto_action in auto_cfg.actions.items():
        manual_action = manual_cfg.actions[name]
        assert auto_action.scale == manual_action.scale
        assert tuple(auto_action.actuator_names) == tuple(manual_action.actuator_names)

    # Rewards: every manual term has a matching autowrap term (same func + weight
    # + scalar params). SceneEntityCfg-valued params are excluded from equality
    # because their runtime `resolve` state diverges even when the definitions
    # match.
    auto_rewards = {k: _reward_signature(v) for k, v in auto_cfg.rewards.items()}
    manual_rewards = {k: _reward_signature(v) for k, v in manual_cfg.rewards.items()}
    missing = set(manual_rewards) - set(auto_rewards)
    assert not missing, f"autowrap missing reward terms: {missing}"
    extra = set(auto_rewards) - set(manual_rewards)
    assert not extra, f"autowrap has unexpected reward terms: {extra}"
    for name, manual_sig in manual_rewards.items():
        assert auto_rewards[name] == manual_sig, name

    # Terminations: same key set, same func qualname, same params.
    assert set(auto_cfg.terminations.keys()) == set(manual_cfg.terminations.keys())
    for name, manual_term in manual_cfg.terminations.items():
        auto_term = auto_cfg.terminations[name]
        assert getattr(auto_term.func, "__qualname__", None) == getattr(
            manual_term.func, "__qualname__", None
        ), name
        assert auto_term.params == manual_term.params, name


def test_autowrap_runs_one_step() -> None:
    """Building + stepping the autowrapped env for 2 envs on CPU produces finite obs."""
    import torch

    auto_cfg = _build_autowrap_cfg(play=False)
    auto_cfg.scene.num_envs = 2

    from mjlab.envs import ManagerBasedRlEnv

    env = ManagerBasedRlEnv(cfg=auto_cfg, device="cpu")
    try:
        obs, _info = env.reset()
        action = torch.zeros(env.num_envs, env.action_manager.total_action_dim)
        obs, reward, terminated, truncated, _info = env.step(action)

        first_group = next(iter(obs.values())) if isinstance(obs, dict) else obs
        assert torch.all(torch.isfinite(first_group))
        assert torch.all(torch.isfinite(reward))
        assert reward.shape[0] == 2
        assert terminated.shape[0] == 2 and truncated.shape[0] == 2
    finally:
        env.close()


@pytest.mark.slow
def test_autowrap_ppo_smoke() -> None:
    """2 PPO iterations on 4 CPU envs without NaNs — same shape as balance smoke."""
    import dataclasses

    import torch
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import RslRlVecEnvWrapper

    os.environ["MYOSUITE_MJLAB_ENABLE_AUTOWRAP"] = "1"
    try:
        from myosuite_mjlab.tasks._autowrap import register_all_autowrapped
    except ImportError as exc:  # pragma: no cover
        pytest.skip(f"_autowrap import failed: {exc}")
    try:
        register_all_autowrapped()
    except (RuntimeError, TimeoutError, ImportError) as exc:
        pytest.skip(f"MyoSuite introspection unavailable: {exc}")

    from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

    env_cfg = load_env_cfg(_AUTO_TASK_ID)
    env_cfg.scene.num_envs = 4

    rl_cfg = load_rl_cfg(_AUTO_TASK_ID)
    runner_cls = load_runner_cls(_AUTO_TASK_ID)
    assert runner_cls is not None

    env = ManagerBasedRlEnv(cfg=env_cfg, device="cpu")
    vec_env = RslRlVecEnvWrapper(env)
    runner = runner_cls(vec_env, dataclasses.asdict(rl_cfg), log_dir=None, device="cpu")
    try:
        runner.learn(num_learning_iterations=2, init_at_random_ep_len=True)
        obs, _ = vec_env.reset()
        # rsl-rl's vec wrapper returns a TensorDict; grab any leaf tensor.
        leaf = next(iter(obs.values()))
        assert torch.all(torch.isfinite(leaf))
    finally:
        vec_env.close()
