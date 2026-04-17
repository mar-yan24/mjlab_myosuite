"""One-shot introspection of a MyoSuite Gym task in an isolated subprocess.

MyoSuite reward/obs dicts are instance methods that require a fully
initialized Gym env (``self.sim``, keyframes, heading body, etc.). Source
parsing cannot recover ``rwd_keys_wt`` or termination thresholds because they
are frequently passed as ``gym.make`` kwargs. So we spawn a child process,
instantiate the env once, capture the metadata, and return.

Running in a spawn subprocess is load-bearing:

* It keeps the global ``gym.envs.registry`` clean for mjlab's later env
  construction (MyoSuite's registration side effects otherwise leak).
* It isolates the transitive ``dm_control`` / old-``gym`` imports MyoSuite
  pulls in, which can shadow ``gymnasium`` in the parent process.
* CPU MuJoCo stays off the GPU, so this does not fight with a later
  MuJoCo-Warp env.

One ``env.reset()`` call is enough to populate ``obs_dict`` and ``rwd_dict``.
Results are cached to ``autowrap/_cache/<task_id>.json`` so repeated builds
skip the subprocess entirely.
"""

from __future__ import annotations

import dataclasses
import importlib
import json
import multiprocessing as mp
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_CACHE_DIR = Path(__file__).parent / "_cache"


@dataclass(frozen=True)
class IntrospectionResult:
    """Frozen snapshot of a MyoSuite task's static structure."""

    task_id: str
    entry_point: str  # e.g. "myosuite.envs.myo.myobase.walk_v0:WalkEnvV0"
    family: str  # derived from entry_point's module tail, e.g. "walk_v0"
    xml_abspath: str
    action_dim: int
    ctrl_range: list[list[float]]  # shape (action_dim, 2)
    obs_keys: list[str]
    reward_keys: list[str]
    rwd_keys_wt: dict[str, float]
    frame_skip: int
    horizon: int  # steps per episode
    dt: float  # sim step * frame_skip
    init_qpos: list[float]
    init_qvel: list[float]
    jnt_names: list[str]
    body_names: list[str] = field(default_factory=list)
    site_names: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> IntrospectionResult:
        return cls(**d)


def _derive_family(entry_point: str) -> str:
    """Extract the task family from an entry_point string.

    ``myosuite.envs.myo.myobase.walk_v0:WalkEnvV0`` → ``walk_v0``.
    """
    if ":" not in entry_point:
        return entry_point.rsplit(".", 1)[-1]
    module_path = entry_point.split(":", 1)[0]
    return module_path.rsplit(".", 1)[-1]


def _introspect_worker(task_id: str, conn) -> None:
    """Runs in the child process. Sends back a dict or an exception string."""
    try:
        # MyoSuite registers its envs on import.
        import myosuite  # noqa: F401  # registration side effect

        try:
            import gymnasium as gym
        except ImportError:  # pragma: no cover - fallback for old envs
            import gym  # type: ignore

        env = gym.make(task_id)
        unwrapped = env.unwrapped

        # Drive one reset to populate obs/reward dicts.
        reset_out = env.reset()
        # gymnasium returns (obs, info); old gym returns obs alone.
        if isinstance(reset_out, tuple) and len(reset_out) == 2:
            obs = reset_out[0]
        else:
            obs = reset_out
        del obs

        # MyoSuite's mjx branch exposes ``mj_model``/``mj_data`` directly on
        # the unwrapped env (older releases went through ``self.sim``). Prefer
        # the new attrs; fall back to ``sim`` for older installs.
        mj_model = getattr(unwrapped, "mj_model", None)
        mj_data = getattr(unwrapped, "mj_data", None)
        if mj_model is None or mj_data is None:
            sim = getattr(unwrapped, "sim", None)
            if sim is not None:
                mj_model = getattr(sim, "model", mj_model)
                mj_data = getattr(sim, "data", mj_data)
        if mj_model is None:
            raise RuntimeError(
                "Could not locate mj_model on MyoSuite env "
                f"(tried unwrapped.mj_model and unwrapped.sim.model on {type(unwrapped).__name__})"
            )

        obs_dict = getattr(unwrapped, "obs_dict", None)
        if obs_dict is None and hasattr(unwrapped, "get_obs_dict"):
            obs_dict = unwrapped.get_obs_dict(mj_model, mj_data)
        obs_keys = sorted(obs_dict.keys()) if obs_dict else []

        rwd_dict = getattr(unwrapped, "rwd_dict", None)
        if rwd_dict is None and hasattr(unwrapped, "get_reward_dict"):
            rwd_dict = unwrapped.get_reward_dict(obs_dict or {})
        reward_keys = sorted(rwd_dict.keys()) if rwd_dict else []

        rwd_keys_wt = {
            k: float(v) for k, v in getattr(unwrapped, "rwd_keys_wt", {}).items()
        }

        spec = env.spec
        entry_point = ""
        if spec is not None and spec.entry_point:
            ep = spec.entry_point
            entry_point = ep if isinstance(ep, str) else f"{ep.__module__}:{ep.__name__}"
        horizon = int(getattr(spec, "max_episode_steps", 0) or 0)
        if horizon == 0:
            horizon = int(getattr(unwrapped, "horizon", 1000))
        action_dim = int(env.action_space.shape[0])
        ctrl_range = [[float(a), float(b)] for a, b in mj_model.actuator_ctrlrange]

        frame_skip = int(getattr(unwrapped, "frame_skip", 1))
        dt = float(mj_model.opt.timestep) * frame_skip

        init_qpos = list(map(float, getattr(unwrapped, "init_qpos", [])))
        init_qvel = list(map(float, getattr(unwrapped, "init_qvel", [])))

        jnt_names = [mj_model.joint(i).name for i in range(mj_model.njnt)]
        body_names = [mj_model.body(i).name for i in range(mj_model.nbody)]
        site_names = [mj_model.site(i).name for i in range(mj_model.nsite)]

        xml_abspath = ""
        model_path = getattr(unwrapped, "model_path", None) or getattr(
            unwrapped, "fullpath", None
        )
        if model_path:
            xml_abspath = str(Path(model_path).resolve())

        result = {
            "task_id": task_id,
            "entry_point": entry_point,
            "family": _derive_family(entry_point),
            "xml_abspath": xml_abspath,
            "action_dim": action_dim,
            "ctrl_range": ctrl_range,
            "obs_keys": obs_keys,
            "reward_keys": reward_keys,
            "rwd_keys_wt": rwd_keys_wt,
            "frame_skip": frame_skip,
            "horizon": horizon,
            "dt": dt,
            "init_qpos": init_qpos,
            "init_qvel": init_qvel,
            "jnt_names": jnt_names,
            "body_names": body_names,
            "site_names": site_names,
        }
        env.close()
        conn.send(("ok", result))
    except BaseException as exc:  # noqa: BLE001 - we forward everything
        import traceback

        conn.send(("err", f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"))
    finally:
        conn.close()


def introspect(
    task_id: str,
    *,
    use_cache: bool = True,
    cache_dir: Path | None = None,
    timeout_s: float = 120.0,
) -> IntrospectionResult:
    """Introspect a MyoSuite task by running it once in a spawn subprocess."""
    cache_root = cache_dir or _CACHE_DIR
    cache_path = cache_root / f"{task_id}.json"
    if use_cache and cache_path.exists():
        with cache_path.open("r", encoding="utf-8") as f:
            return IntrospectionResult.from_dict(json.load(f))

    ctx = mp.get_context("spawn")
    parent_conn, child_conn = ctx.Pipe(duplex=False)
    proc = ctx.Process(target=_introspect_worker, args=(task_id, child_conn))
    proc.start()
    try:
        if not parent_conn.poll(timeout_s):
            proc.terminate()
            proc.join(5)
            raise TimeoutError(
                f"MyoSuite introspection timed out after {timeout_s}s for {task_id!r}"
            )
        status, payload = parent_conn.recv()
    finally:
        proc.join(5)
        if proc.is_alive():
            proc.kill()

    if status != "ok":
        raise RuntimeError(
            f"MyoSuite introspection failed for {task_id!r}:\n{payload}"
        )

    cache_root.mkdir(parents=True, exist_ok=True)
    with cache_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return IntrospectionResult.from_dict(payload)


def _main() -> int:
    """CLI entry point: ``python -m myosuite_mjlab.autowrap.introspect <task_id>``."""
    if len(sys.argv) < 2:
        print("usage: python -m myosuite_mjlab.autowrap.introspect <task_id>")
        return 2
    task_id = sys.argv[1]
    result = introspect(task_id, use_cache=False)
    print(json.dumps(result.to_dict(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())


# Hint to keep ``importlib`` imported for the pyright-relaxed env (avoids unused warning).
_ = importlib
