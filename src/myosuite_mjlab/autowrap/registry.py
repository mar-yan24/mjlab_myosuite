"""Translator registry for autowrapping MyoSuite rewards → mjlab ``RewardTermCfg``.

MyoSuite reward dict keys (``vel_reward``, ``cyclic_hip``, ``act_reg``, ...)
are not directly mjlab reward names. A few can be mapped 1:1 (``vel_reward``
→ the base velocity env's ``track_linear_velocity``), others require wrapping
YAML params into ``SceneEntityCfg`` (e.g. ``cyclic_hip`` takes a list of
joint names that must be lifted into a ``SceneEntityCfg(joint_names=...,
preserve_order=True)``), and some compose out to multiple mjlab terms or
belong as terminations rather than rewards.

This module encodes those per-family translations. Lookup is ``<family>.<name>``
first, then ``*.<name>`` for cross-family primitives like ``act_reg``.
Unknown keys raise when ``TaskAnnotation.strict_unknown_rewards`` is ``True``
(the default) so a new MyoSuite release that grows a reward breaks loudly
instead of silently dropping it.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any, Callable


class UnknownMyoSuiteFamilyError(KeyError):
    """Raised when introspection yields a family with no registered translators."""


class UnknownRewardKeyError(KeyError):
    """Raised when a reward key has neither a family-specific nor wildcard entry."""


# Family allow-list: widen as new task families are onboarded. An entry_point
# whose tail module is not here short-circuits with a clear error instead of
# sliding onto wildcards.
_KNOWN_FAMILIES: set[str] = {"walk_v0"}


@dataclass(frozen=True)
class ExtraRewardSpec:
    """Parsed ``extra_rewards`` entry from a TaskAnnotation YAML.

    ``translator`` is the registry key (e.g. ``"walk_v0.cyclic_hip"``).
    ``weight`` and ``params`` come verbatim from the YAML.
    """

    translator: str
    weight: float
    params: dict[str, Any]


# RewardTermCfg factory signature. We can't type-annotate the return as
# ``RewardTermCfg`` at module import time without pulling mjlab in, so we keep
# it as ``Any`` and let the builder rely on duck typing.
RewardTranslator = Callable[[ExtraRewardSpec], Any]


def _load_sym(dotted: str):
    """Import ``module.path:name`` or ``module.path.name`` and return the symbol."""
    if ":" in dotted:
        mod, sym = dotted.split(":", 1)
    else:
        mod, sym = dotted.rsplit(".", 1)
    return getattr(importlib.import_module(mod), sym)


# --- Concrete translators ---------------------------------------------------


def _translate_cyclic_hip(spec: ExtraRewardSpec) -> Any:
    """walk_v0.cyclic_hip → myosuite_mjlab.mdp.cyclic_hip_flexion_penalty."""
    from mjlab.managers.reward_manager import RewardTermCfg
    from mjlab.managers.scene_entity_config import SceneEntityCfg

    from myosuite_mjlab.mdp import cyclic_hip_flexion_penalty

    joint_names = tuple(spec.params.get("joint_names", ()))
    if not joint_names:
        raise ValueError(
            "walk_v0.cyclic_hip requires params.joint_names = [hip_flexion_l, ...]"
        )
    return RewardTermCfg(
        func=cyclic_hip_flexion_penalty,
        weight=spec.weight,
        params={
            "hip_period": int(spec.params.get("hip_period", 100)),
            "amplitude": float(spec.params.get("amplitude", 0.8)),
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=joint_names,
                preserve_order=True,
            ),
        },
    )


# --- Registry ---------------------------------------------------------------


REWARD_TRANSLATORS: dict[str, RewardTranslator] = {
    "walk_v0.cyclic_hip": _translate_cyclic_hip,
    # Intentionally absent: walk_v0.vel_reward, walk_v0.ref_rot,
    # walk_v0.joint_angle_rew, *.act_reg, *.done, *.sparse. Those either map
    # onto terms already constructed by make_velocity_env_cfg() (weights come
    # from the annotation's ``rewards`` override block) or are represented as
    # terminations — not as reward terms here.
}


# Termination ``kind`` → func resolver. Kept alongside the reward registry so
# the autowrap "vocabulary" of known primitives lives in one place.
def _termination_func_for(kind: str) -> Any:
    from mjlab.envs import mdp as envs_mdp

    if kind == "root_height_below_minimum":
        return envs_mdp.root_height_below_minimum
    if kind == "fell_over":
        # mjlab's velocity env already ships a fell_over termination (the
        # annotation override only touches its ``limit_angle`` param); the
        # base term is overwritten by the builder rather than reconstructed.
        return None
    if kind == "bad_orientation":
        return envs_mdp.bad_orientation
    if kind == "custom":
        return None
    raise ValueError(f"Unknown termination kind {kind!r}")


def resolve_reward_translator(
    family: str, reward_name: str, *, strict: bool = True
) -> RewardTranslator:
    """Look up a reward translator. Raises on unknown family or reward key."""
    if family not in _KNOWN_FAMILIES:
        raise UnknownMyoSuiteFamilyError(
            f"MyoSuite family {family!r} has no registered translators. "
            f"Add to autowrap.registry._KNOWN_FAMILIES and write translators."
        )
    key = f"{family}.{reward_name}"
    if key in REWARD_TRANSLATORS:
        return REWARD_TRANSLATORS[key]
    wildcard = f"*.{reward_name}"
    if wildcard in REWARD_TRANSLATORS:
        return REWARD_TRANSLATORS[wildcard]
    if strict:
        raise UnknownRewardKeyError(
            f"No translator for {key!r} (or {wildcard!r}). Either add one, or "
            f"set strict_unknown_rewards=false in the annotation to skip."
        )
    return lambda _spec: None  # type: ignore[return-value]


def resolve_obs_func(dotted: str) -> Any:
    """Load an obs-term callable from the annotation's ``obs_add[*].func`` string."""
    return _load_sym(dotted)


def resolve_robot_factory(dotted: str) -> Any:
    """Load the ``robot_cfg_factory`` callable named in the annotation."""
    return _load_sym(dotted)


def resolve_termination_func(kind: str) -> Any:
    return _termination_func_for(kind)
