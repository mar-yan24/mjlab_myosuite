"""Pydantic schemas + loader for per-task autowrap annotations.

Each MyoSuite task the autowrapper targets carries a YAML sidecar under
``autowrap/annotations/<task_id>.yaml``. The YAML expresses the parts of the
mjlab env config that cannot be mechanically derived from MyoSuite
introspection — reward weights, termination thresholds, joint-filter regex,
action scale, per-reward parameter overrides, and explicit hook-ins for
primitives the translator registry does not know how to convert.

The schema is deliberately flat and pydantic-validated so missing keys or
typos fail loudly at load time rather than surfacing as silent divergences in
a trained policy.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

_ANNOTATIONS_DIR = Path(__file__).parent / "annotations"


class RewardOverride(BaseModel):
    """Override for a single reward term in the assembled env cfg."""

    model_config = ConfigDict(extra="forbid")

    weight: float | None = None
    params: dict[str, Any] = Field(default_factory=dict)
    enabled: bool = True


class ExtraReward(BaseModel):
    """A reward term not present in the base mjlab velocity env cfg.

    The ``translator`` field names a key in
    ``autowrap.registry.REWARD_TRANSLATORS`` (e.g. ``walk_v0.cyclic_hip``).
    """

    model_config = ConfigDict(extra="forbid")

    translator: str
    weight: float
    params: dict[str, Any] = Field(default_factory=dict)


class TerminationOverride(BaseModel):
    """Override for a termination term."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal[
        "root_height_below_minimum",
        "fell_over",
        "bad_orientation",
    ]
    params: dict[str, Any] = Field(default_factory=dict)


class ActionSpec(BaseModel):
    """Action configuration for the assembled env cfg."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["tendon_effort", "synergy_tendon_effort", "joint_position"]
    actuator_names: list[str] = Field(default_factory=list)
    scale: float = 1.0


class ObsAdd(BaseModel):
    """Extra observation term to splice into both actor and critic groups."""

    model_config = ConfigDict(extra="forbid")

    func: str  # "module.path:callable"
    params: dict[str, Any] = Field(default_factory=dict)


class TaskAnnotation(BaseModel):
    """Top-level annotation for one MyoSuite task."""

    model_config = ConfigDict(extra="forbid")

    mjlab_base: Literal["velocity", "balance"] = "velocity"
    mjlab_task_id: str | None = None
    robot_cfg_factory: str  # "module.path:callable" returning EntityCfg
    action: ActionSpec
    joint_filter_regex: str = ".*"
    episode_length_s: float

    rewards: dict[str, RewardOverride] = Field(default_factory=dict)
    extra_rewards: dict[str, ExtraReward] = Field(default_factory=dict)
    terminations: dict[str, TerminationOverride] = Field(default_factory=dict)

    obs_drop: list[str] = Field(default_factory=list)
    obs_add: dict[str, ObsAdd] = Field(default_factory=dict)

    command_ranges: dict[str, tuple[float, float]] = Field(default_factory=dict)
    foot_site_names: tuple[str, ...] = ("l_foot_touch", "r_foot_touch")
    foot_contact_pattern: str = r"^(calcn_l|calcn_r)$"

    strict_unknown_rewards: bool = True

    def default_mjlab_task_id(self, task_id: str) -> str:
        """Derive ``MjlabMyoSuite-Auto-<Camel>`` from a MyoSuite task id.

        ``myoLegWalk-v0`` → ``MjlabMyoSuite-Auto-MyoLegWalk``.
        """
        if self.mjlab_task_id:
            return self.mjlab_task_id
        base = task_id.split("-v")[0]
        # Capitalize first letter only so camelCase stays readable.
        base = base[:1].upper() + base[1:]
        return f"MjlabMyoSuite-Auto-{base}"


def load_annotation(
    task_id: str, annotations_dir: Path | None = None
) -> TaskAnnotation:
    """Load the annotation YAML for a MyoSuite task id.

    Args:
      task_id: MyoSuite task id, e.g. ``myoLegWalk-v0``.
      annotations_dir: Override for the annotations directory. Defaults to the
        package's ``annotations/`` subfolder.
    """
    directory = annotations_dir or _ANNOTATIONS_DIR
    path = directory / f"{task_id}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"No autowrap annotation for {task_id!r} at {path}")
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return TaskAnnotation.model_validate(raw)
