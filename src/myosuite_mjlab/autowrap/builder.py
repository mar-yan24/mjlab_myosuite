"""Assemble ``ManagerBasedRlEnvCfg`` from an annotation + introspection result.

The algorithm mirrors the manual ``myoleg_flat_env_cfg`` port in
``src/myosuite_mjlab/tasks/velocity/myoleg/env_cfgs.py`` but drives every
param from the YAML annotation. For each base-env field the builder either:

* **copies** from the annotation (reward weights, termination thresholds,
  action config, command ranges, episode length),
* **resolves** a translator from ``autowrap.registry`` (extra rewards
  introducing MyoSuite-specific primitives like ``cyclic_hip``),
* **rewrites** with a walking-task default the annotation does not yet
  expose (viewer/base body = ``pelvis``, terrain = plane, foot geom pattern
  ``.*_col``). These defaults are consistent with the first-target
  ``myoLegWalk`` manual port; generalize to the YAML when a second task
  needs a different value.
"""

from __future__ import annotations

import warnings
from dataclasses import replace
from typing import TYPE_CHECKING, Any

from myosuite_mjlab.autowrap.annotations import TaskAnnotation
from myosuite_mjlab.autowrap.introspect import IntrospectionResult
from myosuite_mjlab.autowrap.registry import (
    ExtraRewardSpec,
    resolve_obs_func,
    resolve_reward_translator,
    resolve_robot_factory,
    resolve_termination_func,
)

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnvCfg


def _make_action_term(annotation: TaskAnnotation) -> tuple[str, Any]:
    """Return ``(action_name, action_cfg)`` for the env cfg's ``actions`` dict."""
    if annotation.action.type == "tendon_effort":
        from mjlab.envs.mdp.actions import TendonEffortActionCfg

        cfg = TendonEffortActionCfg(
            entity_name="robot",
            actuator_names=tuple(annotation.action.actuator_names),
            scale=annotation.action.scale,
        )
        return "tendon_effort", cfg
    if annotation.action.type == "synergy_tendon_effort":
        from myosuite_mjlab.mdp import SynergyTendonEffortActionCfg

        cfg = SynergyTendonEffortActionCfg(
            entity_name="robot",
            scale=annotation.action.scale,
        )
        return "tendon_synergy", cfg
    if annotation.action.type == "joint_position":
        from mjlab.envs.mdp.actions import JointPositionActionCfg

        cfg = JointPositionActionCfg(
            entity_name="robot",
            joint_names=tuple(annotation.action.actuator_names) or (".*",),
            scale=annotation.action.scale,
        )
        return "joint_pos", cfg
    raise ValueError(f"Unsupported action.type: {annotation.action.type!r}")


def _replace_velocity_obs_terms(cfg: Any) -> None:
    """Use ``mjlab.tasks.velocity.mdp`` base_lin_vel/ang_vel, preserving noise."""
    from mjlab.managers.observation_manager import ObservationTermCfg
    from mjlab.managers.scene_entity_config import SceneEntityCfg
    from mjlab.tasks.velocity import mdp as vel_mdp

    for group in ("actor", "critic"):
        terms = cfg.observations[group].terms
        if "base_lin_vel" in terms:
            terms["base_lin_vel"] = ObservationTermCfg(
                func=vel_mdp.base_lin_vel,
                params={"asset_cfg": SceneEntityCfg("robot")},
                noise=terms["base_lin_vel"].noise,
            )
        if "base_ang_vel" in terms:
            terms["base_ang_vel"] = ObservationTermCfg(
                func=vel_mdp.base_ang_vel,
                params={"asset_cfg": SceneEntityCfg("robot")},
                noise=terms["base_ang_vel"].noise,
            )


def _configure_sensors(cfg: Any, annotation: TaskAnnotation) -> None:
    """Point the terrain-scan sensor at the robot pelvis and add foot contacts."""
    from mjlab.sensor import ContactMatch, ContactSensorCfg, ObjRef

    terrain_scan = [
        s for s in cfg.scene.sensors if getattr(s, "name", None) == "terrain_scan"
    ]
    feet_ground = ContactSensorCfg(
        name="feet_ground_contact",
        primary=ContactMatch(
            mode="subtree",
            pattern=annotation.foot_contact_pattern,
            entity="robot",
        ),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found", "force"),
        reduce="netforce",
        num_slots=1,
        track_air_time=True,
    )
    if terrain_scan:
        pelvis_scan = replace(
            terrain_scan[0],
            frame=ObjRef(type="body", name="pelvis", entity="robot"),
        )
        cfg.scene.sensors = (pelvis_scan, feet_ground)
    else:
        cfg.scene.sensors = (feet_ground,)


def _apply_reward_overrides(cfg: Any, annotation: TaskAnnotation) -> None:
    """Apply annotation.rewards weight/param overrides to existing reward terms."""
    from mjlab.envs import mdp as envs_mdp
    from mjlab.managers.reward_manager import RewardTermCfg

    for name, override in annotation.rewards.items():
        if not override.enabled:
            cfg.rewards.pop(name, None)
            continue
        if name not in cfg.rewards:
            # Allow promoting a known "mjlab primitive" name (e.g. ``alive``)
            # when the base velocity cfg does not ship it.
            if name == "alive":
                cfg.rewards["alive"] = RewardTermCfg(
                    func=envs_mdp.is_alive,
                    weight=override.weight if override.weight is not None else 10.0,
                )
                continue
            warnings.warn(
                f"Reward override {name!r} has no matching term in the base cfg",
                RuntimeWarning,
                stacklevel=2,
            )
            continue
        if override.weight is not None:
            cfg.rewards[name].weight = override.weight
        for k, v in override.params.items():
            cfg.rewards[name].params[k] = v


def _apply_extra_rewards(
    cfg: Any,
    annotation: TaskAnnotation,
    introspection: IntrospectionResult,
) -> None:
    """Add registry-translated rewards (e.g. ``walk_v0.cyclic_hip``)."""
    for name, extra in annotation.extra_rewards.items():
        family_part, _, reward_part = extra.translator.partition(".")
        if not reward_part:
            # Bare name → use introspection's family.
            family_part, reward_part = introspection.family, family_part
        if family_part == "*":
            family_part = introspection.family
        translator = resolve_reward_translator(
            family_part,
            reward_part,
            strict=annotation.strict_unknown_rewards,
        )
        spec = ExtraRewardSpec(
            translator=extra.translator,
            weight=extra.weight,
            params=dict(extra.params),
        )
        cfg.rewards[name] = translator(spec)


def _apply_terminations(cfg: Any, annotation: TaskAnnotation) -> None:
    from mjlab.managers.termination_manager import TerminationTermCfg

    # ``resolve_termination_func`` returns None for kinds that the base velocity
    # cfg already ships (``fell_over``): in that case we only patch params.
    # Otherwise we add/replace the term with a fresh TerminationTermCfg.
    for name, override in annotation.terminations.items():
        func = resolve_termination_func(override.kind)
        if func is None:
            if name in cfg.terminations:
                cfg.terminations[name].params.update(override.params)
            continue
        cfg.terminations[name] = TerminationTermCfg(
            func=func,
            params=dict(override.params),
        )


def _apply_observations(cfg: Any, annotation: TaskAnnotation) -> None:
    from mjlab.managers.observation_manager import ObservationTermCfg

    # Velocity-task base obs swap (shared across all walking autowraps).
    _replace_velocity_obs_terms(cfg)

    # Drop.
    for drop_name in annotation.obs_drop:
        for group in ("actor", "critic"):
            cfg.observations[group].terms.pop(drop_name, None)

    # Add (to both actor and critic groups by convention).
    for obs_name, obs_add in annotation.obs_add.items():
        func = resolve_obs_func(obs_add.func)
        from mjlab.managers.scene_entity_config import SceneEntityCfg

        params = dict(obs_add.params)
        params.setdefault("asset_cfg", SceneEntityCfg("robot"))
        term = ObservationTermCfg(func=func, params=params)
        for group in ("actor", "critic"):
            cfg.observations[group].terms[obs_name] = term

    # Foot site names for critic foot_height.
    if "foot_height" in cfg.observations["critic"].terms:
        cfg.observations["critic"].terms["foot_height"].params[
            "asset_cfg"
        ].site_names = tuple(annotation.foot_site_names)


def _apply_commands_and_curriculum(cfg: Any, annotation: TaskAnnotation) -> None:
    from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg

    twist = cfg.commands["twist"]
    assert isinstance(twist, UniformVelocityCommandCfg)
    for key, (lo, hi) in annotation.command_ranges.items():
        setattr(twist.ranges, key, (lo, hi))

    if "command_vel" in cfg.curriculum:
        for stage in cfg.curriculum["command_vel"].params.get("velocity_stages", []):
            if "iteration" in stage:
                stage["step"] = stage.pop("iteration")


def _apply_scene_defaults(cfg: Any, annotation: TaskAnnotation) -> None:
    """Walking-task assumptions shared with the ``myoLeg`` manual port."""
    assert cfg.scene.terrain is not None
    cfg.scene.terrain.terrain_type = "plane"
    cfg.scene.terrain.terrain_generator = None
    cfg.curriculum.pop("terrain_levels", None)

    cfg.events["base_com"].params["asset_cfg"].body_names = ("pelvis",)
    cfg.events["foot_friction"].params["asset_cfg"].geom_names = ".*_col"
    cfg.events["reset_robot_joints"].params["position_range"] = (-0.05, 0.05)

    if "upright" in cfg.rewards:
        cfg.rewards["upright"].params["asset_cfg"].body_names = ("pelvis",)
    if "body_ang_vel" in cfg.rewards:
        cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = ("pelvis",)

    # Pose joint filter
    if "pose" in cfg.rewards:
        cfg.rewards["pose"].params["asset_cfg"].joint_names = (
            annotation.joint_filter_regex,
        )

    # Foot site names for *foot_clearance / foot_swing_height / foot_slip*.
    for reward_name in ("foot_clearance", "foot_swing_height", "foot_slip"):
        if reward_name in cfg.rewards:
            cfg.rewards[reward_name].params["asset_cfg"].site_names = tuple(
                annotation.foot_site_names
            )

    cfg.viewer.body_name = "pelvis"


def _apply_play_mode(cfg: Any) -> None:
    from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg

    cfg.episode_length_s = int(1e9)
    cfg.observations["actor"].enable_corruption = False
    cfg.events.pop("push_robot", None)
    cfg.curriculum.pop("command_vel", None)
    twist = cfg.commands["twist"]
    assert isinstance(twist, UniformVelocityCommandCfg)
    twist.ranges.lin_vel_x = (-0.5, 1.0)
    twist.ranges.ang_vel_z = (-0.5, 0.5)


def build_env_cfg(
    annotation: TaskAnnotation,
    introspection: IntrospectionResult,
    play: bool = False,
) -> ManagerBasedRlEnvCfg:
    """Assemble a ManagerBasedRlEnvCfg from an annotation + introspection."""
    if annotation.mjlab_base != "velocity":
        raise NotImplementedError(
            f"autowrap only supports mjlab_base='velocity' today, "
            f"got {annotation.mjlab_base!r}"
        )
    from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg

    cfg = make_velocity_env_cfg()

    # Robot (XML + articulation).
    robot_factory = resolve_robot_factory(annotation.robot_cfg_factory)
    cfg.scene.entities = {"robot": robot_factory()}

    _configure_sensors(cfg, annotation)

    action_name, action_cfg = _make_action_term(annotation)
    cfg.actions = {action_name: action_cfg}

    _apply_observations(cfg, annotation)
    _apply_scene_defaults(cfg, annotation)
    _apply_reward_overrides(cfg, annotation)
    _apply_extra_rewards(cfg, annotation, introspection)
    _apply_terminations(cfg, annotation)
    _apply_commands_and_curriculum(cfg, annotation)

    cfg.episode_length_s = annotation.episode_length_s

    if play:
        _apply_play_mode(cfg)

    return cfg


def register_autowrapped(
    annotation: TaskAnnotation,
    introspection: IntrospectionResult,
    rl_cfg: Any,
    runner_cls: Any,
    mjlab_task_id: str | None = None,
) -> str:
    """Register the autowrapped task with ``mjlab.tasks.registry``.

    Returns the registered task id. Idempotent: if the task id is already in
    the registry, we skip re-registration rather than swallowing errors.
    """
    from mjlab.tasks.registry import list_tasks, register_mjlab_task

    task_id = mjlab_task_id or annotation.default_mjlab_task_id(introspection.task_id)
    if task_id in list_tasks():
        return task_id
    register_mjlab_task(
        task_id,
        env_cfg=build_env_cfg(annotation, introspection, play=False),
        play_env_cfg=build_env_cfg(annotation, introspection, play=True),
        rl_cfg=rl_cfg,
        runner_cls=runner_cls,
    )
    return task_id
