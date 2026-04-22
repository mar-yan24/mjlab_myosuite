"""Autowrap: lift MyoSuite Gym tasks into mjlab ManagerBasedRlEnvCfg.

Public surface:
  - ``introspect(task_id)`` — run MyoSuite task once in a subprocess, return
    the task family (plus diagnostic task_id / xml_abspath).
  - ``load_annotation(task_id)`` — load the per-task YAML annotation.
  - ``build_env_cfg(annotation, introspection, play=False)`` — assemble the
    ManagerBasedRlEnvCfg from the annotation + registry of translated MDP
    primitives.
  - ``register_autowrapped(...)`` — register the result with mjlab.

This subpackage coexists with the hand-ported tasks under
``myosuite_mjlab/tasks/`` and is auto-registered at ``myosuite_mjlab.tasks``
import time via ``tasks/_autowrap.py::_register_on_import``.
"""

from myosuite_mjlab.autowrap.annotations import (
    RewardOverride,
    TaskAnnotation,
    TerminationOverride,
    load_annotation,
)
from myosuite_mjlab.autowrap.builder import build_env_cfg, register_autowrapped
from myosuite_mjlab.autowrap.introspect import IntrospectionResult, introspect
from myosuite_mjlab.autowrap.registry import (
    UnknownMyoSuiteFamilyError,
    UnknownRewardKeyError,
)

__all__ = [
    "IntrospectionResult",
    "RewardOverride",
    "TaskAnnotation",
    "TerminationOverride",
    "UnknownMyoSuiteFamilyError",
    "UnknownRewardKeyError",
    "build_env_cfg",
    "introspect",
    "load_annotation",
    "register_autowrapped",
]
