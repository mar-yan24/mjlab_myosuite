"""Autowrap-based task registration.

Two entry points:

- :func:`register_all_autowrapped` — explicit call that iterates every
  annotation YAML under ``myosuite_mjlab/autowrap/annotations/`` and registers
  an ``MjlabMyoSuite-Auto-*`` task for each one. Raises on introspection
  failure, so tests use this form.
- :func:`_register_on_import` — wraps the above in a try/except and is called
  from ``myosuite_mjlab.tasks.__init__``. Cache hits make the import path a
  pure file read (no subprocess), which is safe on Windows where spawn
  refuses to start a child during module import. On a missing cache /
  subprocess failure we log a warning and return ``[]`` rather than breaking
  the non-autowrap tasks.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable

from myosuite_mjlab.autowrap import (
    introspect,
    load_annotation,
    register_autowrapped,
)
from myosuite_mjlab.rl import MyosuiteVelocityRunner
from myosuite_mjlab.tasks.velocity.myoleg.rl_cfg import myoleg_ppo_runner_cfg

_log = logging.getLogger(__name__)

# task_id -> (rl_cfg_factory, runner_cls). Extend when a new annotation is
# added with its own PPO hyperparams or runner.
_RL_CFG_BY_TASK: dict[str, tuple[Callable[[], Any], type]] = {
    "myoLegWalk-v0": (myoleg_ppo_runner_cfg, MyosuiteVelocityRunner),
}


def _annotation_ids() -> list[str]:
    from myosuite_mjlab.autowrap.annotations import _ANNOTATIONS_DIR

    return sorted(p.stem for p in Path(_ANNOTATIONS_DIR).glob("*.yaml"))


def register_all_autowrapped() -> list[str]:
    """Register every annotation YAML. Returns the list of registered task ids.

    Relies on the cached introspection JSON under ``autowrap/_cache/`` when
    present. On a cache miss this will spawn a MyoSuite subprocess, which is
    only safe outside of module-import on Windows; see ``_register_on_import``
    for the guarded import-time path.
    """
    registered: list[str] = []
    for task_id in _annotation_ids():
        if task_id not in _RL_CFG_BY_TASK:
            _log.warning(
                "autowrap: annotation %s has no rl_cfg binding in _RL_CFG_BY_TASK; "
                "skipping registration.",
                task_id,
            )
            continue
        rl_cfg_factory, runner_cls = _RL_CFG_BY_TASK[task_id]
        annotation = load_annotation(task_id)
        introspection = introspect(task_id)
        mjlab_task_id = register_autowrapped(
            annotation=annotation,
            introspection=introspection,
            rl_cfg=rl_cfg_factory(),
            runner_cls=runner_cls,
        )
        registered.append(mjlab_task_id)
    return registered


def _register_on_import() -> list[str]:
    """Import-time wrapper around :func:`register_all_autowrapped`.

    Catches the expected failure modes (missing cache, failed/slow MyoSuite
    introspection) and logs a warning so that a broken autowrap environment
    does not break the non-autowrap tasks that ship alongside.
    """
    try:
        return register_all_autowrapped()
    except (FileNotFoundError, RuntimeError, TimeoutError) as exc:
        _log.warning(
            "autowrap: skipping auto-registration (%s: %s). Populate the cache via "
            "`python -m myosuite_mjlab.autowrap.introspect <task_id>` to enable.",
            type(exc).__name__,
            exc,
        )
        return []
