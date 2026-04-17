"""Autowrap-based task registration (opt-in, explicit).

Call :func:`register_all_autowrapped` from a script's ``if __name__ == "__main__"``
block (or from inside a test function) to register every annotation YAML under
``myosuite_mjlab/autowrap/annotations/`` as an ``MjlabMyoSuite-Auto-*`` task.

Registration is **not** triggered on module import. On Windows, the
``introspect()`` subprocess relies on ``multiprocessing.get_context("spawn")``,
which calls ``_check_not_importing_main()`` in the parent and refuses to start
a child if the ``.start()`` call is inside a module that is currently being
imported. Doing the registration at import time therefore crashes pytest and
any other importer on Windows. Explicit invocation avoids the whole class of
problem.
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

    Must be called from an ``if __name__ == "__main__"`` block or equivalent
    guarded context on Windows — ``introspect()`` spawns a subprocess.
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
