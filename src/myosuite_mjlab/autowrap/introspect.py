"""One-shot introspection of a MyoSuite Gym task in an isolated subprocess.

The builder needs to know which reward/termination translators to apply. The
only piece of information that cannot be derived from the annotation YAML is
the task *family* — extracted from ``env.spec.entry_point`` after the env is
instantiated. ``xml_abspath`` is recorded alongside it as a diagnostic so the
cache file stays human-inspectable.

Running in a spawn subprocess is load-bearing:

* It keeps the global ``gym.envs.registry`` clean for mjlab's later env
  construction (MyoSuite's registration side effects otherwise leak).
* It isolates the transitive ``dm_control`` / old-``gym`` imports MyoSuite
  pulls in, which can shadow ``gymnasium`` in the parent process.
* CPU MuJoCo stays off the GPU, so this does not fight with a later
  MuJoCo-Warp env.

Results are cached to ``autowrap/_cache/<task_id>.json`` so repeated builds
skip the subprocess entirely.
"""

from __future__ import annotations

import dataclasses
import json
import multiprocessing as mp
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_CACHE_DIR = Path(__file__).parent / "_cache"


@dataclass(frozen=True)
class IntrospectionResult:
    """Frozen snapshot of a MyoSuite task's static structure.

    Only ``family`` is consumed by the builder (to select reward translators).
    ``task_id`` and ``xml_abspath`` are kept as diagnostic identifiers in the
    on-disk cache.
    """

    task_id: str
    family: str
    xml_abspath: str

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> IntrospectionResult:
        return cls(
            task_id=d["task_id"],
            family=d["family"],
            xml_abspath=d.get("xml_abspath", ""),
        )


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
        import myosuite  # noqa: F401  # registration side effect
        import gymnasium as gym

        env = gym.make(task_id)
        # reset() is load-bearing: MyoSuite fully initializes ``self.sim`` /
        # ``mj_model`` only after the first reset, and we rely on ``env.spec``
        # being populated on a live env.
        env.reset()
        unwrapped = env.unwrapped

        spec = env.spec
        entry_point = ""
        if spec is not None and spec.entry_point:
            ep = spec.entry_point
            entry_point = (
                ep if isinstance(ep, str) else f"{ep.__module__}:{ep.__name__}"
            )

        xml_abspath = ""
        model_path = getattr(unwrapped, "model_path", None) or getattr(
            unwrapped, "fullpath", None
        )
        if model_path:
            xml_abspath = str(Path(model_path).resolve())

        result = {
            "task_id": task_id,
            "family": _derive_family(entry_point),
            "xml_abspath": xml_abspath,
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
    timeout_s: float = 120.0,
) -> IntrospectionResult:
    """Introspect a MyoSuite task by running it once in a spawn subprocess."""
    cache_path = _CACHE_DIR / f"{task_id}.json"
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
        raise RuntimeError(f"MyoSuite introspection failed for {task_id!r}:\n{payload}")

    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
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
