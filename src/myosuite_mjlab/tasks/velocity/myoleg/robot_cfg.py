"""MyoLeg robot config using public myosuite assets and public mjlab APIs."""

from __future__ import annotations

import os
from pathlib import Path

import mujoco

from mjlab.actuator import XmlMuscleActuatorCfg
from mjlab.actuator.actuator import TransmissionType
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.spec_config import CollisionCfg


def _resolve_myoleg_xml_path() -> Path:
    """Resolve MyoLeg XML path from env override or installed myosuite assets."""
    env_override = os.environ.get("MYOSUITE_MJLAB_MYOLEG_XML")
    if env_override:
        xml_path = Path(env_override)
        if xml_path.exists():
            return xml_path

    try:
        import myosuite  # type: ignore
    except ImportError as exc:  # pragma: no cover - runtime dependency
        raise FileNotFoundError(
            "myosuite is required to resolve MyoLeg XML. "
            "Install myosuite or set MYOSUITE_MJLAB_MYOLEG_XML."
        ) from exc

    myosuite_root = Path(myosuite.__file__).resolve().parent

    # Try system path first if that's where assets are
    system_path = Path("/opt/conda/lib/python3.12/site-packages/myosuite")
    if system_path.exists():
        myosuite_root = system_path

    candidates = (
        myosuite_root / "simhive" / "myo_sim" / "leg" / "myolegs_mjlab.xml",
        myosuite_root / "simhive" / "myo_sim" / "leg" / "myolegs.xml",
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        "Could not locate MyoLeg XML from myosuite package assets; "
        "set MYOSUITE_MJLAB_MYOLEG_XML explicitly."
    )


def get_myoleg_spec() -> mujoco.MjSpec:
    """Load the MyoLeg entity-only MjSpec (robot, no terrain)."""
    spec = mujoco.MjSpec.from_file(str(_resolve_myoleg_xml_path()))
    # Force SPARSE Jacobian to trigger the correct branch in mujoco_warp/mjlab
    # This avoids a reshape ValueError for models with many tendons.
    spec.option.jacobian = mujoco.mjtJacobian.mjJAC_SPARSE
    # myolegs_mjlab.xml ships 4 keyframes ("init_state" + 3 unnamed). mjlab's
    # Scene._add_entities uses only the first and deletes it before attach,
    # but the remaining unnamed keyframes all get prefixed to "robot/" on
    # attach, triggering "repeated name 'robot/' in key" at compile. Keep
    # only the first to match mjlab's single-keyframe contract.
    while len(spec.keys) > 1:
        spec.delete(spec.keys[-1])
    return spec


MYOLEG_ARTICULATION = EntityArticulationInfoCfg(
    actuators=(
        XmlMuscleActuatorCfg(
            transmission_type=TransmissionType.TENDON,
            target_names_expr=(r".*_tendon",),
        ),
    ),
)

MYOLEG_COLLISION = CollisionCfg(
    geom_names_expr=(".*",),
    contype=0,
    conaffinity=1,
    condim=3,
)

def _joint_pos_from_keyframe() -> dict[str, float]:
    """Extract non-root joint positions from the MyoLeg XML's first keyframe.

    Passing an explicit ``joint_pos`` dict forces mjlab's Entity init down the
    ``resolve_expr`` path (float32 tensors) instead of the keyframe path
    (``torch.tensor(mj_model.key('init_state').qpos, ...)``, which inherits
    numpy's float64 and later fails a dtype check in
    ``write_joint_state_to_sim`` during reset events).

    The XML's keyframes are unnamed; mjlab's ``Entity.build`` later renames
    the first to ``init_state``. We read it here by index before that happens.
    """
    spec = get_myoleg_spec()
    if not spec.keys:
        return {}
    model = spec.compile()
    qpos = model.key(0).qpos
    pos: dict[str, float] = {}
    for i in range(model.njnt):
        joint = model.joint(i)
        if joint.type[0] == mujoco.mjtJoint.mjJNT_FREE:
            continue
        pos[joint.name] = float(qpos[model.jnt_qposadr[i]])
    return pos


MYOLEG_INIT_STATE = EntityCfg.InitialStateCfg(
    pos=(0.0, 0.0, 0.92),
    rot=(1.0, 0.0, 0.0, 0.0),
    joint_pos=_joint_pos_from_keyframe(),
    joint_vel={".*": 0.0},
)


def get_myoleg_robot_cfg() -> EntityCfg:
    """Get MyoLeg robot entity config (muscle controlled; XML actuators)."""
    return EntityCfg(
        spec_fn=get_myoleg_spec,
        articulation=MYOLEG_ARTICULATION,
        init_state=MYOLEG_INIT_STATE,
        collisions=(MYOLEG_COLLISION,),
    )
