"""Rope + ascender rail for MuJoCo — plain `mujoco`, no RL dependency.

Adds to a robot MjSpec (g1_unitree_ascender.xml):

  world
  ├── rope            visual cylinder along +x through the ascender channel (no contact)
  └── rope_carriage   slide joint `rope_slide` (along +x) + hinge `rope_spin` (about +x)
        weld  rope_carriage <-> right_wrist_yaw_link, so the rope always runs
        through the ascender's channel: the tool slides along the rope and spins
        around it, nothing else — a real ascender. `ratchet()` is the cam.

The channel = the tool-frame Z axis through the collision-mesh centre, mapped
through the mount pose (assets/ascender/MOUNT.md: the cam head is centred on
the wrist joint). `solve_wrist()` finds wrist angles that put the channel
parallel to the rope for the reset pose.

Used by rl/chloe/task/robot.py (mjlab training) and usable by app/harness.
"""

from __future__ import annotations

import re
from pathlib import Path

import mujoco
import numpy as np

HERE = Path(__file__).resolve().parent
ASCENDER_MESH = HERE / "meshes/ascender_collision.obj"

ROPE_BODY = "rope"
CARRIER_BODY = "rope_carriage"
SLIDE_JOINT = "rope_slide"  # no "_joint" suffix so ".*_joint" regexes skip it
SPIN_JOINT = "rope_spin"
WRIST_BODY = "right_wrist_yaw_link"
RIGHT_WRIST = ("right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint")

ROPE_LENGTH = 30.0  # m uphill of the start
ROPE_TAIL = 1.0  # m downhill of the start
ROPE_RADIUS = 0.0055  # 11 mm static rope (Petzl Basic: 8-13 mm)
CARRIER_MASS = 0.05
CAM_FRICTION_N = 3.0  # push needed to slide the cam up the rope


def ascender_channel(model: mujoco.MjModel) -> tuple[np.ndarray, np.ndarray]:
  """(channel centre, channel axis) in the wrist-link frame."""
  wb = model.body(WRIST_BODY).id
  gid = next(
    i for i in range(model.ngeom)
    if model.geom_bodyid[i] == wb
    and model.geom_type[i] == mujoco.mjtGeom.mjGEOM_MESH
    and "ascender_collision" in model.mesh(model.geom_dataid[i]).name
  )
  verts = np.array(
    [list(map(float, l.split()[1:4])) for l in open(ASCENDER_MESH) if l.startswith("v ")]
  )
  centre_mesh = (verts.min(0) + verts.max(0)) / 2
  rot = np.zeros(9)
  mujoco.mju_quat2Mat(rot, model.geom_quat[gid])
  rot = rot.reshape(3, 3)
  return model.geom_pos[gid] + rot @ centre_mesh, rot @ np.array([0.0, 0.0, 1.0])


def set_pose(model, data, root_pos, root_quat, joint_pos: dict) -> None:
  """Write a pose given as regex->angle dict (mjlab style) and run forward kinematics."""
  data.qpos[:3] = root_pos
  data.qpos[3:7] = root_quat
  for j in range(model.njnt):
    if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
      continue
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
    for pat, val in joint_pos.items():
      if re.fullmatch(pat, name):
        data.qpos[model.jnt_qposadr[j]] = val
  mujoco.mj_forward(model, data)


def solve_wrist(model, data) -> dict:
  """Wrist roll/pitch/yaw (from the current pose) that make the channel axis parallel to +x."""
  from scipy.optimize import minimize

  wb = model.body(WRIST_BODY).id
  _, axis_link = ascender_channel(model)
  adr = [model.jnt_qposadr[model.joint(n).id] for n in RIGHT_WRIST]
  bounds = [tuple(model.jnt_range[model.joint(n).id] * 0.9) for n in RIGHT_WRIST]

  def cost(q):
    data.qpos[adr] = q
    mujoco.mj_forward(model, data)
    axis_w = data.xmat[wb].reshape(3, 3) @ axis_link
    return (1.0 - axis_w[0]) + 0.02 * float(np.sum(q**2))

  res = minimize(cost, np.zeros(3), bounds=bounds, method="L-BFGS-B")
  data.qpos[adr] = res.x
  mujoco.mj_forward(model, data)
  return {n: float(v) for n, v in zip(RIGHT_WRIST, res.x)}


def add_rope_rail(spec: mujoco.MjSpec, root_pos, root_quat, joint_pos: dict) -> dict:
  """Add rope + carriage + weld to `spec`, sized for the given reset pose.

  `joint_pos` may omit the wrist: it is solved here so the channel is parallel
  to the rope. Returns the full joint_pos dict used (wrist + rail joints at 0).
  """
  model = spec.compile()
  data = mujoco.MjData(model)
  set_pose(model, data, root_pos, root_quat, joint_pos)
  wrist_q = solve_wrist(model, data)
  wb = model.body(WRIST_BODY).id
  grip_link, axis_link = ascender_channel(model)
  rot = data.xmat[wb].reshape(3, 3)
  grip_w = data.xpos[wb] + rot @ grip_link
  axis_w = rot @ axis_link
  assert axis_w[0] > 0.99, f"ascender channel not aligned with the rope: {axis_w}"

  wrist = spec.body(WRIST_BODY)
  wrist.add_site(name="ascender_anchor", pos=grip_link.tolist(), group=5)
  chan = wrist.add_geom(  # the "cylinder inside the hole": shows the channel in the viewer
    name="ascender_channel",
    type=mujoco.mjtGeom.mjGEOM_CYLINDER,
    size=[0.006, 0.055, 0],
    rgba=[0.1, 1.0, 0.2, 0.6],
    contype=0,
    conaffinity=0,
    mass=0.0,
    group=2,
  )
  chan.fromto = np.concatenate([grip_link - 0.055 * axis_link, grip_link + 0.055 * axis_link]).tolist()

  wb_spec = spec.worldbody
  rope = wb_spec.add_body(name=ROPE_BODY, pos=grip_w)
  rg = rope.add_geom(
    name="rope_geom",
    type=mujoco.mjtGeom.mjGEOM_CYLINDER,
    size=[ROPE_RADIUS, 1.0, 0.0],
    rgba=[0.9, 0.3, 0.1, 1.0],
    contype=0,
    conaffinity=0,
    group=2,
  )
  rg.fromto = [-ROPE_TAIL, 0, 0, ROPE_LENGTH, 0, 0]

  carrier = wb_spec.add_body(name=CARRIER_BODY, pos=grip_w)
  carrier.add_joint(
    name=SLIDE_JOINT,
    type=mujoco.mjtJoint.mjJNT_SLIDE,
    axis=[1, 0, 0],
    range=[-ROPE_LENGTH, ROPE_LENGTH],  # symmetric: soft limits (x0.9) must include 0
    damping=2.0,
    frictionloss=CAM_FRICTION_N,
  )
  carrier.add_joint(name=SPIN_JOINT, type=mujoco.mjtJoint.mjJNT_HINGE, axis=[1, 0, 0], damping=0.05)
  carrier.add_geom(
    name="carrier_geom",
    type=mujoco.mjtGeom.mjGEOM_SPHERE,
    size=[0.02, 0, 0],
    mass=CARRIER_MASS,
    rgba=[1.0, 0.9, 0.0, 1.0],
    contype=0,
    conaffinity=0,
    group=2,
  )
  carrier.add_site(name="carrier_anchor", pos=[0, 0, 0], group=5)
  eq = spec.add_equality(
    type=mujoco.mjtEq.mjEQ_WELD,
    name="ascender_grip",
    objtype=mujoco.mjtObj.mjOBJ_BODY,
    name1=CARRIER_BODY,
    name2=WRIST_BODY,
  )
  # relpose = wrist pose in the carriage frame (carriage frame = world, at the channel).
  eq.data[:3] = [0.0, 0.0, 0.0]
  eq.data[3:6] = (data.xpos[wb] - grip_w).tolist()
  eq.data[6:10] = data.xquat[wb].tolist()
  eq.solref = [0.004, 1.0]
  eq.solimp = [0.99, 0.999, 0.001, 0.5, 2.0]

  out = dict(joint_pos)
  out.update(wrist_q)
  out[SLIDE_JOINT] = 0.0
  out[SPIN_JOINT] = 0.0
  return out


def ratchet(model: mujoco.MjModel, data: mujoco.MjData, prev_slide: float, prefix: str = "") -> None:
  """The cam: call after each mj_step with the slide qpos from before the step."""
  jid = model.joint(prefix + SLIDE_JOINT).id
  q, v = model.jnt_qposadr[jid], model.jnt_dofadr[jid]
  data.qvel[v] = max(data.qvel[v], 0.0)
  data.qpos[q] = max(data.qpos[q], prev_slide)


def rope_state(model, data, prefix: str = "", slide_joint: str = SLIDE_JOINT) -> dict:
  """Telemetry with the real end-effector's keys (assets/ascender/ELECTRONICS.md)."""
  jid = model.joint(prefix + slide_joint).id
  dof = int(model.jnt_dofadr[jid])
  tension = float(abs(data.qfrc_constraint[dof]))
  return {
    "rope_progress_m": float(data.qpos[model.jnt_qposadr[jid]]),
    "tension_N": tension,
    "engaged": bool(tension > 5.0),
  }
