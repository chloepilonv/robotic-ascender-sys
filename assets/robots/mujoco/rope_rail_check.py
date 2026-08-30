"""Check the rope <-> ascender alignment contract (ROPE_ASCENDER_ALIGNMENT.md).

    python assets/robots/mujoco/rope_rail_check.py [--slope 20]

Builds the robot + rail in plain MuJoCo, then: channel on the rope? aligned?
cam holds under the hanging robot? Exit code 1 if any check fails.
"""

import argparse
import math
import sys
from pathlib import Path

import mujoco
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import rope_rail as rail  # noqa: E402

KNEES_BENT = {".*_hip_pitch_joint": -0.312, ".*_knee_joint": 0.669, ".*_ankle_pitch_joint": -0.363,
              ".*_elbow_joint": 0.6, "left_shoulder_roll_joint": 0.2, "left_shoulder_pitch_joint": 0.2,
              "right_shoulder_roll_joint": -0.2, "right_shoulder_pitch_joint": 0.2}


def main() -> int:
  ap = argparse.ArgumentParser()
  ap.add_argument("--slope", type=float, default=20.0)
  a = ap.parse_args()
  s = math.radians(a.slope)
  rot = (math.cos(s / 2), 0.0, math.sin(s / 2), 0.0)
  pos = (0.0, 0.0, 0.75)

  spec = mujoco.MjSpec.from_file(str(HERE / "g1_unitree_ascender.xml"))
  for act in list(spec.actuators):
    spec.delete(act)
  for key in list(spec.keys):
    spec.delete(key)
  joint_pos = rail.add_rope_rail(spec, pos, rot, KNEES_BENT)
  m = spec.compile()
  m.opt.gravity[:] = (-9.81 * math.sin(s), 0.0, -9.81 * math.cos(s))
  d = mujoco.MjData(m)
  rail.set_pose(m, d, pos, rot, joint_pos)
  rail.ratchet_reset(m, d)

  anc, car = m.site("ascender_anchor").id, m.site("carrier_anchor").id
  wb = m.body("right_wrist_yaw_link").id
  grip_link, axis_link = rail.ascender_channel(m)
  axis_w = d.xmat[wb].reshape(3, 3) @ axis_link
  gap0 = np.linalg.norm(d.site_xpos[anc] - d.site_xpos[car])
  tilt0 = math.degrees(math.acos(min(1.0, abs(axis_w[0]))))
  print(f"channel centre (tool frame)  {np.round(rail.CHANNEL_CENTRE_TOOL, 4).tolist()}  pitch {rail.CHANNEL_PITCH_DEG} deg")
  print(f"channel centre (wrist frame) {np.round(grip_link, 4).tolist()}")
  print(f"rope height at reset         {d.site_xpos[anc][2]:.3f} m (target {rail.ROPE_HEIGHT})")
  print(f"reset: channel-rope gap {gap0*100:.2f} cm, channel-vs-rope tilt {tilt0:.1f} deg")
  print("arm reset angles:", {k: round(v, 3) for k, v in joint_pos.items() if k.startswith("right_")})

  # Hang the whole robot from the tool for 2 s (no floor): the cam must hold.
  m.geom_contype[:] = 0
  m.geom_conaffinity[:] = 0
  sq = m.jnt_qposadr[m.joint("rope_slide").id]
  for _ in range(int(2.0 / m.opt.timestep)):
    rail.ratchet(m, d)
    mujoco.mj_step(m, d)
  axis_w = d.xmat[wb].reshape(3, 3) @ axis_link
  gap = np.linalg.norm(d.site_xpos[anc] - d.site_xpos[car])
  tilt = math.degrees(math.acos(min(1.0, abs(axis_w[0]))))
  print(f"hanging 2 s: gap {gap*100:.2f} cm, tilt {tilt:.1f} deg, slide {d.qpos[sq]*100:+.1f} cm  ({rail.rope_state(m, d)})")

  ok = gap0 < 0.005 and tilt0 < 5 and gap < 0.01 and tilt < 5 and d.qpos[sq] > -0.05
  print("ALIGNMENT OK" if ok else "ALIGNMENT FAILED")
  return 0 if ok else 1


if __name__ == "__main__":
  sys.exit(main())
