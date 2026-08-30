"""Sim2sim: run an exported climb policy (ONNX) in plain CPU MuJoCo.

Training used MuJoCo Warp through mjlab's managers; here the observation, PD
targets and ratchet are re-implemented by hand on `mujoco.MjModel/MjData`, the
way the Jetson deployment will do it. If the policy behaves the same here as in
`play_mjlab.py`, the export + obs pipeline is right.

    .venv-mjlab/bin/mjpython -m rl.chloe.scripts.sim2sim rl/chloe/policies/g1_ascender_slope20.onnx   # mjpython on macOS (python on Linux)
    ... --headless --seconds 10 --slope 20 --wind 5
"""

from __future__ import annotations

import argparse
import math
import time

import mujoco
import mujoco.viewer
import numpy as np
import onnxruntime as ort


def quat_inv_rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
  """Rotate v by the inverse of quaternion q (w,x,y,z)."""
  out = np.zeros(3)
  qi = np.array([q[0], -q[1], -q[2], -q[3]])
  mujoco.mju_rotVecQuat(out, v, qi)
  return out


def build(slope_deg: float):
  """Compile the training scene once through mjlab; return plain-MuJoCo pieces."""
  import rl.chloe.task as A

  cfg = A.make_env_cfg(slope_deg, play=True)
  cfg.scene.num_envs = 1
  env = A.RatchetEnv(cfg=cfg, device="cpu")
  m = env.sim.mj_model
  term = env.action_manager.get_term("joint_pos")
  scale = term.scale[0].cpu().numpy().copy()
  offset = term.offset[0].cpu().numpy().copy()  # = default joint pos, joint order
  robot = env.scene["robot"]
  jnames = ["robot/" + n for n in robot.joint_names if n.endswith("_joint")]
  key = mujoco.MjData(m)
  mujoco.mj_resetDataKeyframe(m, key, 0)
  qpos0 = key.qpos.copy()
  env.close()
  return m, jnames, scale, offset, qpos0


def main() -> None:
  p = argparse.ArgumentParser()
  p.add_argument("onnx")
  p.add_argument("--slope", type=float, default=20.0)
  p.add_argument("--wind", type=float, default=0.0, help="m/s along -x (downhill headwind)")
  p.add_argument("--headless", action="store_true")
  p.add_argument("--seconds", type=float, default=15.0)
  args = p.parse_args()

  m, jnames, scale, offset, qpos0 = build(args.slope)
  d = mujoco.MjData(m)
  d.qpos[:] = qpos0
  mujoco.mj_forward(m, d)

  jid = np.array([m.joint(n).id for n in jnames])
  qadr, dadr = m.jnt_qposadr[jid], m.jnt_dofadr[jid]
  act_joint = m.actuator_trnid[:, 0]  # actuator i drives joint act_joint[i]
  joint_to_action = {int(j): k for k, j in enumerate(jid)}
  ctrl_map = np.array([joint_to_action[int(j)] for j in act_joint])
  slide = m.joint("robot/rope_slide").id
  s_q, s_d = m.jnt_qposadr[slide], m.jnt_dofadr[slide]
  carrier = m.body("robot/rope_carriage").id
  anchor = m.site("robot/ascender_anchor").id
  torso = m.body("robot/torso_link").id
  g_dir = m.opt.gravity / np.linalg.norm(m.opt.gravity)
  wind_f = 0.5 * 0.55 * 1.0 * 0.45 * args.wind**2

  sess = ort.InferenceSession(args.onnx)
  last_action = np.zeros(len(jid), dtype=np.float32)
  decimation, dt = 4, m.opt.timestep

  def obs() -> np.ndarray:
    q, w = d.qpos[3:7], d.qvel[3:6]  # free joint: ang vel already in body frame
    grav_b = quat_inv_rotate(q, g_dir)
    rel = quat_inv_rotate(q, d.xpos[carrier] - d.qpos[0:3])
    return np.concatenate(
      [w, grav_b, d.qpos[qadr] - offset, d.qvel[dadr], last_action, rel]
    ).astype(np.float32)[None]

  viewer = None if args.headless else mujoco.viewer.launch_passive(m, d)
  t0, step = time.time(), 0
  while (d.time < args.seconds) if args.headless else viewer.is_running():
    action = sess.run(None, {"obs": obs()})[0][0]
    last_action = action.astype(np.float32)
    target = offset + scale * action
    d.ctrl[:] = target[ctrl_map]
    d.xfrc_applied[torso, 0] = -wind_f
    for _ in range(decimation):
      prev = d.qpos[s_q]
      mujoco.mj_step(m, d)
      d.qvel[s_d] = max(d.qvel[s_d], 0.0)  # the ascender cam
      d.qpos[s_q] = max(d.qpos[s_q], prev)
    step += 1
    if step % 50 == 0:
      gap = np.linalg.norm(d.site_xpos[anchor] - d.xpos[carrier])
      print(f"t={d.time:5.1f}s x={d.qpos[0]:+.2f} z={d.qpos[2]:.2f} rope={d.qpos[s_q]:+.2f} m  channel-rope gap={gap*100:.1f} cm")
    if viewer is not None:
      viewer.sync()
      time.sleep(max(0.0, d.time - (time.time() - t0)))
  print("sim2sim done", "(fell)" if d.qpos[2] < 0.4 else "(standing)")


if __name__ == "__main__":
  main()
