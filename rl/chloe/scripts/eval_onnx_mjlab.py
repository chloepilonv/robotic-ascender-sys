"""Run an exported ONNX policy inside the mjlab training env (CPU) — the reference
for sim2sim: if it climbs here but not in sim2sim.py, the deploy loop is wrong;
if it fails in both, the model/policy is the problem.

    .venv-mjlab/bin/python -m rl.chloe.scripts.eval_onnx_mjlab rl/chloe/policies/g1_ascender_slope20_v1.onnx --slope 20
"""

import argparse

import numpy as np
import onnxruntime as ort
import torch

import rl.chloe.task as A


def main() -> None:
  p = argparse.ArgumentParser()
  p.add_argument("onnx")
  p.add_argument("--slope", type=float, default=20.0)
  p.add_argument("--envs", type=int, default=4)
  p.add_argument("--seconds", type=float, default=12.0)
  p.add_argument("--no-dr", action="store_true", help="disable wind / friction / mass DR")
  a = p.parse_args()

  cfg = A.make_env_cfg(a.slope, play=True)
  cfg.scene.num_envs = a.envs
  if a.no_dr:
    for k in ("wind", "ice_friction", "motor_strength", "torso_mass", "torso_com"):
      cfg.events.pop(k, None)
  env = A.RatchetEnv(cfg=cfg, device="cpu")
  sess = ort.InferenceSession(a.onnx)
  obs, _ = env.reset()
  robot = env.scene["robot"]
  slide = robot.joint_names.index(A.env_cfg.R.SLIDE_JOINT)
  x0 = robot.data.root_link_pos_w[:, 0].clone()
  steps = int(a.seconds / env.step_dt)
  falls = torch.zeros(a.envs, dtype=torch.bool)
  for t in range(steps):
    act = sess.run(None, {"obs": obs["actor"].numpy().astype(np.float32)})[0]
    obs, rew, term, trunc, _ = env.step(torch.as_tensor(act))
    falls |= term
    if t % int(2.0 / env.step_dt) == 0:
      x = robot.data.root_link_pos_w[:, 0] - x0
      print(f"t={t*env.step_dt:5.1f}s  dx={np.round(x.numpy(), 2).tolist()}  rope={np.round(robot.data.joint_pos[:, slide].numpy(), 2).tolist()}  z={np.round(robot.data.root_link_pos_w[:, 2].numpy(), 2).tolist()}  fell={falls.tolist()}")
  env.close()


if __name__ == "__main__":
  main()
