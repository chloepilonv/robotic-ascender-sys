"""Render a trained ascender policy to an mp4, headless — no display, no viewer.

`play_mjlab.py` opens an interactive viewer, which blocks forever on a machine
with no display (it falls back to the viser web server), so it cannot be used to
produce a file in a batch job. This does the rollout itself and writes frames.

    python -m rl.chloe.scripts.render_video Himalayas-Ascender-Slope20-G1 \
        rl/chloe/policies/g1_ascender_slope20_v3_2026-08-30_04-35-59.pt climb.mp4 \
        --seconds 12 --num-envs 1

Needs a CUDA GPU (mjlab pulls mujoco-warp); set `MUJOCO_GL=egl` when there is no
display. The camera tracks the torso, per `env_cfg.py`'s ViewerConfig.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict

import mediapy
import numpy as np
import torch

from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg

import rl.chloe.task as ascender

CONTROL_HZ = 50


def main() -> None:
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument("task", help="e.g. Himalayas-Ascender-Slope20-G1")
  p.add_argument("checkpoint", help="a .pt from rl/chloe/policies/ or a run dir")
  p.add_argument("out", nargs="?", default="climb.mp4", help="output mp4")
  p.add_argument("--seconds", type=float, default=12.0)
  p.add_argument("--num-envs", type=int, default=1)
  p.add_argument("--width", type=int, default=960)
  p.add_argument("--height", type=int, default=540)
  p.add_argument("--device", default="cuda:0")
  p.add_argument(
    "--corrupt",
    action="store_true",
    help="keep observation noise on (play mode disables it). Use this to see the "
    "policy under the randomisation it was trained against, not a clean one.",
  )
  args = p.parse_args()

  env_cfg = load_env_cfg(args.task, play=not args.corrupt)
  env_cfg.scene.num_envs = args.num_envs
  env_cfg.viewer.width = args.width
  env_cfg.viewer.height = args.height

  env = ascender.RatchetEnv(cfg=env_cfg, device=args.device, render_mode="rgb_array")
  wrapped = RslRlVecEnvWrapper(env)
  runner = MjlabOnPolicyRunner(
    wrapped, asdict(load_rl_cfg(args.task)), device=args.device
  )
  runner.load(args.checkpoint, load_cfg={"actor": True}, strict=True)
  policy = runner.get_inference_policy(device=args.device)

  obs = wrapped.get_observations()
  frames, n = [], int(args.seconds * CONTROL_HZ)
  with torch.inference_mode():
    for i in range(n):
      obs, _, _, _ = wrapped.step(policy(obs))
      frame = env.render()
      if frame is not None:
        frames.append(np.asarray(frame))
      if (i + 1) % (CONTROL_HZ * 2) == 0:
        print(f"  {(i + 1) / CONTROL_HZ:.0f}s / {args.seconds:.0f}s", flush=True)

  env.close()
  if not frames:
    raise SystemExit("no frames rendered — is MUJOCO_GL set (egl when headless)?")
  mediapy.write_video(args.out, frames, fps=CONTROL_HZ)
  print(f"wrote {args.out}  {len(frames)} frames  {frames[0].shape[1]}x{frames[0].shape[0]}")


if __name__ == "__main__":
  main()
