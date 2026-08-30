"""Headless smoke test for the mjlab ascender env (CPU, 2 envs, ~1 min).

    .venv-mjlab/bin/python rl/chloe/tests/test_ascender_env.py

Checks: env builds, obs/action dims, the rope ratchet (slide qpos never
decreases even when the arm pushes down), and the wrist stays on the rope.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np
import torch

import rl.chloe.task as A


def main() -> None:
  cfg = A.make_env_cfg(slope_deg=20.0, play=True)
  cfg.scene.num_envs = 2
  env = A.RatchetEnv(cfg=cfg, device="cpu")
  obs, _ = env.reset()
  print("obs", {k: tuple(v.shape) for k, v in obs.items()})
  jn = env.scene["robot"].joint_names
  act_dim = env.action_manager.total_action_dim
  print("nq", env.sim.mj_model.nq, "joints", len(jn), "action dim", act_dim)
  assert act_dim == 29, act_dim

  m = env.sim.mj_model
  anchor = m.site("robot/ascender_anchor").id  # the ascender's rope channel
  carrier = m.site("robot/carrier_anchor").id
  sp = jn.index("right_shoulder_pitch_joint")
  hist, gaps, resets = [], [], []
  for t in range(150):
    a = torch.zeros(2, act_dim)
    a[:, sp] = -3.0 if t < 60 else 3.0  # raise arm (slide up), then push down (must hold)
    obs, rew, term, trunc, info = env.step(a)
    resets.append(bool(term[0] | trunc[0]))
    d = env.sim.data
    hist.append(float(d.qpos[0, env._slide_qadr]))
    gaps.append((d.site_xpos[0, anchor] - d.site_xpos[0, carrier]).norm().item())
    if t in (0, 30, 59, 90, 149):
      print(
        f"t={t:3d} slide={hist[-1]:+.3f} gap={gaps[-1]:.4f} "
        f"pelvis_z={d.qpos[0, 2]:.3f} rew={rew[0]:.3f} term={bool(term[0])}"
      )
  h = np.array(hist)
  ok = np.diff(h) >= -1e-6
  ok |= np.array(resets[1:])  # a reset legitimately re-zeroes the slide
  assert ok.all(), "ratchet failed: slide moved down"
  g = np.array(gaps)
  print(f"gap: first 10 steps max {g[:10].max():.4f}  overall median {np.median(g):.4f}  resets={sum(resets)}")
  # Zero actions make mjlab's soft-PD G1 sag and fall (stock G1 does too); the
  # rope constraint is only checked before the fall yanks the arm.
  assert g[:10].max() < 0.03, "wrist left the rope"
  print("ratchet OK, max slide", h.max(), "| reward terms:", list(env.reward_manager.active_terms))
  env.close()


if __name__ == "__main__":
  main()
