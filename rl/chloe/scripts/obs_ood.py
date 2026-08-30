"""Per-dimension out-of-distribution check against a policy's observation normaliser.

The exported policy carries the running mean/std mjlab accumulated during training
(`obs_normalization=True`). Those statistics are a record of what the policy actually
saw, so they give a sim-to-real check for free -- offline, no robot, no GPU:

    z = (live_obs - obs_mean) / obs_std        # past ~3 sigma is extrapolation

Reports **per dimension, never a max across all 96**. During the 2026-08-30 gantry
session a single scalar `|q - target|max` produced two confident and completely wrong
diagnoses before per-joint reporting made the answer obvious in one run.

    # what the policy saw (widths only)
    python -m rl.chloe.scripts.obs_ood rl/chloe/policies/<policy>.pt

    # how far a real/synthetic observation falls outside it
    python -m rl.chloe.scripts.obs_ood <policy>.pt --obs telemetry.csv

`--obs` takes a CSV of raw actor observations, one row of 96 floats per control step.
Pure stdlib: reads the checkpoint zip directly, so it runs on a laptop.
"""

from __future__ import annotations

import argparse
import struct
import sys
import zipfile

# Actor observation layout, in order (env_cfg.py actor_terms).
GROUPS = (
  ("base_ang_vel", 3),
  ("projected_gravity", 3),
  ("joint_pos", 29),
  ("joint_vel", 29),
  ("actions", 29),
  ("ascender_pos_b", 3),
)

# MJCF joint order (assets/robots/mujoco/g1_unitree_ascender.xml), used to label the
# 29-wide groups. sim2sim.py reads the same order off `robot.joint_names`.
JOINTS = (
  "left_hip_pitch", "left_hip_roll", "left_hip_yaw", "left_knee",
  "left_ankle_pitch", "left_ankle_roll",
  "right_hip_pitch", "right_hip_roll", "right_hip_yaw", "right_knee",
  "right_ankle_pitch", "right_ankle_roll",
  "waist_yaw", "waist_roll", "waist_pitch",
  "left_shoulder_pitch", "left_shoulder_roll", "left_shoulder_yaw", "left_elbow",
  "left_wrist_roll", "left_wrist_pitch", "left_wrist_yaw",
  "right_shoulder_pitch", "right_shoulder_roll", "right_shoulder_yaw", "right_elbow",
  "right_wrist_roll", "right_wrist_pitch", "right_wrist_yaw",
)
AXES = ("x", "y", "z")


def labels() -> list[str]:
  out = []
  for name, n in GROUPS:
    tags = JOINTS if n == 29 else AXES
    out += [f"{name}.{tags[i]}" for i in range(n)]
  return out


def read_normalizer(path: str) -> tuple[list[float], list[float]]:
  """(mean, std) of the actor observation, straight out of the rsl_rl checkpoint."""
  with zipfile.ZipFile(path) as z:
    root = z.namelist()[0].split("/")[0]
    # data/0,1,2 = obs_normalizer._mean, ._var, ._std of actor_state_dict (pickle order).
    def f32(i: int) -> list[float]:
      b = z.read(f"{root}/data/{i}")
      return list(struct.unpack(f"<{len(b) // 4}f", b))

    mean, std = f32(0), f32(2)
  if len(mean) != sum(n for _, n in GROUPS):
    raise SystemExit(f"expected {sum(n for _, n in GROUPS)} dims, checkpoint has {len(mean)}")
  return mean, std


def main() -> int:
  ap = argparse.ArgumentParser(description=__doc__)
  ap.add_argument("checkpoint", help="a .pt from rl/chloe/policies/")
  ap.add_argument("--obs", help="CSV of raw actor observations, 96 floats per row")
  ap.add_argument("--sigma", type=float, default=3.0, help="OOD threshold (default 3)")
  args = ap.parse_args()

  mean, std = read_normalizer(args.checkpoint)
  names = labels()

  if not args.obs:
    print(f"{'dimension':<28}{'mean':>10}{'std':>10}")
    for n, m, s in sorted(zip(names, mean, std), key=lambda r: r[2]):
      print(f"{n:<28}{m:10.4f}{s:10.4f}")
    print("\nnarrowest dimensions first: those are where a real-robot offset costs the")
    print("most sigma, and where randomisation most needs widening.")
    return 0

  with open(args.obs) as fh:
    rows = [[float(v) for v in line.split(",")] for line in fh if line.strip()]
  if any(len(r) != len(mean) for r in rows):
    raise SystemExit(f"--obs rows must have {len(mean)} columns")

  peak = [max(abs((r[i] - mean[i]) / std[i]) for r in rows) for i in range(len(mean))]
  bad = [(n, z) for n, z in zip(names, peak) if z > args.sigma]

  print(f"{len(rows)} samples, threshold {args.sigma} sigma\n")
  print(f"{'dimension':<28}{'peak |z|':>10}")
  for n, z in sorted(zip(names, peak), key=lambda r: -r[1]):
    flag = "  <-- OOD" if z > args.sigma else ""
    print(f"{n:<28}{z:10.2f}{flag}")

  if bad:
    print(f"\n{len(bad)} dimension(s) outside {args.sigma} sigma. Each one names the")
    print("randomisation that is still too narrow -- widen those, not all of them.")
    return 1
  print(f"\nAll {len(mean)} dimensions within {args.sigma} sigma.")
  return 0


if __name__ == "__main__":
  sys.exit(main())
