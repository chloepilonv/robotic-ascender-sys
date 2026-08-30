# Hackathon plan (12 h) — one robust climbing policy

Goal: a G1 that climbs a 20° slope on the fixed rope without falling, shown in the
MuJoCo viewer + exported ONNX. One slope, one policy. No curriculum.

## Done
- [x] Rope + one-way ascender, wrist on the rope (`task/robot.py`, `RatchetEnv`)
- [x] DR that makes it *robust*: wind 0–15 m/s, friction 0.4–0.9, PD gains ±20 %,
      torso mass ±10 %, CoM ±3 cm, action delay 0–2 steps
- [x] Train / play / export scripts, CPU smoke test, HF Jobs script

## Remaining (in order)
1. [x] `Slope20` trained on HF (`a10g-large`, 3000 it, ~1 h) → `policies/g1_ascender_slope20_v3_2026-08-30_04-35-59.onnx`
2. [x] Curves rising; verified in mjlab env and sim2sim (climbs, no falls)
3. [ ] Record the video: `mjpython -m rl.scripts.sim2sim rl/policies/g1_ascender_slope20_v3_2026-08-30_04-35-59.onnx`

## Not in scope today
Slope curriculum (10→40°), gait-quality rewards, storms (30 m/s), bare ice (0.05),
Jetson deployment. Listed so nobody starts them.
