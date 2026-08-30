# Hackathon plan (12 h) — one robust climbing policy

Goal: a G1 that climbs a 20° slope on the fixed rope without falling, shown in the
MuJoCo viewer + exported ONNX. One slope, one policy. No curriculum.

## Done
- [x] Rope + one-way ascender, wrist on the rope (`task/robot.py`, `RatchetEnv`)
- [x] DR that makes it *robust*: wind 0–15 m/s, friction 0.4–0.9, PD gains ±20 %,
      torso mass ±10 %, CoM ±3 cm, action delay 0–2 steps
- [x] Train / play / export scripts, CPU smoke test, HF Jobs script

## Remaining (in order)
1. [ ] HF token with *Manage Jobs* → launch `Slope20` on `a10g-large`, 3000 it (~3 h, ~$3)
2. [ ] Check TensorBoard after ~30 min: `ascender_progress` and `uphill_velocity` rising.
       If flat: set `hand_behind` weight to -0.5 and relaunch (once).
3. [ ] Download `model_3000.pt` + `policy.onnx`, record a video with `play_mjlab.py --video`

## Not in scope today
Slope curriculum (10→40°), gait-quality rewards, storms (30 m/s), bare ice (0.05),
Jetson deployment. Listed so nobody starts them.
