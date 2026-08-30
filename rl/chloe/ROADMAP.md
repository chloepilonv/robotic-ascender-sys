# Roadmap — ascender climb policy

Status legend: [x] done  [ ] todo  (~time = on the GPU box)

## v0 — make it climb (now)
- [x] Rope + one-way ascender (`rope_slide` + ratchet), wrist on the rope
- [x] DR: wind 0–15 m/s, friction 0.4–0.9, slope per run (10 → 20 → 30 → 40 by resume)
- [x] Sim-to-real DR: PD gains ±20 %, torso mass ±10 %, CoM ±3 cm, action delay 0–2 steps
- [ ] Train Slope10 on HF Jobs (a10g-large, ~3 h) — needs a token with *Manage Jobs*
- [ ] Watch `ascender_progress` and `uphill_velocity` rise in TensorBoard; if flat after 500 it, lower `hand_behind` / `rope_side` weights

## v1 — robustness (after v0 climbs)
- [ ] Widen DR: wind to 25 m/s, friction down to 0.2, then Slope20/30/40 by resume (~3 h each)
- [ ] Gait quality: feet-slip / air-time rewards (need a foot contact sensor cfg, see mjlab velocity task)
- [ ] Rope cam friction random 1–6 N (`dr.dof_frictionloss` on `rope_slide`)
- [ ] Observation noise on `ascender_pos_b` (wrist FK error ~1 cm)

## v2 — sim-to-real
- [ ] Per-joint torque limits ±10 % (`dr.effort_limits`)
- [ ] IMU bias / gravity-vector noise; joint-velocity filtering as on the SDK
- [ ] Export ONNX → run on Jetson at 50 Hz (`rl/chloe/scripts/export_onnx.py`) and compare obs order with the SDK
- [ ] FSM on the robot: `climb` (this policy) ↔ `walk`, gated by `engaged` from the ascender (ELECTRONICS.md)

## Open questions
- Rope tilt / lateral offset randomisation (rope is not exactly aligned with the slope in reality)
- Per-env slope (needs sloped terrain + rope per env) vs. the current per-run gravity tilt
- Second ascender / left hand: decided **no** (rl/PLAN.md)
