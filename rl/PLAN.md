# RL plan — MuJoCo / MJX (no Isaac Lab)

## Stack
- MuJoCo + MJX + mujoco_playground (Brax PPO) → ONNX → Jetson Orin
- Asset: mujoco_menagerie/unitree_g1 (29 DoF) + our sensor sites (2 IMUs, D435i, Mid-360)

## Policies
1. `walk`  — Playground G1 joystick env + curriculum: slope (heightfield), friction 0.1–1.0 (ice), wind (random torso force)
2. `climb` — `walk` env + right-hand rail: world → slide joint (rope axis) → ball → wrist weld.
             Ratchet = per-step clamp (qvel<0 → 0, qpos = max(qpos, prev)). Reward = slide progress.
3. Runtime FSM on Jetson: `climb` until rail limit → `walk`

## Layout
rl/
  assets/     g1 MJCF + rope rail xml
  envs/       walk.py, climb.py (Playground-style envs)
  train.py    PPO entry
  export.py   → ONNX

## Open questions
- ~~Second ascender on left hand?~~ → No. Right hand only (decided 2026-08-29).
- Rope angle randomization range (±15°?)
- Rope stiffness/damping values

## Next
- [ ] `pip install mujoco mujoco_mjx playground`; run G1 joystick example
