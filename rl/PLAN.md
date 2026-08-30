# RL plan — MuJoCo / MJX (no Isaac Lab)

## Stack
- MuJoCo + MJX + mujoco_playground (Brax PPO) → ONNX → Jetson Orin
- Asset: mujoco_menagerie/unitree_g1 (29 DoF) + our sensor sites (2 IMUs, D435i, Mid-360)

## Policies
1. `walk`  — G1JoystickWalkDR (rl/environment/walk_dr_env.py): fine-tunes the
             pretrained mels joystick policy with domain randomization —
             launch-time per-env slopes 0-40 deg, upstream G1 dynamics recipe,
             OU wind (baseline up to 150 kmph, gusts/wander) — on parallel
             vector environments. Slope done via per-env floor tilt in the DR
             wrapper (heightfield plan dropped). Status: implemented + smoke
             tested (2026-08-29); full training run pending.
2. `climb` — `walk` env + right-hand rail: world → slide joint (rope axis) → ball → wrist weld.
             Ratchet = per-step clamp (qvel<0 → 0, qpos = max(qpos, prev)). Reward = slide progress.
3. Runtime FSM on Jetson: `climb` until rail limit → `walk`

## Layout
rl/
  environment/  wind_env.py, walk_dr_env.py (DR walk), climb_env.py
  scripts/      train_jax_ppo.py (vendored playground PPO), viewer.py
  policies/     mels_g1_joystick.npz (pretrained baseline)
  tests/        headless smoke tests (plain scripts)
  (planned: export.py → ONNX)

## Open questions
- ~~Second ascender on left hand?~~ → No. Right hand only (decided 2026-08-29).
- Rope angle randomization range (±15°?)
- Rope stiffness/damping values
- Fine-tune hyperparams: LR and entropy cost for mels-init (current
  defaults are the from-scratch G1 recipe: lr 3e-4, entropy 5e-3).

## Next
- [x] `pip install mujoco mujoco_mjx playground`; run G1 joystick example
- [x] walk DR env + fine-tune wiring (`--init_from_policy mels`,
      `--domain_randomization`, parallel vector envs)
- [ ] Full fine-tune training run on GPU (user-run)
- [ ] `export.py` → ONNX → Jetson
