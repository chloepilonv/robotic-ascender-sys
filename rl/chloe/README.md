# rl/chloe — mjlab ascender climb (PPO)

## What we did (2026-08-29/30, hackathon)

Goal: a Unitree G1 that climbs a Himalayan slope on a fixed rope with the ascender on its
right wrist — walk, push the ascender up, walk, push — and stays up in wind and on ice.

1. **The ascender on the rope, as a real mechanism** — `assets/robots/mujoco/rope_rail.py`
   (shared, plain MuJoCo, see `ROPE_ASCENDER_ALIGNMENT.md` there). The rope passes through the
   tool's channel (measured on the mesh); the tool is welded to a carriage that has ONE prismatic
   joint along the rope; the cam = the joint's lower limit follows the highest point reached, so
   it never goes back down. Rigid to 0.1 mm, verified by `rope_rail_check.py`.
2. **An RL task on top of it** (`task/`): mjlab (MuJoCo Warp, 4096 robots on one GPU) + rsl_rl PPO.
   Observations = what the real G1 measures (IMU, encoders, last action, wrist position from FK).
   Rewards: go uphill, push the ascender up, stay upright, stay on one side of the rope, keep the
   ascender ahead of the hips. Slope = tilted gravity (one task per slope, 0/10/20/30/40°).
3. **Domain randomisation for robustness and sim-to-real**: wind 0–15 m/s random heading,
   foot friction 0.4–0.9 (snow → crampons), PD gains ±20 %, torso mass ±10 %, CoM ±3 cm,
   action delay 0–2 steps.
4. **Training on HF Jobs** (`scripts/hf_job.sh`): clone → install → train → export ONNX → upload to
   `iteratehack/g1-ascender`. Final run (`Slope20`, v3, 3000 it, ~1 h on an A10G): full-length
   episodes, uphill speed at target, ascender progressing; verified in plain MuJoCo (sim2sim).
5. **Deployment path**: `scripts/export_onnx.py` (obs → 29 joint targets, 50 Hz) and
   `scripts/sim2sim.py` (runs the ONNX in plain CPU MuJoCo with hand-written obs/PD/ratchet — the
   same loop the Jetson will run).
6. **Climb mime for the real robot today** (`deterministic/`): RL walking policy + scripted
   right-arm reach/pull cycle, MuJoCo twin and Unitree-SDK runner (arm_sdk + LocoClient).

What is *not* done: per-env slope, gait-quality rewards, hardware deployment of the climb policy.
See `ROADMAP.md`.


Files:
- `robot.py` — `assets/robots/mujoco/g1_unitree_ascender.xml` as an mjlab entity, plus a
  `rope` (visual cylinder along world +x) and an `rope_carriage` body with one slide
  joint, welded (site connect) to the wrist origin — the rope axis per `assets/ascender/MOUNT.md`.
- `env_cfg.py` — `RatchetEnv` (after every physics substep the slide velocity is clamped
  ≥ 0: the cam), observations / rewards / terminations / domain randomisation, PPO config.
- `mdp.py` — task terms: `ascender_pos_b` obs, `uphill_velocity`, `ascender_progress`,
  `rope_side`, `hand_behind_pelvis` rewards, `wind_on_torso` event.
- `__init__.py` — registers `Himalayas-Ascender-Slope{0,10,20,30,40}-G1`.

Slope is done by **tilting gravity** (world +x = uphill), not the floor, so the rope is a
plain +x line. Gravity is a global MuJoCo option, so slope is fixed per run; wind
(0–30 m/s, random heading, per episode) and foot friction (0.05 ice … 0.9 crampons,
per env) are randomised inside a run.

Train / play / export (GPU box; `--gpu-ids None` for a CPU dry run):

```bash
.venv-mjlab/bin/python -m rl.chloe.scripts.train_mjlab_ppo Himalayas-Ascender-Slope10-G1
.venv-mjlab/bin/python -m rl.chloe.scripts.train_mjlab_ppo Himalayas-Ascender-Slope20-G1 \
    --agent.resume --agent.load-run logs/rsl_rl/g1_ascender_slope10/<run>   # curriculum
.venv-mjlab/bin/python -m rl.chloe.scripts.play_mjlab Himalayas-Ascender-Slope20-G1 \
    --checkpoint-file logs/rsl_rl/g1_ascender_slope20/<run>/model_5000.pt
.venv-mjlab/bin/python -m rl.chloe.scripts.export_onnx Himalayas-Ascender-Slope20-G1 \
    logs/rsl_rl/g1_ascender_slope20/<run>/model_5000.pt policy.onnx
```

Policy I/O (50 Hz): obs = ang_vel(3) + projected_gravity(3) + joint_pos(29) +
joint_vel(29) + last_action(29) + ascender_pos_in_pelvis_frame(3) = 96;
action = 29 joint-position offsets (`default + action * G1_ACTION_SCALE`).
Everything in the obs is available on the real G1 (IMU, encoders, wrist FK).

Next steps: see `ROADMAP.md`.

## Policies (`rl/chloe/policies/`) — read this before using one

Each policy = `.onnx` (deploy / sim2sim, 96 obs → 29 joint targets at 50 Hz) + `.pt` (resume training).

| File | Trained | Rope model it was trained on | Behaviour | Use it for |
|---|---|---|---|---|
| `g1_ascender_slope20_SMOKE` | 20 iterations | old | random, falls | plumbing tests only |
| `g1_ascender_slope20_v1` | 3000 iterations | **old** (rope at the wrist joint, soft attachment) | climbed in its own world; **falls on the fixed rope** | record only — do not demo |
| **`g1_ascender_slope20`** (v3) | 3000 iterations | **final** = `assets/robots/mujoco/rope_rail.py` | climbs: ~0.3 m/s uphill, ascender pushed 3–4 m in 10 s, no falls (4/4 envs with wind+ice DR); sim2sim +4.4 m in 12 s, standing | **the demo policy** → `sim2sim.py`, deployment |

**Rule: a policy is only valid with the rope model it was trained on.** The network's inputs (wrist
position, joint angles) change meaning when the rope/anchor moves, so any change to `rope_rail.py`
(see `assets/robots/mujoco/ROPE_ASCENDER_ALIGNMENT.md`) requires retraining. v2 was trained on an
intermediate rope and crashed; nothing kept.

How to check a policy: `python -m rl.chloe.scripts.eval_onnx_mjlab <policy.onnx>` (inside the training
env, the reference) and `mjpython -m rl.chloe.scripts.sim2sim <policy.onnx>` (plain MuJoCo, like the
Jetson). If both fall the policy is wrong; if only sim2sim falls the deploy loop is wrong.

Also here: `rl/policies/mels_g1_joystick.npz` — colleagues' pretrained G1 walker (JAX), legs of the
climb mime (`deterministic/`). Full runs + all checkpoints: https://huggingface.co/iteratehack/g1-ascender
(org members). ONNX I/O and obs order: `scripts/export_onnx.py` docstring.
