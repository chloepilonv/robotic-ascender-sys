# rl/chloe — mjlab ascender climb (PPO)


Fixed-rope climb with the wrist-mounted ascender, built on
[mjlab](https://github.com/mujocolab/mjlab) (MuJoCo Warp + rsl_rl PPO).
Separate from the JAX/Playground envs above: it needs its own venv.

```bash
uv venv -p 3.11 .venv-mjlab && uv pip install -p .venv-mjlab/bin/python mjlab onnx onnxscript
python assets/robots/mujoco/build.py --fetch        # stock Unitree STLs, once per clone
.venv-mjlab/bin/python rl/chloe/tests/test_ascender_env.py  # CPU smoke test (~1 min)
```

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

## Policies (`rl/policies/`)
| File | What |
|---|---|
| `mels_g1_joystick.npz` | pretrained G1 walker (JAX) — legs of the climb mime |
| `g1_ascender_slope20_SMOKE.{pt,onnx}` | climb policy, **20 iterations only** (pipeline test, does not climb) — same I/O as the real one |
| `g1_ascender_slope20.{pt,onnx}` | the trained climb policy — lands here when the HF job finishes |

Full runs + checkpoints: https://huggingface.co/iteratehack/g1-ascender (org members only).
ONNX I/O: input `obs` float32 [1, 96], output `action` float32 [1, 29] (see `scripts/export_onnx.py` docstring for the obs order).
Load in Python: `onnxruntime.InferenceSession("rl/policies/g1_ascender_slope20_SMOKE.onnx").run(None, {"obs": obs})[0]`
