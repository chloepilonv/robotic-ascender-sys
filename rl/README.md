# rl/ — reinforcement learning for the G1 project

Everything RL lives here: MuJoCo Playground env registrations, the PPO
trainer, the interactive viewer, saved policy weights, and headless smoke
tests. Run everything from the repo root in the `everest` conda env
(`/home/mrinal/miniconda3/envs/everest/bin/python`).

## Layout

- `environment/` — envs registered into `mujoco_playground.registry` on
  import (`import rl.environment` is enough):
  - `wind_env.py` — `G1JoystickWind`: upstream G1 joystick task + a
    quadratic-drag wind force on the torso, written to `xfrc_applied` each
    control step. Registered as `G1JoystickWindFlatTerrain` /
    `G1JoystickWindRoughTerrain`.
  - `climb_env.py` — `G1ClimbAscender`: fixed-rope / ascender climb task on
    the Lhotse Face terrain (see `assets/environments/lhotse_face/README.md`).
- `scripts/` — entry points (run as modules from the repo root):
  - `viewer.py` — interactive WASD viewer with live wind:
    `python -m rl.scripts.viewer --policy mels`
  - `train_jax_ppo.py` — vendored playground v0.2.0 PPO trainer:
    `python rl/scripts/train_jax_ppo.py --env_name G1JoystickWindFlatTerrain ...`
- `policies/` — saved policy weights. `mels_g1_joystick.npz` is the
  pretrained baseline (`--policy mels` in the viewer); brax PPO checkpoints
  from training land in `logs/<exp>/checkpoints/<step>` (repo root, gitignored).
- `tests/` — headless smoke tests, plain scripts (no pytest needed):
  `python rl/tests/test_wind_env.py`, `test_climb_env.py`,
  `test_viewer_internals.py [CKPT_DIR]`.

## Importing the envs

`rl/environment/__init__.py` registers `G1JoystickWindFlatTerrain`,
`G1JoystickWindRoughTerrain`, and `G1ClimbAscender` in the playground
registry:

```python
import rl.environment  # registers the envs
from mujoco_playground import registry
env = registry.load("G1JoystickWindFlatTerrain")
```

`rl/scripts/train_jax_ppo.py` does this bootstrap itself when
`--env_name` starts with `G1JoystickWind`.

## mjlab ascender climb (PPO) — `rl/mjlab_tasks/ascender/`

Fixed-rope climb with the wrist-mounted ascender, built on
[mjlab](https://github.com/mujocolab/mjlab) (MuJoCo Warp + rsl_rl PPO).
Separate from the JAX/Playground envs above: it needs its own venv.

```bash
uv venv -p 3.11 .venv-mjlab && uv pip install -p .venv-mjlab/bin/python mjlab onnx onnxscript
python assets/robots/mujoco/build.py --fetch        # stock Unitree STLs, once per clone
.venv-mjlab/bin/python rl/tests/test_ascender_env.py  # CPU smoke test (~1 min)
```

Files:
- `robot.py` — `assets/robots/mujoco/g1_unitree_ascender.xml` as an mjlab entity, plus a
  `rope` (visual cylinder along world +x) and an `ascender_carrier` body with one slide
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
.venv-mjlab/bin/python -m rl.scripts.train_mjlab_ppo Himalayas-Ascender-Slope10-G1
.venv-mjlab/bin/python -m rl.scripts.train_mjlab_ppo Himalayas-Ascender-Slope20-G1 \
    --agent.resume --agent.load-run logs/rsl_rl/g1_ascender_slope10/<run>   # curriculum
.venv-mjlab/bin/python -m rl.scripts.play_mjlab Himalayas-Ascender-Slope20-G1 \
    --checkpoint-file logs/rsl_rl/g1_ascender_slope20/<run>/model_5000.pt
.venv-mjlab/bin/python -m rl.scripts.export_onnx Himalayas-Ascender-Slope20-G1 \
    logs/rsl_rl/g1_ascender_slope20/<run>/model_5000.pt policy.onnx
```

Policy I/O (50 Hz): obs = ang_vel(3) + projected_gravity(3) + joint_pos(29) +
joint_vel(29) + last_action(29) + ascender_pos_in_pelvis_frame(3) = 96;
action = 29 joint-position offsets (`default + action * G1_ACTION_SCALE`).
Everything in the obs is available on the real G1 (IMU, encoders, wrist FK).
