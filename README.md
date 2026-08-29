# G1 Wind Locomotion — MuJoCo Playground

Unitree G1 locomotion on MuJoCo Playground with a continuous-wind variant of
the G1 joystick task, an interactive WASD viewer, and a vendored PPO trainer
wired for wind-robustness RL.

Everything runs in the conda env `everest` (Python 3.12; `playground==0.2.0`,
`mujoco==3.12.0`, `brax==0.14.2`). Menagerie assets were downloaded once into
site-packages (automatic on first env load); nothing else to install.

## Layout

- `rl/` — all reinforcement-learning code:
  - `rl/environment/wind_env.py` — `G1JoystickWind` env: upstream G1 joystick
    task + quadratic-drag wind force on the torso (`F = ½·ρ·Cd·A·
    |v_wind − v_torso|·(v_wind − v_torso)`), written to `xfrc_applied` each
    control step. Random impulse pushes are disabled; wind is the
    perturbation source. Registered as `G1JoystickWindFlatTerrain` /
    `G1JoystickWindRoughTerrain`. `rl/environment/climb_env.py` registers the
    fixed-rope `G1ClimbAscender` env the same way.
  - `rl/scripts/viewer.py` — interactive viewer (below).
  - `rl/scripts/train_jax_ppo.py` — upstream playground v0.2.0 trainer with
    wind-env aliases and a `--num_videos 0` video guard.
  - `rl/policies/` — saved policy weights (`mels_g1_joystick.npz` baseline).
  - `rl/tests/` — headless smoke tests (`test_wind_env.py`,
    `test_climb_env.py`, `test_viewer_internals.py`).
- `terrain/` — real Everest terrain for the fixed-rope / ascender task. A
  25 x 15 m patch of the **Lhotse Face between Camp II and Camp III**
  (6907 m, 38.9 deg), from Copernicus GLO-30 + OpenStreetMap route nodes.
  Loaded into MuJoCo as an `hfield`. See `terrain/README.md` — in particular
  the REAL vs SYNTHETIC section before quoting the terrain anywhere.

```bash
python -m terrain.mujoco_scene              # viewer
python -m terrain.mujoco_scene --headless   # physics check
```

## Interactive viewer (WASD + live wind)

From the repo root, with the baseline walking policy:

```bash
/home/mrinal/miniconda3/envs/everest/bin/python -m rl.scripts.viewer --policy mels
```

With a trained policy checkpoint instead:

```bash
/home/mrinal/miniconda3/envs/everest/bin/python -m rl.scripts.viewer \
    --policy logs/<experiment>/checkpoints/<step> --wind_speed 10
```

Without `--policy` the G1 runs zero actions and will sag/fall — expected.
The baseline policy is described in the next section.

Keys (press = set, press again = clear; no key-release events exist in the
passive viewer):

| Key | Action |
|-----|--------|
| `W` / `S` | forward / backward, lin_vel_x = ±1·mult |
| `Q` / `E` | strafe left / right, lin_vel_y = ±0.5·mult |
| `A` / `D` | turn left / right, ang_vel_yaw = ±1·mult |
| `X` | zero all commands |
| `R` | cycle speed multiplier 1× / 2× |
| `↑` / `↓` | wind speed ±2 m/s (clip 0–40) |
| `←` / `→` | wind heading ±15° |
| `0` | wind off |

HUD: red arrow = wind (world frame, above origin), blue arrow = command (at
pelvis; vertical component = yaw). Current command/wind prints to stdout on
every keypress. Hold-to-move semantics would require `pynput` (not installed).

## Baseline walking policy (`--policy mels`)

`rl/policies/mels_g1_joystick.npz` is the Unitree G1 joystick policy from the
official MuJoCo Playground live demo (research.mels.ai), extracted from the
demo's published config: MLP 103→512→256→128→58 with swish activations and
an obs normalizer. Its observation layout matches the playground
G1Joystick `state` obs exactly, so it drops into `G1JoystickWind*` unmodified.
Verified: stands indefinitely, walks forward/backward/strafes at command
(0.75 m/s at cmd 1.0), survives ~8 m/s wind, falls at ~10 m/s sustained
wind — the baseline your wind-robustness training should beat.

The viewer also accepts any brax PPO checkpoint dir from the vendored
trainer via `--policy logs/<exp>/checkpoints/<step>`.

## Wind config

`wind_config` keys (set at launch via `--wind_speed`/`--wind_heading` in the
viewer, or `--playground_config_overrides` in the trainer):
`enable` (bool), `wind_speed` (m/s), `wind_heading` (rad),
`rho` (1.225 kg/m³), `cd_torso` (1.2), `area_torso` (0.5 m²).
Reference: 15 m/s on a stationary G1 ⇒ ~69 N on a 33.3 kg robot.

## Training (PPO)

JAX is currently CPU-only in this env. Install `jax[cuda12]` before real
training; the RTX 4070 (8 GB) may need `--num_envs 2048–4096` (default 8192)
to avoid OOM. Also set `JAX_DEFAULT_MATMUL_PRECISION=highest` (upstream
recommendation for Ampere+ GPUs).

```bash
/home/mrinal/miniconda3/envs/everest/bin/python rl/scripts/train_jax_ppo.py \
  --env_name G1JoystickWindFlatTerrain \
  --playground_config_overrides '{"wind_config.enable": true, "wind_config.wind_speed": 10.0}' \
  --num_timesteps 200_000_000 --num_videos 0 --logdir logs
```

The wind env reuses the tuned G1Joystick PPO recipe (512/256/128 networks,
privileged value obs). Checkpoints land in `logs/<exp>/checkpoints/<step>`;
load them in the viewer with `--policy`.

### Randomized wind during training (not yet implemented)

Static wind only right now. For wind randomization add
`wind_speed_range=[min,max]` / `wind_heading_range` to `wind_config`, sample
in `reset()` into `info["wind"]`, and call `env.use_wind_from_info(True)` —
the step-time wind path already reads from `info["wind"]`.

### Local brax patch note

`brax==0.14.2` calls `jax.device_put_replicated`, removed in JAX 0.11.
The call in `brax/training/agents/ppo/train.py` (~line 756) was replaced
in-place with a `jax.device_put` + stack shim (semantics verified). A brax
or JAX upgrade may obsolete or revert this patch.
