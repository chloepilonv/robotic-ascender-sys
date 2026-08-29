# G1 Wind Locomotion — MuJoCo Playground

Unitree G1 locomotion on MuJoCo Playground with a continuous-wind variant of
the G1 joystick task, a fixed-line climbing task (smooth slope + rope
ascender), an interactive WASD viewer, and a vendored PPO trainer wired for
wind-robustness RL.

Everything runs in the conda env `everest` (Python 3.12; `playground==0.2.0`,
`mujoco==3.12.0`, `brax==0.14.2`). Menagerie assets were downloaded once into
site-packages (automatic on first env load); nothing else to install.

## Layout

- `wind_g1/wind_env.py` — `G1JoystickWind` env: upstream G1 joystick task +
  quadratic-drag wind force on the torso (`F = ½·ρ·Cd·A·|v_wind − v_torso|·
  (v_wind − v_torso)`), written to `xfrc_applied` each control step.
  Random impulse pushes are disabled; wind is the perturbation source.
  Registered as `G1JoystickWindFlatTerrain` / `G1JoystickWindRoughTerrain`.
- `wind_g1/climb_env.py` — `G1ClimbAscender` env (below): the G1 climbs a
  smooth slope while its right hand rides a fixed rope through an idealized
  ascender. Registered as `G1ClimbAscender`.
- `wind_g1/viewer.py` — interactive viewer (below).
- `learning/train_jax_ppo.py` — upstream playground v0.2.0 trainer with
  wind-env aliases and a `--num_videos 0` video guard.
- `tests/test_wind_env.py` / `tests/test_climb_env.py` — headless smoke tests
  for the wind and climb envs.
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
/home/mrinal/miniconda3/envs/everest/bin/python -m wind_g1.viewer --policy mels
```

With a trained policy checkpoint instead:

```bash
/home/mrinal/miniconda3/envs/everest/bin/python -m wind_g1.viewer \
    --policy logs/<experiment>/checkpoints/<step> --wind_speed 10
```

For the climbing task instead (default env is the wind task):

```bash
/home/mrinal/miniconda3/envs/everest/bin/python -m wind_g1.viewer \
    --env_name G1ClimbAscender --policy mels --slope_deg 30
```

The wind keys and HUD arrow apply only to wind envs; in climb mode
(`G1ClimbAscender`) they are inert — the climb env has no wind.

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

`policies/mels_g1_joystick.npz` is the Unitree G1 joystick policy from the
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

## Climbing task (`G1ClimbAscender`)

`G1ClimbAscender` (in `wind_g1/climb_env.py`) is the G1 on a smooth slope
with its right hand fixed to a rope:

- **Slope**: the plane floor is tilted `climb_config.slope_deg` (default 30°)
  about the world +y axis, rising toward +x. Explicit foot-pair friction
  (`foot_friction`, default 0.8 — the scene's `<pair>` contacts override
  geom-level friction) lets the robot stand on the incline.
- **Rope**: a static, collision-free cylinder parallel to the slope at the
  height of the rest palm (roughly waist) — visual only.
- **Ascender**: a light carrier body with one slide joint along the rope,
  stiffly `connect`-equality-attached to the `right_palm` site, so the hand
  can never leave the line. Upward sliding stays near-free; a ratchet clamps
  the slide qpos non-decreasing / qvel non-negative every physics substep via
  `jax.lax.scan` (MJX has no `set_mjcb_control` callback), so the hand rides
  up but never slips down, even under full body load.

The appended slide joint is the last qpos/qvel coordinate; obs slices are
trimmed accordingly, so observations still match the upstream G1Joystick
103-dim `state` layout exactly and the mels demo policy loads unmodified.
Reset is deterministic in the `knees_bent` pose with the palm on the carrier
(the stiff grip would yank randomized base poses onto the line); only the
upstream base-velocity randomization is kept. `push_config` is disabled —
velocity impulses are meaningless while gripping the line. The `flat_terrain`
plane floor is required (the rough-terrain hfield cannot be tilted).

`climb_config` keys: `slope_deg` (30.0), `rope_radius` (0.02 m, visual),
`rope_length` (15.0 m upslope), `rope_tail` (0.5 m downslope),
`line_offset_y` (0.0), `carrier_mass` (0.1 kg), `slide_damping` (1.0),
`slide_frictionloss` (0.2), `grip_solref` / `grip_solimp` (equality solver),
`foot_friction` (0.8). Override with `--slope_deg` (viewer) or
`--playground_config_overrides '{"climb_config.slope_deg": 30.0}'`
(trainer).

Training is not wired yet: the trainer imports `wind_g1` only for
`G1JoystickWind*` env names, and `_RL_ENV_ALIASES` covers only the wind envs.
To train `G1ClimbAscender`, widen the import condition and add an alias to
`G1JoystickFlatTerrain` (reuse of the tuned G1Joystick PPO recipe).

## Training (PPO)

JAX is currently CPU-only in this env. Install `jax[cuda12]` before real
training; the RTX 4070 (8 GB) may need `--num_envs 2048–4096` (default 8192)
to avoid OOM. Also set `JAX_DEFAULT_MATMUL_PRECISION=highest` (upstream
recommendation for Ampere+ GPUs).

```bash
/home/mrinal/miniconda3/envs/everest/bin/python learning/train_jax_ppo.py \
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
