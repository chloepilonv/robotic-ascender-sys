# rl/ — reinforcement learning for the G1 project

Everything RL lives here: MuJoCo Playground env registrations, the PPO
trainer, the interactive viewer, saved policy weights, and headless smoke
tests. Run everything from the repo root in the `everest` conda env
(`/home/mrinal/miniconda3/envs/everest/bin/python`).

## Envs

All envs are registered into `mujoco_playground.registry` on import
(`import rl.environment` is enough) and share the G1Joystick obs/action
layout (103-dim `state`, 216-dim `privileged_state`, 29 actions), so the
pretrained mels policy loads into all of them:

- `G1JoystickWalkDR` (`walk_dr_env.py`) — **the fine-tuning env for the
  walking policy**. Upstream G1 joystick task with domain randomization:
  launch-time per-env slopes (default 0–40°, `dr_config.slope_min_deg` /
  `slope_max_deg`), the upstream G1 dynamics recipe (floor/foot friction
  U(0.4, 1.0), frictionloss ×U(0.5, 2), armature ×U(1.0, 1.05), link
  masses ×U(0.9, 1.1), torso mass ±1 kg, qpos0 jitter ±0.05), and
  realistic wind (per-episode baseline up to 150 km/h with smooth
  Ornstein–Uhlenbeck gusts/heading wander; same quadratic drag as
  `wind_env`). Slope-aware internals (reset pose, termination,
  terrain-relative gravity obs). `--domain_randomization` uses the
  registered randomizer.
- `climb_terrain_env.py` — `G1ClimbTerrain`: the **ascender climb task on
  the merged Lhotse terrain** (`climb_scene.py`'s merged model: measured
  Lhotse heightfield patches, draped rope, single-slide ascender carrier
  with the jit-safe ratchet). `terrain_config.patch` selects the patch
  (`A`–`D` measured; `B_flat0`, `B_slope25..50` synthetic curriculum).
  `climb_env.py` is its unregistered machinery base (do not use directly).
- `wind_env.py` — `G1JoystickWindFlatTerrain` / `G1JoystickWindRoughTerrain`:
  upstream G1 joystick task + a quadratic-drag wind force on the torso,
  written to `xfrc_applied` each control step.

`terrain.py` / `ascender.py` are plain numpy (scene building, viewing,
CPU validation) and import without jax; the playground-backed envs import
best-effort (`rl.environment.PLAYGROUND_IMPORT_ERROR` records why on a
machine without jax).

## Training (GPU)

`rl/scripts/train_gpu.py` — PPO with all envs in parallel on the GPU via
MJX/brax (no CPU env loop). Asserts CUDA visibility at startup, caps XLA
memory allocation for a shared laptop GPU, and wraps every flag of the
vendored `train_jax_ppo.py` (env registration, `--init_from_policy` npz
fine-tune seeding, domain randomization, TensorBoard/W&B logging,
checkpoints, rollout videos).

CUDA setup (one-time): `pip install -r requirements.txt` (jax 0.11.1 +
CUDA 12 plugin). The trainer preloads the pip NVIDIA wheels itself — no
`LD_LIBRARY_PATH` needed.

### The standard fine-tune run

Domain-randomized slope + wind walk, fine-tuned from the pretrained mels
policy, W&B logging to `project-yeti/ascender-rl` with eval videos:

```bash
python rl/scripts/train_gpu.py \
  --env_name G1JoystickWalkDR \
  --domain_randomization \
  --init_from_policy mels \
  --num_envs 2048 \
  --num_timesteps 100_000_000 \
  --use_tb --use_wandb \
  --wandb_entity project-yeti --wandb_project ascender-rl \
  --wandb_eval_videos 1
```

Domain-randomization ranges live in `dr_config` and are overridden as
dotted keys:

- slopes 0–15 deg: `{"dr_config.slope_min_deg": 0.0, "dr_config.slope_max_deg": 15.0}`
- max wind 10 m/s: `dr_config.wind_max_speed_kmph` is in **km/h** — 10 m/s
  = `36.0` (gusts go ±`gust_fraction` = ±25% beyond the baseline)
- fixed foot friction 1.0: `"dr_config.friction_range": [1.0, 1.0]`

Full example:

```bash
python rl/scripts/train_gpu.py \
  --env_name G1JoystickWalkDR \
  --domain_randomization --init_from_policy mels \
  --num_envs 2048 \
  --num_timesteps 100_000_000 \
  --use_tb --use_wandb \
  --wandb_entity project-yeti --wandb_project ascender-rl \
  --wandb_eval_videos 1 \
  --playground_config_overrides \
  '{"dr_config.slope_min_deg": 0.0, "dr_config.slope_max_deg": 15.0, "dr_config.wind_max_speed_kmph": 36.0, "dr_config.friction_range": [1.0, 1.0]}'
```

Notes:

- **Intermediate evals are on by default** (`--run_evals true`): every
  `num_timesteps / num_evals` steps (G1 recipe: `num_evals=20`), 128
  dedicated eval envs roll out the deterministic policy; the reward is
  logged (`eval/episode_reward`) and a checkpoint saved at every eval
  point (`logs/<exp>/checkpoints/<step>`).
- `--wandb_eval_videos 1` renders a rollout of the current policy at
  every eval point and logs it to W&B as `eval/video` (env must render,
  i.e. EGL available).
- W&B entity/project: `--wandb_entity project-yeti --wandb_project
  ascender-rl` (defaults: entity = your account, project `mjxrl`).
- VRAM sizing (8 GB laptop GPU): 512 envs fits comfortably; try 2048 for
  throughput. OOM → drop `--num_envs` to 1024 or `--unroll_length` to 10.
- `--init_from_policy mels` resolves to `rl/policies/mels_g1_joystick.npz`
  (or pass a path / any brax-layout npz). Mutually exclusive with
  `--load_checkpoint_path`.

### Terrain climb training

```bash
python rl/scripts/train_gpu.py --env_name G1ClimbTerrain \
  --num_timesteps 100_000_000 --use_tb \
  --wandb_entity project-yeti --wandb_project ascender-rl --wandb_eval_videos 1 \
  --playground_config_overrides '{"terrain_config.patch": "B"}'
```

`terrain_config.patch` selects the terrain: measured Lhotse patches
`A`–`D` (33.7–38.6°), `B_flat0` (flat reference), `B_slope25/30/35/45/50`
(curriculum). Same obs layout as G1Joystick → `--init_from_policy mels`
works.

### Other envs

```bash
python rl/scripts/train_gpu.py --env_name G1JoystickWindFlatTerrain ...   # flat + wind
python rl/scripts/train_gpu.py --env_name G1JoystickWindRoughTerrain ...  # rough + wind
```

## Previewing an env with the base policy

```bash
python rl/scripts/viewer.py --policy mels --env_name G1JoystickWindFlatTerrain
python rl/scripts/viewer.py --policy mels --env_name G1JoystickWalkDR
```

WASD/QE drive the joystick command, `X` stops, `R` toggles 2x speed;
arrow keys set wind on the wind envs (red arrow in-scene), `0` wind off.
`G1JoystickWalkDR` samples wind per episode from `dr_config` (arrow keys
don't apply); `G1ClimbTerrain` has no live knobs — set the patch when
loading. Without `--policy` the robot sags and falls (no policy). First
launch JIT-compiles (~1 min for the climb env) before the window opens.

Trained checkpoints: `python rl/scripts/viewer.py --policy
logs/<exp>/checkpoints`.

The browser harness (teammate's interactive climber) previews the merged
scene with the same policy: `python -m app.harness.runtime --live --world
lhotse_B` (see `app/harness/README.md`).

## Layout

- `environment/` — envs registered into `mujoco_playground.registry` on
  import (`import rl.environment` is enough):
  - `walk_dr_env.py` — `G1JoystickWalkDR` (fine-tuning env; see above)
  - `climb_terrain_env.py` — `G1ClimbTerrain`: the ascender climb task on
    the merged Lhotse terrain (via `climb_scene.build_scene`)
  - `climb_env.py` — unregistered machinery base for the terrain env
  - `wind_env.py` — `G1JoystickWind{Flat,Rough}Terrain`
  - `climb_scene.py`, `terrain.py`, `ascender.py`, `robot.py` — the
    merged-scene builder and its numpy support (terrain patches, rope
    route, carrier); `climb_terrain_env` composes them
  - `walk_policy.py` — numpy reproduction of the mels policy + gait
    clock, used by the scene viewer
- `scripts/` — entry points (run from the repo root):
  - `train_gpu.py` — GPU trainer (see above)
  - `train_jax_ppo.py` — the underlying trainer (same flags; CPU
    fallback if no GPU is visible)
  - `viewer.py` — interactive WASD viewer with live wind:
    `python rl/scripts/viewer.py --policy mels --env_name G1JoystickWindFlatTerrain`
  - `climb_scene.py` — build/inspect/export/view the merged scene
    (`python -m rl.scripts.climb_scene --list`)
- `policies/` — `mels_g1_joystick.npz`, the pretrained baseline
  (`--policy mels` in the viewer, `--init_from_policy mels` for
  fine-tuning). PPO checkpoints land in `logs/<exp>/checkpoints/<step>`
  (repo root, gitignored).
- `tests/` — headless smoke tests, plain scripts:
  `python rl/tests/test_walk_dr_env.py`,
  `python rl/tests/test_climb_scene.py`,
  `python rl/tests/test_wind_env.py`,
  `python rl/tests/test_viewer_internals.py [CKPT_DIR]`.

## Importing the envs

```python
import rl.environment  # registers the envs
from mujoco_playground import registry
env = registry.load("G1JoystickWalkDR")
```

`rl/scripts/train_jax_ppo.py` does this bootstrap itself when `--env_name`
starts with `G1JoystickWind`, `G1JoystickWalkDR`, or `G1Climb`.
