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
  - `walk_dr_env.py` — `G1JoystickWalkDR`: the fine-tuning env for the
    walking policy — upstream G1 joystick task with domain
    randomization (launch-time per-env slopes 0-40 deg + upstream
    dynamics recipe) and realistic wind (per-episode baseline up to
    150 kmph with smooth OU gusts/heading wander; same quadratic drag
    as `wind_env`). Includes `domain_randomize` registered for
    `--domain_randomization` training. See "Fine-tuning the walking
    policy" below.
  - `climb_env.py` — `G1ClimbAscender`: fixed-rope / ascender climb task on
    the Lhotse Face terrain (see `terrain/README.md`).
- `scripts/` — entry points (run as modules from the repo root):
  - `viewer.py` — interactive WASD viewer with live wind:
    `python -m rl.scripts.viewer --policy mels`
  - `train_jax_ppo.py` — vendored playground v0.2.0 PPO trainer:
    `python rl/scripts/train_jax_ppo.py --env_name G1JoystickWalkDR ...`
- `policies/` — saved policy weights. `mels_g1_joystick.npz` is the
  pretrained baseline (`--policy mels` in the viewer, and the
  `--init_from_policy mels` fine-tune seed); brax PPO checkpoints
  from training land in `logs/<exp>/checkpoints/<step>` (repo root, gitignored).
- `tests/` — headless smoke tests, plain scripts (no pytest needed):
  `python rl/tests/test_wind_env.py`, `test_walk_dr_env.py`,
  `test_climb_env.py`, `test_viewer_internals.py [CKPT_DIR]`.

## Importing the envs

`rl/environment/__init__.py` registers `G1JoystickWindFlatTerrain`,
`G1JoystickWindRoughTerrain`, `G1JoystickWalkDR`, and `G1ClimbAscender`
in the playground registry (plus the domain randomizer for
`G1JoystickWalkDR`):

```python
import rl.environment  # registers the envs
from mujoco_playground import registry
env = registry.load("G1JoystickWalkDR")
```

`rl/scripts/train_jax_ppo.py` does this bootstrap itself when
`--env_name` starts with `G1JoystickWind`, `G1JoystickWalkDR`, or
`G1Climb`.

## Fine-tuning the walking policy

`G1JoystickWalkDR` fine-tunes the pretrained walking policy under
domain randomization, on parallel vector environments:

- **Slope (launch time, per parallel env)**: each env's floor is
  permanently tilted by a sample from U(0, 40 deg) — baked into
  vectorized mjx models by the playground DR wrapper, so thousands of
  envs with different slopes train simultaneously.
- **Dynamics (launch time)**: the upstream G1 recipe — floor/foot
  friction U(0.4, 1.0), frictionloss x U(0.5, 2), armature x
  U(1.0, 1.05), link masses x U(0.9, 1.1), torso mass +-1 kg, qpos0
  jitter +-0.05.
- **Wind (per episode + in-episode)**: at reset a baseline speed
  U(0, 150 kmph) and heading U(0, 360 deg) are drawn; during the
  episode an Ornstein-Uhlenbeck process makes the heading wander
  (sigma 12 deg, ~8 s persistence) and the speed gust/dip (+-25%,
  ~3 s persistence) around that baseline, pushing the torso via
  quadratic drag. The baseline persists across auto-resets (a fixed
  "weather" per parallel env).
- **Slope-aware env**: reset places the robot standing on the slope;
  termination and the orientation reward use the terrain normal; the
  `state` obs gravity block is terrain-relative (upright-on-slope
  reads like upright-on-flat), while the privileged obs keeps the
  true world tilt for the value function. Obs layout is unchanged
  (103/216), so the pretrained networks load as-is.

All ranges live in `dr_config` and are overridable:
`--config_overrides '{"dr_config.slope_max_deg": 20.0}'`.

Fine-tune run (from the repo root, GPU):

```bash
python rl/scripts/train_jax_ppo.py \
  --env_name G1JoystickWalkDR \
  --domain_randomization \
  --init_from_policy mels \
  --suffix wind-slope-ft
```
`--init_from_policy mels` seeds the policy weights and the observation
normalizer from `policies/mels_g1_joystick.npz` (verified bit-exact
against the npz forward pass); the value network is freshly
initialized (the export has no value head, and the value function
must be re-learned for the randomized domain anyway). Omit the flag
to train from scratch; use `--load_checkpoint_path` to resume a brax
checkpoint instead (mutually exclusive).
