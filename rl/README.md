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
