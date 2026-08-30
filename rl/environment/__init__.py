"""G1 wind/climb locomotion built on MuJoCo Playground.

Importing this package registers `G1JoystickWindFlatTerrain`,
`G1JoystickWindRoughTerrain`, `G1JoystickWalkDR`, and `G1ClimbTerrain` in the
playground registry so `mujoco_playground.registry.load` resolves them.
`G1JoystickWalkDR` also gets a domain randomizer registered in
`locomotion._randomizer` for `--domain_randomization` training.

Registration needs jax + mujoco_playground, which are only installed on the
training box. `terrain` and `ascender` are plain numpy and are useful on their
own (scene building, viewing, CPU validation), so the playground-backed envs
are imported best-effort: on a machine without jax you still get
`from rl.environment import terrain, ascender`, and `PLAYGROUND_IMPORT_ERROR`
records why the envs are missing.
"""

from rl.environment import ascender  # noqa: F401  numpy only
from rl.environment import terrain  # noqa: F401  numpy only

import functools

from mujoco_playground._src import locomotion

PLAYGROUND_IMPORT_ERROR: ImportError | None = None

try:
    from rl.environment import climb_terrain_env  # noqa: F401
    from rl.environment import walk_dr_env  # noqa: F401  registers the DR walk env on import
    from rl.environment import wind_env  # noqa: F401  registers the wind envs

    climb_terrain_env.register()  # registers G1ClimbTerrain

    # Domain randomizer for --domain_randomization training of G1JoystickWalkDR.
    # The partial pins the DEFAULT dr_config; per-run `--config_overrides` on
    # dr_config fields still apply because the training script's DR path
    # re-binds the randomizer from the LOADED env config (see train_jax_ppo).
    locomotion._randomizer["G1JoystickWalkDR"] = functools.partial(
        walk_dr_env.domain_randomize, dr_cfg=walk_dr_env.default_config().dr_config
    )
except ImportError as exc:  # jax / mujoco_playground absent
    PLAYGROUND_IMPORT_ERROR = exc
