"""G1 wind/climb locomotion built on MuJoCo Playground.

Importing this package registers `G1JoystickWindFlatTerrain`,
`G1JoystickWindRoughTerrain`, and `G1ClimbAscender` in the playground registry
so `mujoco_playground.registry.load` resolves them.

Registration needs jax + mujoco_playground, which are only installed on the
training box. `terrain` and `ascender` are plain numpy and are useful on their
own (scene building, viewing, CPU validation), so the playground-backed envs
are imported best-effort: on a machine without jax you still get
`from rl.environment import terrain, ascender`, and `PLAYGROUND_IMPORT_ERROR`
records why the envs are missing.
"""

from rl.environment import ascender  # noqa: F401  numpy only
from rl.environment import terrain  # noqa: F401  numpy only

PLAYGROUND_IMPORT_ERROR: ImportError | None = None

try:
    from rl.environment import climb_env  # noqa: F401  registers the climb env
    from rl.environment import wind_env  # noqa: F401  registers the wind envs
except ImportError as exc:  # jax / mujoco_playground absent
    PLAYGROUND_IMPORT_ERROR = exc
