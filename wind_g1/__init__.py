"""G1 wind-locomotion project built on MuJoCo Playground.

Importing this package registers `G1JoystickWindFlatTerrain`,
`G1JoystickWindRoughTerrain`, and `G1ClimbAscender` in the playground
registry so `mujoco_playground.registry.load` resolves them.
"""

from wind_g1 import climb_env  # noqa: F401  registers the climb env on import
from wind_g1 import wind_env  # noqa: F401  registers the wind envs on import
