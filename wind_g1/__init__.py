"""G1 wind-locomotion project built on MuJoCo Playground.

Importing this package registers `G1JoystickWindFlatTerrain` and
`G1JoystickWindRoughTerrain` in the playground registry so
`mujoco_playground.registry.load` resolves them.
"""

from wind_g1 import wind_env  # noqa: F401  registers the envs on import
