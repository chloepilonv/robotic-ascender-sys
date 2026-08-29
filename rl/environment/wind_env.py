"""Wind-enabled G1 joystick task for MuJoCo Playground.

Subclass of the upstream `Joystick` task with a continuous quadratic-drag
wind force applied to the torso:

    F_xy = 0.5 * rho * Cd * A * |v_wind - v_torso| * (v_wind - v_torso)

The force is written into `data.xfrc_applied` each control step and acts
across all physics substeps (mjx_env.step does not clear xfrc_applied).
Random velocity-impulse pushes from the upstream `push_config` are disabled
in this env; wind is the perturbation source.
"""

import functools
import math

import jax
import jax.numpy as jp
from ml_collections import config_dict

from mujoco_playground._src import locomotion
from mujoco_playground._src.locomotion.g1 import joystick as g1_joystick


def default_config() -> config_dict.ConfigDict:
  """Upstream G1 joystick config plus `wind_config`."""
  cfg = g1_joystick.default_config()
  cfg.wind_config = config_dict.create(
      enable=False,
      wind_speed=0.0,  # m/s, magnitude of wind velocity.
      wind_heading=0.0,  # radians, world-frame XY direction.
      # Quadratic-drag coefficients.
      rho=1.225,  # air density kg/m^3.
      cd_torso=1.2,  # torso drag coefficient.
      area_torso=0.5,  # torso frontal area m^2.
  )
  return cfg


class G1JoystickWind(g1_joystick.Joystick):
  """G1 joystick task with continuous wind forces on the torso."""

  def __init__(
      self,
      task: str = "flat_terrain",
      config: config_dict.ConfigDict | None = None,
      config_overrides: dict | None = None,
  ):
    config = config or default_config()
    # Wind replaces the upstream random velocity-impulse pushes.
    config.push_config.enable = False
    super().__init__(
        task=task, config=config, config_overrides=config_overrides
    )
    wc = self._config.wind_config
    self._wind_enable = bool(wc.enable)
    self._drag_coeff_torso = (
        0.5 * float(wc.rho) * float(wc.cd_torso) * float(wc.area_torso)
    )
    self._static_wind = jp.array([
        float(wc.wind_speed) * math.cos(float(wc.wind_heading)),
        float(wc.wind_speed) * math.sin(float(wc.wind_heading)),
    ])
    # Interactive mode: viewer updates wind host-side via info["wind"].
    self._wind_from_info = False

  def set_wind(self, speed: float, heading: float) -> None:
    """Host-side wind update for interactive use (viewer)."""
    self._static_wind = jp.array([
        speed * math.cos(heading),
        speed * math.sin(heading),
    ])

  def use_wind_from_info(self, enabled: bool) -> None:
    """Read wind from `info["wind"]` each step (interactive mode)."""
    self._wind_from_info = bool(enabled)

  def _wind_velocity(self, info: dict) -> jax.Array | None:
    """(2,) world-frame XY wind velocity, or None if wind disabled."""
    if not self._wind_enable:
      return None
    if self._wind_from_info:
      return info["wind"]
    return self._static_wind

  def reset(self, rng: jax.Array):
    state = super().reset(rng)
    # Stable info signature across modes: always carry the wind vector.
    info = {**state.info, "wind": self._static_wind}
    return state.replace(info=info)

  def step(self, state, action):
    wind = self._wind_velocity(state.info)
    if wind is not None:  # host-side check, fixed at trace time.
      torso_linvel = self.get_global_linvel(state.data, "torso")
      rel = wind - torso_linvel[:2]  # v_wind - v_torso.
      speed = jp.linalg.norm(rel)
      force_xy = self._drag_coeff_torso * speed * rel
      xfrc = state.data.xfrc_applied
      xfrc = xfrc.at[self._torso_body_id, :2].set(force_xy)
      xfrc = xfrc.at[self._torso_body_id, 2:].set(jp.zeros(4))
      state = state.replace(data=state.data.replace(xfrc_applied=xfrc))
    return super().step(state, action)


# Registered on import; see rl/environment/__init__.py.
locomotion.register_environment(
    "G1JoystickWindFlatTerrain",
    functools.partial(G1JoystickWind, task="flat_terrain"),
    default_config,
)
locomotion.register_environment(
    "G1JoystickWindRoughTerrain",
    functools.partial(G1JoystickWind, task="rough_terrain"),
    default_config,
)
