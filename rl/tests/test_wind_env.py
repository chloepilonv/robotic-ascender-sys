"""Headless smoke test for the wind env (no GUI).

Checks: registry resolution, force magnitude/direction at 15 m/s headwind
on a stationary torso, zero force when disabled, downwind acceleration.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np

import jax
import jax.numpy as jp

import rl.environment  # noqa: F401  registers the envs
from mujoco_playground import registry
from rl.environment import wind_env

# --- Verification 2: registry -------------------------------------------
cfg = registry.get_default_config("G1JoystickWindFlatTerrain")
assert hasattr(cfg, "wind_config"), "default config missing wind_config"
env = registry.load(
    "G1JoystickWindFlatTerrain",
    config_overrides={
        "wind_config.enable": True,
        "wind_config.wind_speed": 15.0,
        "wind_config.wind_heading": 0.0,
    },
)
assert isinstance(env, wind_env.G1JoystickWind), type(env)
print("registry OK: G1JoystickWindFlatTerrain ->", type(env).__name__)

# --- Verification 1: wind-on rollout ------------------------------------
jit_reset = jax.jit(env.reset)
jit_step = jax.jit(env.step)
state = jit_reset(jax.random.PRNGKey(0))

torso_id = env._torso_body_id
qvel0 = np.asarray(state.data.qvel[:2])
action = jp.zeros(env.action_size)

for i in range(50):
  state = jit_step(state, action)

xfrc = np.asarray(state.data.xfrc_applied[torso_id])
linvel = np.asarray(env.get_global_linvel(state.data, "torso"))
rel = np.array([15.0, 0.0]) - linvel[:2]
speed = np.linalg.norm(rel)
expected = 0.5 * 1.225 * 1.2 * 0.5 * speed * rel
print(f"step 50 torso xfrc[:2] = {xfrc[:2]}, expected ~{expected}")
assert np.all(xfrc[2:] == 0.0), "torque components should be zero"
assert abs(xfrc[0]) > 50.0 and xfrc[0] > 0, "force should be large and +x"
qvel50 = np.asarray(state.data.qvel[:2])
print(f"qvel[:2] start {qvel0} -> step50 {qvel50}")
assert qvel50[0] > qvel0[0] + 0.1, "robot should accelerate downwind (+x)"
print(f"reward={float(state.reward):.4f} done={bool(state.done)}")
print("wind-on rollout OK")

# --- Verification 1c: wind-off -------------------------------------------
env_off = registry.load(
    "G1JoystickWindFlatTerrain",
    config_overrides={"wind_config.enable": False},
)
jit_reset = jax.jit(env_off.reset)
jit_step = jax.jit(env_off.step)
state = jit_reset(jax.random.PRNGKey(0))
state = jit_step(state, jp.zeros(env_off.action_size))
xfrc = np.asarray(state.data.xfrc_applied)
assert np.all(xfrc == 0.0), "xfrc must stay zero when wind disabled"
print("wind-off rollout OK: xfrc all zero")

print("ALL SMOKE TESTS PASSED")
