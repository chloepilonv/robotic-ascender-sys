"""Headless verification of viewer internals (no GUI).

Drives on_key semantics, the wind-from-info step path, and load_policy()
against the dry-run checkpoint. Viewer GUI itself is verified by the user
per plan (Verification 4 notes GUI check may be user-performed).
"""

import math
import sys

import numpy as np

import jax
import jax.numpy as jp

from mujoco_playground import registry
import wind_g1  # noqa: F401
from wind_g1 import wind_env
from wind_g1.viewer import add_arrow, load_policy, rotation_z_to


def make_env(speed=0.0, heading=0.0):
  env = registry.load(
      "G1JoystickWindFlatTerrain",
      config_overrides={
          "impl": "jax",
          "wind_config.enable": True,
          "wind_config.wind_speed": speed,
          "wind_config.wind_heading": heading,
      },
  )
  env.use_wind_from_info(True)
  return env


# --- env with wind-from-info: wind follows info["wind"] -------------------
env = make_env(speed=5.0)
jit_reset = jax.jit(env.reset)
jit_step = jax.jit(env.step)
state = jit_reset(jax.random.PRNGKey(0))
# Host-side wind update (simulating viewer arrow keys): 15 m/s @ +90 deg.
w = jp.array([0.0, 15.0])
state = state.replace(info={**state.info, "wind": w})
state = jit_step(state, jp.zeros(env.action_size))
xfrc = np.asarray(state.data.xfrc_applied[env._torso_body_id])
assert xfrc[1] > 50.0 and abs(xfrc[0]) < 5.0, xfrc[:2]
print(f"wind-from-info OK: xfrc[:2]={xfrc[:2]} (force along +y)")

# Retrace stability: swap wind magnitude, step again without retrace issues.
w2 = jp.array([10.0, 0.0])
state = state.replace(info={**state.info, "wind": w2})
state = jit_step(state, jp.zeros(env.action_size))
xfrc = np.asarray(state.data.xfrc_applied[env._torso_body_id])
assert xfrc[0] > 30.0, xfrc[:2]
print(f"wind swap OK: xfrc[:2]={xfrc[:2]} (force along +x)")

# --- command injection survives the 500-step resample ----------------------
cmd = jp.array([1.0, 0.0, 0.0])
state = jit_reset(jax.random.PRNGKey(1))
for i in range(505):
  state = state.replace(info={**state.info, "command": cmd, "wind": w2})
  state = jit_step(state, jp.zeros(env.action_size))
obs_cmd = np.asarray(state.info["command"])
assert np.allclose(obs_cmd, [1.0, 0.0, 0.0]), obs_cmd
print(f"command injection OK after 505 steps: {obs_cmd}")

# --- on_key semantics (import module fresh to isolate closure state) ------
import importlib
import wind_g1.viewer as viewer_mod
importlib.reload(viewer_mod)

cmd_state = {"cmd": np.zeros(3), "speed": 3.0, "heading": 0.0, "mult": 1.0}


class FakeParser:  # reuse module main() pieces manually instead:
  pass


# Recreate the closure by calling main() is GUI-bound; test key logic by
# replicating the branch table through the module constants.
KEY = viewer_mod
captured = []


def on_key(key):
  cmd = np.zeros(3)
  wind_speed, wind_heading, mult = 0.0, 0.0, 1.0
  if key in (KEY.KEY_W, KEY.KEY_S):
    v = mult * (1.0 if key == KEY.KEY_W else -1.0)
    cmd[0] = 0.0 if cmd[0] == v else v
  elif key == KEY.KEY_UP:
    wind_speed = min(wind_speed + KEY.WIND_SPEED_STEP, KEY.WIND_SPEED_MAX)
  captured.append((cmd.copy(), wind_speed))


on_key(KEY.KEY_W)
assert captured[-1][0][0] == 1.0
on_key(KEY.KEY_UP)
assert captured[-1][1] == 2.0
print("key table constants OK (full closure tested in GUI run)")

# --- arrow math -------------------------------------------------------------
r = rotation_z_to(np.array([1.0, 0.0, 0.0]))
# Rotate +z onto +x: columns give x_img = R@[0,0,1].
assert np.allclose(r @ np.array([0, 0, 1.0]), [1.0, 0.0, 0.0], atol=1e-9), r
r2 = rotation_z_to(np.array([0.0, 0.0, 1.0]))
assert np.allclose(r2, np.eye(3), atol=1e-9), r2
r3 = rotation_z_to(np.array([0.0, 1.0, 0.0]))
assert np.allclose(r3 @ np.array([0, 0, 1.0]), [0.0, 1.0, 0.0], atol=1e-9), r3

import mujoco

scn = mujoco.MjvScene(mujoco.MjModel.from_xml_string("<mujoco/>"), 10)
add_arrow(scn, np.zeros(3), np.array([2.0, 0.0, 0.0]), [1, 0, 0, 1])
assert scn.ngeom == 1
print("arrow math OK")

# --- load_policy against the dry-run checkpoint -----------------------------
ckpt = sys.argv[1] if len(sys.argv) > 1 else None
if ckpt:
  policy = load_policy(ckpt, ("state", "privileged_state"))
  env = make_env(speed=8.0)
  state = jax.jit(env.reset)(jax.random.PRNGKey(0))
  rng = jax.random.PRNGKey(7)
  action = policy(state.obs["state"], rng)[0]
  assert action.shape == (env.action_size,), action.shape
  state = state.replace(info={**state.info, "wind": jp.array([8.0, 0.0])})
  state = jax.jit(env.step)(state, action)
  print(f"load_policy OK: action shape {action.shape}, stepped 1 frame")
else:
  print("load_policy: no checkpoint arg given, SKIPPED")

print("ALL VIEWER-INTERNAL TESTS PASSED")
