"""Interactive WASD viewer for the wind-enabled G1 locomotion task.

Runs the MuJoCo Playground G1JoystickWind env in the passive MuJoCo viewer.
Keyboard drives the joystick command and the wind:

  Command (toggle on press, press again to clear):
    W / S      forward / backward  (lin_vel_x  = +1 / -1  * mult)
    Q / E      strafe left / right (lin_vel_y  = +0.5/-0.5* mult)
    A / D      turn left / right   (ang_vel_yaw= +1 / -1  * mult)
    X          zero all commands
    R          cycle speed multiplier (1x / 2x)
  Wind (live):
    Up / Down  wind speed +-2 m/s  (clip [0, 40])
    Left/Right wind heading +-15 deg
    0          wind off

Usage:
  python -m wind_g1.viewer [--policy CKPT_DIR] [--wind_speed M/S]
  [--wind_heading DEG] [--env_name NAME] [--impl jax|warp]

Without --policy the G1 runs with zero actions: it will sag and fall —
that is expected (a policy checkpoint is needed to walk).
"""

import argparse
import math
import os
import sys

# Bootstrap: allow `python wind_g1/viewer.py` in addition to `-m wind_g1.viewer`.
_PKG_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PKG_ROOT not in sys.path:
  sys.path.insert(0, _PKG_ROOT)


import functools  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402

import jax  # noqa: E402
import jax.numpy as jp  # noqa: E402
import mujoco  # noqa: E402
import mujoco.viewer  # noqa: E402
from brax.training import checkpoint as brax_checkpoint  # noqa: E402
from brax.training.agents.ppo import networks as ppo_networks  # noqa: E402
from etils import epath  # noqa: E402

from mujoco_playground import registry  # noqa: E402
import wind_g1  # noqa: E402,F401  registers G1JoystickWind* envs

# GLFW keycodes (press events only; launch_passive has no release events).
KEY_W, KEY_S, KEY_A, KEY_D = 87, 83, 65, 68
KEY_Q, KEY_E, KEY_X, KEY_R = 81, 69, 88, 82
KEY_UP, KEY_DOWN, KEY_LEFT, KEY_RIGHT = 265, 264, 263, 262
KEY_0 = 48

WIND_SPEED_STEP = 2.0
WIND_SPEED_MAX = 40.0
WIND_HEADING_STEP = math.radians(15.0)


def load_policy(policy_path: str, obs_keys: tuple[str, str]):
  """Restore a brax PPO checkpoint -> jit'd deterministic policy."""
  path = epath.Path(policy_path).resolve()
  params = brax_checkpoint.load(path)
  config = brax_checkpoint.load_config(path / "config.json")
  kwargs = dict(config.network_factory_kwargs)
  kwargs.setdefault("policy_obs_key", obs_keys[0])
  kwargs.setdefault("value_obs_key", obs_keys[1])
  network = brax_checkpoint.get_network(
      config, functools.partial(ppo_networks.make_ppo_networks, **kwargs)
  )
  inference_fn = ppo_networks.make_inference_fn(network)(
      params, deterministic=True
  )
  return jax.jit(inference_fn)


def load_mels_policy(npz_path: str):
  """Load the mels.ai demo G1 joystick policy (npz) -> jit'd policy.

  MLP 103->512->256->128->58 with obs normalization; outputs are 29 action
  means (first half) + 29 log-stds (ignored; deterministic eval). Layout
  matches the playground G1Joystick `state` observation exactly.
  """
  z = np.load(npz_path)
  def forward(obs):
    x = (obs - z['obs_mean']) / z['obs_std']
    swish = lambda v: v / (1.0 + np.exp(-v))  # brax PPO default activation.
    for i in range(3):
      x = swish(x @ z[f'hidden_{i}_kernel'] + z[f'hidden_{i}_bias'])
    out = x @ z['hidden_3_kernel'] + z['hidden_3_bias']
    return out[:29]

  def policy_fn(obs, rng):
    del rng
    return jp.array(forward(np.asarray(obs))), None

  return policy_fn


def rotation_z_to(vec: np.ndarray) -> np.ndarray:
  """3x3 rotation matrix mapping +z onto a unit XY(Z) vector."""
  v = np.asarray(vec, dtype=float)
  if v.shape == (2,):  # Wind vectors are XY.
    v = np.array([v[0], v[1], 0.0])
  v = v / np.linalg.norm(v)
  z = np.array([0.0, 0.0, 1.0])
  axis = np.cross(z, v)
  s = np.linalg.norm(axis)
  c = float(z @ v)
  if s < 1e-9:  # Parallel or antiparallel.
    return np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
  axis = axis / s
  k = np.array([
      [0.0, -axis[2], axis[1]],
      [axis[2], 0.0, -axis[0]],
      [-axis[1], axis[0], 0.0],
  ])
  return np.eye(3) + k + k @ k * ((1.0 - c) / (s * s))


def add_arrow(scn, pos, vec, rgba):
  """Append an mjGEOM_ARROW to a user scene, guarded by maxgeom."""
  if scn.ngeom >= scn.maxgeom or np.linalg.norm(vec) < 1e-6:
    return
  geom = scn.geoms[scn.ngeom]
  geom.type = mujoco.mjtGeom.mjGEOM_ARROW
  geom.mat = rotation_z_to(vec)
  geom.size = [0.02, 0.02, max(float(np.linalg.norm(vec)), 0.05)]
  geom.rgba = rgba
  scn.ngeom += 1


def main():
  parser = argparse.ArgumentParser(
      description=__doc__,
      formatter_class=argparse.RawDescriptionHelpFormatter,
  )
  parser.add_argument("--policy", default=None, help="brax checkpoint dir")
  parser.add_argument("--env_name", default="G1JoystickWindFlatTerrain")
  parser.add_argument("--wind_speed", type=float, default=0.0)
  parser.add_argument("--wind_heading", type=float, default=0.0, help="deg")
  parser.add_argument("--impl", default="jax", choices=["jax", "warp"])
  args = parser.parse_args()

  heading0 = math.radians(args.wind_heading)
  env = registry.load(
      args.env_name,
      config_overrides={
          "impl": args.impl,
          "wind_config.enable": True,
          "wind_config.wind_speed": args.wind_speed,
          "wind_config.wind_heading": heading0,
      },
  )
  env.use_wind_from_info(True)  # live wind via info["wind"]

  policy = None
  if args.policy == "mels":
    npz = epath.Path(__file__).parent.parent / "policies" / "mels_g1_joystick.npz"
    policy = load_mels_policy(npz.as_posix())
    print(f"mels demo policy loaded from {npz}")
  elif args.policy:
    policy = load_policy(args.policy, ("state", "privileged_state"))
    print(f"policy loaded from {args.policy}")
  else:
    print("WARNING: no policy: G1 will not walk (zero actions).")

  # Host-side interactive state.
  cmd = np.zeros(3)  # lin_vel_x, lin_vel_y, ang_vel_yaw
  wind_speed = float(args.wind_speed)
  wind_heading = heading0
  speed_multiplier = 1.0

  def wind_vec():
    return np.array([
        wind_speed * math.cos(wind_heading),
        wind_speed * math.sin(wind_heading),
    ])

  def on_key(key):
    nonlocal wind_speed, wind_heading, speed_multiplier
    if key in (KEY_W, KEY_S):
      v = speed_multiplier * (1.0 if key == KEY_W else -1.0)
      cmd[0] = 0.0 if cmd[0] == v else v
    elif key in (KEY_A, KEY_D):
      v = speed_multiplier * (1.0 if key == KEY_A else -1.0)
      cmd[2] = 0.0 if cmd[2] == v else v
    elif key in (KEY_Q, KEY_E):
      v = speed_multiplier * (0.5 if key == KEY_Q else -0.5)
      cmd[1] = 0.0 if cmd[1] == v else v
    elif key == KEY_X:
      cmd[:] = 0.0
    elif key == KEY_R:
      speed_multiplier = 2.0 if speed_multiplier == 1.0 else 1.0
    elif key == KEY_UP:
      wind_speed = min(wind_speed + WIND_SPEED_STEP, WIND_SPEED_MAX)
    elif key == KEY_DOWN:
      wind_speed = max(wind_speed - WIND_SPEED_STEP, 0.0)
    elif key == KEY_LEFT:
      wind_heading += WIND_HEADING_STEP
    elif key == KEY_RIGHT:
      wind_heading -= WIND_HEADING_STEP
    elif key == KEY_0:
      wind_speed = 0.0
    print(
        f"cmd=[{cmd[0]:+.2f},{cmd[1]:+.2f},{cmd[2]:+.2f}]"
        f" wind={wind_speed:5.1f} m/s @ {math.degrees(wind_heading):6.1f} deg"
        f" mult={speed_multiplier}x"
    )

  mj_data = mujoco.MjData(env.mj_model)
  handle = mujoco.viewer.launch_passive(
      env.mj_model, mj_data, key_callback=on_key
  )

  jit_reset = jax.jit(env.reset)
  jit_step = jax.jit(env.step)
  rng = jax.random.PRNGKey(0)
  state = jit_reset(rng)
  zero_action = jp.zeros(env.action_size)

  print(
      "viewer running - WASD/QE/X/R commands, arrows/0 wind,"
      " close window to quit"
  )
  while handle.is_running():
    if bool(state.done):
      rng, reset_rng = jax.random.split(rng)
      state = jit_reset(reset_rng)
    rng, act_rng = jax.random.split(rng)
    if policy is not None:
      action = policy(state.obs["state"], act_rng)[0]
    else:
      action = zero_action
    # Host-side injection of live command + wind (array swap, no retrace).
    info = {
        **state.info,
        "wind": jp.array(wind_vec()),
        "command": jp.array(cmd),
    }
    state = state.replace(info=info)
    state = jit_step(state, action)

    # Mirror mjx state -> mj_data for the viewer.
    mj_data.qpos[:] = np.asarray(state.data.qpos)
    mj_data.qvel[:] = np.asarray(state.data.qvel)
    mj_data.time = float(state.data.time)
    mj_data.ctrl[:] = np.asarray(state.info["motor_targets"])
    mujoco.mj_forward(env.mj_model, mj_data)

    # HUD arrows: wind (red, world frame) + command (blue, at pelvis).
    scn = handle.user_scn
    if scn is not None:
      scn.ngeom = 0
      w = wind_vec()
      if np.linalg.norm(w) > 1e-3:
        add_arrow(
            scn, np.array([0.0, 0.0, 1.5]), 0.1 * w, [0.9, 0.1, 0.1, 0.8]
        )
      if np.linalg.norm(cmd[:2]) > 1e-3:
        theta = math.atan2(cmd[1], cmd[0])
        arrow = 0.3 * np.array([math.cos(theta), math.sin(theta), 0.0])
        if abs(cmd[2]) > 1e-3:  # Yaw command: vertical arrow, up = CCW.
          arrow += np.array([0.0, 0.0, 0.1 * cmd[2]])
        add_arrow(
            scn,
            np.asarray(mj_data.xpos[1]) + np.array([0.0, 0.0, 0.2]),
            arrow,
            [0.1, 0.2, 0.9, 0.8],
        )

    handle.sync()
    time.sleep(env.dt)

  print("viewer closed; exiting")


if __name__ == "__main__":
  main()
