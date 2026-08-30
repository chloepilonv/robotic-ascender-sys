"""Sim twin of the climb mime: RL walking policy + scripted right arm, in the MuJoCo viewer.

    ~/Documents/Code/mujoco-playground/.venv/bin/python -m rl.chloe.deterministic.mime_sim
    ... --headless --seconds 12      # smoke test, prints phase / pelvis height / hand height

The walking policy (mels G1 joystick, rl/policies) controls all 29 joints; we
overwrite the 7 right-arm actions each step so the motor targets follow
`choreo.step(t)`, and set the forward-velocity command during the pull phase.
"""

from __future__ import annotations

import argparse
import time

import jax
import jax.numpy as jp
import mujoco
import numpy as np
from etils import epath
from mujoco_playground import registry

import rl.environment  # noqa: F401  registers G1JoystickWind*
from rl.chloe.deterministic import choreo
from rl.scripts.viewer import load_mels_policy


def main() -> None:
  p = argparse.ArgumentParser()
  p.add_argument("--headless", action="store_true")
  p.add_argument("--seconds", type=float, default=12.0)
  p.add_argument("--wind_speed", type=float, default=0.0)
  args = p.parse_args()

  env = registry.load(
    "G1JoystickWindFlatTerrain",
    config_overrides={"wind_config.enable": True, "wind_config.wind_speed": args.wind_speed},
  )
  npz = epath.Path(__file__).resolve().parents[2] / "policies" / "mels_g1_joystick.npz"
  policy = load_mels_policy(npz.as_posix())
  jit_reset, jit_step = jax.jit(env.reset), jax.jit(env.step)

  m = env.mj_model
  jnames = [m.joint(i).name for i in range(1, m.njnt)]  # skip the freejoint
  arm_idx = np.array([jnames.index(n) for n in choreo.RIGHT_ARM])
  waist_idx = jnames.index("waist_pitch_joint")
  default_pose = np.asarray(env._default_pose)
  scale = float(env._config.action_scale)
  hand = m.body("right_wrist_yaw_link").id

  rng = jax.random.PRNGKey(0)

  def reset_facing_x(rng):
    """Upstream reset randomises yaw; start squared up to the rope (+x)."""
    st = jit_reset(rng)
    qpos = st.data.qpos.at[3:7].set(jp.array([1.0, 0.0, 0.0, 0.0]))
    return st.replace(data=st.data.replace(qpos=qpos))

  state = reset_facing_x(rng)
  mj_data = mujoco.MjData(m)
  viewer = None if args.headless else mujoco.viewer.launch_passive(m, mj_data)
  t0, t, dt = time.time(), 0.0, float(env.dt)
  last_phase = None
  while t < args.seconds if args.headless else viewer.is_running():
    if bool(state.done):
      state = reset_facing_x(rng)
    c = choreo.step(t)
    q = np.asarray(state.data.qpos[3:7])  # pelvis quat (w,x,y,z) -> yaw
    yaw = np.arctan2(2 * (q[0] * q[3] + q[1] * q[2]), 1 - 2 * (q[2] ** 2 + q[3] ** 2))
    cmd = np.array([c["speed"], 0.0, choreo.yaw_rate_cmd(yaw)])
    rng, k = jax.random.split(rng)
    action = np.array(policy(state.obs["state"], k)[0])
    # Right arm: motor_target = default + action*scale  ->  action = (target-default)/scale
    action[arm_idx] = (np.array(c["arm"]) - default_pose[arm_idx]) / scale
    action[waist_idx] = (c["waist_pitch"] - default_pose[waist_idx]) / scale
    state = state.replace(info={**state.info, "command": jp.array(cmd)})
    state = jit_step(state, jp.array(action))
    t += dt
    mj_data.qpos[:] = np.asarray(state.data.qpos)
    mj_data.qvel[:] = np.asarray(state.data.qvel)
    mujoco.mj_forward(m, mj_data)
    if c["phase"] != last_phase:
      last_phase = c["phase"]
      print(
        f"t={t:5.2f} {c['phase']:5s} walk={int(c['walk'])} engaged={int(c['engaged'])} "
        f"tension={c['tension_N']:5.0f}N pelvis_z={mj_data.qpos[2]:.2f} hand_z={mj_data.xpos[hand][2]:.2f} "
        f"yaw={np.degrees(yaw):+5.1f}deg x={mj_data.qpos[0]:.2f}"
      )
    if viewer is not None:
      viewer.sync()
      time.sleep(max(0.0, t - (time.time() - t0)))
  if args.headless:
    assert mj_data.qpos[2] > 0.5, "robot fell"
    print("mime_sim OK")


if __name__ == "__main__":
  main()
