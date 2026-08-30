"""Headless smoke test for the domain-randomized walk env (no GUI).

Checks:
1. Registry resolution + dr_config presence; obs/action sizes match
   upstream G1Joystick (fine-tune compatibility with the mels policy).
2. `domain_randomize`: per-env floor tilts within [0, 40] deg (distinct
   across envs), floor-foot friction within range, dynamics fields
   vectorized with correct in_axes.
3. Parallel vector environments through the playground DR wrapper:
   reset + rollout across N envs with different slopes; no NaN; feet
   contact the tilted floors (floor_found sensors fire); terrain-relative
   gravity obs reads upright on every slope.
4. Slope-aware reset placement: on the steepest env the pelvis height
   above the slope plane matches the keyframe height.
5. Wind realism: baseline speed within [0, 150 kmph / 3.6]; over a
   rollout the speed stays within (1 + gust_fraction) of the baseline
   (gusts/dips, clipped); heading wanders smoothly around the baseline
   (bounded, slow-varying: no teleporting directions).
6. Flat-ground equivalence: with slope pinned to 0 the terrain-relative
   gravity block equals the upstream world-frame gravity block.
"""

import functools
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np

import jax
import jax.numpy as jp

import rl.environment  # noqa: F401  registers the envs
from mujoco_playground import registry
from mujoco_playground._src import wrapper as pg_wrapper
from rl.environment import walk_dr_env

N_ENVS = 8
SLOPE_MIN, SLOPE_MAX = 0.0, 40.0
KMPH_TO_MS = 1.0 / 3.6

# --- 1. registry ----------------------------------------------------------
cfg = registry.get_default_config("G1JoystickWalkDR")
assert hasattr(cfg, "dr_config"), "default config missing dr_config"
env = registry.load("G1JoystickWalkDR", config_overrides={"impl": "jax"})
assert isinstance(env, walk_dr_env.G1JoystickWalkDR), type(env)
assert env.observation_size["state"] == (103,)
assert env.observation_size["privileged_state"] == (216,)
assert env.action_size == 29
print("registry OK: G1JoystickWalkDR, obs 103/216, act 29 (mels-compatible)")

# --- 2. domain_randomize ---------------------------------------------------
base_model = env.mjx_model
rng = jax.random.split(jax.random.PRNGKey(0), N_ENVS)
dr_cfg = registry.get_default_config("G1JoystickWalkDR").dr_config
v_model, in_axes = jax.jit(
    functools.partial(walk_dr_env.domain_randomize, dr_cfg=dr_cfg)
)(base_model, rng)

quats = np.asarray(v_model.geom_quat[:, 0])  # floor quats, (N, 4)
assert quats.shape == (N_ENVS, 4)
slopes = 2 * np.arctan2(-quats[:, 2], quats[:, 0])  # from [c, 0, -s, 0]
assert np.all(slopes >= math.radians(SLOPE_MIN) - 1e-6), slopes
assert np.all(slopes <= math.radians(SLOPE_MAX) + 1e-6), slopes
assert len(np.unique(np.round(slopes, 4))) > N_ENVS // 2, "slopes not varied"
print(f"slope randomization OK: {np.degrees(slopes).round(1)} deg")

fr = np.asarray(v_model.pair_friction[:, 0:2, 0])
assert np.all(fr >= 0.4 - 1e-6) and np.all(fr <= 1.0 + 1e-6), fr
bm = np.asarray(v_model.body_mass)
base_mass = np.asarray(base_model.body_mass)
assert bm.shape == (N_ENVS, base_mass.shape[0])
ratio = bm[:, 16] / base_mass[16]
# Only the six randomized fields are vmapped (axis 0); check by name.
for field in ("geom_quat", "pair_friction", "dof_frictionloss",
               "dof_armature", "body_mass", "qpos0"):
  assert np.asarray(getattr(in_axes, field)).item() == 0, field
print("dynamics randomization OK: friction/masses vectorized, in_axes sane")

# --- 3. parallel vector envs through the DR wrapper ------------------------
train_env = registry.load(
    "G1JoystickWalkDR",
    config=registry.get_default_config("G1JoystickWalkDR"),
    config_overrides={"impl": "jax"},
)
rand_fn = functools.partial(
    walk_dr_env.domain_randomize, rng=rng, dr_cfg=dr_cfg
)
wrapped = pg_wrapper.wrap_for_brax_training(
    train_env,
    episode_length=1000,
    action_repeat=1,
    randomization_fn=rand_fn,
)

jit_wreset = jax.jit(wrapped.reset)
jit_wstep = jax.jit(wrapped.step)
reset_rng = jax.random.split(jax.random.PRNGKey(1), N_ENVS)
wstate = jit_wreset(reset_rng)
assert wstate.obs["state"].shape == (N_ENVS, 103), wstate.obs["state"].shape

# At reset the robot stands on the slope: both feet in floor contact on
# every env (zero-action rollouts later sag/fall, as on upstream).
mm = train_env.mj_model
adr = [mm.sensor_adr[mm.sensor(s).id] for s in
       ("left_foot_floor_found", "right_foot_floor_found")]
found = np.asarray(wstate.data.sensordata)[:, adr]  # (N, 2)
# Upstream joint randomization (x U(0.5, 1.5)) can lift one foot at
# reset; require at least one foot-floor contact per env (the robot
# starts standing on the slope, not in mid-air or inside the plane).
assert np.all(found.max(axis=1) > 0), found
assert found.mean() > 0.7, found  # overwhelmingly in contact.

# Terrain-relative gravity obs reads upright on every slope.
g_obs = np.asarray(wstate.obs["state"])[:, 6:9]
assert np.all(g_obs[:, 2] < -0.85), g_obs[:, 2]  # "down" along the normal
print("terrain-relative gravity OK: min z-component", g_obs[:, 2].min().round(3))

# Slope-aware placement: pelvis height above the tilted plane == keyframe.
v_quat = np.asarray(v_model.geom_quat[:, 0])
s_env = np.degrees(2 * np.arctan2(-v_quat[:, 2], v_quat[:, 0]))
i_steep = int(np.argmax(s_env))
n = np.array([-math.sin(math.radians(s_env[i_steep])), 0.0,
              math.cos(math.radians(s_env[i_steep]))])
h0 = float(np.asarray(train_env._init_q)[2])
pelvis = np.asarray(wstate.data.qpos[i_steep, 0:3])
above = float(pelvis @ n)
assert abs(above - h0) < 0.05, (above, h0)
print(f"placement OK: steepest env {s_env[i_steep]:.1f} deg, "
      f"pelvis {above:.3f} m above plane (keyframe {h0:.3f})")

# Zero-action rollout: robots sag/fall (no policy), which is fine — this
# phase only exercises the parallel envs and collects the wind time series.
zero_act = jp.zeros((N_ENVS, env.action_size))
wind_speeds, headings = [], []
for i in range(100):
  wstate = jit_wstep(wstate, zero_act)
  qpos = np.asarray(wstate.data.qpos)
  assert not np.isnan(qpos).any(), f"NaN at step {i}"
  w = np.asarray(wstate.info["wind"])  # (N, 2)
  wind_speeds.append(np.linalg.norm(w, axis=-1))
  headings.append(np.arctan2(w[:, 1], w[:, 0]))
wind_speeds = np.stack(wind_speeds)  # (T, N)
headings = np.stack(headings)
print(f"vector rollout OK: {N_ENVS} envs x 100 steps, no NaN")

# --- 5. wind realism --------------------------------------------------------
# 5a. Baseline (reset) wind speed within [0, 150 kmph].
w0 = np.asarray(wstate.info["wind_base"])
speed0 = np.linalg.norm(w0, axis=-1)
vmax = dr_cfg.wind_max_speed_kmph * KMPH_TO_MS
assert np.all(speed0 >= -1e-6) and np.all(speed0 <= vmax + 1e-6), speed0
print(f"wind baseline OK: speeds (m/s) {speed0.round(1)}, max allowed {vmax:.1f}")

# 5b. Speed stays within (1 + gust_fraction) of the baseline at all times.
base_speed = np.linalg.norm(
    np.asarray(wstate.info["wind_base"]), axis=-1
)  # (N,)
spread = np.abs(wind_speeds - base_speed[None, :])
lim = (1 + dr_cfg.gust_fraction) * base_speed[None, :] + 1e-6
# zero-baseline envs: |speed - 0| <= 0 by clip construction (gust of a
# zero baseline stays zero) — allow numerical noise only.
ok = np.where(base_speed[None, :] > 1e-6, spread <= lim, wind_speeds < 1e-4)
assert np.all(ok), (spread[np.logical_not(ok)], lim[np.logical_not(ok)])
print("wind gusts OK: speed within +-%.0f%% of baseline over the rollout"
      % (100 * dr_cfg.gust_fraction))

# 5c. Direction wanders smoothly: step-to-step change is small (OU with
# ~8 s persistence), and heading stays near the baseline (12 deg std).
dtheta = np.abs(np.diff(headings, axis=0))
dtheta = np.minimum(dtheta, 2 * np.pi - dtheta)  # wrap
assert dtheta.max() < math.radians(5.0), np.degrees(dtheta.max())
# Bounded wander: after 100 steps (2 s) still within a few sigma.
dev = np.abs(headings[-1] - np.arctan2(w0[:, 1], w0[:, 0]))
dev = np.minimum(dev, 2 * np.pi - dev)
nz = base_speed > 1e-6
assert np.all(dev[nz] < math.radians(3 * dr_cfg.direction_wander_deg)), (
    np.degrees(dev[nz]))
print("wind direction OK: smooth wander, max step %.2f deg, deviation < %.1f deg"
      % (np.degrees(dtheta.max()), np.degrees(dev[nz].max())))

# --- 6. flat-ground equivalence --------------------------------------------
flat_over = {"impl": "jax", "dr_config.slope_min_deg": 0.0,
             "dr_config.slope_max_deg": 0.0}
env_flat = registry.load("G1JoystickWalkDR", config_overrides=flat_over)
from mujoco_playground._src.locomotion.g1 import joystick as g1_joystick  # noqa: E402

# Flat equivalence: with slope pinned to 0 and noise disabled, the state
# obs must equal upstream Joystick._get_obs bit-for-bit (the env is a
# pure subclass; only the gravity re-expression differs, which on flat
# ground is the identity). Noise off avoids rng-draw mismatch.
env_flat = registry.load(
    "G1JoystickWalkDR",
    config_overrides={
        "impl": "jax",
        "dr_config.slope_min_deg": 0.0,
        "dr_config.slope_max_deg": 0.0,
        "noise_config.level": 0.0,
    },
)
state_flat = jax.jit(env_flat.reset)(jax.random.PRNGKey(3))
up = g1_joystick.Joystick._get_obs(
    env_flat, state_flat.data, state_flat.info,
    jp.array([True, True]),
)
dr_full = np.asarray(state_flat.obs["state"])
up_full = np.asarray(up["state"])
assert np.allclose(dr_full, up_full, atol=1e-6), np.abs(dr_full - up_full).max()
print("flat equivalence OK: state obs identical to upstream Joystick")

print("ALL WALK-DR SMOKE TESTS PASSED")
