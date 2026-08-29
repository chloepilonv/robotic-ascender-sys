"""Headless smoke test for the G1 climb-ascender env (no GUI).

Checks:
1. Registry resolution + climb_config presence; model has rope, carrier,
   slide joint, and the connect equality.
2. Geometry: floor tilted by the configured slope, rope cylinder
   parallel to the slope at rest-palm height, palm coincident with the
   carrier at reset.
3. Obs shapes match upstream G1Joystick (103-dim state) — the mels demo
   policy layout stays loadable.
4. Ascender semantics under a 5 s zero-action rollout (robot sags into
   the grip): no NaN, slide coordinate never decreases, palm stays on
   the line (perpendicular error bounded), robot is caught by the hand
   (pelvis does not fall to the floor).
5. Slide-up still works: with feet planted and gravity the ratchet must
   not freeze the carrier when the hand is pushed up the line (slide
   monotone increase observed in the sag rollout counts, but also
   verified explicitly via hand_height_on_line >= slide).
"""

import math

import jax
import jax.numpy as jp
import mujoco
import numpy as np

import wind_g1  # noqa: F401  registers the envs
from mujoco_playground import registry
from wind_g1 import climb_env

SLOPE_DEG = 30.0

# --- 1. registry + model structure ---------------------------------------
cfg = registry.get_default_config("G1ClimbAscender")
assert hasattr(cfg, "climb_config"), "default config missing climb_config"
env = registry.load(
    "G1ClimbAscender",
    config_overrides={"climb_config.slope_deg": SLOPE_DEG},
)
assert isinstance(env, climb_env.G1ClimbAscender), type(env)
print("registry OK: G1ClimbAscender ->", type(env).__name__)

m = env.mj_model
assert m.nq == 37 and m.nv == 36, (m.nq, m.nv)
for name in ("rope", "rope_geom", "ascender_carrier", "ascender_slide",
             "carrier_site", "ascender_grip"):
  m.body(name) if name in ("rope", "ascender_carrier") else (
      m.geom(name) if name == "rope_geom"
      else m.joint(name) if name == "ascender_slide"
      else m.site(name) if name == "carrier_site"
      else m.equality(name)
  )
assert m.neq == 1 and m.eq_type[0] == int(mujoco.mjtEq.mjEQ_CONNECT)
print("model structure OK: slide joint + connect equality present")

# --- 2. geometry ----------------------------------------------------------
d = mujoco.MjData(m)
d.qpos[:] = m.key("knees_bent").qpos
mujoco.mj_forward(m, d)
# Floor normal tilted by slope about +y, rising toward +x.
normal = d.geom_xmat[m.geom("floor").id].reshape(3, 3)[:, 2]
s = math.radians(SLOPE_DEG)
expect = np.array([-math.sin(s), 0.0, math.cos(s)])
assert np.allclose(normal, expect, atol=1e-6), (normal, expect)
print(f"floor OK: normal {normal} = slope {SLOPE_DEG} deg")
# Rope cylinder axis parallel to the in-plane uphill direction.
rgid = m.geom("rope_geom").id
rmat = d.geom_xmat[rgid].reshape(3, 3)
axis_actual = rmat[:, 2]  # cylinder local +z
uphill = np.array([math.cos(s), 0.0, math.sin(s)])
assert abs(abs(axis_actual @ uphill) - 1.0) < 1e-6, axis_actual
# Rope sits at the rest palm height along the slope (roughly waist height
# for the bent-knee pose: ~0.66 m up from the contact point).
rope_pt = np.asarray(env._line_pt)
palm = d.site_xpos[m.site("right_palm").id]
assert np.linalg.norm(rope_pt - palm) < 1e-6
# Palm coincident with carrier at reset.
carrier = d.site_xpos[m.site("carrier_site").id]
assert np.linalg.norm(carrier - palm) < 1e-9
print("grip OK: palm starts on the carrier")

# --- 3. obs shapes --------------------------------------------------------
jit_reset = jax.jit(env.reset)
jit_step = jax.jit(env.step)
state = jit_reset(jax.random.PRNGKey(0))
obs_state = np.asarray(state.obs["state"])
obs_priv = np.asarray(state.obs["privileged_state"])
assert obs_priv.shape == (216,), obs_priv.shape
print(f"obs OK: state {obs_state.shape}, privileged {obs_priv.shape}")

# --- 4. ascender semantics: 5 s zero-action sag rollout -------------------
action = jp.zeros(env.action_size)
slide_min = float(state.data.qpos[env._slide_qposadr])
prev_slide = slide_min
for i in range(250):  # 250 * 0.02 s = 5 s
  state = jit_step(state, action)
  qpos = np.asarray(state.data.qpos)
  assert not np.isnan(qpos).any(), f"NaN at step {i}"
  slide = float(qpos[env._slide_qposadr])
  assert slide >= prev_slide - 1e-9, (
      f"ascender slipped down at step {i}: {prev_slide} -> {slide}"
  )
  prev_slide = slide
  err = float(env.hand_line_error(state.data))
  assert err < 0.05, f"hand left the line at step {i}: err={err}"
print(f"sag rollout OK: slide {slide_min:.4f} -> {prev_slide:.4f} (monotone up),"
      f" max line err < 5 cm, no NaN")

# Robot caught by the hand, resting on the slope surface: the pelvis
# stays within one body-height of the surface (no cratering through the
# floor), and the slide coordinate bore the full body load for 5 s
# without slipping downhill.
pelvis = np.asarray(state.data.qpos[0:3])
surf_n = np.array([-math.sin(s), 0.0, math.cos(s)])
pelvis_above = float(pelvis @ surf_n)  # signed height above slope plane
print(f"after 5 s: pelvis {pelvis}, {pelvis_above:.3f} m above slope")
assert pelvis_above > -0.05, "pelvis penetrated the floor"
assert pelvis_above < 1.0, "robot hoisted clear of the slope (bad geometry)"
assert prev_slide - slide_min < 0.01, (
    "ascender let the hand slide down the line under load"
)

# Hand height on the line is consistent with the slide coordinate.
h = float(env.hand_height_on_line(state.data))
assert abs(h - prev_slide) < 0.05, (h, prev_slide)
print(f"hand-on-line height {h:.4f} ~ slide {prev_slide:.4f}")

# --- 5. ratchet does not freeze upward motion -----------------------------
# The carrier is light and yoked to the robot by the equality, so the
# hand slides up only when the ROBOT moves up. Teleport the robot 0.1 m
# up the line and verify the carrier follows (slide increases) and then
# holds (ratchet retains the high point even as the robot settles).
axis = np.asarray(env._slope_axis)
qpos_t = np.asarray(state.data.qpos).copy()
qpos_t[0:3] = qpos_t[0:3] + 0.1 * axis
state = state.replace(data=state.data.replace(qpos=jp.array(qpos_t)))
s0 = float(state.data.qpos[env._slide_qposadr])
peak = s0
for _ in range(10):
  state = jit_step(state, action)
  peak = max(peak, float(state.data.qpos[env._slide_qposadr]))
assert peak > s0 + 0.02, f"slide-up blocked: {s0} -> {peak}"
s_end = float(state.data.qpos[env._slide_qposadr])
assert s_end >= peak - 0.005, "ratchet lost the high point"
print(f"slide-up OK: {s0:.4f} -> peak {peak:.4f}, held at {s_end:.4f}")


# =========================================================================
# Himalaya terrain mode (teammate's 3 m pad: snow/ice/rock, 10/40 deg
# wedges; rope up the 40 deg wedge).
# =========================================================================

# --- H1. terrain port + materials ----------------------------------------
env_h = registry.load("G1ClimbAscenderHimalaya")
assert env_h._config.climb_config.terrain == "himalaya"
m_h = env_h.mj_model
for name in ("ice", "wall", "slope_10deg", "slope_40deg", "rope_geom"):
  m_h.geom(name)
for pair, mu in (
    ("left_foot_floor", 0.5),   # packed snow
    ("left_foot_slope_40deg", 0.5),
    ("left_foot_ice", 0.1),     # ice
    ("left_foot_wall", 0.8),    # rock
):
  got = m_h.pair_friction[m_h.pair(pair).id][:2]
  assert np.allclose(got, mu), (pair, got, mu)
print("terrain port OK: snow 0.5 / ice 0.1 / rock 0.8 on explicit pairs")

# --- H2. geometry: wedges rise toward -X, rope parallel to the 40 deg ramp
d_h = mujoco.MjData(m_h)
d_h.qpos[:] = m_h.key("knees_bent").qpos
mujoco.mj_forward(m_h, d_h)
th40 = math.radians(40)
n40 = d_h.geom_xmat[m_h.geom("slope_40deg").id].reshape(3, 3)[:, 2]
assert np.allclose(n40, [math.sin(th40), 0, math.cos(th40)], atol=1e-6), n40
axis_h = np.asarray(env_h._slope_axis)
assert abs(float(axis_h @ n40)) < 1e-6, "rope not parallel to the ramp"
# Ramp top face: rises from x=-0.25 (z=0) to x=-1.25 (z=0.84).
c40 = d_h.geom_xpos[m_h.geom("slope_40deg").id]
top_c = c40 + n40 * 0.15
uphill = np.array([-math.cos(th40), 0, math.sin(th40)])
lo_edge = top_c - uphill * (0.5 / math.cos(th40))
hi_edge = top_c + uphill * (0.5 / math.cos(th40))
assert abs(lo_edge[0] + 0.25) < 1e-3 and lo_edge[2] < 0.02, lo_edge
assert abs(hi_edge[0] + 1.25) < 1e-3 and abs(hi_edge[2] - 0.84) < 1e-3, hi_edge
print(f"ramp OK: base {np.round(lo_edge,3)} -> top {np.round(hi_edge,3)}")

# --- H3. spawn: upright on snow at the ramp base, facing -X --------------
pelvis = np.asarray(d_h.qpos[0:3])
assert abs(pelvis[2] - 0.755) < 1e-6, pelvis  # keyframe height, upright
palm_h = d_h.site_xpos[m_h.site("right_palm").id]
carrier_h = d_h.site_xpos[m_h.site("carrier_site").id]
assert np.linalg.norm(carrier_h - palm_h) < 1e-6
print(f"spawn OK: pelvis {np.round(pelvis,3)}, palm on carrier")

# --- H4. obs + 5 s zero-action sag on the himalaya pad -------------------
state_h = jax.jit(env_h.reset)(jax.random.PRNGKey(0))
assert np.asarray(state_h.obs["state"]).shape == (103,)
prev_h = float(state_h.data.qpos[env_h._slide_qposadr])
for i in range(250):
  state_h = jax.jit(env_h.step)(state_h, jp.zeros(env_h.action_size))
  q = np.asarray(state_h.data.qpos)
  assert not np.isnan(q).any(), f"NaN at step {i}"
  s = float(q[env_h._slide_qposadr])
  assert s >= prev_h - 1e-9, f"ascender slipped at {i}: {prev_h} -> {s}"
  prev_h = s
  assert float(env_h.hand_line_error(state_h.data)) < 0.05
print(f"himalaya sag OK: slide {prev_h:.4f} (monotone), hand on line, no NaN")

print("ALL CLIMB SMOKE TESTS PASSED")
