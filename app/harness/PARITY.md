# PARITY — is the harness running the real training environment?

The harness (`app/harness/`) is an interactive, browser-driven demo of the G1
on the fixed line. Its whole value depends on one claim:

> **the physics you drive in the browser is the physics Chloe's policy is
> trained against — `rl/environment/climb_env.py::G1ClimbAscender` — not a
> lookalike we rebuilt.**

This file is the evidence for that claim, and the honest list of where it does
NOT hold.

The rule the code follows: **load their physics, never rebuild it.** We
instantiate their env class, take the `mujoco.MjModel` it compiled, and read
every constant the loop needs off their env/config object. Where a JAX/MJX
function cannot be called from a plain-MuJoCo loop (the observation builder,
the ratchet), it is PORTED line for line and then TESTED against the original.

Nothing about the robot is named in our code. Foot geoms, the palm and carrier
sites, the torso and pelvis bodies, the ascender joint and the grip equality
are all derived from the loaded model — so a training robot that grows snow
boots, an ascender end-effector, or a jacket is inherited by reloading, not by
editing the harness. `fingerprint.json` carries the full body / geom / site /
joint / actuator / sensor tables so any such change is a visible diff.

---

## How to reproduce every number in this file

```bash
cd g1-himalayas
../.venv_everest/bin/python app/harness/team_env.py        # prints + writes fingerprint.json
../.venv_everest/bin/python -m app.harness.test_parity     # (a) and (b) below
../.venv_everest/bin/python rl/tests/test_climb_env.py     # their own baseline
../.venv_everest/bin/python -m app.harness.runtime --duration 15 --hold-w --keep-going --no-render
../.venv_everest/bin/python -m app.harness.runtime --live  # http://localhost:8766/app/web/index.html
```

Environment: `/Users/dengjingxi/Documents/code/himalaya_hack/.venv_everest`,
python 3.12.8, **exactly the requested pins** — `playground==0.2.0`,
`mujoco==3.12.0`, `brax==0.14.2`, plus `jax/jaxlib 0.11.1` (CPU),
`mujoco-mjx 3.12.0`, `warp-lang 1.16.0`, `numpy 2.5.2`, `websockets 17.1`,
`pillow 12.3.0`. **No pin deviations were needed on macOS.** `mujoco_menagerie`
is cloned automatically by playground on first import (commit
`1b86ece576591213e2b666ebf59508454200ca97`).

---

## The parity table

| Piece | Source of truth (their code) | How we consume it | Verified how | Status |
|---|---|---|---|---|
| **The whole MjModel** (slope, rope cylinder, carrier, equality, feet, keyframes) | `rl/environment/climb_env.py:130-262` `_build_model` | **IMPORT.** `registry.load("G1ClimbAscender")`, then `env.mj_model`. We never call `MjSpec`, never compile anything. | `fingerprint.json` — nq 37 / nv 36 / nu 29, neq 1, nsensor 29, 33 bodies / 74 geoms / 7 sites, total mass 33.4411 kg | ✅ identical object |
| Slope (floor tilt) | `climb_env.py:132-148` (`floor.quat` about +y) | Import (in the model). Read back as `slope_radians` / `slope_axis_world`. | fingerprint `geometry.floor_normal_world = [-0.5, -0.0, 0.8660254]` = exactly 30° | ✅ |
| Feet on the incline | `climb_env.py:151-154` (`condim=3`, `friction=(0.8, 0.005, 0.0001)`) | Import. Foot geom **ids** come from THEIR `_feet_geom_id` (`joystick.py:209`), never from a name we typed. | fingerprint `feet`: geoms `['left_foot','right_foot']` on `['left_ankle_roll_link','right_ankle_roll_link']`, condim `[3,3]`, friction `[[0.8,0.005,0.0001]]×2`, body mass `[0.608, 0.608]` kg | ✅ |
| Rope (visual cylinder) | `climb_env.py:169-186` | Import. | fingerprint `geometry.line_point_world = [0.14357, -0.2268, 0.66346]`, length 15.0 m + 0.5 m tail | ✅ |
| `ascender_carrier` + `ascender_slide` | `climb_env.py:190-208` | Import. The joint is found **by its qpos address** (`jnt_qposadr == slide_qposadr`), not by name; the carrier body is `jnt_bodyid` of that joint. | fingerprint `ascender_slide`: qpos[36] dof[35], axis `[0.866025, 0, 0.5]`, range `[-0.5, 15.0]`, damping 1.0, frictionloss 0.2, carrier mass 0.1 | ✅ |
| The grip (`connect` equality) | `climb_env.py:212-220` | Import. The equality is found **by its two site endpoints**, not by the name `ascender_grip`. | fingerprint `grip_equality`: `mjEQ_CONNECT`, `right_palm` ↔ `carrier_site`, solref `[0.004, 1.0]`, solimp `[0.95, 0.99, 0.001, 0.5, 2.0]` | ✅ **point** constraint — wrist free to rotate, intended |
| Timestep / integrator | `climb_env.py:227` (`opt.timestep = sim_dt`) | Import. | fingerprint `model`: timestep 0.002 s, `mjINT_EULER`, gravity `[0,0,-9.81]` | ✅ |
| Actuators (29 position servos) | upstream G1 MJCF via `consts.task_to_xml` | Import. | fingerprint `control.actuator_kp` min 2.0 max 75.0, per-joint kp/kv/ctrlrange/forcerange tables in the json | ✅ |
| Reset pose | `climb_env.py:237` / `:291-312` (`knees_bent` keyframe, palm on carrier) | Import (`model.keyframe("knees_bent").qpos`). | fingerprint: palm at reset `[0.14357, -0.2268, 0.66346]`, **\|palm − carrier\| = 0.00e+00 m**; runtime prints palm-on-line error `1.04e-08 m` at spawn | ✅ |
| `default_pose` (29,) | `climb_env.py:247-249` (`qpos[7:36]` of the keyframe) | Read off `env._default_pose`. | in fingerprint `reset.default_pose_radians`, with `reset.joint_names` | ✅ |
| Control law `ctrl = default_pose + 0.5·action` | `climb_env.py:359` | **PORT** (one line, `runtime.py::Episode.step`). `action_scale` read from their config, never typed. | parity test (b) rolls both; behaviours agree | ✅ |
| ctrl_dt / sim_dt / 10 substeps | `joystick.py:34-35`, `climb_env.py:124` | Read off `env.dt`, `env.sim_dt`, `env._n_substeps`. | fingerprint `control`: 0.02 s × 10 substeps, action_scale 0.5 | ✅ |
| **Ascender ratchet** | `climb_env.py:268-285` `_step_physics` | **PORT** — `app/harness/ratchet.py`, verbatim order: latch `prev_slide`, `mj_step`, `qvel = max(qvel, 0)`, `qpos = max(qpos, prev)`, **once per physics substep**, no `mj_forward` in between (so sensors are one substep stale on both sides). | parity (b): slide divergence **max 0.0066 m, final 0.0039 m** over 2 s. 15 s runs: rope travel is monotone and freezes at 0.187 m / 0.129 m — it never slips back. | ✅ |
| **103-d `state` observation** | `climb_env.py:419-480` | **PORT** — `playground_policy.PlaygroundObservation`. Sensors resolved by **role** through `team_env.sensor_names_by_role`, whose names are built from `g1_constants.py:53-58`, not typed. | parity (a): **max abs diff 8.7e-08** (reset state) and **1.8e-07** (random state) vs their `_get_obs` on identical inputs. Tolerance 1e-4. Per-block numbers printed. | ✅ **PASS** |
| — pelvis local linvel, gyro | `climb_env.py:422/462` via `base.py:86-102` | sensordata slices `local_linvel_pelvis` / `gyro_pelvis` | parity (a) block rows: max 1.4e-07 | ✅ |
| — projected gravity | `climb_env.py:431` (`site_xmat[imu_in_pelvis].T @ [0,0,-1]`) | same expression on the C `site_xmat` | parity (a): max 1.8e-07 | ✅ |
| — joint slices `qpos[7:36]`, `qvel[6:35]` | `climb_env.py:440/449` | slices built from `env._slide_qposadr` / `_slide_dofadr` | parity (a): max 2.9e-08 | ✅ |
| — gait phase (cos,cos,sin,sin) | `climb_env.py:458-460`, advance `:388-389`, `phase_dt` `:317-319` | `playground_policy.GaitPhase` | parity (a): max 8.7e-08 | ✅ (frequency pinned — see gap G4) |
| Termination / fall | `joystick.py:426-442` (`upvector_torso.z < 0`, three foot/shin self-collision sensors, NaN) | **PORT** — `playground_policy.TerminationCheck`. Self-collision sensor **ids** come from their `_post_init` (`joystick.py:239-247`); the up-vector sensor name from `consts.GRAVITY_SENSOR`. | 15 s runs report `fell at 0.66 s reason=self_collision`, matching their own zero-action collapse | ✅ |
| Command layout `[lin_vel_x, lin_vel_y, ang_vel_yaw]` | `joystick.py:95-103`, `:802-822` | Our `HeadingController` emits that 3-vector. Ranges printed in fingerprint `training_config.command_limits`. | live socket shows `command = [0.5, 0.0, 1.0]` with W held | ✅ |
| mels demo policy | `rl/scripts/viewer.py:80-100` `load_mels_policy` | **PORT** — numpy body lifted out, JAX wrapper dropped. Same npz file (`rl/policies/mels_g1_joystick.npz`). | shapes checked at load (103→512→256→128→58, first 29 used); parity (b) feeds the SAME policy object to both loops | ✅ |
| Wind drag law | `rl/environment/wind_env.py:28-36` (constants), `:57-59` (coefficient), `:92-103` (application) | **PORT**, constants imported at run time from `wind_env.default_config()`. Runtime prints `0.5*rho*Cd*A = 0.3675`. | live socket: 18 m/s dial → `wind_force_world_newtons = [120.3, -0.01, 0]`; 0.3675·18² = 119.1 N ✔ | ⚠️ **see gap G1** |

---

## Test (a) — observation parity, verbatim numbers

Both sides run with `noise_config.level = 0.0` (comparing determinism, not
RNGs). Two states: their `knees_bent` reset, and a random perturbation
(joints ±0.25 rad, base ±0.1 m, random quaternion, random qvel ±0.6, slide
0.37 m, non-zero command / last_action / phase).

```
reset knees_bent state: overall max abs diff 8.742e-08   (their obs norm 1.7321)
random perturbed state: overall max abs diff 1.835e-07   (their obs norm 4.3868)
  linear_velocity_pelvis    max 1.408e-07
  angular_velocity_pelvis   max 1.407e-07
  projected_gravity         max 1.835e-07
  command                   max 1.192e-08
  joint_angle_minus_default max 2.342e-08
  joint_velocity            max 2.847e-08
  last_action               max 2.970e-08
  gait_phase                max 6.817e-08
==> worst 1.835e-07  vs tolerance 1e-4   PASS
```

The residual is float32-vs-float64: their MJX pipeline is float32, ours reads
float64 off the C struct. It is three orders of magnitude below tolerance.

## Test (b) — rollout divergence, verbatim numbers

100 control steps (2.0 s), command `(0.5, 0, 0)`, the same mels policy driving
both, one shared start state (theirs, copied into ours — their reset draws base
velocity from a JAX RNG numpy cannot reproduce, and the point of this test is
the dynamics, not the RNG). Gait clock pinned to 1.375 Hz on both sides.

```
 step  t(s)  pelvis err(m)  slide err(m)  our slide  their slide
    1  0.02        0.00155       0.00006     0.0001       0.0000
    5  0.10        0.01558       0.00401     0.0131       0.0171
   10  0.20        0.02465       0.00271     0.1215       0.1242
   20  0.40        0.02404       0.00388     0.1452       0.1491
   50  1.00        0.22695       0.00388     0.1452       0.1491
  100  2.00        0.08908       0.00388     0.1452       0.1491

pelvis divergence: final 0.089 m, max 0.230 m, mean 0.091 m
slide  divergence: final 0.0039 m, max 0.0066 m
travel over 2 s:        ours +0.1452 m   theirs +0.1491 m   (2.6% apart)
pelvis displacement:    ours  0.7770 m   theirs  0.7913 m   (1.8% apart)
```

**Read this as the curve, not a threshold.** MJX and the MuJoCo C engine are
different numerical implementations of the same model; a falling humanoid is
chaotic, so per-step positions separate. What matters is that both do the *same
thing*: both collapse, both end within 2% on total travel and displacement, and
the ascender coordinate — the quantity the task is scored on — tracks to
**4 mm over 2 s**. The 0.23 m mid-roll peak is the two bodies falling out of
phase by a few tens of milliseconds, then re-converging.

## Test (c) — 15 s of mels on the team env, vs their own baseline

`--duration 15 --keep-going --no-render`, observation noise off, no wind,
deterministic reset (zero base velocity).

| arm | fell at | reason | rope travel | pelvis displacement | height gained | max grip force | hand off line |
|---|---|---|---|---|---|---|---|
| **ours, W held** (`lin_vel_x = 0.5`) | **0.66 s** | self_collision | **+0.187 m** (monotone, then frozen) | 0.787 m | −0.703 m | 1162 N | 1.8e-04 m |
| **ours, no command** | **0.48 s** | self_collision | **+0.130 m** | 0.794 m | −0.730 m | 1097 N | 1.8e-04 m |
| **theirs**, `rl/tests/test_climb_env.py`, zero actions 5 s | n/a (test does not check falling) | — | 0.0000 m | — | pelvis ends 0.000 m above the slope | — | < 0.05 m (asserted) |

**mels does not climb, and does not even stand.** It is a flat-ground joystick
walking policy; on a 30° incline with one hand roped to a fixed line it
self-collides within half a second, drops ~0.7 m, and then **dangles from the
ascender for the remaining 14 s at a steady 250–320 N** (peak 1162 N at the
catch). This agrees with their own test's zero-action arm, which also just sags.
It is the correct, useful baseline for Chloe: the harness is healthy, the
policy is not trained for the task.

The ascender itself is provably working in every run: rope travel is monotone
and freezes (never slips back), and the palm stays on the line to 0.18 mm.

---

## GAPS / ASKS

Everything below is either something we could **not** guarantee, or something
missing on the `rl/` side that would make the harness (and the training) better.
Precise references; paste straight to the team.

**G1 — the climb env has no wind, so the demo's wind dial is not trained.**
`rl/environment/climb_env.py` never touches `xfrc_applied`; wind lives only in
`rl/environment/wind_env.py`, a *sibling* subclass of `Joystick`
(`wind_env.py:40`). The harness wires the dial with wind_env's own law and
constants (`wind_env.py:28-36`, `:57-59`, `:92-103`), applied to the torso body
— but the policy has never seen it. Flagged in the live state message as
`"wind_in_training": false` and on the page itself.
**ASK:** if wind matters for the demo story, either give `G1ClimbAscender` a
`wind_config` (the two classes could share a `WindMixin` — the drag code is 12
lines) or say explicitly that wind is out of scope for the climb task.

**G2 — no Lhotse terrain / hfield in the climb env.** `G1ClimbAscender.__init__`
*rejects* anything but `flat_terrain` (`climb_env.py:107-111`: "the
rough_terrain hfield cannot be tilted"). Meanwhile `terrain/` and
`assets/environments/lhotse_face/` build a real mesh of the face. The harness
therefore has exactly ONE world, `team_climb_30` (a tilted plane), and the map
selector says so; a `{"type":"world"}` message for any other name is logged and
ignored.
**ASK:** is real terrain in scope? If yes, the blocker is that a tilted plane
and an hfield cannot both be the floor — the fix is to tilt the *robot spawn +
line* instead of the floor, or to bake the slope into the hfield. Worth a
decision before more is built on the tilted-plane assumption.

**G3 — no snowshoes / boots, no jacket, no ascender end-effector body.** The
model is the stock menagerie G1: feet are `left_foot` / `right_foot` capsules on
`*_ankle_roll_link` at 0.608 kg, total mass 33.4411 kg. (Our previous rig,
`pemba_bench`, carried 1.6 kg of snowshoes and a 0.3 kg ascender body, for
33.65 kg — for comparison, not as a request.) The harness is written to inherit
these changes automatically: no geom/body/site name is hard-coded, and
`fingerprint.json` holds the full body/geom/mass tables so a changed robot shows
up as a diff.
**ASK:** when the robot changes, nothing needs doing on our side except a
rerun — but please keep new foot geoms listed in
`g1_constants.FEET_GEOMS`/`FEET_SITES` (that is where we read them from,
`joystick.py:202-212`) rather than only in the MJCF.

**G4 — the gait frequency is randomised per reset and we have to pin it.**
`climb_env.py:317` draws `gait_freq ~ U(1.25, 1.5)` and stores only the derived
`phase_dt` in `info`. The harness pins 1.375 Hz (the midpoint) so runs are
reproducible and the parity test is meaningful.
**ASK:** none required — just be aware the demo runs one fixed gait clock.

**G5 — observation noise is ON at training time (level 1.0) and OFF in the
harness.** `joystick.py:42-51`: gyro 0.2, gravity 0.05, joint_pos 0.03,
joint_vel 1.5, linvel 0.1, all × level 1.0. We run at 0.0 (a demo wants the
policy's best behaviour, and a deterministic obs is what makes test (a)
meaningful). The switch exists — `PlaygroundObservation(noise_level=...)`.
**ASK:** none. Noted so nobody reads the demo as a noise-robustness result.

**G6 — brax PPO checkpoints cannot be loaded without JAX; we need an npz
export.** `viewer.py:63-77` `load_policy` restores an orbax/brax checkpoint and
builds a jitted JAX inference function. The harness deliberately runs pure numpy
(the whole point of the plain-MuJoCo loop is that it is fast and dependency-light
on the demo laptop), so it can only load the mels-style npz
(`viewer.py:80-100`).
**ASK (highest value one here):** add
`rl/scripts/export_policy_npz.py --checkpoint DIR --out policy.npz` that writes
the same key layout the mels npz uses — `obs_mean`, `obs_std`,
`hidden_{0..N}_kernel`, `hidden_{0..N}_bias` — so a trained checkpoint drops
straight into `--policy`. Until that exists, every trained policy has to be
demoed through JAX.

**G7 — `_build_model` is a method entangled with `self`, so nothing else can
build the climb model.** `climb_env.py:130-262` reads `self._config`,
`self._task`, `self.sim_dt` and writes `self._mj_model`, `self._mjx_model`,
`self._init_q`, `self._slide_qposadr`, `self._default_pose`, `self._lowers`…
Consequence: getting an `MjModel` costs a full `G1ClimbAscender.__init__`,
which imports JAX, `mjx.put_model`s the whole thing, and takes ~25 s. The
harness pays that once per launch.
**ASK:** split out a module-level
`build_climb_spec(config) -> mujoco.MjSpec` (or `build_climb_model(config) ->
mujoco.MjModel`) with `_build_model` calling it. Pure win: the env keeps
working, and viewers/harnesses/tests get the model in ~1 s with no JAX.

**G8 — `PLAN.md` and the code disagree about the grip.** `rl/PLAN.md:9` says
"right-hand rail: world → slide joint (rope axis) → **ball → wrist weld**". The
code does a `connect` equality on the `right_palm` **site**
(`climb_env.py:212-218`) — a point constraint with **no** rotational coupling.
We have kept `connect` (it is what trains, and a free wrist is the right model
for a hand through an ascender handle).
**ASK:** update PLAN.md, or say if the weld was actually wanted — it changes the
arm's load path materially.

**G9 — the rope's height and lateral position are derived from the rest palm,
not chosen.** `climb_env.py:157-166`: compile a throwaway model, forward the
`knees_bent` keyframe, take `site_xpos[right_palm]`, offset by
`line_offset_y` (default 0). So the line sits at 0.663 m — waist height — and
0.227 m to the robot's right, wherever the keyframe happens to put the palm.
Anyone re-tuning `knees_bent` silently moves the rope.
**ASK:** is waist height intended? A real fixed-line ascent has the hand
*above* the head. If the rope should be higher, the reset pose has to reach up
to it, which is a different keyframe — worth deciding before reward shaping.

**G10 — the observation has no rope terms, and the layout will change if it
gains any.** `climb_env.py:471-480` is exactly the upstream 103-d joystick
vector; nothing tells the policy where it is on the line, how loaded the grip
is, or where the line runs. `hand_line_error` / `hand_height_on_line`
(`climb_env.py:536-547`) exist but feed nothing.
**ASK:** if Chloe adds rope terms (slide travel, grip force, line direction in
the pelvis frame), please (a) append them at the END of `state` and (b) tell us
— `playground_policy.PlaygroundObservation` is a hand port and will need the
same edit, plus `OBSERVATION_SIZE` bumped. Test (a) will catch it loudly, but
only if someone runs it.

**G11 — the reset randomises base velocity through a JAX RNG we cannot
reproduce.** `climb_env.py:299-302`: `qvel[0:6] ~ U(-0.5, 0.5)`. Our reset
defaults to zero (deterministic demo) with `--randomise-reset-velocity` to draw
the same distribution from numpy — same distribution, different stream. This is
why parity test (b) copies their start state instead of re-deriving it.
**ASK:** none.

**G12 — `env.step` is never used by the harness, so their reward/metrics path
is untested here.** We reimplement the *physics* half of `climb_env.py:355-413`
and skip `_get_reward` entirely (a demo has no reward). If a reward bug ever
looks like a physics bug, this harness will not see it.
**ASK:** none — just do not read a good demo as evidence the reward is right.

**G13 — the fall check fires while the robot is visibly still upright.**
`joystick.py:428-436` terminates on foot/shin self-collision, which in our runs
triggers at `torso_upvector_z = +0.789` (i.e. ~38° from vertical, still
standing). The HUD therefore shows "FELL" before it looks like a fall.
**ASK:** none, but worth knowing if the demo's banner ever looks premature —
it is their termination condition, faithfully reported.

**G14 — `njmax` is sized by hand and could bite on new contacts.**
`climb_env.py:79`: `njmax = 29*2 + 8*4 + 8`, commented as "3 (connect) + 1
(slide frictionloss) + margin". Plain MuJoCo grows constraints dynamically so
the harness is unaffected, but MJX will silently drop constraint rows if the
count is exceeded — which is exactly the kind of thing that shows up as "the
policy learned something weird".
**ASK:** if geometry gets added (boots, a second ascender, terrain contacts),
re-check this number.

**G15 — no `ffmpeg` dependency is declared, and the live recorder is capped.**
`app/harness/recorder.py` skips `episode.mp4` with a warning when ffmpeg is
missing (`frames.npz` / `hud.json` are always written). A live session holds
every JPEG in RAM to mux at the end — ~2 MB/s — so the harness caps
`episode.mp4` at 6000 frames (2 minutes) and keeps recording the numeric rows.
**ASK:** none; ours to own.

---

## What is ours, so nobody looks for it in `rl/`

`W` → `lin_vel_x = 0.5`; the camera-heading yaw controller (gain 2.0/rad,
±1.0 rad/s, 2° deadband); the third-person orbit and its half-turn azimuth
offset; the wind dial; the friction slider; pause/reset/record/replay; the
websocket protocol and the page. None of it exists on their side, and none of it
touches the physics except through the command 3-vector, `xfrc_applied` (wind)
and `geom_friction` (the slider) — both clearly marked in the state message and
on the page.
