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
../.venv_everest/bin/python -m app.harness.runtime --world climb_30 --duration 10 --hold-w --keep-going
../.venv_everest/bin/python -m app.harness.runtime --live --world free_0   # http://localhost:8766/
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

## The four worlds — a stock-walking baseline, built from their env

The map selector offers four worlds (`app/harness/worlds.py`). All four are
**their** `G1ClimbAscender`; nothing is rebuilt by us. Two knobs separate them:

* **slope** — a `config_overrides` entry, `climb_config.slope_deg`, handed
  straight to their constructor. Nothing else is overridden.
* **rope** — `data.eq_active[grip] = 0/1`, MuJoCo's own per-step enable for an
  equality constraint. Their `climb_config` has no "disable the grip" key and
  adding one would mean editing `rl/`, so the free worlds simply run the
  identical model with the constraint off. The carrier body and the visual rope
  stay where their `_build_model` put them; we set the two apparatus geoms'
  **alpha to 0** so the picture is not misleading, which changes nothing
  physical (both are already `contype=0/conaffinity=0`, climb_env.py:181-182
  and :206-207). Those two geoms are **derived**, not named: the carrier body is
  the one owning the slide joint, and the rope body is the static body sitting
  at `line_point_world`.

| world | slope | rope | what it isolates |
|---|---|---|---|
| `climb_30` | 30° | on | the training task itself (the harness default) |
| `free_30` | 30° | **off** | how much of the behaviour is the slope alone |
| `free_0` | 0° | **off** | the stock mels flat-walking baseline |
| `climb_0` | 0° | on | what the grip alone costs a walker, slope removed |

**Model cache.** Models are built lazily on first selection and cached. The
cache key is the **full frozen `config_overrides` dict**, never a subset — a key
that omits an override that changes the model is how a run silently reads a
stale one. Because the rope flag is not an override, `climb_30`/`free_30` share
one model and `free_0`/`climb_0` share another: **four worlds, two builds.**

Measured build cost, warm (`WorldLibrary` in a fresh process): **free_0 1.64 s,
climb_30 0.21 s, the two cache hits 0.00 s — 1.92 s for all four.** This
corrects an earlier "~25 s per world" figure in this file, which was a
**cold-start artifact**: the first run in a fresh venv also clones
`mujoco_menagerie` and compiles bytecode. The sim loop is still what blocks
during a build, so it broadcasts one state carrying `"loading": true` first, and
the page's toast timeout was raised from 8 s to 60 s — headroom for a cold
start, not a measured wait. (Measured against `app/web/index.html`, which was
the front end at the time; `app/web/render3d.html` carries the same 60 s.)

## Test (d) — the four worlds with mels, W held (lin_vel_x 0.5), 10 s

`--duration 10 --hold-w --keep-going --no-render`, noise off, no wind,
deterministic reset.

| world | slope | rope | fell? | fell at | pelvis displacement | rope travel | height gained | max grip force |
|---|---|---|---|---|---|---|---|---|
| `climb_30` | 30° | on | **yes** | 0.66 s | 0.765 m | **+0.187 m** | −0.708 m | 1162 N |
| `free_30` | 30° | off | **yes** | 0.50 s | **25.124 m** | 0.000 m | **−13.510 m** | 0 N |
| `free_0` | 0° | off | **no** | — | **4.320 m** | 0.000 m | −0.004 m | 0 N |
| `climb_0` | 0° | on | **no** | — | 4.377 m | **+4.369 m** | −0.001 m | 22 N |

What the four rows say together, which no single run could:

* **mels walks fine on the flat** (`free_0`): 10 s upright, 4.320 m =
  **0.432 m/s at a commanded 0.5 m/s** (86% of command). At
  `--command-speed 1.0` it does **0.854 m/s** (85%), against
  `README.md:82`'s claim of "0.75 m/s at cmd 1.0" — same ballpark, ours ~14%
  faster; the claim is not exactly reproduced but the policy's ~15%
  under-tracking of its command is.
* **The rope is not what breaks it** (`climb_0`): with the grip on but the
  slope removed, it still walks 4.377 m without falling, and the ascender
  tracks the walk to **4.369 m of travel at only 22 N** of grip force. On flat
  ground the fixed line costs a walker essentially nothing.
* **The slope is what breaks it** (`free_30`): unroped on the incline it falls
  in 0.50 s and then tumbles **25 m down and 13.5 m below** the spawn for the
  rest of the run — the plane is infinite, so there is nothing to stop it.
* **The rope catches it** (`climb_30`): the same fall, at 0.66 s, but the robot
  ends **0.765 m** from spawn instead of 25 m, hanging at 250–320 N (1162 N at
  the catch). The ascender does exactly its job; what is missing is a policy
  that can stand on a 30° slope.

## ClimbScene worlds — the merged scene (PR #8)

Twelve of the sixteen worlds now run `rl/environment/climb_scene.py`: the
Lhotse Face heightfield, a rope polyline draped over it, a bead-on-a-wire
carrier, the jacketed G1. `app/harness/climb_worlds.py` calls `build_scene`
and drives what comes back. **It builds nothing.**

### What is his, and what is ours

| piece | source of truth | how we consume it |
|---|---|---|
| the whole model | `climb_scene.build_scene(...)` | **called**. Terrain, rope, carrier, grip equality, slope-fitted spawn all come back compiled |
| the physics step | `ClimbScene.step(wind)` | **called**, 10× per 50 Hz control tick. It is one `mj_step` + the carrier projection + the arc-length ratchet, with wind drag written into `xfrc_applied` first |
| the 103-d observation | `walk_policy.WalkController.observe()` | **called**. His is the contract now; ours is measured against it (below) and used only for the legacy worlds |
| the policy + gait clock + `last_act` | `walk_policy.WalkController.substep()` | **called**. It writes `data.ctrl` and evaluates the net once per decimation |
| wind | `climb_scene.WindParams` | **constructed** from the dial (m/s + heading) and handed to `step()` |
| friction | `FrictionParams.from_scalar` at build; live knob writes `geom_friction` | see the note on contact pairs below |
| the robot | `robot.resolve` / `robot.adapt` | selected by name (`himalaya` / `playground`), never by a path we typed |
| command clamp | `walk_policy.CMD_LIMITS` | **imported**, not restated |
| climb metric | `RopeCarrier.progress` (his ratchet state) | arc length along the rope since spawn |

### `policy_compat` — what it actually does

`robot.adapt(spec, policy_compat=True)` is on by default and is **not
cosmetic**. The jacketed robot ships stock menagerie dynamics; Playground
retuned them for RL and the mels policy learned against *that* plant. Measured
on the built model, `adapt_report` reports:

```
added   : site right_palm, sensor local_linvel_pelvis, sensor gyro_pelvis,
          keyframe knees_bent
retuned : foot contact: 4 spheres -> playground box
          actuator kp/kv, dof damping/armature/frictionloss
```

Concretely it rewrites actuator kp (500 uniform → 75/20/2 per joint), kv, dof
damping, armature and frictionloss, and swaps the four 5 mm foot spheres for
Playground's single contact box. Our compiled scene reads back
**`actuator kp min/max 2/75`**.

**This is the fix for the kp=500 problem the previous report raised as ASK P1.**
That ask is resolved upstream: the plant mismatch is handled by an explicit,
documented, default-on shim, and `--stock-plant` keeps the robot exactly as the
project specifies it (his measurement: the walking policy then falls at ~1.8 s).

### Gates — both run, both printed

    python -m app.harness.climb_worlds --world lhotse_B

**Joint parity.** The scene's 29 actuated joints against Playground's own
model, in order. Result: **29 vs 29, identical order.** Re-run here rather than
inherited, because the merged scene is a different build path.

**Observation parity.** Our `PlaygroundObservation` against his `observe()` at
the same state, reset and perturbed:

```
obs parity (reset state):     max |ours - his| 0.000e+00   his norm 3.5787
obs parity (perturbed state): max |ours - his| 0.000e+00   his norm 3.5618
observation parity worst 0.000e+00  PASS
```

**Bit-identical.** One input has to be handed over rather than derived, and it
is load-bearing: `default_pose`. His is pinned to `robot.KNEES_BENT_QPOS`; ours
reads the compiled keyframe. On a slope `build_scene` *leans the base and
re-pitches the ankles* so the soles lie flat, so the scene's keyframe is no
longer the pose the policy's action deltas are about. Reading it — which is
correct for the flat legacy env — would silently move the policy's operating
point with the terrain. His is right; ours takes his value.

**His own acceptance suite:** `python -m rl.scripts.climb_scene --check` →
**ALL CHECKS PASSED** (13 checks).

### What the scene does not have, and what we do instead

| absent | what we do |
|---|---|
| `upvector_torso` sensor (`adapt` adds only `local_linvel_pelvis`, `gyro_pelvis`) | read the same quantity off `site_xmat[imu_in_torso]` column z — that IS what a `framezaxis` sensor on that site reports |
| the seven `..._found` contact sensors | **our fall test is narrower than the training env's**: upright < 0 or non-finite state, with no foot/shin self-collision term. Stated here because a "did not fall" on a ClimbScene world is a weaker claim than on a legacy one |
| a rope-off switch in `build_scene` | the free world deactivates the `ascender_grip` equality through `data.eq_active`, re-applied after every reset (`mj_resetDataKeyframe` restores it from the model) |

### mels on the merged scene, W held (`lin_vel_x` 0.5), 10 s

| world | robot | rope | slope | fell | at | arc climbed | height | max rope | hand-off-rope | pelvis displ |
|---|---|---|---|---|---|---|---|---|---|---|
| `flat_0` | himalaya | on | 0.4° | **no** | — | **+1.855 m** | −0.076 m | 137 N | 0.0003 m | 1.803 m |
| `lhotse_B` | himalaya | on | 38.6° | **no** | — | +0.002 m | −0.741 m | 309 N | 0.0023 m | 0.768 m |
| `lhotse_B_playground` | playground | on | 38.6° | **no** | — | +0.002 m | −0.796 m | 580 N | 0.0038 m | 0.828 m |
| `lhotse_B_free` | himalaya | **off** | 38.6° | **yes** | 3.14 s | 0.000 m | **−276.4 m** | 0 N | 272 m | 276.6 m |

What the four rows say together:

* **On the flat the ascender tracks a walking robot.** `flat_0` walks and the
  carrier climbs 1.855 m of arc, hand 0.3 mm off the rope throughout.
* **On the real face the policy cannot climb, and the rope catches it.**
  `lhotse_B` slips to −0.74 m and hangs at ~300 N; it never tips past upright,
  and the hand stays 2 mm off the line. This reproduces his own documented
  baseline ("slips, then hangs from the rope, dz −0.83 m").
* **The jacketed robot and the Playground robot behave the same** on the same
  scene — −0.741 m vs −0.796 m, arc 0.0020 vs 0.0023 m. That is
  `policy_compat` working: his "103-dim observation bit-identical between the
  two robots" claim holds dynamically, not just statically.
* **Without the rope it is a 276 m slide.** `lhotse_B_free` tips at 3.14 s and
  ends a quarter-kilometre down the face. The rope is the entire difference
  between "hangs, −0.74 m" and "gone".

**Caveat, his and worth repeating:** the command is *body-frame* forward
velocity and yaw drifts freely, so world-frame pelvis displacement is not a
tracking measure. Read the arc-length column, not the displacement column.

### Fingerprints

`fingerprint_lhotse_B.json`, `fingerprint_lhotse_B_playground.json`. Highlights
for B:

```
model     nq 39  nv 38  nu 29   (7 base + 29 robot + 3 carrier slides)
          nbody 32  neq 1  nsensor 7  nhfield 1   dt 0.002  mass 34.4411 kg
terrain   patch B, 38.60 deg (REAL, Copernicus GLO-30)
          hfield 12.5 x 7.5 x 0.698 m, 300 x 500 grid, friction 0.9
rope      29.476 m, 9 waypoints, rise 18.409 m over run 23.002 m
          => mean slope 38.67 deg (matches the terrain to 0.07 deg)
ascender  carrier 1.0 kg, ratchet on, arc at spawn 6.368 m
          grip CONNECT solref [0.004, 1.0] solimp [0.95, 0.99, 0.001, 0.5, 2.0]
spawn     lean 12.4 deg, ankle -47.0 deg, hand-rope distance 5.6e-17 m
control   action_scale 0.5, ctrl_dt 0.02 x 10 substeps, kp 2..75
```

The three carrier slide joints are appended AFTER the robot's, so the robot's
joints are `qpos[7:36]` / `qvel[6:35]` — **bounded slices only**. An open-ended
`qpos[7:]` picks the carrier up as three phantom joints. Ours are bounded and
the obs parity result proves it.

### Contact pairs — the trap that does NOT apply here

His doc records that `climb_env.py`'s `foot_friction` knob is inert, because
the G1 XML declares `<pair name="left_foot_floor" friction="0.6 0.6"/>` and an
explicit pair overrides both geoms' friction. **The merged scene compiles with
`npair 0`** — verified on the built model, not assumed — so the live friction
knob writing `geom_friction` on the foot and terrain geoms does take effect
here. The legacy worlds still have the bug, upstream.

### Gaps this raised

**C1 — neither fetch path for the scene's assets works on a clean machine.**
`rl/tools/fetch_reference_model.py` dies with
`URLError: [SSL: CERTIFICATE_VERIFY_FAILED]`, and it writes to `rl/.reference`
(`__file__.parent.parent`) while `robot.py` reads `<repo>/.reference` — so even
a successful fetch lands in the wrong directory. Separately,
`assets/robots/mujoco/build.py --fetch` still dies with
`ModuleNotFoundError: No module named 'pxr'`. `app/harness/provision_assets.py`
sidesteps all three by copying from the `mujoco_playground` already installed in
the venv (same pinned menagerie commit, no network). **ASK:** fix the
`parent.parent` path, and either vendor the reference or make the fetch
tolerate a proxy.

**C2 — the trainer is still on the old env.** His doc's "Not yet done:
training" — `rl/environment/{climb_env,wind_env}.py` still use the flat tilted
plane and the slide joint. So **wind and this terrain are demo-only**, and the
state message keeps `wind_in_training: false` and adds
`terrain_in_training: false`. The four `legacy_*` worlds exist precisely
because that is still the thing being trained.

**C3 — no self-collision termination on the merged scene.** See the table
above; our fall test is narrower than the training env's. If `adapt` gained
Playground's seven `..._found` contact sensors the two would match.

**C4 — `B_slope*` slopes are synthetic overrides.** `--list` says so and the
world list carries `slope_provenance`, but it is worth repeating where the
numbers get quoted: `slope_45` is *not* the Lhotse Face at 45°. Only
`lhotse_A/B/C/D` have measured slopes, and even there everything finer than
~30 m is synthetic (one patch covers 0.447 of a DEM cell).

### Asks from earlier reports that PR #8 RESOLVED

* **P1 — actuator gains (was the blocker).** Resolved by `policy_compat`.
* **P2 — point `climb_env` at the jacketed MJCF.** Superseded: `climb_scene`
  takes `robot_scene=` directly and defaults to the jacketed robot, so no
  monkeypatch is needed for the new path. (Our `robot_variants.py` monkeypatch
  survives for the legacy env only.)
* **P3 — the robot needs Playground's names.** Resolved by `robot.adapt`,
  which adds `right_palm`, the two sensors and the `knees_bent` keyframe.
* **P4 — `knees_bent` was Playground's, not the robot's.** Resolved: `adapt`
  installs it, and `build_scene` re-poses it per slope while `walk_policy`
  keeps the policy's `default_pose` pinned to the training value.
* **P6 — `build.py --fetch` broken.** Still broken; now tracked as C1.

Still open from earlier: **G6** (brax → npz export for trained checkpoints),
**G10** (obs layout will change if rope terms are added — his doc says the same:
"keep the observation at 103 dims"), **G14** (`njmax` sizing, MJX only),
**P5** (MJX CCD warning on mesh/cylinder pairs — now more relevant, since the
merged scene has a rope cylinder *and* mesh collision geoms).

## Graphics, natural wind, the sandbox, and A/D — all visual or input-side

### Physics is untouched, and that is MEASURED

The alpine look sets `model.vis`, `geom_rgba`, `light_*` and render-time flags,
and installs a skybox by recompiling the spec. None of it is read by the
solver, but a spec recompile is exactly the kind of change that could move
something quietly, so it is checked rather than asserted. Same world, same
seed, 6 s, W held, with and without `--plain-graphics`:

| metric | alpine | plain | delta |
|---|---|---|---|
| pelvis displacement | 0.768703118 m | 0.768703118 m | **0** |
| height gained | −0.750794943 m | −0.750794943 m | **0** |
| rope travel | 0.002006123 m | 0.002006123 m | **0** |
| max rope force | 304.229107608 N | 304.229107608 N | **0** |
| hand off line | 0.002243983 m | 0.002243983 m | **0** |
| `frames.npz` root_position_world | | | **0** |
| `frames.npz` rope_force_newtons | | | **0** |

**Bit-identical.** The graphics pass cannot change a result.

### The skybox, and why a recompile is safe

A skybox is a TEXTURE and textures are fixed at compile time; the merged scene
compiles with `ntex 2`, both `mjTEXTURE_2D`. Without a skybox texture
`mjRND_SKYBOX` does nothing and the background renders BLACK — which is what
the first attempt looked like.

`build_scene` returns `scene.spec`, so `graphics.add_skybox` adds a gradient
skybox there and recompiles. A texture is an ASSET, not structure: measured,
all 14 structural fields come back bit-identical (`nq nv nu nbody njnt neq
ngeom nsite nsensor nkey`, `jnt_qposadr`, `jnt_dofadr`, `body_mass`, actuator
targets), with only `ntex` going 2 → 3. `add_skybox` re-checks that list on
every call and REFUSES the swap if anything moved, so the day this stops being
true it fails loudly. One real gotcha it also handles: `vis.global_.offwidth`
lives on the compiled model, not the spec, so a recompile resets the offscreen
framebuffer to 640 and the next 1920-wide renderer raises
`Image width 1920 > framebuffer width 640`. It is carried across explicitly.

### Exposure — the first pass was worse than stock

Snow 0.90 + ambient 0.42 + sun 1.00 summed past 1.0 everywhere and clipped: the
rendered face came back a **featureless white sheet with less visible relief
than the stock grey**. Looking at the frame is what caught it; the fps numbers
were fine. Snow is bright but a camera is not, and what sells snow is the
CONTRAST between the lit and shaded sides of the roughness. Now: snow 0.82,
ambient 0.20, sun 0.78, total near 1.0 and mostly directional, sun at 16° so it
rakes. Measured saturation (pixels > 250) is **0.0%**.

### Render cost, 1920×1080, lhotse_B

| | ms/frame | fps |
|---|---|---|
| stock visuals | 16.2 | 61.9 |
| alpine, shadows ON | 14.9 | 67.1 |
| alpine, shadows OFF | 9.2 | 109.0 |

Shadows cost **5.7 ms/frame** against a 20 ms control tick, and the alpine look
is *faster* than stock because fog culls distant geometry. So shadows stayed ON
at every size we rendered — the "off above 1280 wide" contingency was not
needed. **Live at 1920×1080 with shadows: realtime factor 1.00.**

> **SUPERSEDED 2026-08-30 — the render these numbers measure no longer exists.**
> The third-person render, `--no-shadows` and `graphics.shadows_affordable` went
> with the 2-D page; `app/web/render3d.html` draws its own shadows in three.js.
> The measurements stay because they are why the eye cameras render with
> `shadows=False`: the 4096² shadow pass costs the same whatever the output
> size, so it is the most expensive thing in a 320×240 eye render and it buys a
> block matcher nothing. `--plain-graphics` is still live, and is no longer
> cosmetic — the eyes render through the same model.

No noise texture on the snow, deliberately: a texture needs a compile-time
asset for the same reason a skybox does, and the heightfield's own 12 cm
roughness under a raking sun already gives the surface its texture — real
geometry rather than a painted-on pattern.

### Natural wind

`wind_natural` (0/1, default 0) makes the dial a TARGET rather than a constant.

    speed   = target * clamp(1 + OU(sigma 0.25, tau 4 s), 0.4, 1.6) * (1 + gust)
    heading = target + OU(sigma 15 deg, tau 6 s)
    gust    = raised cosine, arrivals every 3-8 s, +20-60%, lasting 0.5-1.5 s

The OU processes are integrated EXACTLY (`x <- x e^(-dt/tau) + sigma sqrt(1 -
e^(-2dt/tau)) N(0,1)`), not Euler-stepped, so sigma and tau do not drift with
the tick rate. Everything draws from one generator seeded from `--seed` and
advances once per control tick, so a replay at the same seed sees the same
weather. Measured over 300 s at a 12 m/s target:

| seed | speed mean | sd | min | max | heading sd | gusts |
|---|---|---|---|---|---|---|
| 0 | 12.54 | 2.90 | 4.80 | 27.48 | 16.09° | 45 in 300 s |
| 1 | 11.78 | 2.94 | 4.80 | 24.18 | 15.83° | 47 in 300 s |

Determinism at equal seed: PASS. Disabled: **bit-exact pass-through** of the
dial value — verified, not assumed.

Live, the state message carries `wind_speed_mps`, `wind_heading_degrees`,
`wind_gain`, `wind_gust` and `wind_natural` alongside the existing
`wind_velocity_world_meters_per_second` and `wind_force_world_newtons`; all of
them are recorded, because a replay that stored only the dial could not
reproduce the gust that knocked the robot over. Measured live at a 12 m/s dial:
off → exactly 12.00 every tick; on → mean 13.53, sd 1.47, range 10.81–18.03,
heading sd 8°.

### The sandbox map

**The shipped DEM patches are fixed at 25 × 15 m.** `terrain.load_patch` reads
a whole `.npz`; there is no crop or window argument anywhere in that module, so
a bigger map cannot come from the DEM without new code. `terrain.make_terrain`
DOES take an arbitrary `length_m`/`width_m` and builds from the same octave
recipe, so the sandbox uses that — **synthetic, not measured**, and it must
never be quoted as terrain evidence.

Measured at 1920×1080 with an idle robot:

| size | res | grid | fps | hfield MB |
|---|---|---|---|---|
| 25 × 15 m | 0.05 | 300 × 500 | 60.9 | 0.6 |
| 60 × 60 m | 0.10 | 600 × 600 | 54.4 | 1.4 |
| 120 × 120 m | 0.15 | 800 × 800 | 39.0 | 2.6 |
| **120 × 120 m** | **0.20** | **600 × 600** | **48.8** | **1.4** |
| 200 × 200 m | 0.25 | 800 × 800 | 37.9 | 2.6 |
| 200 × 200 m | 0.30 | 667 × 667 | 43.7 | 1.8 |

**Physics was 20–29× realtime at every size** — the heightfield never bound the
solver. Render cost is what buys area, and it tracks the GRID, not the metres.
200 × 200 m is affordable only by making cells so coarse (0.30 m) that the
finest roughness octave (0.6 m correlation) spans two cells and stops being
resolved. 120 × 120 m at 0.20 m keeps three cells per finest octave, leaves
headroom for the graphics pass, and is still **38× the area of a patch**
(14 400 m² against 375). `sandbox_free` and `sandbox_rope` (the scene's own
route builder lays a 120.7 m rope across it); live realtime factor 1.00.

### Uneven-terrain ladder, rope off

`load_patch` runs a least-squares DE-PLANE, returning patch B's real
micro-roughness as a mean-zero grid (RMS 0.1138 m) with the macro tilt on the
geom's quaternion. So `dataclasses.replace(patch, slope_deg=X)` is the
roughness-preserving path — verified: `rough` stays bit-identical to patch B's.
This is used instead of the `B_slope*` files because those exist only at 0, 25,
30, 35, 45, 50 **and each carries its own noise seed** (roughness correlation
between B and B_slope25 is −0.06, i.e. unrelated draws). Reusing B's actual
roughness makes slope the only variable that changes across the ladder.

mels, W held, 10 s, rope off, jacketed robot, spawned at the bottom facing
uphill:

| world | slope | fell | at | displacement | height |
|---|---|---|---|---|---|
| `terrain_free_5` | 5° | **no** | — | 2.04 m | −0.36 m |
| `terrain_free_10` | 10° | **no** | — | 1.97 m | −0.52 m |
| `terrain_free_15` | 15° | yes | 3.00 s | 2.60 m | −1.46 m |
| `terrain_free_20` | 20° | yes | 2.04 s | 149.95 m | −149.65 m |
| `terrain_free_25` | 25° | yes | 2.88 s | 2.43 m | −1.69 m |
| `terrain_free_30` | 30° | *no* | — | 2.31 m | −1.74 m |

**The flat-ground walker gives up between 10° and 15°** on real micro-roughness
with nothing to hold. 20° is the one that finds a clean fall line and slides
150 m. Read the 30° "no" with the C3 caveat above: our fall test is upright < 0
with no self-collision term, and at −1.74 m of height that row is a robot
sitting on the slope, not one walking.

### A/D turn-in-place — works, but the policy tracks yaw poorly

A = +1.0 rad/s, D = −1.0 rad/s, both held cancel, all inside the policy's
trained `ang_vel_yaw` range. While A or D is held the camera-follow
`HeadingController` is SUSPENDED and its target re-seated to the robot's
current yaw every tick, so releasing does not snap the robot back toward the
camera.

The wiring is right — the issued command is exactly `[0, 0, ±1.0]` — but the
achieved turn is far below it. flat_0, 3 s, measured:

| rope | command | achieved | note |
|---|---|---|---|
| off | +1.0 (A) | **+9.4 °/s** | correct direction |
| off | −1.0 (D) | −1.9 °/s | correct direction, weak |
| on | +1.0 (A) | +1.7 °/s | tether dominates |
| on | +1.0 (A) + W | **−7.9 °/s** | **wrong direction** |

Commanded +1.0 rad/s is 57.3 °/s, so the policy delivers roughly a sixth of it
at best, asymmetrically. Two causes, both his and both documented upstream:
the mels policy only initiates a gait above ~0.4 command and a yaw-only command
leaves it shuffling in place; and his `HeadingController` docstring already
warns that the rope point-attaches the palm to a fixed line, so a heading
change drags the robot around its own hand. **Flagged, not fixed** — it is a
policy/tether limitation, not a wiring bug, and per the standing rule the fix
is not mine to choose. Yaw also oscillates ±2° per gait step, so "monotonic"
is ~0.5 at tick level even when the trend is clean.

## Pemba robot variant — the real demo robot on the rope

The four `*_pemba` worlds fly the robot the team actually built
(`assets/robots/mujoco/g1_unitree_ascender.xml`: jacket, snow boots, ascender
end-effector in place of the right hand) through the same
`G1ClimbAscender` env, the same surgery, the same everything else.

### How it is built

`app/harness/robot_variants.py` generates a Playground-compatible scene wrapping
the real robot, then redirects the one line their builder uses to pick a
starting scene — `consts.task_to_xml`, called at climb_env.py:136 and :231 and
joystick.py:121 — at that file for the duration of one constructor. Their
`_build_model` then tilts the floor, adds the rope and carrier, connects the
palm and sets foot friction, exactly as it does for the stock robot. **Nothing
under `assets/robots/mujoco/` is edited.** Generated files carry absolute mesh
paths, so they are machine-specific and gitignored under
`app/harness/generated/`.

The generated scene adds only what their code looks up **by name** and would
crash without. Every addition is a NAME on something that already exists, or a
sensor — never new collision geometry, because that would change the physics:

| what | how |
|---|---|
| `floor` plane geom, `groundplane` material, visual block | copied from Playground's `scene_mjx_feetonly_flat_terrain.xml` |
| keyframe `knees_bent` | Playground's qpos/ctrl **verbatim** — legal only because joint parity passes (below) |
| site `right_palm` | the hand is gone; placed at the ascender's own bounding-box centre, where the rope runs through the device |
| site `left_palm` | left arm untouched, so Playground's exact position (0.08, 0, 0) |
| geoms `left_foot`/`right_foot` | **named** one of the four existing collision spheres per foot |
| geoms `left/right_shin`, `left/right_thigh`, `left/right_hand_collision` | **named** the existing collision mesh on each link. On the right arm that is the **ascender**, which is what is out there now |
| 22 state sensors | Playground's, verbatim — they only reference sites the robot already has |
| 7 `..._found` contact sensors | re-aimed at **bodies** instead of Playground's geom names. More faithful, not less: `body1="left_ankle_roll_link"` covers all four foot spheres, where a geom name would cover one |
| mesh paths | the robot points at `../g1/_menagerie/...` (a 34 MB gitignored fetch that needs `usd-core`); the identical menagerie STLs already ship inside `mujoco_playground`, so we point there and skip the fetch — the same rewrite their own `_rewrite_mesh_paths` does (climb_env.py:83-95) |

### Joint parity — the gate the whole variant rests on

`verify_joint_parity()` runs before anything else and **raises** if it fails,
because copying Playground's `knees_bent` and feeding the policy's 29 actions to
these actuators is only meaningful if the joint lists match name-for-name in
order. Result: **29 vs 29, identical names, identical order.** The ascender does
**not** replace any wrist joint — all three right wrist joints are present; the
ascender is a geom mounted on `right_wrist_yaw_link`.

### The measured diff (bare Playground G1 -> Pemba G1)

```
total mass        33.4411 kg -> 33.5411 kg   (+0.1000 kg)
bodies   33 -> 33     geoms 74 -> 92     sites 7 -> 8
joints   31 -> 31     actuators 29 -> 29     sensors 29 -> 34

per-body mass changes: 1 of 33 bodies
  right_wrist_yaw_link        0.2546 kg -> 0.3546 kg     (the ascender)

feet   bare : 1 named box  0.09 x 0.03 x 0.008, condim 3, mu 0.8
       pemba: 4 spheres per foot, r 0.005, condim 3, mu 0.8 (one named per foot)

joints    identical order: True
actuators identical order: True
actuator kp   bare 2-75 per joint      pemba 500 for all 29      IDENTICAL: False

grip   bare  palm/line at [0.14357, -0.2268, 0.66346]
       pemba palm/line at [0.13970, -0.22663, 0.66425]
       the line moves 4.0 mm
```

Jacket, boots and logos are visual-only exactly as their README says: they add
18 geoms and **zero** mass. The only mass change in the whole robot is the
ascender's +0.1 kg on the wrist. The grip point moves 4.0 mm, so the rope sits
essentially where it did.

Full tables: `fingerprint_pemba.json` (30°) and `fingerprint_pemba_slope_0.json`
against `fingerprint_slope_30.json` / `fingerprint_slope_0.json`.

### Does the observation still work, and does mels still fly it?

Observation is **still 103-d** and finite; `default_pose`, `action_scale`,
ctrl_dt and the substep count are identical; the mels npz loads and produces a
finite 29-vector. So the policy runs. It just runs badly:

| run (mels, W held = lin_vel_x 0.5, 10 s) | fell? | fell at | pelvis displ | rope travel | max grip |
|---|---|---|---|---|---|
| `free_0` (bare) | **no** | — | **4.320 m** | — | — |
| `free_0_pemba` (as built) | **yes** | 2.16 s | 0.729 m | — | — |
| `free_0_pemba` + Playground gains *(diagnostic)* | **no** | — | **2.732 m** | — | — |
| `climb_30` (bare) | yes | 0.66 s | 0.765 m | +0.187 m | 1162 N |
| `climb_30_pemba` (as built) | yes | 1.54 s | 0.636 m | +0.167 m | 333 N |
| `climb_30_pemba` + Playground gains *(diagnostic)* | yes | 0.50 s | 0.656 m | +0.124 m | 378 N |

**The gear is not what stops it walking — the actuator gains are.** With the
robot exactly as built it falls on flat ground at 2.16 s. Copy Playground's
per-joint gains onto the same robot, change nothing else, and it walks the full
10 s for 2.73 m. (0.273 m/s against the bare robot's 0.432 m/s at the same
command: still degraded by the jacket-era inertia and the four-sphere feet, but
walking.) That diagnostic is `build_pemba_scene(playground_gains=True)`; it
writes its own scene file and is **not** the team's robot.

On the rope the ascender does its job on the real robot too: `climb_30_pemba`
holds the palm **0.09 mm** off the line, travel is monotone, and the grip peaks
at 333 N instead of the bare robot's 1162 N — it is caught earlier and more
gently.

### What was looked at, not just measured

A frame of `climb_30_pemba` was rendered and inspected: purple jacket with the
Everest Robotics chest logo, yellow snow boots, the brown rope cylinder running
up the 30° slope, and the **orange ascender sitting on the rope** with the
carrier sphere at its centre — right arm ending at the ascender, left hand
still a hand. The rope renders on every climb world, as before.

### ASKS — what the team must do to make this official

**P1 — decide the actuator gains (the blocker).**
`assets/robots/mujoco/g1_unitree_ascender.xml:8` sets
`<position kp="500" dampratio="1" inheritrange="1"/>` for all 29 joints.
Playground's G1 uses per-joint gains from 2 (ankle roll, wrist roll) to 75
(hips, knees, waist, shoulders), and every policy trained in `rl/` is trained
against those. At kp 500 the ankle roll is **250x** stiffer than the policy
expects. The measurement above isolates this as the cause of the walking
failure. Either bring the MJCF's gains to Playground's, or train on 500 and
accept that the mels baseline will never look good on this robot — but it has
to be a decision, not an accident.

**P2 — point `climb_env` at this MJCF for real.** The clean version of what
this module does by monkeypatch: give `climb_env.default_config()` a
`robot_xml` key (default `None` = Playground's scene) and have `_build_model`
use `MjSpec.from_file(cfg.robot_xml or consts.task_to_xml(self._task))`. Then
`registry.load("G1ClimbAscender", config_overrides={"robot_xml": ...})` is all
anyone needs and this file's generation step is the team's, not ours. Whoever
does it inherits the checklist in the table above — those names are load-bearing.

**P3 — the robot needs Playground's names, or Playground needs the robot's.**
Every row in that table exists because the two disagree on naming. The
cheapest fix is on the robot side: name the collision geoms
(`left_shin`, `right_thigh`, ...), add `left_palm`/`right_palm` sites, and
name one foot geom per foot. That is ~10 attributes in `build.py` and it makes
the robot drop into Playground with no wrapper at all.

**P4 — `knees_bent` is Playground's, not the robot's.** The robot ships one
keyframe, `stand` (`g1_unitree_ascender.xml:338`), at pelvis z 0.79. We inject
Playground's `knees_bent` (z 0.755). It is the right pose for the task — the
palm lands on the rope at waist height — but if the team ever re-tunes the
demo pose, it must be added to the MJCF as `knees_bent` or this variant keeps
using Playground's.

**P5 — MJX warns on the mesh collision pairs.** Loading the pemba scene under
MJX prints: *"MULTICCD is enabled, but the scene contains CCD pairs without
multicontact support: [('CYLINDER','CYLINDER'), ('CYLINDER','MESH')]. At most 1
contact will be generated for these pairs."* The harness runs the plain MuJoCo
C engine and is unaffected, but **training** on this robot runs MJX — worth
checking before a long run, since the ascender and the rope are exactly those
mesh/cylinder pairs.

**P6 — `build.py --fetch` is broken without `usd-core`.** It imports
`../g1/build_g1_usd.py`, which imports `pxr`, so the documented step 1 dies with
`ModuleNotFoundError: No module named 'pxr'` on a plain install. The harness
sidesteps it (the same STLs ship inside `mujoco_playground`), but the README's
two-step instructions do not work as written.

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

## Test (c) — 15 s of mels on `climb_30`, vs their own baseline

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

## Guide follower — what is vision, what is stand-in, what is cheat

`app/harness/guide.py`. A human guide walks ahead along the rope; the robot
measures its distance with two head cameras and drives itself. Reproduce every
number below with `python -m app.harness.test_guide`.

### The ledger

| piece | status | what it actually is |
|---|---|---|
| the two eye images | **REAL** | 320×240 RGB renders of the scene from two MuJoCo cameras 6 cm apart, copied from the `d435i` mount already in `assets/robots/mujoco/g1_unitree_ascender.xml` (pos `0.0789635 0 0.386` on `torso_link`, fovy 58°). Cameras are visual-only; MuJoCo integrates nothing from them. |
| the DISTANCE | **REAL passive stereo** | OpenCV `StereoSGBM` on that pair → disparity → `depth = focal_pixels × baseline / disparity`, `focal_pixels = (height/2)/tan(fovy/2) = 216.5 px`, `baseline = 0.06 m`. Sub-pixel from SGBM's own 1/16-px fixed point. **No simulator state is read anywhere in this path.** |
| the BEARING | **REAL** | the matched pixels' centroid column through the same intrinsics. |
| WHICH PIXELS ARE THE HUMAN | **STAND-IN** | an HSV colour threshold on her deliberately distinctive orange BACKPACK (hue 9–14, saturation ≥ 180, value ≥ 40), largest connected component. Re-measured with a segmentation render: **100.0% of pack pixels in, 0.0% of every other material and of the scene** — the table is below. A person detector goes here; the seam is `guide.detect_guide(image) -> (box, mask)`. It is not vision in any interesting sense — it knows the answer's colour. It targeted her JACKET until 2026-08-30, when the user put her in blue with an orange pack. |
| `true_distance_meters` | **LABELLED CHEAT** | read straight out of `data.cam_xpos` and the guide's own pose. HUD and grading only; the follower never sees it. Recorded as `guide_true_distance_meters`. |
| the guide's motion | **not physics, and says so** | Chloe's hiker (`assets/humans/human.xml`) as one mocap root plus six WELDED limb bodies (zero DOF, `nq`/`nv`/`njnt` untouched), driven along `RopeRoute` arc length, height snapped to `terrain.surface_z` every tick. It cannot fall, be pushed, or be walked into: every geom is `contype=0, conaffinity=0`. |
| the guide's WALK | **synthetic animation, geometric not learned** | six hinge angles written into `model.body_quat`, phase locked to distance travelled (`2π × travel / 1.05 m`). No joint, no actuator, no integrator. See "The animated guide" below. |
| the command | ours, as everything in the app layer is | the follower writes the same 3-vector the keyboard writes. No new policy, no retraining, no change to `rl/`. |

### The animated guide, and the 23 cm it nearly cost

`guide.py`'s six limbs are hinges IN NAME ONLY: `_add_guide_body` adds no joint,
each limb body is welded to its parent, and `Guide.write` turns it by writing
`model.body_quat` every control tick. `mj_kinematics` reads that field for a
welded body, so the figure poses exactly as if it had joints, and the state
vector the solver integrates does not grow by one number.

**THE FIRST VERSION USED REAL HINGES, AND IT MOVED THE ROBOT.** Six `mjJNT_HINGE`
joints grew `nq` 39 -> 45 and `nv` 38 -> 44. The guide's limbs cannot touch the
robot -- no contacts (`contype=0`), no constraint, a mocap root welded to the
world -- so they exert no force on it. They still changed the answer: two 6 s
same-seed `flat_0` runs, `--hold-w`, came back with **1.447 m** and **1.675 m**
of rope travel. That is not a force, it is a walking robot amplifying a
floating-point difference in the solver over 300 ticks, and a run that does not
reproduce is exactly what this file exists to forbid.

**AND SO DID THE CAMERA REFRESH, which had been there all along.** The guide
runs `mj_kinematics` + `mj_camlight` after it moves the human, so the eyes see
where she IS rather than where she was a tick ago. `mj_step` is
forward-then-integrate, so when it returns `data.qpos` is the NEW state while
`data.xpos` still describes the OLD one -- and the next control tick reads those
stale frames. Refreshing them hands the next step a fresher world than it would
have had: **0.95 rad** of joint-angle difference in six seconds, measured. The
frames are now snapshotted before the refresh and put back after the eye render
(`GuideSystem._freeze` / `_restore`, `KINEMATICS_OUTPUT_FIELDS` -- the exact
output set of those two functions), so the cameras get the fresh pose and the
physics gets the frames it would have got.

**MEASURED, both claims, and both reproducible:**

`python -m app.harness.test_guide`, section D -- the same scripted command flown
twice from the same reset, once with the guide OFF and once with it ON and the
human WALKING (mocap moving, limbs swinging, eyes rendering every fifth tick):

| array | max abs difference, `flat_0` | `terrain_free_10` |
|---|---|---|
| `qpos` | 0.000e+00 | 0.000e+00 |
| `qvel` | 0.000e+00 | 0.000e+00 |
| `ctrl` | 0.000e+00 | 0.000e+00 |
| `sensordata` | 0.000e+00 | 0.000e+00 |
| `qfrc_constraint` | 0.000e+00 | 0.000e+00 |
| `cfrc_ext` | 0.000e+00 | 0.000e+00 |

And at the runtime level, two 6 s same-seed runs with the guide's bodies in the
model and with `--no-guide-body` (which skips the surgery entirely):

    python -m app.harness.runtime --world flat_0 --duration 6 --hold-w \
        --no-render --keep-going --seed 0 --output-name paritytest_guidebody
    python -m app.harness.runtime --world flat_0 --duration 6 --hold-w \
        --no-render --keep-going --seed 0 --no-guide-body \
        --output-name paritytest_noguidebody

300 ticks, 39 recorded arrays, of which **35 are physics/robot: max absolute
difference 0.000e+00**. The four that differ are the guide's own HUD columns
(`guide_human_progress_meters` and friends), which do not exist in a run with no
guide in it.

### The gait, and why the feet do not skate

Distance-locked: phase = `2π × arclength / GUIDE_STRIDE_METERS`, a function of
how far she has walked and never of the clock. One stride of ground is one
stride of animation at any speed, and S (walking back down the rope) runs the
same cycle in reverse for free. Within a stride the planted foot's offset from
the hip ramps linearly from +stride/4 to -stride/4 while the root advances
stride/2, so the two cancel and the boot holds still in the world; the swing
half returns it on a raised cosine. The hip angle that puts a boot a given
distance in front is one arcsine, because with the knee straight the hip-to-boot
vector is rigid (`Guide._hip_for_foot_offset`).

**STRIDE IS SET FROM CADENCE.** Speed is 1.0 m/s (the user's ruling), so
cadence = 2 x speed / stride. The 0.70 m the figure was first drawn with gives
171 steps/min -- a jog, and it read as one. 1.05 m gives **114 steps/min** and
hips swinging +/-17 deg, which is where a real brisk walker's are.

`python -m app.harness.guide_walk_sheet` writes the contact sheet and prints the
audit. `flat_0`, 200 samples around one cycle:

| leg | stance skate | stance spread | swing clearance | lowest sole vs snow |
|---|---|---|---|---|
| left | 0.0007 m | 0.0007 m | 0.084 m | -0.026 m |
| right | 0.0007 m | 0.0007 m | 0.125 m | -0.010 m |

0.7 mm of skate per stance phase. The sole sits between -2.6 cm and +2.2 cm of
the surface over the cycle, on terrain whose roughness is 10.9 cm rms: the root
height is solved from the lowest boot corner assuming a LOCALLY FLAT surface
under the root, so a boot half a stride away can be a couple of centimetres out.

**The neutral pose is baked, not authored.** Chloe drew the hiker mid-stride, so
every limb is rotated back to vertical once at surgery time and the angles that
were subtracted become the joints' zero. Printed on attach, `flat_0`:

| hinge | baked out |
|---|---|
| `hip_l` | +20.9 deg |
| `knee_l` | -14.8 deg |
| `hip_r` | -19.3 deg |
| `knee_r` | +4.5 deg |
| `shoulder_l` | -15.9 deg |
| `elbow_l` | +42.5 deg |

Her two legs also came out 2.7 cm different in length and 3.2 cm apart along the
stride, which in a symmetric gait means a limp: one leg carries the whole walk
and the other paws the air (swing clearances 7.6 cm against 5.1 cm). The knee
anchor and the boot are nudged to the mean of the two sides in the sagittal
plane -- 1.3 cm and 1.6 cm, half the difference each way. The left/right
offsets in y are untouched: those are her stance width, not an error.

### The outfit, and the colour window re-measured on HER BACKPACK

**THE OUTFIT IS OURS, APPLIED AT ATTACH TIME** (user's ruling, 2026-08-30: "make
the human wear a blue outfit with a bright orange backpack"). `guide.
GUIDE_OUTFIT_RGBA` overrides the `rgba` of her materials as `_add_guide_body`
copies them onto the spec; `assets/humans/human.xml` is shared with
`human-safety/` and is not edited for a wardrobe choice. Cobalt jacket
`0.13 0.30 0.72`, navy trousers `0.09 0.14 0.34`, slate boots `0.16 0.19 0.28`,
teal beanie `0.05 0.62 0.55`, safety-orange pack `1.00 0.38 0.02`. Only `rgba`
moves -- `specular` and `shininess` still come from her XML and no geom, body,
mass or contact property is touched -- so the compiled model, the GLB the
browser draws and the eye images the robot detects with all read one source.

The detector therefore keys on the PACK. Choosing the clothes and choosing the
window is one decision: the boots left brown because brown renders at hue 12-14,
inside the pack's own window.

`test_guide` section A0 renders the left eye twice at each test range -- once in
colour, once in SEGMENTATION -- so every pixel is attributed to the geom that
painted it before its hue is counted. Window: hue 9-14, saturation >= 180,
value >= 40. `flat_0`, pooled over 1/2/4/8 m (`terrain_free_10` is the same
verdict: pack 5,220 px at hue 11-11, 100.0%, everything else 0.0%):

| material | pixels | hue 1-99% | saturation 1-99% | value 1-99% | inside the window |
|---|---|---|---|---|---|
| **pack (target)** | 6,380 | 11-12 | 246-250 | 51-239 | **100.0%** |
| jacket | 21,546 | 111-113 | 204-219 | 60-198 | 0.0% |
| skin | 563 | 0-178 | 39-74 | 52-223 | 0.0% |
| beanie | 6,395 | 86-91 | 224-237 | 47-170 | 0.0% |
| glove | 25 | 110-116 | 73-118 | 11-31 | 0.0% |
| pants | 4,301 | 114-115 | 181-208 | 26-92 | 0.0% |
| boots | 519 | 113-113 | 109-124 | 33-70 | 0.0% |
| rope carrier (orange, on the palm) | 0 | | | | not visible |
| everything else (snow, sky, rope, robot) | 267,471 | 1-111 | 33-238 | 74-230 | 0.0% |

Nothing sits on an edge: hue is 2 units either side of the pack's own 11-12;
saturation is 66 clear of its 246 and is the barrier that keeps SKIN out, whose
hue wraps straight through the band; value is 11 clear of the pack's darkest
face. The OTHER reds, by arithmetic (OpenCV hue = degrees / 2): the **rope**
`0.85 0.08 0.05` computes to 2.25 deg -> **hue 1**, eight units below the window,
and hue is its only barrier since it clears both floors. The browser's **wind
pennants** `0xc41414` (196,20,20) compute to 0 deg -> **hue 0**, nine units
below -- and they are Three.js-only decoration that exists in no MuJoCo model,
so they cannot reach an eye image at all. The **ascender carrier**, translucent
orange on the robot's own palm, computes to **hue ~17**, three units above the
window's high end, which is why that end stops at 14 rather than opening up; it
renders 0 px in these poses, so it is out of frame rather than absent.

**WHAT THE SMALLER MARKER COST AND BOUGHT**, measured jacket -> pack:

| | jacket | pack |
|---|---|---|
| maximum detection range, `flat_0` | 10.00 m | **18.25 m** |
| maximum detection range, `terrain_free_10` | 10.00 m | **15.87 m** |
| stereo error at 2 m / 5 m | -5.8% / -6.0% | -7.0% / -4.4% |
| standing at 2 m, back turned | 100% detected, WAIT at 0.89 m | 87.0% detected, WAIT at 0.78 m |
| standing at 5 m, back turned | 79.5% detected, ends in SEARCH | **99.0% detected, ends in FOLLOW** |
| off-axis re-acquire (`test_search` J) | 0.20 s / 0.40 s | 0.20 s / 0.60 s |

The range went UP: the pack is small but it is a solid, uniformly-lit box, where
the jacket's thin limbs anti-aliased into the snow. Detected is not the same as
usable, though -- past about 10 m the disparity is 1-1.5 px and the range wanders
(12 m true reads 13.07 m; 16 m true reads 13.05 m).

**TWO NEW LIMITS, measured rather than left for a demo to find** (`test_guide`
A1, A2, A2b):

* **She is invisible facing the robot.** 0 mask pixels at 2 m and at 5 m, and
  the follower sits in SEARCH rather than inventing a range -- 0.0% of ticks for
  a whole 20 s run at 5 m on both worlds, and at 2 m on `terrain_free_10`. The
  one exception is printed as a first-detection range: at 2 m on `flat_0` the
  robot walks BLIND from 2.00 m down to 1.01 m with 0 mask pixels the whole way,
  and only sees the pack's edge at 0.98 m once it is round her shoulder.
* **A close-range hole on the approach.** She walks 0.6 m left of the rope, so
  inside about 1.5 m the bearing to her crosses the +/-29 deg frame edge and a
  narrow marker on her back goes with it, where a whole jacket still filled the
  picture. The follower recovers through SEARCH (WAIT at 13.7 s from a 2 m
  start), but on `flat_0` it is what stops `test_search`'s REALIGN handing over
  to the ordinary follower, which the jacket managed at 0.24 s. On
  `terrain_free_10` the hand-over still completes, at 1.10 s with a
  camera-bearing error of 0.8 deg.

Physics is untouched by all of it: `test_guide` D and `test_search` M both come
back **0.000e+00 across every array**.

### The model surgery, and why it is safe

`guide.attach_guide(scene)` adds one mocap body (3 geoms) and two cameras to
**his** `MjSpec` and recompiles — the same mechanism `graphics.add_skybox` uses,
with the same refusal rule. Every joint qpos/dof address, every actuator target,
every existing body's mass and name, and `nq nv nu njnt neq nsite nsensor nkey`
are compared before and after; if any of them moves the swap is **refused** and
the scene is left exactly as it was. Measured on `flat_0`: bodies 32→39, geoms
101→126, cameras 1→3, mocap 0→1, joints 33→33, `nq` 39→39, `nv` 38→38,
**all 13 structural fields unchanged**. The new bodies are appended after the
existing tree, so no existing id shifts — and `nq`, `nv` and `njnt` are IN the
checked list, which is the whole reason the limbs are welded rather than jointed.

TWO ORDERING RULES, both learned by breaking them:

1. **The surgery runs before the graphics dressing.** `apply_alpine_look` writes
   to the COMPILED model (lights, fog, the snow colour) and a recompile throws
   that away; doing it the other way round gives a dark, unlit picture.
2. **The camera orientation is read off the compiled model, not the spec.** The
   MJCF writes `d435i` as `xyaxes="0 -1 0  0 0 1"`, and MjSpec keeps that in the
   element's `alt` field while leaving `quat` at IDENTITY. Copying `source.quat`
   produced two cameras pointing straight down — a black picture and zero
   detections at every range. `model.cam_quat` is the compiler's resolved answer.

### Stereo accuracy vs the simulator's own answer

Two truths are printed because the measurement sits between them, and neither
alone is the whole story: a dense matcher's median disparity over a convex body
reads its **near face**, so the surface column is the like-for-like comparison,
while the axis column is the literal "distance to the human" the HUD reports.

`flat_0`:

| true to axis | true to surface | measured | err vs axis | err vs surface | disparity |
|---|---|---|---|---|---|
| 1.000 m | 0.810 m | 0.798 m | −20.2% | −1.5% | 17.88 px |
| 2.000 m | 1.810 m | 1.884 m | **−5.8%** | **+4.1%** | 6.94 px |
| 4.000 m | 3.810 m | 4.002 m | +0.0% | +5.0% | 3.31 px |
| 8.000 m | 7.810 m | 7.235 m | −9.6% | −7.4% | 1.81 px |

Re-measured on the animated hiker (the numbers moved a little because the figure
did: the reference point and the visible silhouette are hers, not the
placeholder's). `terrain_free_10`, against the axis: −19.2% / **−3.6%** /
+12.5% / +69.6% — the 8 m row there matches on a single disparity pixel, which
is the honest end of a 6 cm baseline. The 8 m
row is the honest limit of a 6 cm baseline at this focal length: the whole
disparity there is 1.25 px, so one quantisation step is metres.

### The decision, and its hysteresis

`FOLLOW` above 1.3 m (1.0 m once already following), `WAIT` at or below 1.0 m
(1.3 m once already waiting), `LOST` after a second with no detection. Both WAIT
and LOST command zero, so a lost human and a close human stop the robot alike —
the conservative direction. `ang_vel_yaw = clamp(2 × bearing, ±1)`, 2° deadband.

### One notion of "a human is there"

`human-safety/human_gate.py` already owns the auditable rule "no climbing UP
while a human is in front", with a SIM ORACLE detector. Running the follower's
vision alongside that oracle would give the demo two detectors free to disagree.
So while the guide is on, the gate is driven from the same measurement
(`guide.GuideVisionDetector` returns *her* `Detection` type, with `seen` true
exactly when the follower says WAIT) and its own hysteresis is set to 0.0,
because the follower already has two. Her file is imported, not edited.
**ASK (low priority):** if the oracle-driven `--human` spawns and the guide are
ever wanted at once, the two gates need merging rather than switching.

### Cost, measured

Per stereo pair on this machine: two 320×240 renders 10.9 ms, SGBM 1.1 ms,
detection 0.7 ms, annotate + JPEG 1.2 ms. At one vision tick in five that is
**2.8 ms per control tick**. Dropping to 256×192 saves 0.1 ms — the cost is the
GL round trip, not the pixels, so the resolution dial is not the lever.

One real win found on the way: `mujoco.Renderer` sizes its offscreen buffer from
`model.vis.global_`, so a second small renderer on a model whose `offwidth` was
raised to 1920×1080 for the main view allocates a **1920×1080 8×-MSAA** buffer to
read 320×240 out of. Per pair: 17.07 ms that way, 13.23 ms with the buffer at
320×240, 11.09 ms with MSAA off too. `StereoEyes` sets those three fields around
the context creation and restores them immediately.

Realtime factor on `lhotse_B`, live, W held (this machine, three agents running):

| viewport | guide off | guide on |
|---|---|---|
| 960×540 (the page's default) | 1.00 | **1.00** |
| 1280×720 | 1.00 | **1.00** |
| 1920×1080 | 0.90–0.99 | 0.79 |
| 1920×1080, `--no-shadows` | 1.00 | **1.00** |

At 1920×1080 with shadows the main render alone already fills the 20 ms tick
(0.90–0.99 with the guide OFF, load-dependent), so there is no headroom there
with or without the eyes; `--no-shadows` (a documented 5.7 ms at that size)
restores it and holds 1.00 with the guide on.

### What the follower cannot fix, because it is the plant

Both measured with `test_guide` and a direct command sweep, and both are
properties of the team's walking policy in these scenes, not of this layer:

* **Ground speed is ~0.15 m/s whatever `lin_vel_x` says.** `flat_0`, 20 s,
  straight-ahead command: 3.15 m at cmd 0.5, 4.18 m at cmd 0.8, 4.22 m at
  cmd 1.0. The guide walks at **1.0 m/s** (the user's ruling), so holding W
  indefinitely opens the gap at ~0.85 m/s and she leaves the ±29° field of view
  in about three seconds — `test_guide` B now goes FOLLOW → LOST at t = 2.5 s
  and stays there. The demo that works is a two-second tap of W and then
  release, or **S**, which walks her back down the rope to the robot (measured:
  −2.00 m of arc length in 2 s, exactly the 1.0 m/s, gait running in reverse
  with her yaw still uphill). Closing a gap the robot opened is now beyond it,
  and that is the plant, not the follower.
* **Yaw authority is nearly nil while the palm grips the rope.** `flat_0`, 3 s
  of a constant yaw command on top of `lin_vel_x` 0.5: +1.0 → −23.6°, +0.5 →
  −24.2°, 0.0 → −29.6°, −0.5 → −15.1°, −1.0 → −13.4°. The robot yaws about −25°
  regardless of what is asked. `HeadingController`'s docstring already warned
  about this for the mouse-look controller; the follower inherits it, and the
  visible symptom is FOLLOW→LOST flapping when the human ends up outside the
  ±36° horizontal field of view.
* On `terrain_free_10` the walker drifts **−50° of yaw in 3 s with a zero yaw
  command** and walks backwards (−2.12 m in 20 s at cmd 0.5), which is why the
  follower cannot hold a gap there at all. **ASK:** if the follower is to be
  demoed *steering*, it wants a world where the walker tracks its command — a
  rope-off gentle slope with a retuned policy would be the fix, and that lives
  in `rl/`.

---

## SEARCH -- turning the cameras, and what that costs

`app/harness/guide.py`. When the follower loses the human it sweeps its cameras
to find her again. The G1 has no neck, so the cameras are panned by WAIST YAW.

### The mount, checked rather than assumed

The `d435i` camera sits on `torso_link`, and the parent chain is

    pelvis -> waist_yaw_link (waist_yaw_joint, +z, +/-150 deg, actuator 12)
           -> waist_roll_link -> torso_link   [the cameras]

so `waist_yaw_joint` is above the cameras in the tree and is the joint that pans
them. `WaistYaw.bind` looks the actuator up by name and turns the search off with
a message if it is not there.

### What is written where

| piece | status | what it actually is |
|---|---|---|
| the waist offset | ours, a supervisory command | one number added to the walking policy's OWN waist-yaw PD target, at `ClimbSceneEpisode.control_hooks` -- after `WalkController.substep` writes `data.ctrl` and before the `mj_step` that acts on it. The policy is not retrained, not consulted and not modified. |
| the rate limit | ours | 1.5 rad/s on the offset. A step change in a PD target is a kick, and this robot hangs off a rope by one palm. |
| the clamp | ours, and MEASURED | +/-60 deg (see below), applied on ASSIGNMENT to `target_radians`. |
| `theta_waist` | **read from `data.qpos`**, not from the command | the achieved joint angle. See the windup note. |
| `ang_vel_yaw` | **zero unless a policy can use it** | `yaw_command_available`, False for every climb world. |

### Two bugs this found, both worth keeping written down

**1. Feeding the COMMAND back is a windup loop.** The bearing to the human in the
body's frame is `theta_waist + beta`. The first version used the COMMANDED
offset for `theta_waist` -- but the offset is added to a PD target the policy is
also writing, so the policy pulls back and the joint settles short. The image
bearing therefore never closes, the target grows every vision tick, and on
`flat_0` the waist wound to **168 degrees** and the robot fell at **7.3 s**.
`WaistYaw.measure` reads the achieved angle out of `qpos` instead.

**2. Tracking her in FOLLOW/WAIT is a fall.** Keeping the waist on her after the
search ended looks obviously right and is not: with the palm clipped to the rope,
twisting the waist counter-rotates the PELVIS, so the image bearing never closes
and the waist chases it to the clamp. On `flat_0` that is a fall at **1.9 s**.
The waist straightens once REALIGN hands over, and REALIGN does not hand over
until she is inside the cone the straightened cameras will still see.

### The clamp is measured, not chosen

Sweeping `flat_0`, roped, 25 s, 1.5 rad/s:

| peak sweep | outcome |
|---|---|
| 90 deg | fell at 9.5 s |
| 80 deg | fell at 8.3 s |
| 75 deg | fell at 5.6 s |
| 70 deg | survived |
| 65 deg | fell at 21.6 s |
| **60 deg** | **survived, upright 0.96** |

So `WAIT_LIMIT` is 60 deg. The 20/60/90 ladder stays in the source because it is
the design; the clamp is what binds. A robot that can hold 90 gets it by raising
one line.

### The acquisition, measured

`python -m app.harness.test_search`. The human is placed **60 degrees** off the
robot's axis -- the camera's horizontal half-FOV is 36.5 deg, so she is outside
it and the detector sees nothing at t = 0. The camera-bearing error is read from
the simulator (a LABELLED CHEAT, grading only) so the same detector that did the
aiming cannot flatter it.

| world | rope | ACQUIRE | hand-over | camera-bearing error at hand-over | waist peak | fell |
|---|---|---|---|---|---|---|
| `flat_0` | on | 0.20 s | 0.24 s -> FOLLOW | **10.2 deg** | 58.4 deg | no |
| `terrain_free_10` | off | 0.40 s | 1.00 s -> FOLLOW | **3.2 deg** | 20.1 deg | no |

Rope-off is the cleaner of the two, as expected: nothing is counter-rotating the
pelvis, so the waist barely has to move and the error is 3 deg.

### The limit of a neck that only turns 60 degrees

The searchable cone is 60 (waist) + 36.5 (half-FOV) = **+/-96.5 deg**. A human
BEHIND the robot cannot be found. Held **S** on `flat_0` until she walks back
past the robot: FOLLOW -> WAIT (2.5 s) -> SEARCH/sweep, and she is never
re-acquired, correctly. **ASK to Mrinal: randomise `ang_vel_yaw` in the training
commands if a steerable turn is wanted.** Everything here is a workaround for a
policy that cannot turn; `realign_mode: "body+waist"` is already written for one
that can.

### The physics claim, stated the only way it can be true

A SEARCHING robot's physics is NOT identical to a still one's, and must not be:
the waist offset is a real command on a real actuator, and the feature IS that
the torso turns. What must be identical is the OFF case -- the machinery built,
the hook registered and running on every substep, the knob off:

    no guide system at all   vs   guide built + waist hook registered, knob OFF

Same reset, same scripted command, 6 s, `flat_0`: `qpos`, `qvel`, `ctrl`,
`sensordata`, `qfrc_constraint` all **0.000e+00**.

---

## The two wind pennants and the camera subject -- browser only

`app/web/three/flag.js`, `three/stage.js`, `three/first_person_camera.js`. Both
rulings of 2026-08-30 -- "the flag needs to be a lot bigger" / "put the same
flag on the human too", and "when the guide is turned on the camera should be on
the guide" -- landed entirely in the browser.

| piece | status | what it actually is |
|---|---|---|
| the two pennants | **decoration** | Three.js `Group`s parented to the `torso_link` and `guide` nodes of the GLB. No MuJoCo body, no mocap slot, no geom, no mass, no collision shape. The pose stream they ride on is read-only, so the model the solver integrates is byte for byte the model it integrated before this file existed. |
| the flag's clearance | ours, measured | from `model.cam_pos` / `cam_quat` / `cam_fovy` on the compiled `flat_0`: eyes' half diagonal **42.7°**; pole base **131.4°** off axis, pole top **98.0°**, and the worst point of the swept cloth over every lift angle and heading **56.2°**. 13.5° of margin, so even an in-model version could not enter an eye image. |
| the hiker's pennant in the eye images | **absent** | it is browser geometry; the eye cameras render the MuJoCo model, which has no flag in it. Its colour therefore cannot reach the detector at all. Stated because the obvious worry -- a red flag inside an orange-red colour window -- is a worry about a thing that does not exist. |
| who the camera follows | **input side** | the chase boom's target and the first-person mount swap between the robot's pelvis/torso and the guide's mocap root. The two numbers that go up the socket are still the CHASE camera's azimuth and elevation, exactly as before, so the steering command is unchanged. |
| the hiker's 360° head turn | **input side** | her first-person yaw WRAPS instead of clamping, because a person has a neck and the robot's ±150° is the waist joint's real travel. Camera-only: nothing here writes a joint. |

---

## Visibility -- a white-out, and here is the proof it is only a picture

`app/harness/storm.py`. The `visibility` knob is a DISTANCE IN METRES. It
degrades what the robot SEES and nothing else.

**WHAT CHANGED 2026-08-30** (user's ruling): this used to be a `storm` 0/1
switch whose thickness came out of the WIND SPEED, `100 m x exp(-wind / 6)`.
Two independent things were one control -- no still white-out, no clear gale,
and every wind experiment silently moved what the robot could see. The coupling
is deleted; wind does force, flag, sound and gusting, and visibility does only
visibility. An older page's `storm` knob is stored and ignored rather than
rejected, so a stale client cannot crash the runtime.

### The ledger

| piece | status | what it actually is |
|---|---|---|
| the visibility itself | **the knob** | metres, straight off the page. Not derived from anything. 100 m = clear, 3 m = white-out. |
| the clear end | **an identity** | at 100 m `degrade` returns the image it was handed, un-fogged and un-grained, and does not advance its own generator -- so "clear" is a genuine control arm, not a nearly-clear one. |
| the log share | ours, stated | `ln(100 / v) / ln(100 / 3)`, 0 at clear and 1 at white-out; centre 17.3 m. It drives the page's fog whiteness, the far-field flake count and the sensor grain, and it sets the slider's own scale. Shared with `whiteoutShare` in `app/web/three/world.js`. |
| the fog on the eye images | **synthetic degradation** | composited per pixel from the eye renderer's own depth buffer: `out = colour*(1-f) + white*f`, `f` a linear ramp from `0.15 x visibility` to `visibility`. The same law GL's `GL_LINEAR` fog uses. |
| the sensor grain | **synthetic degradation** | Gaussian noise, sigma 1.0 grey levels at 100 m rising to 7.0 at 3 m, drawn INDEPENDENTLY per eye from a seeded generator. It used to scale with the WIND; that was the last place the two dials were tied together. |
| the fog on the 3D page | **synthetic degradation** | `FogExp2` at density `1.73 / visibility` (FogExp2 is 95% opaque at `1.73 / density`), background set to the fog colour so there is no horizon. Driven by the same knob. |
| the physics | **untouched** | nothing here writes to the model or to `MjData`. |

### Why the fog is composited, and not left to MuJoCo

MuJoCo has linear GL fog and it CANNOT BE MOVED at runtime from Python. Four
measurements, each one a dead end:

* `model.vis.map.fogstart` / `fogend` are multiples of `model.stat.extent`, and
  `mjr_render(viewport, scene, context)` takes **no model** -- it cannot read
  them. Writing them mid-run changes the rendered picture by **0.000**.
* `context.fogStart` / `fogEnd` are metres and ARE writable from Python, but
  they only reach GL through `glFogf` inside `mjr_makeContext`. Writing them
  mid-run also changes the picture by **0.000**.
* `mjr_makeContext` is not exposed in the Python bindings, so the context cannot
  be rebuilt to pick them up.
* `context.fogRGBA` IS read every frame (**14.06** mean pixel change). That is
  the trap: the fog COLOUR is live while the fog DISTANCE is frozen at whatever
  the model compiled with -- **2534 m** on `flat_0`, i.e. no fog at all. Driving
  only the colour puts a flat white wash over every pixel at every depth, near
  objects included, which is the "particles on the glass" look arrived at from
  the other direction. The first build of this feature was doing exactly that.

**A FINDING WHILE WE ARE HERE, left unfixed on purpose.** The same arithmetic
says the scene's CLEAR-weather fog has never been visible either:
`graphics.apply_alpine_look` writes `1.35 x terrain diagonal` into a field
measured in EXTENTS, which on `flat_0` is a fog end of 2534 m. What reads as
haze is the haze layer and the skybox, not the fog. Changing it would change the
look of every view, so it is written down rather than touched.

### The white-out is by DISTANCE, not a wash

`test_storm` section E0. The guide is at 5 m; the near half of the frame (depth
<= 6 m) and the far half are reported apart, sensor noise off:

| visibility | mean change NEAR | mean change FAR | mean brightness |
|---|---|---|---|
| 100 m (clear) | 0.0 | 0.0 | 151.3 |
| 30 m | 0.5 | 42.9 | 176.2 |
| 10 m | 30.1 | 58.6 | 197.8 |
| 3 m | 126.8 | 63.8 | 241.9 |

A flat colour wash would move both columns equally. Fog eats the far half first
and only reaches the near half once the visibility falls below the subject's own
range.

The page's fog colour carries **2.6x linear headroom** on purpose: the renderer
tone-maps ACES-filmic at exposure 0.92, ACES rolls a 1.0 off to about 0.77, and
a fog painted at plain "white" renders as a mid grey. Measured before the
headroom: a full white-out frame came back at RGB 150.

(A screenshot trap worth writing down: `#lockOverlay` is `rgba(9,9,11,.55)`
across the viewport whenever the page does not hold the pointer, and **headless
Chrome can never take pointer lock**, so it is always up. It dims the render by
55% -- a white-out measured RGB 118 instead of 245 for that reason alone, and
two rounds of chasing the tone mapper were spent on it.)

A consequence of the bottom of the dial, stated so nobody reads it as a bug: the
chase camera sits on a **4.3 m boom**, so at 3 m of visibility the CAMERA is
itself outside the visibility and the third-person view goes fully white --
`render3d_shots/visibility_3.png` is a blank frame. That is what a 3 m white-out
looks like from four metres away, and it is the same weather the robot's eyes
are in.

### What it does to the follower

`test_storm` sections E and F, on `flat_0`, with the guide's orange BACKPACK as
the detector's target:

| visibility | detected at 2 m | detected at 5 m | max detection range |
|---|---|---|---|
| 100 m (clear) | 100% | 100% | 16 m |
| 30 m | 100% | 100% | 10 m |
| 10 m | 100% | **0%** | **4 m** |
| 3 m | **0%** | **0%** | **never seen** |

Stereo error over the frames where she IS seen stays around -4% to -7%; what the
white-out takes is DETECTION, not accuracy, which is what a contrast-destroying
fog should do to a colour threshold.

And the follower's own verdict, human parked at 9 m, 6 s of real vision, robot
not stepped -- nothing scripted, the detector simply misses and the 1 s timeout
expires:

| visibility | vision frames with a detection | FOLLOW | LOST |
|---|---|---|---|
| 100 m (clear) | 100% | 100% | 0% |
| 30 m | 100% | 100% | 0% |
| 10 m | **0%** | 0% | 0% |
| 3 m | **0%** | 0% | 0% |

(The 10 m and 3 m rows show 0% FOLLOW and 0% LOST because the follower opens in
WAIT and never sees her at all -- LOST needs a detection to have been lost.)

### Visibility is not wind

`test_storm` section J, the table that says the 2026-08-30 split is real rather
than cosmetic. Visibility held at 10 m; the wind dial at 0 m/s and at 12 m/s,
two settings that under the retired coupling meant 100 m and 13.5 m of
visibility. The wind really blows -- half a second of it through
`episode.step`'s world-frame wind velocity, from the same reset -- and then the
guide is re-placed at the test range by bisection.

    `StormVision.update` parameters: ['visibility_meters']
    J1  same rendered eye, degraded in each arm: max per-pixel difference 0

| wind m/s | detected at 2 m | detected at 5 m | max detection range |
|---|---|---|---|
| 0 | 100% | 0% | 4.0 m |
| 12 | 100% | 0% | 4.0 m |

The detection rates and the maximum range match exactly. The median stereo error
does not (-5.2% against -8.8%), and that is the wind doing the one thing it
still does: it moved the robot, so the two arms measure the same range from
slightly different poses.

### The physics claim

`test_storm` section H -- the same scripted command flown twice from the same
reset, once at 100 m clear and once in a 3 m white-out with the guide on, so the
fog is composited every tick and the eyes render through it every fifth:

| array | max abs difference |
|---|---|
| `qpos` | 0.000e+00 |
| `qvel` | 0.000e+00 |
| `ctrl` | 0.000e+00 |
| `sensordata` | 0.000e+00 |
| `qfrc_constraint` | 0.000e+00 |
| `cfrc_ext` | 0.000e+00 |

And the whole-runtime claim, measured through `runtime.run` itself against
commit `cf03957`, 300 ticks, seed 0, `--hold-w`:

| arm | flat_0 | lhotse_B |
|---|---|---|
| guide OFF, whole tree | **0.000e+00**, POS0 bytes identical | **0.000e+00**, POS0 bytes identical |
| guide ON, weather/flag/camera changes only (`guide.py` held at cf03957) | **0.000e+00**, POS0 sha `b757ad6e` unchanged | **0.000e+00**, POS0 sha `79246087` unchanged |
| guide ON, whole tree | **0.000e+00** | qpos 1.865, qvel 17.97, ctrl 1.290 |

The last cell is the ONLY divergence and it is the intended one: with the guide
on, the robot's command comes from what it SEES, and the detector was
deliberately re-pointed from the jacket to the backpack the same day. Holding
`guide.py` at the old commit and changing everything else brings lhotse_B back
to bit-identical, which is what isolates the claim -- the weather, the flags and
the camera subject move no physics at all.

---

## Snow and footprints -- visual only, and here is the proof

`app/harness/snow.py`. The terrain wears a procedural snow texture and keeps the
footprints the robot leaves in it.

**THE PHYSICS CLAIM.** Three things change on the compiled model: one texture is
added, one material is added, and the terrain geom's `matid` points at it. None
of the three is read by the solver -- MuJoCo's contact model knows about
`geom_friction`, `geom_solref`, `condim` and the collision bitmasks, not about
what a surface looks like -- and the HEIGHTFIELD IS NEVER EDITED, so the ground
the feet actually touch is the same ground. The spec recompile is guarded the
same way `graphics.add_skybox` and the guide's surgery are: `nq nv nu nbody njnt
neq ngeom nsite nsensor nkey`, every joint address, every actuator target, every
body mass, and additionally every geom's `friction`, `contype` and `conaffinity`
are compared before and after, and the swap is refused if any of them moved.
Measured on `flat_0`: **all 17 fields unchanged**.

MEASURED, not asserted. Two 6 s runs, same seed, same world, `--hold-w`, one
with the snow and one with `--no-snow`:

    python -m app.harness.runtime --world flat_0 --duration 6 --hold-w \
        --no-render --keep-going --seed 0 --output-name paritytest_snow
    python -m app.harness.runtime --world flat_0 --duration 6 --hold-w \
        --no-render --keep-going --seed 0 --no-snow --output-name paritytest_nosnow

300 ticks, **39 recorded arrays compared, max absolute difference 0.000e+00** --
bit-identical, joint angles, contact forces, rope force and all. Both runs also
counted the same 18 footsteps.

**Which footprint path was used.** The TEXTURE path, not the geom-pool fallback.
`mujoco.Renderer` exposes `_mjr_context` and `_gl_context`, and
`mjr_uploadTexture` replaces a texture in a live context, so a print is written
into `model.tex_data` (through a reshaped numpy VIEW of it -- a slice of that
buffer is a view, which turns a repaint into one vectorised assignment instead of
a loop over rows) and pushed. The world-to-texel map is exact rather than
approximate: the material carries `texrepeat 1 1` with `texuniform` off, so the
texture maps once across the heightfield, and world x is de-sloped by the same
three fixed-point passes `terrain.surface_z` uses before it is normalised. It was
verified by painting a marker at a computed texel and rendering top-down before a
single footprint was written.

**The fade, and why it is cheap.** A separate alpha channel holds the prints;
`live = base blended toward the shadow colour by alpha` is recomputed from
`base` every time, never from the previous frame, so repeated fading cannot
ratchet the snow grey. Fading is one multiply of the alpha over the union
rectangle of the live prints, every 3 seconds. A ring buffer caps the live
prints at 400 and erases the oldest outright.

**Costs, per 50 Hz control tick, measured on this machine** (`lhotse_B`,
25 x 15 m, texture 1600 x 960 = 4.6 MB):

| step | cost |
|---|---|
| paint a print (only on a landing) | 0.04 ms mean, 0.90 ms worst |
| the fade pass (every 3 s) | 0.002 ms mean, 0.22 ms worst |
| `mjr_uploadTexture`, main context only | 3.69 ms per upload -> 0.44 ms/tick at 6 Hz |
| `mjr_uploadTexture`, main + the guide's eye context | 6.28 ms per upload -> **0.75 ms/tick** at 6 Hz |

The upload is the whole cost of the feature, and it is the RESOLUTION that sets
it, because `mjr_uploadTexture` replaces the entire texture: at 80 texels/m the
same patch is a 7.2 MB texture and 6.52 ms for the pair of contexts. 64 texels/m
still gives a 17 x 8 texel footprint, which reads as a print at demo distance,
and 6 Hz instead of 10 means a print appears at most 170 ms after the foot lands.
`--no-snow` removes all of it.

**A note on the realtime numbers.** The live realtime factor could not be
measured cleanly for this change: three agents were running on this laptop and
the SAME configuration measured 0.99 and 0.85 an hour apart with no code change
at all. The per-tick costs above are the reproducible statement; against them,
the snow is +0.8 ms on a 20 ms tick.

**The `foot_steps` protocol.** A landing is a foot geom gaining contact with the
terrain geom after at least two ticks with none -- the debounce matters, because
a scuffing foot makes and breaks contact several times a second and would
machine-gun both the sound and the paint. Feet are identified by geom id from
`meta["foot_geom_ids"]`, grouped by owning body, and labelled left/right off the
body name (falling back to the sign of its y offset), so nothing here breaks when
the robot's foot contacts change from four spheres to one box. Impact speed is
the foot's own downward speed on the last tick it was still airborne, differenced
from its world height -- the number a sound engine wants, and one that no longer
exists once contact does.

---

## Chloe's ascender worlds (`chloe_v1_20`, `chloe_v1_25`) — a SECOND plant, on purpose

Every parity claim above this line is about ONE plant: `climb_scene.build_scene`
with the walking policy on it. `chloe_v1_20` / `chloe_v1_25` are a second plant and a
second brain, built in `app/harness/chloe_worlds.py` around
`assets/robots/mujoco/rope_rail.py` and driven by
`rl/chloe/policies/g1_ascender_slope20_v3_2026-08-30_04-35-59.onnx`. They exist
because her policy CANNOT be run on the plant above — measured, both directions,
below.

### The divergence ledger, D1–D14

The numbering is established HERE (the harness had no such list before). Each
row is one way the harness's walking plant differs from the plant Chloe's
network was trained in, and what the new worlds do about it.

| # | divergence | the walking worlds | Chloe's training plant | `chloe_*` |
|---|---|---|---|---|
| D1 | actuator stiffness / damping | `robot.adapt`'s retune (kp 75 / kd 2 on the big joints, kp 2 on the wrists) | mjlab `G1_ARTICULATION`, per motor group (40.2/2.56, 99.1/6.31, 28.5/1.81, 14.3/0.91, 16.8/1.07) | **sidestepped** — hers, verbatim |
| D2 | joint armature | the harness's retune | mjlab, per motor group (0.0036–0.0251) | **sidestepped** |
| D3 | action scale | one flat 0.5 for all 29 | per joint: 0.5476 / 0.3507 / 0.4386 / 0.0745 | **sidestepped** |
| D4 | default pose the action deltas are about | `robot.KNEES_BENT_QPOS[7:36]`, a constant | whatever `rope_rail.add_rope_rail` IK-solves for this slope and rope height | **sidestepped** — solved, never transcribed |
| D5 | foot contact geometry | `_swap_feet_for_boxes` — one Playground box per foot | the XML's four spheres per foot | **sidestepped** — no swap |
| D6 | `policy_compat` | on, for the walking policy's sake | off | **sidestepped** — off |
| D7 | the observation | 103-d: linvel, gyro, projected gravity, **command (3)**, joints, last action, **gait phase (4)** | 96-d: pelvis angular velocity, projected gravity, joints, last action, **carriage − pelvis in the pelvis frame (3)**. No command, no phase, no linear velocity | **sidestepped** — her layout |
| D8 | the ascender mechanism | a mocap bead projected onto a draped polyline every substep, arc-length ratchet | a real 1-DoF prismatic `rope_carriage`, mjEQ_WELD to the wrist, ratchet = the joint's moving lower limit | **sidestepped** — `rope_rail.py`, the shared file |
| D9 | rope shape | a 9-waypoint polyline draped over the terrain, weaving ±0.35 m, colliding | ONE straight non-colliding line, 0.60 m above the ground | **sidestepped** |
| D10 | terrain roughness | patch B's measured micro-roughness, RMS 0.114 m, on a heightfield | a plane. No roughness at all | **REMAINS.** Her network has never seen a rough surface, so it is not run on one. Putting her on `lhotse_B` is the open experiment, not a claim |
| D11 | slope | 0.4°–50°, whatever the patch is | 20° (one task per slope: 0/10/20/30/40) | **sidestepped** at 20; `chloe_v1_25` is one rung INSIDE the measured 10–30° band but is not a slope she was trained at |
| D12 | rates | 500 Hz physics, 10 substeps, 50 Hz control | 200 Hz physics, 4 substeps, 50 Hz control | **sidestepped** — 5 ms / 4 |
| D13 | how the slope is expressed | terrain geom quaternion; gravity vertical | flat plane, gravity tilted | **sidestepped by a rotation** — see below; both frames run and both are printed |
| D14 | the `ClimbScene` carrier and its per-substep projection | present on every Lhotse world, and part of what those numbers mean | absent | **REMAINS as a difference between the two world families.** `rope_travel_meters` on a Chloe world is the slide joint's own coordinate; on a Lhotse world it is `RopeCarrier.progress` along a draped polyline. The two are not the same measurement and must never be pooled |

D1 and D3 are not bookkeeping. Each one, swapped alone for the harness's
value, turns +5.3 m of climbing into a fall inside two seconds. That
measurement is why this is a second plant and not a flag on the first.

### D13 — the frame rotation, measured

The shipped worlds rotate the whole world by −slope about +y and restore
vertical gravity, so the page shows a slope that rises and a robot standing on
it. Her observation is invariant under a world rotation term by term, and the
weld's relative pose and the slide's axis are expressed in the carriage frame,
which rotates with everything else — so the numbers in the model never change.

`python -m app.harness.chloe_worlds --equivalence`, 15 s, v3, no wind:

| slope | frame | uphill m | rope m | outcome | |difference| |
|---|---|---|---|---|---|
| 20° | `tilted_gravity` (hers) | +5.301 | +5.504 | STANDING | — |
| 20° | `tilted_plane` (shipped) | +5.300 | +5.502 | STANDING | 0.0018 m (0.03 %) |
| 25° | `tilted_gravity` | +4.208 | +4.495 | STANDING | — |
| 25° | `tilted_plane` | +4.218 | +4.506 | STANDING | 0.0100 m (0.24 %) |

Floating point, over 750 control steps of a contact-rich sim. Nothing else.

The `tilted_gravity` arm also reproduces the from-scratch plain-MuJoCo
reproduction of her sim2sim loop **exactly** — +5.30 m uphill, +5.50 m of rope,
standing — which is what makes the whole plant claim checkable rather than
asserted.

### THE ROPE RAIL MOVED UNDER THE POLICY (2026-08-30)

`rope_rail.py` changed the ascender channel from the mesh hull's Ø16 mm bore
(centre x = −21 mm, pitch −6.4°) to a channel set from renders (x = +15 mm,
pitch +5°). v3 was trained against the old one. Same policy, same seed, 15 s at
20°:

| rope_rail | uphill m | rope m | outcome |
|---|---|---|---|
| before (v3's own) | +5.61 | +5.82 | STANDING |
| after (current, shipped) | +5.30 | +5.50 | STANDING |

About 5 % of climb rate, and no change in stability. The harness runs the
CURRENT rail, because the rail is a shared file and forking it would be worse.
**The team should know that v3 and the rail have drifted apart**; the fix is the
retrain already in flight (v7), not a harness change.

### Straight climb — slope × wind, 15 s

`python -m app.harness.chloe_worlds --matrix`. Wind blows straight DOWN the
slope, through the same `climb_scene.WindParams` law and the same torso body as
every other world.

| slope | wind m/s | uphill m | rope m | height m | torso up_z | outcome |
|---|---|---|---|---|---|---|
| 20° | 0 | 5.30 | 5.50 | 1.81 | +0.94 | STANDING |
| 20° | 6 | 4.78 | 4.99 | 1.63 | +0.95 | STANDING |
| 20° | 12 | 3.02 | 3.23 | 1.03 | +0.93 | STANDING |
| 25° | 0 | 4.22 | 4.51 | 1.78 | +0.96 | STANDING |
| 25° | 6 | 3.78 | 4.02 | 1.60 | +0.95 | STANDING |
| 25° | 12 | 1.90 | 2.17 | 0.80 | +0.96 | STANDING |

No falls anywhere in the band. Wind costs climb rate and nothing else, which is
consistent with her training randomisation (wind 0–15 m/s, random heading).

### THE ONE-WAY STOP — the finding this feature turned up

W released freezes the PD targets and keeps stepping (`AscenderController.go =
False`). 5 s of climbing, 10 s stopped, 10 s with the gate back on:

| slope | wind | stood 10 s? | rope slide while stopped | uphill drift | pelvis sag | uphill in the 10 s after | total |
|---|---|---|---|---|---|---|---|
| 20° | 0 | yes | +0.105 | +0.116 | −0.354 | **−0.04** | 1.65 |
| 20° | 6 | yes | +0.319 | +0.196 | −0.250 | **+0.22** | 1.80 |
| 20° | 12 | yes | +0.036 | −0.112 | −0.445 | **+0.06** | 0.68 |
| 25° | 0 | yes | +0.150 | −0.014 | −0.356 | **+0.07** | 1.23 |
| 25° | 6 | yes | +0.051 | −0.130 | −0.435 | **+0.10** | 0.98 |
| 25° | 12 | yes | +0.005 | −0.090 | −0.442 | **+0.08** | 0.30 |

Read it in two halves.

**The stop works.** She never falls, `torso up_z` stays positive in all six
cells, and the rope slide is **never negative** — the cam does exactly what a
cam is for, and she cannot slide back down. The pelvis sags 25–45 cm: the
frozen targets are a mid-stride pose, so she settles into a lean on the line
rather than standing to attention.

**The resume does not.** In the 10 s after the gate comes back on she climbs
between −0.04 m and +0.22 m, against the ~3.7 m the same 10 s buys from a clean
start. She is not stuck in the mechanism — she is stuck in the POLICY: the
settled lean (torso up_z ~0.55, pelvis ~45 cm below standing height) is a state
her network never saw. v3 was trained on 15 s episodes that always start from a
clean reset and always climb; it has no recovery behaviour because it was never
asked for one.

Three things were tried, and all three change nothing:

1. **The one variant asked for** — blend the frozen targets toward the
   `rope_rail` reset pose over 0.5 s (`--chloe-hold-blend 0.5`). Resume:
   −0.07 to +1.41 m; five of six cells still stuck, one (25° / 12 m/s)
   recovers 1.41 m. Not a fix.
2. **A shorter stop.** 1 s of hold already costs it: resume +0.19 m. 2 s:
   −0.06 m. The door closes almost immediately.
3. **Zeroing `last_action` on resume** instead of feeding the frozen one:
   identical to four decimal places. The stale action is not the problem; the
   stale POSE is.

So `--chloe-hold-blend` stays at 0.0 (a pure freeze) because the variant does
not earn its extra moving part, and the honest description of W on these worlds
is: **it stops her, and R is what starts her again.** The ask on the `rl/` side
is a recovery term or a stand-still mode in the training distribution — v7's
SLIDE/WALK mode bit is the shape that would fix it.

**This bites the guide switch too, and a demo operator should know.** With
`guide` ON the follower owns the command, so it owns the gate. v3 walks uphill
*backwards* (no heading term in its reward — `rl/chloe/README.md` says so), the
human ends up behind the cameras within about three seconds, the follower drops
to WAIT/LOST and commands zero, and the gate closes. Measured on `chloe_v1_20`,
guide ON, W held: 0.83 m of rope by t = 3 s and then flat to the end of the run,
torso `up_z` settling at +0.56. Nothing crashes, the eyes keep rendering and
every readout stays live — but the climb is over. Until the policy can recover,
run these two worlds with the guide OFF if what you want is a climb.

### What the harness does NOT do to her

* **No steering.** `yaw_command_available` is forced False on these worlds and
  the guide's waist-yaw "neck" hook is NOT registered, whatever `--policy`
  says. The follower's eyes still render and its stereo range and bearing are
  real measurements; the guide card's mode pill carries `no steering` beside
  the mode so the page cannot imply otherwise.
* **No observation noise, no command.** Her 96-vector is built from `MjData`
  and the frozen `last_action`, nothing else.
* **No second ratchet.** `rope_rail.ratchet` is `max(lower_limit, qpos)`, so
  the controller and the scene may both call it (they do — the controller
  before the `mj_step` it precedes, the scene before its own) and the result is
  the same as calling it once. A scene stepped without a controller still
  ratchets.

### The existing worlds are untouched — measured, same session

| test | worst difference |
|---|---|
| `test_search` (guide SEARCH vs the same run without it) | 0.000e+00 — bit-identical |
| `test_guide` (guide OFF vs guide ON, human walking) | 0.000e+00 — bit-identical |
| `test_storm` (visibility off vs on) | 0.000e+00 — bit-identical |
| `test_parity` observation parity vs the JAX env | 1.835e-07 (tolerance 1e-4), PASS |

The only file in the shared path that changed behaviour is
`export_scene.py`, twice, and both changes are additive: a PLANE geom now
reports itself as terrain (so a world whose ground is one flat geom still gets
the page's snow shader and its footprint canvas — no `climb_scene` world has a
plane, its floor is a heightfield), and `index.json` is now MERGED rather than
rewritten, so exporting one world no longer empties the map dropdown of the
other twenty-one.

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

**G2 — no Lhotse terrain / hfield in the climb env.**  *(The harness's four
worlds are all the tilted plane at two angles; see "The four worlds" below.)* `G1ClimbAscender.__init__`
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
Consequence: getting an `MjModel` costs a full `G1ClimbAscender.__init__` —
importing JAX and `mjx.put_model`ing the whole model even when only the plain
model is wanted. Measured warm: **1.64 s for the first, 0.21 s for each
additional.** (An earlier draft of this file said ~25 s; that was a cold-start
measurement and is withdrawn.) So this is a papercut, not a blocker.
**ASK (low priority):** split out a module-level
`build_climb_spec(config) -> mujoco.MjSpec` (or `build_climb_model(config) ->
mujoco.MjModel`) with `_build_model` calling it. The env keeps working, and
viewers/harnesses/tests get the model with no JAX import at all.

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

The four worlds and their slope/rope split; the model cache; `W` →
`lin_vel_x = 0.5` (`--command-speed` to change it); the camera-heading yaw controller (gain 2.0/rad,
±1.0 rad/s, 2° deadband); the third-person orbit and its half-turn azimuth
offset; the wind dial; the friction slider; pause/reset/record/replay; the
guide follower and its two head cameras (`app/harness/guide.py` — a mocap body
and two visual-only cameras added to his spec, nothing else); the
websocket protocol and the page. None of it exists on their side, and none of it
touches the physics except through the command 3-vector, `xfrc_applied` (wind)
and `geom_friction` (the slider) — both clearly marked in the state message and
on the page.
