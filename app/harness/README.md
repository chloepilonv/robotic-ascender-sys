# app/harness — the interactive climber, in the browser

Runs the team's **merged climb scene** (`rl/environment/climb_scene.py`) in
plain MuJoCo at 50 Hz on a laptop and streams it to the browser: the Lhotse
Face heightfield, a rope draped over it, the jacketed G1 with the ascender on
the line. Nothing physical is re-implemented — `build_scene` builds it,
`ClimbScene.step(wind)` is the physics step, and `walk_policy.WalkController`
is the policy. See `PARITY.md` and `fingerprint_lhotse_B.json`.

The older flat-plane env (`rl.environment.climb_env.G1ClimbAscender`) is still
reachable as the four `legacy_*` worlds, because the trainer still uses it.

## Run it on localhost

From the repo root, in the `everest` env (the one that runs the trainer/viewer —
it already has `playground==0.2.0`, `mujoco==3.12.0`, `brax==0.14.2`):

    pip install websockets pillow          # the only extras the harness needs
    # optional, for episode.mp4: brew install ffmpeg / apt install ffmpeg

    python -m app.harness.runtime --live --world lhotse_B

then open **http://localhost:8766/app/web/index.html**. Click the view to take
the pointer, hold **W** to walk/climb, move the mouse to look around, **R**
resets, **Esc** pauses. Map selector, twelve **ClimbScene** worlds then four **legacy** ones:

| world | what |
|---|---|
| `lhotse_B` | the default — measured Lhotse patch B, 38.6°, jacketed robot, roped |
| `lhotse_A` / `C` / `D` | the other measured patches, 33.7–36.2° |
| `flat_0` | B's roughness with the macro slope removed — the walking reference |
| `slope_25/30/35/45/50` | patch B with a synthetic slope override (curriculum) |
| `lhotse_B_free` | patch B with the grip equality off |
| `lhotse_B_playground` | patch B with the bare Playground G1, for comparison |
| the four `legacy_*` | the old flat tilted plane + slide joint, which the trainer still uses |

Also `terrain_free_5/10/15/20/25/30` (patch B's measured roughness re-tilted,
rope off — the walker gives up between 10° and 15°) and `sandbox_free` /
`sandbox_rope`, a 120 × 120 m free-roam map.

Keys: **W** walk, **A/D** turn in place (they suspend the camera-follow while
held) — *unless the `guide` knob is on, in which case W walks the HUMAN up the
rope, S walks her back down it, and the robot drives itself; see "Follow the
guide" below.* Wind dial is in m/s and goes through **his** `WindParams` into
`ClimbScene.step`; the `wind_natural` knob turns the dial into a target that
gusts and drifts. Neither the wind nor this terrain is in training yet — the
trainer is still on the old `climb_env` — so the state message keeps saying
`wind_in_training: false` and `terrain_in_training: false`.

Every live session is recorded under `app/harness/episodes/<stamp>_<world>/`
(frames.npz, hud.json, header.json, episode.mp4) and playable from the page's
Replay tab.

Other entry points:

    python -m app.harness.runtime --world lhotse_B --duration 10 --hold-w --no-render  # headless case
    python -m app.harness.climb_worlds --world lhotse_B                                  # parity gates + fingerprint
    python -m app.harness.runtime --live --policy path/to/policy.npz                    # a trained policy (mels npz layout)
    python -m app.harness.test_parity                                                   # obs + rollout parity vs the JAX env

`--port` moves the websocket (HTTP is port+1); `--command-speed` sets the W
forward command (default 0.5 m/s).

`--pose-stream` (ON by default, `--no-pose-stream` to turn it off) adds a second
binary message beside the JPEG: every body's world pose once per control tick,
946 bytes, ~47 kB/s. It is what **http://localhost:8766/app/web/render3d.html**
runs on -- a WebGL third-person view that draws the scene in the browser instead
of showing a picture of it, so it gets a game camera, soft shadows, a snow/ice
terrain shader, blown snow and footprints. It needs the world exported once:

    python -m app.harness.export_scene --world lhotse_B      # or --all

writes `app/harness/scene_assets/<world>.glb` plus a JSON sidecar (gitignored;
lhotse_B is 18.5 MB). The map selector on that page only offers worlds that have
been exported. The JPEG stream, `episode.mp4` and `app/web/index.html` are
unaffected either way -- the pose hook costs a measured 20 us of a 20 ms tick.


## Follow the guide (`app/harness/guide.py`)

Turn the **`guide`** knob on and a human guide appears 2.5 m up the rope —
**Chloe's hiker** from `assets/humans/human.xml`, loaded whole and re-parented
into six moving limbs. Hold **W** and the *human* walks **up** the rope at
1.0 m/s; hold **S** and she walks **back down** it toward the robot, still
facing uphill, animation running in reverse; hold both and she stands. She is
0.6 m to the rope's left, snapped to the terrain, and she never slips or falls,
because she is a mocap body with no degrees of freedom and no collision. The
**robot decides for itself** what to do about that. A/D do nothing while the
guide is on: the follower owns the yaw command, and the camera-follow
controller stands down.

**The walk is distance-locked**, which is why the feet do not skate: the gait
phase is `2π × travel / 1.05 m`, a function of how far she has WALKED and never
of the clock, so one stride of ground is exactly one stride of animation
whatever the speed, and S runs the same cycle backwards. Within a stride the
planted foot is placed by the same arithmetic in reverse — its offset from the
hip ramps from +stride/4 to −stride/4 while the root advances stride/2, so the
two cancel. MEASURED on `flat_0`: the planted boot drifts **0.7 mm** per stance
phase, swing clearance is 8–13 cm, and the lowest boot corner sits between
−2.6 cm and +2.2 cm of the snow surface (on terrain whose roughness is 10.9 cm
rms). Stride is set from the CADENCE, not chosen: 1.05 m at 1.0 m/s is
114 steps/min, a brisk walk; the 0.70 m it was first drawn with gave
171 steps/min, which is a jog, and it looked like one.

    python -m app.harness.guide_walk_sheet     # the contact sheet + the skate audit

writes `render3d_shots/guide_walk_sheet.png` — eight frames across one gait
cycle with the hip/knee/shoulder angles printed on each — and the audit table.

**Her limbs have no joints, on purpose.** Six child bodies hang off the mocap
root (thigh and shin either side, left upper arm and forearm; the right arm is
posed gripping the rope and does not swing), and each is turned by writing
`model.body_quat` every control tick rather than by a hinge. The first version
used real hinges, which grew `nq` 39→45 — and six degrees of freedom the solver
carries is enough to move the walking robot **23 cm in six seconds** through
nothing but floating-point. `PARITY.md` has that number and the fix.

The robot has two RGB cameras on its head, 6 cm apart, at the `d435i` mount
already in the jacketed MJCF. Ten times a second they are rendered at 320×240
and OpenCV's semi-global block matcher turns the pair into a disparity map;
`depth = focal_pixels × baseline / disparity` gives metres. **The distance is
real passive stereo — nothing in the decision path reads the simulator.** Which
pixels are the human is a **stand-in** (a colour threshold on the guide's
orange-red, where a person detector would go); `PARITY.md` has the full ledger
of what is vision, what is stand-in and what is a labelled cheat.

Three states, with hysteresis so it cannot chatter at the threshold it is trying
to hold:

| mode | when | command |
|---|---|---|
| `FOLLOW` | range > 1.3 m, or > 1.0 m while already following | `lin_vel_x` 0.5, `ang_vel_yaw` = clamp(2·bearing, ±1) |
| `WAIT` | range ≤ 1.0 m, or ≤ 1.3 m while already waiting | zero |
| `LOST` | nothing detected for a whole second | zero |

The page gets a `guide` block in every state message and, at the vision rate, a
second binary message — the four bytes `EYE0` then a JPEG of the left eye with
the detection box drawn on it. Episodes record `guide_mode` (0 WAIT, 1 FOLLOW,
2 LOST), `guide_distance_meters`, `guide_true_distance_meters` and
`guide_human_progress_meters`.

    python -m app.harness.test_guide     # colour window, stereo, two rollouts, physics parity
    python -m app.harness.runtime --world flat_0 --duration 20 --hold-w --guide --no-render

**Measured, and worth knowing before you demo it.** The colour window keeps
**97.7%** of her jacket's pixels and **0.0%** of every other material in the
scene and of the scene itself (`flat_0`, pooled over 1/2/4/8 m, attributed with
a segmentation render — `test_guide` section A0). The stereo reads −5.8% at 2 m
and +0.0% at 4 m against the simulator's own answer. The *walker*, not the
follower, is the limit: its real ground speed is about **0.15 m/s** whatever
`lin_vel_x` says, and the guide now walks at 1.0 m/s, so holding W indefinitely
opens the gap at ~0.85 m/s and she is out of the ±29° field of view within
three seconds. **The demo is: tap W for two or three seconds, then let go**, or
hold **S** and let her walk back to the robot. Yaw authority is also nearly nil
while the palm is gripping the rope (commanding +1 rad/s and −1 rad/s for 3 s
ends up 10° apart), so the follower steers much better on rope-off worlds.

The guide is available on the **ClimbScene** worlds only. The four `legacy_*`
worlds hand back a compiled model with no `MjSpec`, so there is nothing to add
the body and the cameras to, and the feature turns itself off there.

## The storm (`app/harness/storm.py`)

Turn the **`storm`** knob on and the weather closes in. **A storm here is FOG**,
not a snow shower: what a white-out does is take DISTANCE away, so the far slope
dissolves into white first, then the middle distance, and at the top of the dial
you cannot see the hiker four metres in front of you. Nothing sits on the lens.

Visibility follows the **instantaneous** wind speed, gusts included, so a gust
really does blind the robot for a second:

    visibility = 100 m x exp(-wind / 6)      100 / 37 / 14 / 3.6 m at 0 / 6 / 12 / 20 m/s

**The same curve in two places.** The page's own fog is `stormVisibilityMeters`
in `app/web/render3d.html` (a `FogExp2` density of `1.73 / visibility`, the sky
mesh hidden and the clear colour set to the fog colour so there is no horizon at
all); the robot's eyes use `storm.visibility_meters`. If one moves, move the
other, or the picture and the robot stop being in the same weather.

**On the robot's eyes** the fog is composited per pixel from the eye renderer's
own DEPTH buffer, `out = colour*(1 - f) + white*f`, plus a couple of grey levels
of Gaussian sensor noise drawn INDEPENDENTLY per eye -- the one thing fog does
not reproduce, and what leaves the block matcher nothing to match. All of it
lands BEFORE the matcher and the detector, which is the only placement that
makes the degradation honest, and the eye-view PiP shows the degraded image
because it IS the image the robot used.

It legitimately breaks the follower, which is the point:

| wind m/s | visibility | detected at 2 m | detected at 5 m | max detection range |
|---|---|---|---|---|
| storm off | -- | 100% | 100% | 10 m |
| 0 | 100.0 m | 100% | 100% | 10 m |
| 6 | 36.8 m | 100% | 100% | 10 m |
| 12 | 13.5 m | 100% | 100% | 6 m |
| 20 | 3.6 m | 100% | **0%** | **2 m** |

and with the human parked at 9 m the follower goes to LOST on its own -- 100%
of detections at 6 m/s, **0%** at 12 and 20, and the mode follows.

    python -m app.harness.test_storm      # the tables above + the eye contact sheet
    python -m app.harness.runtime --world flat_0 --duration 10 --hold-w --guide \
        --storm --wind 20 0 --no-render

**None of it is physics.** The fog is arithmetic on two rendered arrays and
nothing writes to the model or to `MjData`. `test_storm` section H is a
same-seed diff, storm off against a 20 m/s white-out with the guide on:
**0.000e+00** across `qpos`, `qvel`, `ctrl`, `sensordata`, `qfrc_constraint`
and `cfrc_ext`. Storm off is a row in every table and is identical to a clean
run.

`render3d_shots/storm_eyes.png` is the left eye at each speed;
`render3d_shots/storm_after/page3d_{0,12,20}mps.png` is the 3D page.

## Snow, and footprints in it (`app/harness/snow.py`)

The terrain wears a procedurally generated snow texture -- metre-scale drifts,
centimetre-scale wind grain, the odd sparkling crystal -- instead of a flat grey
sheet. Every time a foot lands, an elliptical print (26 x 12 cm, turned to the
foot's own yaw, soft-edged) is painted into that texture's pixels and pushed to
the GPU with `mjr_uploadTexture`, so the robot leaves a trail behind it that
fades away over about half a minute.

**None of it touches physics.** The heightfield is never edited; what changes is
a texture, a material and the terrain geom's `matid`, none of which the solver
reads. Proven, not asserted: two 6 s same-seed runs with and without
`--no-snow` come back **bit-identical across all 39 recorded arrays** (PARITY.md).

The same landing detection does three jobs, which is why they cannot disagree:
it stamps the print, it increments `step_count`, and it puts a `foot_steps`
event on the websocket -- `[{"foot": "left"|"right", "impact_speed_mps": f}]`,
empty on almost every tick -- so the page can play one snow crunch per step at a
volume set by how hard the foot came down. A landing is a foot gaining terrain
contact after at least two ticks in the air; without that debounce a scuffing
foot machine-guns. `hud.json` records `step_count`, so a replay counts the same
steps the live session heard.

Costs, measured: painting 0.04 ms per control tick, the fade 0.002 ms, the GPU
upload 0.75 ms (6 Hz, a 4.6 MB texture, pushed to both the main and the eye
contexts). `--no-snow` turns the texture and the prints off; the step events
stay, because the sound does not depend on the picture.

## The ClimbScene worlds

`app/harness/climb_worlds.py` calls `climb_scene.build_scene(...)` and drives
what comes back. The physics step is his (`ClimbScene.step(wind)` = one
`mj_step` + the carrier projection + the arc-length ratchet); the 103-dim
observation and the walking policy are his (`walk_policy.WalkController`).
Ours is the command, the wind vector, the friction knob, the telemetry and the
bookkeeping.

Two gates run before a world is trusted, both printed:

    python -m app.harness.climb_worlds --world lhotse_B

*joint parity* — the scene's 29 actuated joints against Playground's, in order
(a mismatch means the policy's actions drive the wrong joints) — and
*observation parity* — our builder against his `observe()` at the same state.
Measured **0.000e+00**: bit-identical.

`app/harness/provision_assets.py` puts the gitignored trees the scene needs on
disk (`.reference/`, the jacketed robot's stock link STLs, `~/mujoco_menagerie`)
by copying from the installed `mujoco_playground`, because all of the repo's own
fetch paths fail here. It runs automatically; PARITY.md says what is broken.

## The Pemba G1 (the real demo robot, legacy env only)

**Superseded by the ClimbScene worlds**, which build the jacketed robot by
default through the team's own `robot.resolve`/`adapt`. Kept because it is the
only way to put that robot in the *legacy* env.

`app/harness/robot_variants.py` generates a Playground-compatible scene that
wraps `assets/robots/mujoco/g1_unitree_ascender.xml`, then points the ONE line
their `_build_model` uses to choose a starting scene (`consts.task_to_xml`) at
it for the duration of one env construction. Their builder does everything else
— tilts the floor, adds the rope and carrier, connects the palm, sets foot
friction — unchanged and unaware. Nothing under `assets/robots/mujoco/` is
edited, and the generated files (absolute mesh paths, so machine-specific) live
gitignored in `app/harness/generated/`.

    python -m app.harness.robot_variants     # regenerate + print what it did

Joint parity is checked before anything else and raises if it ever fails: the
demo robot has the same 29 actuated joints in Playground's exact order, which
is the only reason its `knees_bent` keyframe and the policy's 29 actions are
transferable. See `PARITY.md` for the full diff and the open ASK about
actuator gains.

## Chloe: BMS plugs in here

`Episode.physics_step_hooks` is a list of `callable(model, data) -> dict | None`
called after **every** `mj_step` — that is `model.opt.timestep` (2 ms / 500 Hz),
the rate a battery or thermal model integrates at, not the 50 Hz control tick.
The last non-None dict any hook returns during a control tick becomes
`episode.latest_bms`, which is broadcast in the live state message as
`state["bms"]` and written to `hud.json` as a per-tick `bms` list.

    episode.physics_step_hooks.append(my_hook)   # that is the whole seam

`--bms` wires `app/bms/sim/mujoco_monitor.SimMonitor` in for you:

    python -m app.harness.runtime --live --world lhotse_B --bms

It builds `Environment(altitude_m=<world's altitude_meters, default 6907>,
wind_kmh=3.6 × wind dial)` and keeps `wind_kmh` live as the dial moves. Your
`step(data)` signature is adapted at the call site, so nothing in `app/bms`
needs to change. The whole attach is best-effort: if the import fails, the
harness prints why and runs on without a battery readout.
