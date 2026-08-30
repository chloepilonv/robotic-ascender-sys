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
held) — *unless the `guide` knob is on, in which case W drives the HUMAN and the
robot drives itself; see "Follow the guide" below.* Wind dial is in m/s and goes through **his** `WindParams` into
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


## Follow the guide (`app/harness/guide.py`)

Turn the **`guide`** knob on and a human guide appears 2.5 m up the rope. Hold
**W** and the *human* walks (0.5 m/s along the route, 0.6 m to the rope's left,
snapped to the terrain — it never slips and never falls, because it is a mocap
body with no degrees of freedom and no collision). The **robot decides for
itself** what to do about that. A/D do nothing while the guide is on: the
follower owns the yaw command, and the camera-follow controller stands down.

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

    python -m app.harness.test_guide                  # stereo accuracy + two rollouts
    python -m app.harness.runtime --world flat_0 --duration 20 --hold-w --guide --no-render

**Measured, and worth knowing before you demo it.** The stereo reads −5% at 2 m
and +7% at 4 m against the simulator's own answer. The *walker*, not the
follower, is the limit: its real ground speed is about **0.15 m/s** whatever
`lin_vel_x` says, and the guide walks at 0.5 m/s, so holding W indefinitely just
opens the gap. The demo that works is **hold W for a few seconds, then let go**
— the robot closes the gap and stops (measured on `flat_0`: 3.8 m → WAIT at
1.1 m). Yaw authority is also nearly nil while the palm is gripping the rope
(commanding +1 rad/s and −1 rad/s for 3 s ends up 10° apart), so the follower
steers much better on rope-off worlds.

The guide is available on the **ClimbScene** worlds only. The four `legacy_*`
worlds hand back a compiled model with no `MjSpec`, so there is nothing to add
the body and the cameras to, and the feature turns itself off there.

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
