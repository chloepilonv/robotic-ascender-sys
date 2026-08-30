"""The interactive walker harness, running the TEAM's climbing environment.

  ../.venv_everest/bin/python -m app.harness.runtime --live
  ../.venv_everest/bin/python -m app.harness.runtime --duration 15 --hold-w

THE ONLY FRONT END IS app/web/render3d.html (user's ruling, 2026-08-30). This
loop SIMULATES and BROADCASTS NUMBERS; it renders no picture of the scene at
all. What goes out is the `POS0` pose stream (`pose_stream.py`, ~1.1 kB a tick),
the state JSON, and -- while the guide is on -- the `EYE0` JPEG of the robot's
own left eye, which is a SENSOR readout and not a view of the world. The
third-person picture is drawn in the browser in three.js from the poses. What
was deleted with the 2-D page: the per-tick offscreen chase-camera render, its
raw-JPEG websocket frame, the browser-viewport resize negotiation and
`episode.mp4`. That render cost 10-20 ms of a 20 ms control tick.

Two layers, kept strictly apart:

THEIR STEP SEMANTICS (never re-derived; see team_env.py and PARITY.md)
    command (3,)                     -> the joystick command their obs carries
    observation = 103-d `state`      -> playground_policy.PlaygroundObservation
    action = policy(observation)     -> 29 raw values
    ctrl = default_pose + 0.5*action -> climb_env.py:359
    10 x (mj_step + ascender ratchet)-> climb_env.py:268-285
    phase += 2*pi*dt*gait_freq       -> climb_env.py:388-389
    fall = _get_termination          -> joystick.py:426-442
  No `mj_forward` between the substeps and the next observation: their MJX
  `step` reads sensors that are one substep stale, and so do we.

OUR APP LAYER (the demo; none of it exists on their side)
    W held            -> command lin_vel_x = 0.5 m/s (their lin_vel_x range is
                         [-1, 1], so this is a deliberate half-speed demo pace).
    mouse-look        -> HeadingController turns the robot toward the camera
                         heading with command ang_vel_yaw. NOTE: the right palm
                         is point-attached to a fixed line, so yaw authority is
                         genuinely limited -- the robot will fight the rope
                         rather than spin freely. That is the physics, not a bug.
    wind dial         -> the quadratic drag law from rl/environment/wind_env.py,
                         applied to the torso body's xfrc_applied. THE CLIMB ENV
                         HAS NO WIND: the dial is a demo affordance and every
                         state message says so with `wind_in_training: false`.
    friction slider   -> foot geom mu, live. Training pins it at
                         climb_config.foot_friction (0.8).
    reset / pause / recorder / pose stream / websocket.

Rates. Their model compiles at a 2 ms timestep, so physics is 500 Hz and one
control tick is 10 substeps at 50 Hz -- the ctrl_dt 0.02 / sim_dt 0.002 pair
their config declares. Live mode paces itself against the wall clock and prints
a realtime factor; a `--duration` run goes as fast as it can.
"""
import argparse
import math
import os
import sys
import time

import numpy as np

_REPOSITORY_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
if _REPOSITORY_ROOT not in sys.path:
    sys.path.insert(0, _REPOSITORY_ROOT)

# `import mujoco` used to live here for `MjvCamera` and `Renderer`; with the
# chase render gone this file touches no MuJoCo type directly. The modules it
# imports below still do.
from app.harness import ratchet as ratchet_module  # noqa: E402
from app.harness.playground_policy import (  # noqa: E402
    GaitPhase, MelsPolicy, PlaygroundObservation, TerminationCheck,
    default_policy_path,
)
from app.harness.recorder import Recorder  # noqa: E402
from app.harness import worlds as worlds_module  # noqa: E402
from app.harness import climb_worlds as climb_worlds_module  # noqa: E402
from app.harness import chloe_worlds as chloe_worlds_module  # noqa: E402
from app.harness import graphics as graphics_module  # noqa: E402
from app.harness import guide as guide_module  # noqa: E402
from app.harness import hearing as hearing_module  # noqa: E402
from app.harness import snow as snow_module  # noqa: E402
from app.harness import storm as storm_module  # noqa: E402
from app.harness.natural_wind import NaturalWind  # noqa: E402
sys.path.insert(0, os.path.join(  # human-safety/ is a program, not a package
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "human-safety"))
from human_gate import (  # noqa: E402
    HumanGate, HumanWorld, VirtualFrustumDetector)

# THE HARNESS RENDERS NOTHING. The one offscreen renderer left in the process
# belongs to the robot's eye cameras (`guide.StereoEyes`, 320x240), and it makes
# and sizes its own. What used to live here -- RENDER_WIDTH/HEIGHT,
# RENDER_MAXIMUM_*, `clamp_render_size` and `make_renderer` -- served the
# third-person JPEG the retired 2-D page displayed.

# The browser's third-person orbit still steers the robot, so its zero has to
# agree with ours even though nothing here draws it.
CAMERA_AZIMUTH_DEGREES = 180.0
# The protocol declares azimuth 180 to mean "behind the robot, looking uphill".
# MuJoCo's own azimuth 180 puts the camera UPHILL looking back down at the
# robot's face, so the browser's number is rotated half a turn before it
# reaches MjvCamera. A constant offset only moves where the orbit's zero is;
# dragging still feels the same. (Carried over from pemba_bench, and the sign
# is the same here because this floor also rises toward +x.)
BROWSER_AZIMUTH_OFFSET_DEGREES = 180.0

# Camera-relative driving, third-person-game style.
HEADING_GAIN_PER_RADIAN = 2.0
# A/D turn-in-place rate. The policy's ang_vel_yaw was trained over [-1, 1]
# (walk_policy.CMD_LIMITS), so this is the fastest turn it has ever seen.
MANUAL_YAW_RATE_RADIANS_PER_SECOND = 1.0
MAXIMUM_YAW_RATE_RADIANS_PER_SECOND = 1.0
HEADING_DEADBAND_RADIANS = math.radians(2.0)
# Their lin_vel_x training range is [-1, 1] m/s; a demo wants a steady pace.
# Overridable with --command-speed so their README's "0.75 m/s at cmd 1.0"
# claim can be checked rather than taken on faith.
CLIMB_COMMAND_METERS_PER_SECOND = 0.5

FALL_LINGER_SECONDS = 1.0     # timed runs end this long after the fall
PAUSED_BROADCAST_HZ = 5.0     # heartbeat while the browser has let go
GAIT_FREQUENCY_HZ = 1.375     # midpoint of their reset draw U(1.25, 1.5)
# rl/environment/wind_env.py:28-36 -- the ONLY place wind constants come from.
# Imported at load time rather than restated; these are the fallback if the
# import fails (e.g. their config keys get renamed) and the run says so.
WIND_FALLBACK = {"rho": 1.225, "cd_torso": 1.2, "area_torso": 0.5}


def make_battery_plugin(model, substeps):
    """Chloe's `app/bms_ui.BmsPlugin`, or None if it cannot be imported.

    Always on -- the battery readout is part of the demo now, not a flag. Kept
    best-effort anyway: a broken import in someone else's module must not take
    the walker down with it.

    ALTITUDE IS DELIBERATELY LEFT AT 0. Her `Environment` derives ambient from
    altitude by the ISA lapse (`t_amb = t_sea_c - 6.5e-3 * altitude_m`), while
    `set_ambient` treats the `t_amb` knob as the pack's actual temperature. Set
    both and they fight: at 6907 m the environment's ambient lands 44.9 C BELOW
    whatever the knob reads. So the knob is the single source of truth for
    ambient, and Everest conditions are dialled in by setting it to about
    -30 C rather than by declaring an altitude.
    """
    try:
        from app.bms_ui.bridge import BmsPlugin
    except Exception as error:  # pragma: no cover - reporting only
        print(f"[bms] NOT attached: {type(error).__name__}: {error}."
              " The harness runs normally without a battery readout.", flush=True)
        return None
    plugin = BmsPlugin(model, substeps)
    print(f"[bms] BmsPlugin attached: {model.nu} actuators, dt"
          f" {plugin.dt * 1000:.0f} ms (one call per control tick),"
          f" ambient {plugin.t_amb_c:.1f} C, soc0 {plugin.soc0:.0f}%"
          f"  [altitude left at 0 on purpose -- the t_amb knob is the truth]",
          flush=True)
    return plugin


def wind_drag_coefficient():
    """0.5 * rho * Cd * A, read from THEIR wind_config -- wind_env.py:57-59."""
    try:
        from rl.environment import wind_env
        wind_config = wind_env.default_config().wind_config
        values = {
            "rho": float(wind_config.rho),
            "cd_torso": float(wind_config.cd_torso),
            "area_torso": float(wind_config.area_torso),
        }
        source = "rl/environment/wind_env.py"
    except Exception as error:  # pragma: no cover - reporting only
        values, source = dict(WIND_FALLBACK), f"FALLBACK ({error})"
    coefficient = 0.5 * values["rho"] * values["cd_torso"] * values["area_torso"]
    print(f"[wind] drag coefficient 0.5*rho*Cd*A = {coefficient:.4f}"
          f"  from {source}  {values}", flush=True)
    return coefficient


# --------------------------------------------------------------- small math
def wrap_to_pi(angle_radians: float) -> float:
    return float((angle_radians + np.pi) % (2 * np.pi) - np.pi)


def root_yaw_radians(quaternion_wxyz) -> float:
    """Yaw about world +z from a w,x,y,z quaternion (MuJoCo's qpos ordering)."""
    w, x, y, z = [float(v) for v in quaternion_wxyz]
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


class HeadingController:
    """Turns "where the camera looks" into their 3-vector joystick command.

    The browser only ever sends W plus an orbit azimuth, so steering is derived:
    the desired heading is the camera's ground-plane viewing direction and a
    proportional controller closes the gap with ang_vel_yaw. The yaw command is
    issued whether or not W is held, so the robot pivots on the spot to follow
    the camera and walks off the moment W goes down.

    THE ROPE LIMITS THIS. The right palm is point-attached to a fixed line, so a
    large heading change drags the robot around its own hand and the yaw error
    may never close. The deadband stops the gait from being nudged every tick by
    the +/-2 degree yaw wobble walking itself produces.

    Output: (3,) [lin_vel_x m/s, lin_vel_y m/s, ang_vel_yaw rad/s] -- exactly
    their command layout (joystick.py:802-822, config lin_vel_x/lin_vel_y/
    ang_vel_yaw).
    """

    def __init__(self, command_speed=CLIMB_COMMAND_METERS_PER_SECOND):
        self.desired_heading_radians = math.radians(
            CAMERA_AZIMUTH_DEGREES + BROWSER_AZIMUTH_OFFSET_DEGREES)
        self.yaw_error_radians = 0.0
        self.command_speed = float(command_speed)

    def set_browser_azimuth(self, azimuth_degrees) -> None:
        if azimuth_degrees is not None:
            self.desired_heading_radians = math.radians(
                float(azimuth_degrees) + BROWSER_AZIMUTH_OFFSET_DEGREES)

    @property
    def desired_heading_degrees(self) -> float:
        return float(math.degrees(wrap_to_pi(self.desired_heading_radians)))

    def command(self, root_quaternion_wxyz, walking: bool,
                manual_yaw_rate=None) -> np.ndarray:
        """`manual_yaw_rate` (rad/s) SUSPENDS the camera-follow while held.

        A and D are the user steering by hand, and two controllers fighting for
        the same channel is the worst of both: the camera-follow would drag the
        yaw straight back the moment the key came up. So while A or D is down
        the follow's output is ignored, and its target is RE-SEATED to the
        robot's current yaw every tick -- whatever the yaw is at release becomes
        the new target, and the deadband reopens on a zero error. The camera
        only steers when the user isn't.
        """
        current = root_yaw_radians(root_quaternion_wxyz)
        if manual_yaw_rate is not None:
            self.desired_heading_radians = current
            self.yaw_error_radians = 0.0
            forward = self.command_speed if walking else 0.0
            return np.array([forward, 0.0, float(np.clip(
                manual_yaw_rate,
                -MAXIMUM_YAW_RATE_RADIANS_PER_SECOND,
                MAXIMUM_YAW_RATE_RADIANS_PER_SECOND))])
        self.yaw_error_radians = wrap_to_pi(self.desired_heading_radians - current)
        if abs(self.yaw_error_radians) < HEADING_DEADBAND_RADIANS:
            yaw_rate = 0.0
        else:
            yaw_rate = float(np.clip(
                HEADING_GAIN_PER_RADIAN * self.yaw_error_radians,
                -MAXIMUM_YAW_RATE_RADIANS_PER_SECOND,
                MAXIMUM_YAW_RATE_RADIANS_PER_SECOND))
        forward = self.command_speed if walking else 0.0
        return np.array([forward, 0.0, yaw_rate])


def make_header(episode, meta, arguments) -> dict:
    fingerprint_summary = {
        "kind": meta.get("kind", "legacy_climb_env"),
        "nq": int(episode.model.nq), "nv": int(episode.model.nv),
        "nu": int(episode.model.nu),
        "timestep_seconds": float(episode.model.opt.timestep),
        "action_scale": meta["action_scale"],
        "substeps_per_control_step": meta["substeps_per_control_step"],
        "slide_qpos_address": meta["slide_qpos_address"],
        "foot_friction_at_load": meta["foot_friction"],
    }
    if meta.get("kind") == "climb_scene":
        fingerprint_summary.update({
            "patch": meta["patch"], "robot": meta["robot"],
            "rope_length_meters": meta["rope_length_meters"],
            "rope_waypoints": meta["rope_waypoints"],
            "slope_provenance": meta["slope_provenance"],
            "adapt_report": meta["adapt_report"],
        })
    return {
        "backend": "mujoco-c (plain), model from rl.environment.climb_env.G1ClimbAscender",
        "world": episode.world_name,
        "world_label": episode.definition["label"],
        "world_description": episode.definition["description"],
        "config_overrides": dict(episode.definition.get("config_overrides", {})),
        "rope_enabled": episode.rope_enabled,
        # A `chloe_ascender` world is not necessarily a NETWORK world: the
        # `scripted_*` rungs share the plant and run a hand-written gait, whose
        # controller has no checkpoint to name. Ask the controller what it is,
        # rather than assuming every world of this kind carries a `.onnx`.
        "policy": (getattr(episode.controller, "policy_path", None)
                   and os.path.basename(episode.controller.policy_path)
                   or ("scripted gait (no network)"
                       if meta.get("kind") == "chloe_ascender"
                       else "mels_g1_joystick.npz")),
        "autonomous": bool(getattr(episode, "autonomous", False)),
        "seed": arguments.seed,
        "slope_degrees": episode.slope_degrees,
        "control_hz": episode.control_hz,
        "physics_hz": 1.0 / meta["physics_dt_seconds"],
        "joint_names": meta["joint_names"],
        "line_point_world": np.asarray(meta["line_point_world"]).tolist(),
        "slope_axis_world": np.asarray(meta["slope_axis_world"]).tolist(),
        "patch": meta.get("patch"),
        "robot": meta.get("robot", "bare"),
        "spawn_position_world": episode.spawn_position_world.tolist(),
        "command_speed_meters_per_second": arguments.command_speed,
        "observation_noise_level": 0.0,
        "training_observation_noise_level": meta["noise_level"],
        "wind_in_training": meta.get("wind_in_training", False),
        "terrain_in_training": meta.get("terrain_in_training", False),
        "model_fingerprint": fingerprint_summary,
        "wind": [], "commands": [],
        "source": "live" if arguments.live else "timed",
    }


def episode_outcome(episode, realtime_factor: float) -> dict:
    return {
        "fell": episode.fell_at_seconds is not None,
        "fell_at_seconds": episode.fell_at_seconds,
        "fall_reason": episode.fall_reason,
        "distance_meters": float(np.linalg.norm(
            (episode.pelvis_position_world - episode.spawn_position_world)[:2])),
        "pelvis_displacement_meters": float(np.linalg.norm(
            episode.pelvis_position_world - episode.spawn_position_world)),
        "rope_travel_meters": episode.rope_travel_meters,
        "climb_meters": episode.rope_travel_meters,
        "arclength_meters": getattr(episode, "arclength_meters", None),
        "height_gained_meters": episode.height_gained_meters,
        "maximum_rope_force_newtons": episode.maximum_rope_force_newtons,
        "hand_line_error_meters": episode.hand_line_error_meters(),
        "world": episode.world_name,
        "rope_enabled": episode.rope_enabled,
        "slope_degrees": episode.slope_degrees,
        "realtime_factor": realtime_factor,
    }


# ---------------------------------------------------------------- the run
def run(arguments) -> str:
    policy = MelsPolicy(arguments.policy or default_policy_path(_REPOSITORY_ROOT))
    print(policy.describe(), flush=True)
    wind_drag = wind_drag_coefficient()
    climb_library = climb_worlds_module.ClimbSceneLibrary()
    # CHLOE'S WORLDS ARE THE ONE PLACE THE WALKER DOES NOT FLY THE ROBOT.
    # `chloe_worlds` builds her mjlab plant and drives her ONNX ascender
    # policy; every other world is untouched by its existence.
    chloe_library = chloe_worlds_module.ChloeSceneLibrary()

    server = None
    if arguments.live:
        from app.harness.server import Server
        server = Server(arguments.port, worlds=worlds_module.describe_worlds())

    def announce_build(name):
        """Tell the page why the picture is about to freeze.

        A world's first selection costs a full G1ClimbAscender.__init__, and the
        sim loop is what does it, so no poses go out while it runs. Measured
        warm that is ~1.6 s for the first world and ~0.2 s for the second
        distinct model -- brief, but a frozen picture with no explanation is
        still worse than a labelled one, and a cold venv is slower. Push one
        last state carrying `loading: true` before blocking; the page's toast
        waits on it (timeout raised to 60 s, generous headroom).
        """
        if server is None:
            return
        if latest_state[0] is not None:
            server.broadcast(dict(latest_state[0], paused=True, loading=True,
                                  loading_world=name))

    latest_state = [None]

    def open_world(name):
        """Either kind of world, behind one interface.

        `climb_scene` worlds are PR #8's merged model -- real terrain, a rope
        draped over it, the jacketed robot, his physics step and his walking
        policy. `legacy_climb_env` worlds are the older flat tilted plane and
        slide joint, kept because the trainer still uses them. Both return an
        episode with the same interface, so everything below this function is
        shared.
        """
        name = worlds_module.resolve_world_name(name)
        kind = worlds_module.WORLD_DEFINITIONS[name]["kind"]
        if kind in ("climb_scene", "chloe_ascender"):
            # Two libraries, one shape. `chloe_ascender` worlds are her mjlab
            # plant and her ONNX policy (app/harness/chloe_worlds.py); they
            # present the same scene surface -- spec, model, data, terrain,
            # route, ascender, reset -- so every line below this one, the
            # guide surgery and the whole alpine dressing included, is shared.
            library = (chloe_library if kind == "chloe_ascender"
                       else climb_library)
            scene, meta, definition = library.load(
                name, on_build_start=lambda: announce_build(name))
            # THE GUIDE'S SURGERY GOES FIRST, before anything dresses the model.
            # It recompiles the spec, and `apply_alpine_look` writes to the
            # COMPILED model (lights, fog, the snow colour) -- so doing it the
            # other way round throws the alpine look away and the picture comes
            # back dark. `add_skybox` survives either order because a texture
            # lives in the spec, but the ordering rule is the same for both.
            # `--no-guide-body` skips the surgery entirely, which is how the
            # physics-parity claim is measured: a run with no guide bodies in
            # the model at all against a run that has them, same seed.
            if not arguments.no_guide_body:
                guide_module.attach_guide(scene)
            # Snow next, and for the same reason: it adds a texture and a
            # material to the spec and recompiles, so it has to happen before
            # anything writes to the compiled model.
            if not (arguments.plain_graphics or arguments.no_snow):
                snow_module.attach_snow(scene, seed=arguments.seed)
            # Dress the scene BEFORE the episode binds to it: `add_skybox`
            # recompiles the spec, so it must happen while nothing holds a
            # reference to the old model or data. It verifies the swap and
            # refuses if anything structural moved.
            if not arguments.plain_graphics:
                graphics_module.add_skybox(scene)
                look = graphics_module.apply_alpine_look(
                    scene.model, terrain_size_meters=scene.terrain.size_xy)
                print(f"[graphics] {name}: fog"
                      f" {look['fog_start_meters']:.0f}-{look['fog_end_meters']:.0f} m,"
                      f" sun {look['sun']['elevation_degrees']:.0f} deg elevation,"
                      f" shadows {look['shadow_texture']}, snow on", flush=True)
            if kind == "chloe_ascender":
                episode = chloe_worlds_module.ChloeAscenderEpisode(
                    scene, meta, definition, name, seed=arguments.seed,
                    policy_path=arguments.chloe_policy,
                    hold_blend_seconds=arguments.chloe_hold_blend)
                print(f"[runtime] {name}: {episode.controller.describe()}"
                      "  -- W gates the network, A/D and the mouse do nothing",
                      flush=True)
            else:
                episode = climb_worlds_module.ClimbSceneEpisode(
                    scene, meta, definition, name, seed=arguments.seed)
            model = scene.model
            print(f"[runtime] world={name} ({definition['label']})"
                  f"  patch={definition['patch']} robot={definition['robot']}"
                  f"  slope={episode.slope_degrees:.1f} deg"
                  f" ({definition['slope_provenance']})"
                  f"  rope={'ON' if episode.rope_enabled else 'OFF'}"
                  f"  control {episode.control_hz:.0f} Hz  physics"
                  f" {1.0 / meta['physics_dt_seconds']:.0f} Hz"
                  f"  substeps/tick={episode.substeps}", flush=True)
            print(f"[runtime] spawn pelvis"
                  f" {episode.spawn_position_world.round(4).tolist()}"
                  f"  hand-rope distance {episode.hand_line_error_meters():.2e} m"
                  f"  arc length {episode.arclength_meters:.3f} m of"
                  f" {meta['rope_length_meters']:.3f}"
                  f"  lean {meta['lean_degrees']:.1f} deg"
                  f"  ankle {meta['ankle_degrees']:.1f} deg"
                  f"  upright {episode.torso_upright:+.3f}", flush=True)
            return episode, model, meta, scene


    # THE 3-D PAGE'S SEAM (app/web/render3d.html). A wrapper rather than edits
    # inside `open_world`: every episode -- the first and every map switch --
    # gets a pose broadcaster on its physics_step_hooks, and this file keeps one
    # hunk. Arity-agnostic on purpose (`open_world` grew a fourth return value
    # while this was being written); it only ever touches element 0.
    # Costs tens of microseconds of a 20 ms tick and prints that measurement
    # itself. Since the 2-D page was retired this IS the picture: with the pose
    # stream off, a browser gets telemetry and no scene.
    if arguments.pose_stream and server is not None:
        from app.harness import pose_stream as pose_stream_module
        _open_world_without_poses = open_world

        def open_world(name):
            opened = _open_world_without_poses(name)
            pose_stream_module.attach(opened[0], server, opened[0].world_name)
            return opened

    episode, model, meta, scene = open_world(arguments.world)
    print(f"[runtime] observation noise OFF (training level {meta['noise_level']});"
          f" wind NOT in training; friction knob starts at"
          f" {meta['foot_friction']}", flush=True)
    if server is not None:
        server.knobs["friction"] = meta["foot_friction"]

    # HUMAN GATE (human-safety/human_gate.py). Deterministic, outside the policy:
    # the forward (= up-rope) command is clamped to <= 0 while a human is in the
    # d435i frustum. Humans are virtual (no physics; THEIR model is untouched).
    human_world = HumanWorld.from_model(model)   # virtual unless the model has human_* bodies
    human_gate = HumanGate(
        VirtualFrustumDetector(model, human_world, arguments.human_range),
        clear_after_seconds=arguments.human_clear_seconds)
    for distance in arguments.human:
        human_world.spawn_ahead_of(
            episode.spawn_position_world, root_yaw_radians(episode.data.qpos[3:7]),
            distance)
        print(f"[safety] human spawned {distance:.1f} m ahead", flush=True)

    # THE GUIDE FOLLOWER (app/harness/guide.py). A human guide walks ahead along
    # the rope; the robot measures its distance by STEREO from two head cameras
    # and drives itself. Off until the page's `guide` knob (or --guide) says so.
    #
    # ONE NOTION OF "A HUMAN IS THERE". While the guide is on, Chloe's gate is
    # driven from the same vision measurement the follower uses, so the gate
    # blocks UP over exactly the band in which the follower commands zero,
    # instead of a second oracle detector disagreeing with it. Its own
    # hysteresis is off (0.0) because the follower already has two -- the
    # 1.0/1.3 m bands and the 1 s LOST timeout.
    # VISIBILITY, as the robot experiences it (app/harness/storm.py). The page's
    # `visibility` knob is a DISTANCE IN METRES, derived from nothing (user's
    # ruling, 2026-08-30 -- it replaced a `storm` switch whose thickness came
    # out of the wind speed). A fog is composited into the eye images between
    # the render and the block matcher. Visual and sensor only: PARITY.md has
    # the same-seed diff.
    storm_vision = storm_module.StormVision(seed=arguments.seed)

    def make_guide(current_scene, current_model, current_episode):
        # CHLOE'S WORLDS ARE AUTONOMOUS. Her network has no command port at
        # all, so nothing here may claim to steer it: `yaw_command_available`
        # is forced False and the waist "neck" is NOT registered, whatever
        # `--policy` says. The follower still SEES -- both eyes render, the
        # block matcher runs, the distance and bearing are real -- and it may
        # drive the go/stop gate through `command[0]`. It may not aim the robot.
        autonomous = bool(getattr(current_episode, "autonomous", False))
        system = guide_module.GuideSystem(
            current_scene, current_model, current_episode.control_hz,
            enable=not arguments.no_guide_body,
            degradation=storm_vision.degrade,
            # ASK TO MRINAL, and the reason this is a flag rather than a
            # constant: the climb policy was trained with ang_vel_yaw ~ 0, so a
            # commanded turn does nothing and the waist has to do the aiming.
            # A policy trained with randomised ang_vel_yaw would earn a True
            # here and a steerable body with it. `--policy` supplies one.
            yaw_command_available=(arguments.policy is not None
                                   and not autonomous),
            # No rope means no line for her to be on: W/S walk her along her own
            # heading and A/D turn it.
            free_walk=not current_episode.rope_enabled,
            # WALK THE VECTOR on rope-off worlds only (user's ruling,
            # 2026-08-30). With the palm clipped to the rope a lateral command
            # fights the line, and Chloe's autonomous worlds have no command
            # port at all -- both keep the old scalar approach law.
            vector_steering=(not current_episode.rope_enabled
                             and not autonomous))
        gate = HumanGate(guide_module.GuideVisionDetector(system),
                         clear_after_seconds=0.0)
        if system.available:
            system.place(current_episode.spawn_position_world)
        if system.available and not autonomous:
            # THE "NECK", REGISTERED ON THE CONTROL SEAM. `control_hooks` runs
            # after the policy writes `data.ctrl` and before the `mj_step` that
            # acts on it, which is the only place a waist-yaw offset can be
            # added without touching the policy. It is a no-op whenever the
            # offset is zero, i.e. whenever the robot is not searching.
            current_episode.control_hooks.append(system.waist.apply)
        return system, gate

    guide_system, guide_gate = make_guide(scene, model, episode)

    # THE EARS (app/harness/hearing.py). The page streams the Mac microphone
    # down the socket as `MIC0` PCM; the runtime emits that voice at the
    # HIKER'S MOUTH and receives it at four virtual microphones on the robot's
    # head, then decides from those four signals alone whether a human voice is
    # there, whether the word was `stop`, and which way it came from. It is the
    # same honesty as the eyes: truth geometry in, a sensor image out, and every
    # decision downstream of the sensor image. With the knob off it is a no-op
    # and the guide follower's command goes through untouched.
    hearing_system = hearing_module.HearingSystem(
        model, episode.control_hz, seed=arguments.seed,
        walk_speed=arguments.command_speed)
    for specification in arguments.inject_voice:
        path, _, at_seconds = specification.partition("@")
        hearing_system.add_injector(path, float(at_seconds or 0.5))
    if server is not None:
        server.knobs["hearing"] = 1.0   # ears ON by default (user ruling 2026-08-30); --hearing kept for headless
        server.microphone = hearing_system.microphone
    print(f"[hearing] starts ON by default"
          f" (knob `hearing`); it needs the GUIDE on, because the voice comes"
          f" out of her mouth and the hand-over is to her eyes", flush=True)

    # FOOTSTEPS (app/harness/snow.py). The detector reads the solver's own
    # contacts and reports each landing; that one event counts a step and
    # becomes a `foot_steps` message, so the crunch the page plays, the decal it
    # drops and the counter can never disagree about what a step is. The snow
    # TEXTURE was attached to the spec at world build above; the prints that
    # used to be stamped into it are the 3-D page's decals now.
    def make_snow(current_model, current_meta):
        detector = None
        if current_meta.get("foot_geom_ids") is not None and \
                "terrain_geom_id" in current_meta:
            detector = snow_module.TouchdownDetector(
                current_model, current_meta["foot_geom_ids"],
                current_meta["terrain_geom_id"])
            print(f"[snow] touchdowns from {detector.describe()}", flush=True)
        return detector

    touchdowns = make_snow(model, meta)
    if server is not None:
        server.knobs["guide"] = 1.0 if arguments.guide else 0.0
        server.knobs["eyes"] = 1.0   # vision on unless the page says otherwise
    print(f"[guide] {'available' if guide_system.available else 'NOT available'}"
          f" in world {episode.world_name};"
          f" starts {'ON' if arguments.guide else 'OFF'}"
          f" (knob `guide`, W walks the human up the rope, S back down it)",
          flush=True)
    starting_visibility = storm_module.clamp_visibility_meters(
        arguments.visibility)
    print(f"[visibility] starts at {starting_visibility:.1f} m"
          f" (knob `visibility`, metres;"
          f" {storm_module.CLEAR_VISIBILITY_METERS:.0f} m = clear, the eyes"
          f" untouched;"
          f" {storm_module.MINIMUM_VISIBILITY_METERS:.0f} m = white-out)."
          f" A fog composited from the eye renderer's depth buffer before the"
          f" matcher; white-out share"
          f" {', '.join(f'{v:.0f} m -> {storm_module.whiteout_share(v):.2f}' for v in (100, 30, 10, 3))}",
          flush=True)

    heading = HeadingController(arguments.command_speed)

    episodes_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "episodes")
    directories_opened = []

    def new_episode_directory() -> str:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        # --output-name names the FIRST episode only; a map switch must never
        # reopen it.
        if arguments.output_name and not directories_opened:
            name = arguments.output_name
        else:
            name = (f"{stamp}_{'live' if arguments.live else 'timed'}"
                    f"_{episode.world_name}")
        directories_opened.append(name)
        return os.path.join(episodes_root, name)

    output_directory = new_episode_directory()
    header = make_header(episode, meta, arguments)
    recorder = Recorder(output_directory, header, control_hz=episode.control_hz)

    duration_seconds = arguments.duration if arguments.duration is not None else 1e9
    wind_velocity_world = np.zeros(2)
    # The dial gives a TARGET; NaturalWind turns it into gusts and drift when
    # the page asks for it. Seeded from --seed, advanced once per control tick,
    # so a replay at the same seed sees the same weather.
    natural_wind = NaturalWind(seed=arguments.seed)
    last_logged_command = last_logged_wind = None
    wall_start = time.time()
    realtime_factor = 0.0
    applied_friction = meta["foot_friction"]

    while True:
        time_seconds = episode.tick / episode.control_hz
        if time_seconds >= duration_seconds:
            break
        if (not arguments.live and not arguments.keep_going
                and episode.fell_at_seconds is not None
                and time_seconds >= episode.fell_at_seconds + FALL_LINGER_SECONDS):
            break

        guide_enabled = bool(arguments.guide)
        eyes_enabled = True   # headless default; the page's `eyes` knob overrides
        world_frozen = False  # guide reached -> physics held (see the WAIT freeze)
        walking = bool(arguments.hold_w)
        backing = False
        # A/D as the HIKER's steering, +1 = left. Only ever non-zero on a
        # rope-off world with the guide on; on the rope she is ON the line and
        # there is nowhere for her to turn to (user's ruling, 2026-08-30).
        guide_turning = 0.0
        if server is not None:
            latest = server.latest_input
            keys = set(latest.get("keys", []))
            walking = "w" in keys
            # S walks the HUMAN back down the rope toward the robot. Only the
            # human: the robot's own command still comes from the follower.
            backing = "s" in keys
            guide_enabled = bool(server.knobs.get("guide", 0.0))
            eyes_enabled = bool(server.knobs.get("eyes", 1.0))
            # Only the AZIMUTH is read: it is the steering input. Elevation
            # moves the browser's own camera and the server no longer draws
            # anything, so the page still sends it and the harness ignores it.
            browser_camera = latest.get("camera") or {}
            heading.set_browser_azimuth(browser_camera.get("azimuth_degrees"))
            # A = turn left (positive yaw), D = turn right, both down cancel.
            turn = ((MANUAL_YAW_RATE_RADIANS_PER_SECOND if "a" in keys else 0.0)
                    - (MANUAL_YAW_RATE_RADIANS_PER_SECOND if "d" in keys else 0.0))
            steering = ("a" in keys) != ("d" in keys)
            # WITH THE GUIDE ON AND NO ROPE, A AND D ARE HERS. They were already
            # doing nothing to the robot -- the follower owns its command while
            # the guide is on -- so this hands two idle keys to the one body
            # that can use them.
            if guide_enabled and not episode.rope_enabled:
                guide_turning = ((1.0 if "a" in keys else 0.0)
                                 - (1.0 if "d" in keys else 0.0))
            command = heading.command(
                episode.data.qpos[3:7], walking="w" in keys,
                manual_yaw_rate=turn if steering else None)
            if server.paused:
                # Freeze: no physics, no policy, no recorded tick. Keep a
                # heartbeat flowing so the page stays live and knows why nothing
                # moves -- it goes on drawing the last poses it has. The pacing
                # clock is rebased every frame, so unpausing resumes at realtime
                # instead of firing a catch-up burst of ticks for the paused
                # wall seconds.
                if latest_state[0] is not None:
                    server.broadcast(dict(latest_state[0], paused=True))
                wall_start = time.time() - episode.tick / episode.control_hz
                time.sleep(1.0 / PAUSED_BROADCAST_HZ)
                continue
            natural_wind.enabled = bool(server.knobs.get("wind_natural", 0.0))
            wind_velocity_world[:] = natural_wind.step(
                [server.knobs.get("wind_x", 0.0),
                 server.knobs.get("wind_y", 0.0)],
                1.0 / episode.control_hz, episode.tick / episode.control_hz)
            # An older page still sends a `storm` 0/1 knob. `Server._handle`
            # stores any name it is given, so that key simply sits in the dict
            # unread -- accepted, ignored, and unable to crash this loop.
            visibility_meters = server.knobs.get(
                "visibility", storm_module.CLEAR_VISIBILITY_METERS)
            # The battery model's wind chill is live: the dial is m/s, hers is
            # km/h.
            if episode.bms is not None:
                # t_amb / soc0 come from the page; her plugin decides what a
                # change means (a cold-soak jump for ambient, a full reset for
                # soc0), so we just hand the knobs over every tick.
                episode.bms.apply_knobs(server.knobs)
            friction = float(server.knobs.get("friction", applied_friction))
            if abs(friction - applied_friction) > 1e-9:
                episode.set_foot_friction(friction)
                applied_friction = friction
            if server.world_requested is not None:
                requested, server.world_requested = server.world_requested, None
                requested = worlds_module.resolve_world_name(requested)
                if requested not in worlds_module.WORLD_DEFINITIONS:
                    print(f"[runtime] ignoring unknown world {requested!r}; have"
                          f" {worlds_module.world_names()}", flush=True)
                else:
                    # Finalise the episode we are leaving, then open the new
                    # world at its own spawn in a fresh episode folder. Building
                    # a world for the first time blocks this loop (~1.6 s warm)
                    # -- `announce_build` warns the page first.
                    recorder.finalize(episode_outcome(episode, realtime_factor))
                    episode, model, meta, scene = open_world(requested)
                    # A new world is a new compiled model, so the eyes' renderer
                    # and every id the guide cached belong to a model that is
                    # gone. Build both again, and re-place the human 2.5 m ahead
                    # of the new spawn.
                    guide_system.close()
                    guide_system, guide_gate = make_guide(scene, model, episode)
                    # A new world is a new compiled model, so the head body id
                    # the ears cached belongs to a model that is gone.
                    hearing_system.bind(model)
                    hearing_system.reset()
                    touchdowns = make_snow(model, meta)
                    # The friction knob still reads the OLD world; re-sync it or
                    # the next tick paints the previous mu over the new map.
                    applied_friction = meta["foot_friction"]
                    server.knobs["friction"] = applied_friction
                    header = make_header(episode, meta, arguments)
                    recorder = Recorder(new_episode_directory(), header,
                                        control_hz=episode.control_hz)
                    last_logged_command = last_logged_wind = None
                    wall_start = time.time()
                    continue
            if server.reset_requested:
                server.reset_requested = False
                episode.reset()
                guide_system.place(episode.spawn_position_world)
                hearing_system.reset()
                wall_start = time.time()
                continue
        else:
            command = np.array([
                arguments.command_speed if arguments.hold_w else 0.0, 0.0, 0.0])
            visibility_meters = starting_visibility
            natural_wind.enabled = bool(arguments.wind_natural)
            wind_velocity_world[:] = natural_wind.step(
                arguments.wind, 1.0 / episode.control_hz,
                episode.tick / episode.control_hz)

        # VISIBILITY, before the guide: `degrade` has to know this tick's
        # visibility before the eye cameras render, which happens inside
        # `guide_system.update` on the next line. The white-out on the PAGE is
        # the 3D view's own fog (render3d.html + three/world.js), driven from
        # the same knob echoed back in the state message.
        storm_vision.update(visibility_meters)

        # THE GUIDE OWNS THE COMMAND WHILE IT IS ON. W/A/D stop steering the
        # robot: W tells the HUMAN to walk, and what the robot does about that
        # is the follower's business -- which is the whole point of the feature.
        # The camera-follow controller is stood down too (its target is re-seated
        # to the robot's actual yaw every tick, exactly as it is while A or D is
        # held), so switching the guide off does not snap the robot back to a
        # heading it drifted away from ten seconds ago.
        guide_command = guide_system.update(
            episode.data, episode.tick, guide_enabled, walking, backing,
            guide_turning, eyes_enabled=eyes_enabled)
        if guide_command is not None:
            command = guide_command
            heading.desired_heading_radians = root_yaw_radians(
                episode.data.qpos[3:7])
            heading.yaw_error_radians = 0.0

        # THE EARS, AFTER THE EYES AND BEFORE THE GATE. They need this tick's
        # vision verdict (that is what "if the eyes see her, vision drives"
        # means) and they may hand back a different command, which the safety
        # gate must then still be allowed to veto.
        hearing_enabled = bool(arguments.hearing)
        if server is not None:
            hearing_enabled = bool(server.knobs.get("hearing", 0.0))
            hearing_system.monitor_enabled = bool(
                server.knobs.get("ear_monitor", 0.0))
        hearing_command = hearing_system.update(
            episode.data, episode.tick, hearing_enabled, guide_system,
            guide_command, natural_wind.report()["wind_speed_mps"],
            episode.tick / episode.control_hz)
        if hearing_command is not None:
            command = hearing_command

        gate = guide_gate if guide_command is not None else human_gate
        gate.update(episode.data, episode.tick / episode.control_hz)
        # ON AUTONOMOUS (rope-policy) WORLDS THE MASK IS SKIPPED -- user-found
        # bug 2026-08-30: with the guide on, "a human is in front" clamped the
        # forward command to zero, and on these worlds that IS the W gate, so
        # the robot stood still forever while looking at her. The safety job
        # moves to the follower's WAIT world-freeze below, which stops the
        # robot the moment she is inside 1.5 m -- harder than a clamp does.
        if not bool(getattr(episode, "autonomous", False)):
            command = gate.mask(command)

        # GUIDE REACHED -> FREEZE THE WORLD (user's ruling, 2026-08-30). While
        # the follower says WAIT, physics is NOT stepped at all: the robot is
        # held bit-identical mid-stride, so nothing -- wind, drift, a bad
        # half-pose -- can knock it over while it waits, and resuming from an
        # untouched state cannot fall. Everything above this line already ran
        # this iteration, so the HUMAN still walks (she is mocap), the eyes
        # still render and measure, and the ears still listen; the moment she
        # is past the follow band (1.8 m) the follower flips to FOLLOW and the
        # next iteration steps physics again. On autonomous rope worlds this
        # freeze IS the human-safety stop (the gate mask is skipped there).
        follower_waiting = (guide_command is not None
                            and guide_system.follower.mode == "WAIT")
        # A confident STOP freezes the world the same way (user's ruling,
        # 2026-08-30): the ears keep listening while frozen, and the next
        # voice that is not a stop flips the behaviour out of STOPPED, which
        # un-freezes and resumes the climb.
        hearing_stopped = (hearing_enabled and str(getattr(
            hearing_system.behaviour, "mode", "")).upper() == "STOPPED")
        freeze_reason = ("stop" if hearing_stopped
                         else "guide" if follower_waiting else None)
        if freeze_reason is not None:
            if not world_frozen:
                world_frozen = True
                print("[runtime] world FROZEN"
                      + (": STOP heard, call to resume" if freeze_reason == "stop"
                         else ": guide reached, waiting for her to advance"),
                      flush=True)
            if server is not None:
                eye_jpeg = guide_system.take_eye_jpeg()
                if eye_jpeg is not None:
                    server.broadcast(eye_jpeg)
                ear_pcm = hearing_system.take_ear_pcm()
                if ear_pcm is not None:
                    server.broadcast(ear_pcm)
                if latest_state[0] is not None:
                    frozen_state = dict(latest_state[0])
                    frozen_state["world_frozen"] = True
                    frozen_state["world_frozen_reason"] = freeze_reason
                    frozen_state["guide"] = guide_system.state()
                    frozen_state["hearing"] = hearing_system.state()
                    frozen_state["command"] = [0.0, 0.0, 0.0]
                    server.broadcast(frozen_state)
            # THE CLOCK RUNS WHILE THE WORLD IS HELD. The eye render and the
            # ear detectors are both `tick % N` gated, so a freeze that pinned
            # the tick could start on the wrong remainder and then NEVER see
            # her leave or hear the resume voice (and a --duration run would
            # never end -- measured as a 10-minute hang). Advancing the tick
            # without stepping physics keeps every cadence and exit alive.
            episode.tick += 1
            # Rebase the pacing clock every held iteration, exactly as the
            # pause path does, so resuming does not fire a catch-up burst.
            wall_start = time.time() - episode.tick / episode.control_hz
            time.sleep(1.0 / episode.control_hz)
            continue
        if world_frozen:
            world_frozen = False
            print("[runtime] world RESUMED", flush=True)

        row = episode.step(command, wind_velocity_world)

        if row["command"].tolist() != last_logged_command:
            header["commands"].append({
                "time_seconds": row["time_seconds"],
                "lin_vel_x": float(row["command"][0]),
                "lin_vel_y": float(row["command"][1]),
                "ang_vel_yaw": float(row["command"][2])})
            last_logged_command = row["command"].tolist()
        wind_list = row["wind_velocity_world_meters_per_second"].tolist()
        row.update(natural_wind.report())
        if wind_list != last_logged_wind:
            header["wind"].append({"time_seconds": row["time_seconds"],
                                   "wind_velocity_world_meters_per_second": wind_list})
            last_logged_wind = wind_list
        # TOUCHDOWNS. Read after the step, from the solver's own contacts: each
        # landing counts a step and becomes one `foot_steps` event on the wire,
        # which the page turns into a crunch and a decal. Reading contacts is
        # microseconds; the expensive half of this feature -- stamping the print
        # into the ground texture and re-uploading it to every GL context -- was
        # deleted with the 2-D page.
        foot_steps = []
        if touchdowns is not None:
            for landing in touchdowns.update(episode.data, 1.0 / episode.control_hz):
                foot_steps.append({"foot": landing["foot"],
                                   "impact_speed_mps": landing["impact_speed_mps"]})

        recorder.append(**{k: v for k, v in row.items() if k != "observation"})
        recorder.append(**guide_system.recorded())
        recorder.append(**hearing_system.recorded())
        recorder.append(**storm_vision.recorded())
        recorder.append(step_count=float(
            touchdowns.step_count if touchdowns is not None else 0))
        recorder.append_bms(episode.latest_bms)

        elapsed = max(time.time() - wall_start, 1e-9)
        realtime_factor = row["time_seconds"] / elapsed

        if server is not None:
            # The left eye, at the vision rate: the 4 bytes `EYE0` then a JPEG.
            # This is the ROBOT'S SENSOR, not a view of the scene -- the only
            # picture the server still encodes, and the page shows it as the
            # guide card's PiP. Poses are broadcast from the physics hook
            # (pose_stream.py), which is what the 3-D view is drawn from.
            eye_jpeg = guide_system.take_eye_jpeg()
            if eye_jpeg is not None:
                server.broadcast(eye_jpeg)
            # `EAR0` + int16 PCM: the front microphone's own signal, so the
            # operator can listen to WHAT THE ROBOT HEARS rather than to what
            # the room sounds like. Only while the page asks for it.
            ear_pcm = hearing_system.take_ear_pcm()
            if ear_pcm is not None:
                server.broadcast(ear_pcm)
            latest_state[0] = {
                "type": "state", "tick": episode.tick,
                "time_seconds": row["time_seconds"],
                "command": row["command"].tolist(),
                "wind_velocity_world_meters_per_second": wind_list,
                "wind_force_world_newtons": row["wind_force_world_newtons"].tolist(),
                "wind_in_training": meta.get("wind_in_training", False),
                # INSTANTANEOUS, not the dial: with natural wind on these surge
                # and swing with every gust, and the ribbons/sound follow them.
                **natural_wind.report(),
                # How far the ROBOT can see, in metres -- always a number, and
                # the single field the page's fog, flakes and slider read.
                **storm_vision.state(),
                # Whichever gate is live -- the guide's vision while the guide
                # is on, the sim oracle otherwise. One set of `human_*` fields
                # either way, so the page needs no new case.
                **gate.state(),
                "guide": guide_system.state(),
                "hearing": hearing_system.state(),
                # Landings that happened on THIS tick (usually none), and the
                # running total. The page plays one crunch per event, at a
                # volume set by the impact speed.
                "foot_steps": foot_steps,
                "step_count": int(touchdowns.step_count) if touchdowns else 0,
                "fell": bool(row["fell"]),
                "fall_reason": episode.fall_reason,
                "root_position_world": row["root_position_world"].tolist(),
                "rope_travel_meters": row["rope_travel_meters"],
                "climb_meters": row["climb_meters"],
                "arclength_meters": row.get("arclength_meters"),
                "rope_length_meters": meta.get("rope_length_meters"),
                "world_kind": meta.get("kind", "legacy_climb_env"),
                "terrain_in_training": meta.get("terrain_in_training", False),
                "hand_height_on_line_meters": row["hand_height_on_line_meters"],
                "hand_line_error_meters": row["hand_line_error_meters"],
                "height_gained_meters": row["height_gained_meters"],
                "rope_force_newtons": row["rope_force_newtons"],
                "slope_degrees": episode.slope_degrees,
                "robot": meta.get("robot", "bare"),
                "realtime_factor": realtime_factor,
                "heading_degrees": heading.desired_heading_degrees,
                "world": episode.world_name,
                "world_label": episode.definition["label"],
                "rope_enabled": episode.rope_enabled,
                # TRUE ON CHLOE'S WORLDS ONLY. The page reads it for one
                # purpose: to say out loud that nothing steers this robot --
                # W gates her policy, A/D and the mouse are dead, and the
                # guide card's readouts are a measurement, not a control loop.
                "autonomous": bool(getattr(episode, "autonomous", False)),
                "paused": False,
                "loading": False,
                "world_frozen": False,
                # bms + actuator_names + r_int_curve, straight from her plugin.
                **(episode.bms.state() if episode.bms else {}),
            }
            server.broadcast(latest_state[0])
            sleep_for = wall_start + episode.tick / episode.control_hz - time.time()
            if sleep_for > 0:
                time.sleep(sleep_for)

        if (hearing_system.enabled
                and episode.tick % int(episode.control_hz) == 0):
            # ONCE A SECOND, WHAT THE MICROPHONE ACTUALLY DELIVERED. The page is
            # asked for no automatic gain control, so this number really is how
            # loudly the person spoke -- and a shout is what carries through
            # wind.
            print(hearing_system.microphone.describe(), flush=True)
            print(f"[hearing] {hearing_system.behaviour.mode}"
                  f"  voice p={hearing_system.ears.voice_probability:.2f}"
                  f"  heard={hearing_system.ears.heard}"
                  f" ({hearing_system.ears.stop_confidence:.2f})"
                  f"  bearing="
                  + ("--" if hearing_system.ears.bearing_radians is None else
                     f"{math.degrees(hearing_system.ears.bearing_radians):+.0f}°")
                  + f" (conf {hearing_system.ears.bearing_confidence:.2f})"
                  f"  ear level {hearing_system.ears.level_db:.0f} dBFS"
                  f"  | {hearing_system.describe_cost()[10:]}", flush=True)

        if episode.tick % int(episode.control_hz) == 0:
            print(f"[runtime] {episode.world_name:<9}"
                  f" t={row['time_seconds']:6.1f}s "
                  f" rope_travel={row['rope_travel_meters']:6.3f} m "
                  f" height={row['height_gained_meters']:+6.3f} m "
                  f" rope={row['rope_force_newtons']:7.1f} N "
                  f" hand_off_line={row['hand_line_error_meters']:.3f} m "
                  f" up_z={row['torso_upvector_z']:+.2f} "
                  f" fell={bool(row['fell'])} "
                  f" yaw={math.degrees(root_yaw_radians(row['root_quaternion_world_wxyz'])):6.1f}"
                  f"/{heading.desired_heading_degrees:6.1f} deg "
                  f" realtime x{realtime_factor:.2f}", flush=True)

    outcome = episode_outcome(episode, realtime_factor)
    recorder.finalize(outcome)
    guide_system.close()
    print("[runtime] SUMMARY " + "  ".join(
        f"{k}={v}" for k, v in outcome.items()), flush=True)
    return recorder.output_directory


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--live", action="store_true",
                        help="serve the browser harness instead of a timed run")
    parser.add_argument("--duration", type=float, default=None,
                        help="seconds of simulated time (timed runs)")
    parser.add_argument("--keep-going", action="store_true",
                        help="timed runs: keep simulating after a fall (dangle),"
                             " the way live mode does")
    parser.add_argument("--hold-w", action="store_true",
                        help="timed runs: hold the climb command the whole way")
    parser.add_argument("--world", default=worlds_module.DEFAULT_WORLD_NAME,
                        choices=(worlds_module.world_names()
                                 + list(worlds_module.WORLD_ALIASES)),
                        help="which world to start in (see app/harness/worlds.py)")
    parser.add_argument("--command-speed", type=float,
                        default=CLIMB_COMMAND_METERS_PER_SECOND,
                        help="lin_vel_x commanded while W is held, m/s"
                             " (their training range is [-1, 1])")
    parser.add_argument("--plain-graphics", action="store_true",
                        help="skip the alpine look (fog/sky/snow/sun). NOT"
                             " cosmetic any more: the robot's eye cameras"
                             " render through the same model, so this changes"
                             " what the stereo follower SEES. Physics is"
                             " identical either way.")
    parser.add_argument("--bms", action="store_true",
                        help="accepted and ignored: the BMS is always on now")
    parser.add_argument("--policy", default=None, help="path to a policy npz")
    parser.add_argument("--chloe-policy", default=None,
                        help="path to the ONNX rope-ascender policy the"
                             " `chloe_*` worlds run. Defaults to"
                             " rl/chloe/policies/g1_ascender_slope20_v3_*.onnx"
                             " (chloe_policy.default_policy_path). Ignored by"
                             " every other world.")
    parser.add_argument("--chloe-hold-blend", type=float, default=0.0,
                        help="seconds over which a STOPPED Chloe world eases"
                             " its held PD targets toward the reset pose."
                             " 0 (the default, measured sufficient) freezes"
                             " them exactly where the policy left them.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--randomise-reset-velocity", action="store_true",
                        help="reproduce their reset base-velocity draw U(-0.5, 0.5)")
    parser.add_argument("--no-render", action="store_true",
                        help="accepted and ignored: the harness renders no view"
                             " of the scene at all now, so there is nothing to"
                             " turn off. Kept because app/bms_ui/selftest.py"
                             " passes it.")
    parser.add_argument("--no-snow", action="store_true",
                        help="skip the procedural snow texture. Physics is"
                             " identical either way (PARITY.md has the"
                             " same-seed diff), but the eye cameras lose the"
                             " grain their block matcher matches on.")
    parser.add_argument("--hearing", action="store_true",
                        help="start with the EARS on: four virtual microphones"
                             " on the robot's head hear the human's voice, and"
                             " the robot comes when she calls and stops when"
                             " she says stop. Live mode has the same thing as"
                             " the `hearing` knob on the page. Needs the guide"
                             " on. Sensor and command only; the physics is"
                             " identical either way (test_hearing section 5).")
    parser.add_argument("--inject-voice", action="append", default=[],
                        metavar="PATH[@SECONDS]",
                        help="play a wav into the microphone stream at that"
                             " simulated time, as if someone had spoken it"
                             " (repeatable). This is how the demo is driven"
                             " with no microphone and how the screenshots are"
                             " taken; it pushes into the SAME buffer the"
                             " browser pushes into, so nothing downstream can"
                             " tell the difference. Clips come from"
                             " app.harness.hearing_corpus.")
    parser.add_argument("--guide", action="store_true",
                        help="start with the human guide ON: a guide walks the"
                             " rope route and the robot follows it by stereo"
                             " vision. Live mode has the same thing as the"
                             " `guide` knob on the page.")
    parser.add_argument("--visibility", type=float,
                        default=storm_module.CLEAR_VISIBILITY_METERS,
                        metavar="METRES",
                        help="how far the robot can see, in metres."
                             f" {storm_module.CLEAR_VISIBILITY_METERS:.0f}"
                             " (the default) is CLEAR and the eye images are"
                             " handed back untouched;"
                             f" {storm_module.MINIMUM_VISIBILITY_METERS:.0f}"
                             " is a white-out. Live mode has the same thing as"
                             " the `visibility` slider on the page. Visual and"
                             " sensor only; the physics is identical either"
                             " way. Replaces --storm (user's ruling,"
                             " 2026-08-30): visibility is its own dial and owes"
                             " the wind nothing.")
    parser.add_argument("--wind", type=float, nargs=2, default=(0.0, 0.0),
                        metavar=("EAST", "NORTH"),
                        help="headless only: the world-frame wind VELOCITY in"
                             " m/s. Live mode takes it from the page's dial.")
    parser.add_argument("--wind-natural", action="store_true",
                        help="headless only: treat --wind as a target the"
                             " gusting process wanders around, as the page's"
                             " `natural` switch does.")
    parser.add_argument("--no-guide-body", action="store_true",
                        help="skip the guide's model surgery altogether, so the"
                             " model carries no guide bodies, no walking hinges"
                             " and no eye cameras. For the physics-parity diff"
                             " only: the guide feature is unavailable with it.")
    parser.add_argument("--human", type=float, action="append", default=[],
                        help="spawn a virtual human this many metres ahead of"
                             " the spawn point (repeatable)")
    parser.add_argument("--human-range", type=float, default=2.0,
                        help="gate range: a human closer than this blocks UP")
    parser.add_argument("--human-clear-seconds", type=float, default=1.0,
                        help="hysteresis: seconds without a detection before UP re-arms")
    parser.add_argument("--output-name", default=None)
    parser.add_argument("--pose-stream", dest="pose_stream",
                        action="store_true", default=True,
                        help="broadcast per-body world poses for the WebGL page"
                             " app/web/render3d.html. ON by default, and now"
                             " the only thing the page can draw a scene from:"
                             " with it off a browser gets telemetry and an"
                             " empty stage.")
    parser.add_argument("--no-pose-stream", dest="pose_stream",
                        action="store_false")
    parser.add_argument("--port", type=int, default=8765)
    return parser


if __name__ == "__main__":
    run(build_argument_parser().parse_args())
