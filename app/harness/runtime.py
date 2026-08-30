"""The interactive walker harness, running the TEAM's climbing environment.

  ../.venv_everest/bin/python -m app.harness.runtime --live
  ../.venv_everest/bin/python -m app.harness.runtime --duration 15 --hold-w --no-render

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
    reset / pause / recorder / renderer / websocket.

Rates. Their model compiles at a 2 ms timestep, so physics is 500 Hz and one
control tick is 10 substeps at 50 Hz -- the ctrl_dt 0.02 / sim_dt 0.002 pair
their config declares. Live mode paces itself against the wall clock and prints
a realtime factor; a `--duration` run goes as fast as it can.
"""
import argparse
import io
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

import mujoco  # noqa: E402
from PIL import Image  # noqa: E402

from app.harness import ratchet as ratchet_module  # noqa: E402
from app.harness.playground_policy import (  # noqa: E402
    GaitPhase, MelsPolicy, PlaygroundObservation, TerminationCheck,
    default_policy_path,
)
from app.harness.recorder import Recorder  # noqa: E402
from app.harness import worlds as worlds_module  # noqa: E402
from app.harness import climb_worlds as climb_worlds_module  # noqa: E402
from app.harness import graphics as graphics_module  # noqa: E402
from app.harness import guide as guide_module  # noqa: E402
from app.harness import snow as snow_module  # noqa: E402
from app.harness.natural_wind import NaturalWind  # noqa: E402
sys.path.insert(0, os.path.join(  # human-safety/ is a program, not a package
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "human-safety"))
from human_gate import (  # noqa: E402
    HumanGate, HumanWorld, VirtualFrustumDetector)

RENDER_WIDTH, RENDER_HEIGHT = 960, 540    # 16:9 -- the page fills the viewport with it
RENDER_MAXIMUM_WIDTH, RENDER_MAXIMUM_HEIGHT = 1920, 1080   # native-resolution cap (F = fullscreen in the page)


def clamp_render_size(viewport) -> tuple:
    """Browser-reported viewport (css px * devicePixelRatio) -> (width, height) we will render.
    Even numbers (JPEG/mp4 friendly), capped so a 5K display cannot drag the tick below realtime."""
    try:
        width, height = int(viewport["width"]), int(viewport["height"])
    except (TypeError, KeyError, ValueError):
        return RENDER_WIDTH, RENDER_HEIGHT
    if width < 320 or height < 180:
        return RENDER_WIDTH, RENDER_HEIGHT
    scale = min(1.0, RENDER_MAXIMUM_WIDTH / width, RENDER_MAXIMUM_HEIGHT / height)
    return (int(width * scale) // 2) * 2, (int(height * scale) // 2) * 2


def make_renderer(model, width: int, height: int, alpine=True, shadows=None):
    """The offscreen framebuffer is sized from model.vis.global_ at context creation;
    raise it to the cap once so any requested size up to 1920x1080 fits.

    Shadows measured at 1920x1080 on this machine: 14.9 ms/frame with, 9.2 ms
    without -- 5.7 ms, against a 20 ms control tick. They stay ON at every size
    we render, and `--no-shadows` is the escape hatch if a slower machine needs
    it. (`graphics.shadows_affordable` keeps the width rule for that case.)"""
    model.vis.global_.offwidth = max(int(model.vis.global_.offwidth), RENDER_MAXIMUM_WIDTH)
    model.vis.global_.offheight = max(int(model.vis.global_.offheight), RENDER_MAXIMUM_HEIGHT)
    renderer = mujoco.Renderer(model, height, width)
    if alpine:
        flags = graphics_module.apply_render_flags(
            renderer, shadows=True if shadows is None else shadows)
        print(f"[graphics] render flags {flags} at {width}x{height}", flush=True)
    return renderer
JPEG_QUALITY = 80

# Third-person orbit, matching the browser's defaults so live and recorded
# views agree.
CAMERA_DISTANCE_METERS = 3.0
CAMERA_AZIMUTH_DEGREES = 180.0
CAMERA_ELEVATION_DEGREES = -15.0
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
# A live session has no end, and the recorder holds every JPEG in RAM to mux at
# the end: at 50 Hz / ~40 kB a frame that is ~2 MB per second, i.e. 7 GB an
# hour. Cap the VIDEO (the per-tick numeric rows are ~1 kB/s and stay
# uncapped), and say so on stdout when the cap bites.
LIVE_MAXIMUM_RECORDED_FRAMES = 6000   # 2 minutes of episode.mp4 at 50 Hz
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


class ChaseCamera:
    """Third-person orbit around the pelvis. Azimuth/elevation from the browser."""

    def __init__(self):
        self.camera = mujoco.MjvCamera()
        self.camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.camera.distance = CAMERA_DISTANCE_METERS
        self.camera.azimuth = CAMERA_AZIMUTH_DEGREES + BROWSER_AZIMUTH_OFFSET_DEGREES
        self.camera.elevation = CAMERA_ELEVATION_DEGREES

    def aim(self, lookat_world, azimuth_degrees=None, elevation_degrees=None):
        """azimuth/elevation are the BROWSER's numbers, or None to hold."""
        self.camera.lookat[:] = lookat_world
        if azimuth_degrees is not None:
            self.camera.azimuth = float(azimuth_degrees) + BROWSER_AZIMUTH_OFFSET_DEGREES
        if elevation_degrees is not None:
            self.camera.elevation = float(elevation_degrees)
        return self.camera


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


def encode_jpeg(pixels: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    Image.fromarray(pixels).save(buffer, "JPEG", quality=JPEG_QUALITY)
    return buffer.getvalue()


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
        "policy": "mels_g1_joystick.npz",
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


def episode_outcome(episode, realtime_factor: float, frames_rendered: int) -> dict:
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
        "frames_rendered": frames_rendered,
    }


# ---------------------------------------------------------------- the run
def run(arguments) -> str:
    policy = MelsPolicy(arguments.policy or default_policy_path(_REPOSITORY_ROOT))
    print(policy.describe(), flush=True)
    wind_drag = wind_drag_coefficient()
    climb_library = climb_worlds_module.ClimbSceneLibrary()

    server = None
    if arguments.live:
        from app.harness.server import Server
        server = Server(arguments.port, worlds=worlds_module.describe_worlds())

    def announce_build(name):
        """Tell the page why the picture is about to freeze.

        A world's first selection costs a full G1ClimbAscender.__init__, and the
        sim loop is what does it, so no frames go out while it runs. Measured
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
        if latest_jpeg[0] is not None:
            server.broadcast(latest_jpeg[0])

    latest_jpeg = [None]
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
        if kind == "climb_scene":
            scene, meta, definition = climb_library.load(
                name, on_build_start=lambda: announce_build(name))
            # THE GUIDE'S SURGERY GOES FIRST, before anything dresses the model.
            # It recompiles the spec, and `apply_alpine_look` writes to the
            # COMPILED model (lights, fog, the snow colour) -- so doing it the
            # other way round throws the alpine look away and the picture comes
            # back dark. `add_skybox` survives either order because a texture
            # lives in the spec, but the ordering rule is the same for both.
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
    # Costs tens of microseconds of a 20 ms tick, prints that measurement
    # itself, and is invisible to app/web/index.html, which reads only JPEGs.
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
    def make_guide(current_scene, current_model, current_episode):
        system = guide_module.GuideSystem(
            current_scene, current_model, current_episode.control_hz)
        gate = HumanGate(guide_module.GuideVisionDetector(system),
                         clear_after_seconds=0.0)
        if system.available:
            system.place(current_episode.spawn_position_world)
        return system, gate

    guide_system, guide_gate = make_guide(scene, model, episode)

    # SNOW AND FOOTPRINTS (app/harness/snow.py). Visual only: the texture is
    # painted, the heightfield never is. The touchdown detector that stamps the
    # prints is the SAME one that counts steps and feeds the page's `foot_steps`
    # events, so a sound, a print and the counter can never disagree about what
    # a step is -- and it runs even when the snow texture is off, because the
    # audio does not depend on the picture.
    def make_snow(current_scene, current_model, current_meta):
        ground = (snow_module.SnowGround(current_scene, seed=arguments.seed,
                                         verbose=False)
                  if current_scene is not None else None)
        if ground is not None and not ground.available:
            ground = None
        detector = None
        if current_meta.get("foot_geom_ids") is not None and \
                "terrain_geom_id" in current_meta:
            detector = snow_module.TouchdownDetector(
                current_model, current_meta["foot_geom_ids"],
                current_meta["terrain_geom_id"])
            print(f"[snow] touchdowns from {detector.describe()};"
                  f" footprints {'ON' if ground is not None else 'OFF'}",
                  flush=True)
        return ground, detector

    snow_ground, touchdowns = make_snow(scene, model, meta)
    if server is not None:
        server.knobs["guide"] = 1.0 if arguments.guide else 0.0
    print(f"[guide] {'available' if guide_system.available else 'NOT available'}"
          f" in world {episode.world_name};"
          f" starts {'ON' if arguments.guide else 'OFF'}"
          f" (knob `guide`, W drives the human)", flush=True)

    renderer = None
    render_size = (RENDER_WIDTH, RENDER_HEIGHT)   # follows the browser viewport in live mode
    camera = ChaseCamera()
    heading = HeadingController(arguments.command_speed)
    rendered_model = None
    if not arguments.no_render:
        renderer = make_renderer(model, *render_size,
                                 alpine=not arguments.plain_graphics,
                                 shadows=not arguments.no_shadows)
        rendered_model = model

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
    frames_rendered = 0
    applied_friction = meta["foot_friction"]

    while True:
        time_seconds = episode.tick / episode.control_hz
        if time_seconds >= duration_seconds:
            break
        if (not arguments.live and not arguments.keep_going
                and episode.fell_at_seconds is not None
                and time_seconds >= episode.fell_at_seconds + FALL_LINGER_SECONDS):
            break

        azimuth_degrees = elevation_degrees = None
        guide_enabled = bool(arguments.guide)
        walking = bool(arguments.hold_w)
        if server is not None:
            latest = server.latest_input
            keys = set(latest.get("keys", []))
            walking = "w" in keys
            guide_enabled = bool(server.knobs.get("guide", 0.0))
            browser_camera = latest.get("camera") or {}
            azimuth_degrees = browser_camera.get("azimuth_degrees")
            elevation_degrees = browser_camera.get("elevation_degrees")
            heading.set_browser_azimuth(azimuth_degrees)
            # A = turn left (positive yaw), D = turn right, both down cancel.
            turn = ((MANUAL_YAW_RATE_RADIANS_PER_SECOND if "a" in keys else 0.0)
                    - (MANUAL_YAW_RATE_RADIANS_PER_SECOND if "d" in keys else 0.0))
            steering = ("a" in keys) != ("d" in keys)
            command = heading.command(
                episode.data.qpos[3:7], walking="w" in keys,
                manual_yaw_rate=turn if steering else None)
            if server.paused:
                # Freeze: no physics, no policy, no recorded tick. Keep the last
                # picture and a heartbeat flowing so the page stays live and
                # knows why nothing moves. The pacing clock is rebased every
                # frame, so unpausing resumes at realtime instead of firing a
                # catch-up burst of ticks for the paused wall seconds.
                if latest_jpeg[0] is not None:
                    server.broadcast(latest_jpeg[0])
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
                    recorder.finalize(episode_outcome(
                        episode, realtime_factor, frames_rendered))
                    episode, model, meta, scene = open_world(requested)
                    # A new world is a new compiled model, so the eyes' renderer
                    # and every id the guide cached belong to a model that is
                    # gone. Build both again, and re-place the human 2.5 m ahead
                    # of the new spawn.
                    guide_system.close()
                    guide_system, guide_gate = make_guide(scene, model, episode)
                    snow_ground, touchdowns = make_snow(scene, model, meta)
                    if not arguments.no_render and model is not rendered_model:
                        # The GL context is NOT garbage collected, and it is
                        # bound to the model it was made for. Two worlds that
                        # share a model share the renderer; a different model
                        # needs a new one.
                        renderer.close()
                        renderer = make_renderer(model, *render_size,
                                 alpine=not arguments.plain_graphics,
                                 shadows=not arguments.no_shadows)
                        rendered_model = model
                    # The friction knob still reads the OLD world; re-sync it or
                    # the next tick paints the previous mu over the new map.
                    applied_friction = meta["foot_friction"]
                    server.knobs["friction"] = applied_friction
                    header = make_header(episode, meta, arguments)
                    recorder = Recorder(new_episode_directory(), header,
                                        control_hz=episode.control_hz)
                    last_logged_command = last_logged_wind = None
                    frames_rendered = 0
                    wall_start = time.time()
                    continue
            if server.reset_requested:
                server.reset_requested = False
                episode.reset()
                guide_system.place(episode.spawn_position_world)
                if snow_ground is not None:
                    snow_ground.reset()
                wall_start = time.time()
                continue
        else:
            command = np.array([
                arguments.command_speed if arguments.hold_w else 0.0, 0.0, 0.0])

        # THE GUIDE OWNS THE COMMAND WHILE IT IS ON. W/A/D stop steering the
        # robot: W tells the HUMAN to walk, and what the robot does about that
        # is the follower's business -- which is the whole point of the feature.
        # The camera-follow controller is stood down too (its target is re-seated
        # to the robot's actual yaw every tick, exactly as it is while A or D is
        # held), so switching the guide off does not snap the robot back to a
        # heading it drifted away from ten seconds ago.
        guide_command = guide_system.update(
            episode.data, episode.tick, guide_enabled, walking)
        if guide_command is not None:
            command = guide_command
            heading.desired_heading_radians = root_yaw_radians(
                episode.data.qpos[3:7])
            heading.yaw_error_radians = 0.0

        gate = guide_gate if guide_command is not None else human_gate
        gate.update(episode.data, episode.tick / episode.control_hz)
        command = gate.mask(command)
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
        # TOUCHDOWNS. Read after the step, from the solver's own contacts:
        # each landing stamps a print, counts a step, and becomes one
        # `foot_steps` event on the wire. Painting is ~0.06 ms and only happens
        # on a landing; the GPU upload is throttled and lives below, next to
        # the renderers it has to push to.
        foot_steps = []
        if touchdowns is not None:
            for landing in touchdowns.update(episode.data, 1.0 / episode.control_hz):
                foot_steps.append({"foot": landing["foot"],
                                   "impact_speed_mps": landing["impact_speed_mps"]})
                if snow_ground is not None:
                    snow_ground.paint_footprint(
                        float(landing["position_world"][0]),
                        float(landing["position_world"][1]),
                        landing["yaw_radians"])
        if snow_ground is not None:
            snow_ground.decay(row["time_seconds"])

        recorder.append(**{k: v for k, v in row.items() if k != "observation"})
        recorder.append(**guide_system.recorded())
        recorder.append(step_count=float(
            touchdowns.step_count if touchdowns is not None else 0))
        recorder.append_bms(episode.latest_bms)

        jpeg = None
        if renderer is not None and server is not None:
            wanted = clamp_render_size(server.latest_input.get("viewport"))
            if wanted != render_size:
                renderer.close()
                renderer = make_renderer(episode.model, *wanted,
                                         alpine=not arguments.plain_graphics,
                                         shadows=not arguments.no_shadows)
                rendered_model = episode.model
                render_size = wanted
                print(f"[runtime] render size -> {wanted[0]}x{wanted[1]}", flush=True)
        if renderer is not None:
            renderer.update_scene(episode.data, camera.aim(
                row["root_position_world"], azimuth_degrees, elevation_degrees))
            human_world.draw(renderer.scene)
            jpeg = encode_jpeg(renderer.render())
            latest_jpeg[0] = jpeg
            if not arguments.live or frames_rendered < LIVE_MAXIMUM_RECORDED_FRAMES:
                recorder.append_frame(jpeg)
                frames_rendered += 1
                if (arguments.live
                        and frames_rendered == LIVE_MAXIMUM_RECORDED_FRAMES):
                    print(f"[recorder] video cap reached"
                          f" ({LIVE_MAXIMUM_RECORDED_FRAMES} frames ="
                          f" {LIVE_MAXIMUM_RECORDED_FRAMES / episode.control_hz:.0f} s);"
                          " numeric rows keep recording, episode.mp4 stops here",
                          flush=True)

        # One upload per changed texture per context, at most UPLOAD_HZ. Each
        # renderer holds its own GPU copy, so the eye cameras need it too or the
        # robot would look at snow with no prints in it.
        if snow_ground is not None:
            contexts = [renderer] if renderer is not None else []
            if guide_system.available and guide_system.eyes.renderer is not None:
                contexts.append(guide_system.eyes.renderer)
            snow_ground.upload(contexts, row["time_seconds"])

        elapsed = max(time.time() - wall_start, 1e-9)
        realtime_factor = row["time_seconds"] / elapsed

        if server is not None:
            if jpeg is not None:
                server.broadcast(jpeg)
            # The left eye, at the vision rate: `EYE0` then a JPEG. The page
            # tells the two streams apart by the first four bytes -- a main
            # frame is a raw JPEG and starts 0xFFD8.
            eye_jpeg = guide_system.take_eye_jpeg()
            if eye_jpeg is not None:
                server.broadcast(eye_jpeg)
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
                # Whichever gate is live -- the guide's vision while the guide
                # is on, the sim oracle otherwise. One set of `human_*` fields
                # either way, so the page needs no new case.
                **gate.state(),
                "guide": guide_system.state(),
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
                "paused": False,
                "loading": False,
                # bms + actuator_names + r_int_curve, straight from her plugin.
                **(episode.bms.state() if episode.bms else {}),
            }
            server.broadcast(latest_state[0])
            sleep_for = wall_start + episode.tick / episode.control_hz - time.time()
            if sleep_for > 0:
                time.sleep(sleep_for)

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

    outcome = episode_outcome(episode, realtime_factor, frames_rendered)
    recorder.finalize(outcome)
    guide_system.close()
    if renderer is not None:
        renderer.close()
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
                        help="skip the alpine look (fog/sky/snow/sun). Visual"
                             " only either way; physics is identical.")
    parser.add_argument("--no-shadows", action="store_true",
                        help="render without shadows (saves ~5.7 ms/frame at"
                             " 1920x1080)")
    parser.add_argument("--bms", action="store_true",
                        help="accepted and ignored: the BMS is always on now")
    parser.add_argument("--policy", default=None, help="path to a policy npz")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--randomise-reset-velocity", action="store_true",
                        help="reproduce their reset base-velocity draw U(-0.5, 0.5)")
    parser.add_argument("--no-render", action="store_true")
    parser.add_argument("--no-snow", action="store_true",
                        help="skip the procedural snow texture and its"
                             " footprints. Visual only either way; physics is"
                             " identical, and PARITY.md has the same-seed diff"
                             " that says so.")
    parser.add_argument("--guide", action="store_true",
                        help="start with the human guide ON: a guide walks the"
                             " rope route and the robot follows it by stereo"
                             " vision. Live mode has the same thing as the"
                             " `guide` knob on the page.")
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
                             " app/web/render3d.html. ON by default; the JPEG"
                             " stream is unaffected either way.")
    parser.add_argument("--no-pose-stream", dest="pose_stream",
                        action="store_false")
    parser.add_argument("--port", type=int, default=8765)
    return parser


if __name__ == "__main__":
    run(build_argument_parser().parse_args())
