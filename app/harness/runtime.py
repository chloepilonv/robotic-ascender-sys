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
from app.harness import team_env  # noqa: E402
from app.harness.playground_policy import (  # noqa: E402
    GaitPhase, MelsPolicy, PlaygroundObservation, TerminationCheck,
    default_policy_path,
)
from app.harness.recorder import Recorder  # noqa: E402
from app.harness import worlds as worlds_module  # noqa: E402
from app.harness import climb_worlds as climb_worlds_module  # noqa: E402
from app.harness import graphics as graphics_module  # noqa: E402
from app.harness.natural_wind import NaturalWind  # noqa: E402

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


class Episode:
    """One spawn-to-outcome run on the team env. Owns model, data, readouts."""

    def __init__(self, model, meta, policy, wind_drag, definition,
                 world_name, seed=0, randomise_reset_velocity=False):
        self.model = model
        self.meta = meta
        self.definition = definition
        self.world_name = world_name
        # The ONE thing that separates a "free" world from a climbing one: the
        # grip equality's runtime enable. Applied to MjData every reset, never
        # to their model. See app/harness/worlds.py.
        self.rope_enabled = bool(definition["rope"])
        self.policy = policy
        self.wind_drag_coefficient = wind_drag
        self.random = np.random.default_rng(seed)
        self.randomise_reset_velocity = randomise_reset_velocity

        self.data = mujoco.MjData(model)
        self.observation_builder = PlaygroundObservation(model, meta, noise_level=0.0)
        self.termination = TerminationCheck(meta)
        self.ratchet = ratchet_module.AscenderRatchet(
            meta["slide_qpos_address"], meta["slide_dof_address"]
        )
        self.gait_phase = GaitPhase(meta["control_dt_seconds"], GAIT_FREQUENCY_HZ)
        self.substeps = meta["substeps_per_control_step"]
        self.control_hz = 1.0 / meta["control_dt_seconds"]
        self.default_pose = np.asarray(meta["default_pose_radians"])
        self.action_scale = meta["action_scale"]
        self.torso_body_id = meta["torso_body_id"]
        self.pelvis_body_id = meta["pelvis_body_id"]
        self.palm_site_id = meta["palm_site_id"]
        self.slide_qpos_address = meta["slide_qpos_address"]
        self.grip_equality_id = meta["grip_equality_id"]
        # The wind law needs the torso's world velocity; the sensor is looked
        # up by ROLE in team_env, never by a name typed here.
        self.global_linvel_torso_slice = slice(
            *meta["sensor_addresses"]["torso_global_linvel"])
        self.slope_degrees = meta["slope_degrees"]
        # THE PHYSICS-STEP SEAM (Chloe: your BMS plugs in here).
        # Each hook is `callable(model, data) -> dict | None`, called after
        # EVERY mj_step -- i.e. at model.opt.timestep, the rate a battery or
        # thermal model integrates at, not the 50 Hz control rate. The last
        # non-None dict any hook returns during a control tick becomes
        # `latest_bms`, which is broadcast as state["bms"] and recorded as one
        # hud.json entry per tick. Append to this list; nothing else to touch.
        self.physics_step_hooks = []
        self.latest_bms = None
        # Chloe's BMS, always on: one call per CONTROL tick, because her plugin
        # integrates with dt = timestep * substeps. It is NOT a physics-step
        # hook -- calling it per substep would run the battery model ten times
        # too often on a dt ten times too long.
        self.bms = make_battery_plugin(model, self.substeps)
        self.wind_velocity_world = np.zeros(2)
        self.wind_force_world_newtons = np.zeros(3)
        # Two worlds can SHARE an MjModel (climb_30/free_30, climb_0/free_0), and
        # both of these write to the model, so re-apply them for every episode
        # or the previous world's friction slider and rope visibility leak in.
        self.set_foot_friction(meta["foot_friction"])
        self._set_ascender_visible(self.rope_enabled)
        self.reset()

    def _set_ascender_visible(self, visible: bool) -> None:
        """Show/hide the carrier + visual rope. Cosmetic alpha only.

        A "free walk" world still has the carrier body and the rope cylinder in
        the model -- their `_build_model` always makes them and we do not edit
        their model's structure. Drawing a rope the robot is not attached to
        just misleads the viewer, so the apparatus geoms go transparent. Nothing
        about the physics changes: both are already contype=0/conaffinity=0
        (climb_env.py:181-182, :206-207), i.e. collision-free either way.
        """
        if not hasattr(self, "_ascender_geom_ids"):
            self._ascender_geom_ids = worlds_module.ascender_geom_ids(
                self.model, self.meta)
            self._ascender_geom_alpha = [
                float(self.model.geom_rgba[geom_id, 3])
                for geom_id in self._ascender_geom_ids
            ]
        for geom_id, alpha in zip(self._ascender_geom_ids, self._ascender_geom_alpha):
            self.model.geom_rgba[geom_id, 3] = alpha if visible else 0.0

    # ------------------------------------------------------------- state
    def reset(self) -> None:
        """Their deterministic reset -- climb_env.py:291-312.

        qpos = the `knees_bent` keyframe (palm exactly on the carrier, slide 0),
        qvel = 0. Theirs additionally draws base velocity U(-0.5, 0.5) on
        qvel[0:6]; a demo wants the same spawn every time, so that is OFF by
        default and switchable with --randomise-reset-velocity.
        """
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:] = self.meta["keyframe_qpos"]
        self.data.qvel[:] = 0.0
        if self.randomise_reset_velocity:
            self.data.qvel[0:6] = self.random.uniform(-0.5, 0.5, 6)
        self.data.ctrl[:] = self.meta["keyframe_qpos"][7:self.slide_qpos_address]
        self.data.xfrc_applied[:] = 0.0
        # THE ROPE FLAG. mj_resetData restores eq_active from model.eq_active0
        # (their default: on), so a "free" world switches it off here, per
        # MjData, leaving their model untouched. The ratchet keeps running
        # either way -- with the grip off it simply parks the unloaded carrier
        # at travel 0 instead of letting gravity drag it down the line.
        self.data.eq_active[self.grip_equality_id] = 1 if self.rope_enabled else 0
        mujoco.mj_forward(self.model, self.data)
        self.ratchet.reset(self.data)
        self.gait_phase.reset()
        self.last_action = np.zeros(self.meta["action_size"])
        self.spawn_position_world = self.data.qpos[0:3].copy()
        self.fell_at_seconds = None
        self.fall_reason = None
        self.maximum_rope_force_newtons = 0.0
        self.tick = 0
        if getattr(self, "bms", None) is not None:
            self.bms.reset()

    def set_foot_friction(self, friction: float) -> None:
        """Live friction knob. Training pins this at climb_config.foot_friction."""
        for geom_id in self.meta["foot_geom_ids"]:
            self.model.geom_friction[geom_id, 0] = float(friction)

    @property
    def pelvis_position_world(self) -> np.ndarray:
        return self.data.qpos[0:3].copy()

    @property
    def rope_travel_meters(self) -> float:
        """The ascender's own coordinate: metres up the line from the grip point."""
        return float(self.data.qpos[self.slide_qpos_address])

    @property
    def height_gained_meters(self) -> float:
        return float(self.pelvis_position_world[2] - self.spawn_position_world[2])

    @property
    def rope_force_newtons(self) -> float:
        """Magnitude of the `connect` equality's constraint force, newtons.

        The grip is a 3-row equality; MuJoCo puts its rows in efc_* tagged with
        efc_type == mjCNSTR_EQUALITY and efc_id == the equality's id.
        """
        if self.data.nefc == 0:
            return 0.0
        rows = np.where(
            (np.asarray(self.data.efc_type[:self.data.nefc])
             == int(mujoco.mjtConstraint.mjCNSTR_EQUALITY))
            & (np.asarray(self.data.efc_id[:self.data.nefc]) == self.grip_equality_id)
        )[0]
        if rows.size == 0:
            return 0.0
        return float(np.linalg.norm(np.asarray(self.data.efc_force)[rows]))

    def hand_height_on_line_meters(self) -> float:
        return ratchet_module.hand_height_on_line_meters(
            self.data, self.palm_site_id, self.meta["line_point_world"],
            self.meta["slope_axis_world"])

    def hand_line_error_meters(self) -> float:
        return ratchet_module.hand_line_error_meters(
            self.data, self.palm_site_id, self.meta["line_point_world"],
            self.meta["slope_axis_world"])

    # ------------------------------------------------------- one control tick
    def apply_wind(self, wind_velocity_world) -> None:
        """Quadratic drag on the torso -- wind_env.py:92-103, verbatim law.

        F_xy = 0.5*rho*Cd*A * |v_wind - v_torso| * (v_wind - v_torso), written
        into xfrc_applied once per control step; mj_step does not clear it, so
        it acts across all 10 substeps exactly as it does on their side.
        """
        self.wind_velocity_world[:] = wind_velocity_world
        torso_velocity = np.zeros(2)
        if self.global_linvel_torso_slice is not None:
            torso_velocity = np.asarray(
                self.data.sensordata[self.global_linvel_torso_slice][:2])
        relative = self.wind_velocity_world - torso_velocity
        force_xy = self.wind_drag_coefficient * np.linalg.norm(relative) * relative
        self.wind_force_world_newtons[:2] = force_xy
        self.wind_force_world_newtons[2] = 0.0
        self.data.xfrc_applied[self.torso_body_id, :3] = self.wind_force_world_newtons
        self.data.xfrc_applied[self.torso_body_id, 3:] = 0.0

    def step(self, command, wind_velocity_world) -> dict:
        observation = self.observation_builder.build(
            self.data, command, self.last_action, self.gait_phase)
        action = self.policy.act(observation)
        self.data.ctrl[:] = self.default_pose + self.action_scale * action
        self.apply_wind(wind_velocity_world)

        readings = ratchet_module.step_with_ratchet(
            mujoco, self.model, self.data, self.ratchet, self.substeps,
            self.physics_step_hooks)
        if readings:
            self.latest_bms = readings[-1]

        self.gait_phase.advance()
        self.last_action = action
        self.tick += 1
        time_seconds = self.tick / self.control_hz

        rope_force = self.rope_force_newtons
        self.maximum_rope_force_newtons = max(self.maximum_rope_force_newtons, rope_force)
        reasons = self.termination.reasons(self.data)
        if self.fell_at_seconds is None and (
                reasons["tipped_over"] or reasons["self_collision"]
                or reasons["not_finite"]):
            self.fell_at_seconds = time_seconds
            self.fall_reason = ("not_finite" if reasons["not_finite"]
                                else "tipped_over" if reasons["tipped_over"]
                                else "self_collision")
            print(f"[runtime] FELL at t={time_seconds:.2f}s reason={self.fall_reason}"
                  f" torso_upvector_z={reasons['torso_upvector_z']:+.3f}", flush=True)

        return {
            "time_seconds": time_seconds,
            "root_position_world": self.pelvis_position_world,
            "root_quaternion_world_wxyz": self.data.qpos[3:7].copy(),
            "root_velocity_world": self.data.qvel[0:3].copy(),
            "joint_positions_radians": self.data.qpos[7:self.slide_qpos_address].copy(),
            "joint_velocities_radians_per_second":
                self.data.qvel[6:self.meta["slide_dof_address"]].copy(),
            "action": action,
            "target_positions_radians": self.data.ctrl.copy(),
            "command": np.asarray(command, dtype=np.float64),
            "observation": observation,
            "wind_velocity_world_meters_per_second": self.wind_velocity_world.copy(),
            "wind_force_world_newtons": self.wind_force_world_newtons.copy(),
            "projected_gravity_body": observation[6:9],
            "rope_travel_meters": self.rope_travel_meters,
            # Alias kept for the HUD's existing contract: on this env the climb
            # IS the ascender's travel up the line.
            "climb_meters": self.rope_travel_meters,
            "hand_height_on_line_meters": self.hand_height_on_line_meters(),
            "hand_line_error_meters": self.hand_line_error_meters(),
            "height_gained_meters": self.height_gained_meters,
            "rope_force_newtons": rope_force,
            "torso_upvector_z": reasons["torso_upvector_z"],
            "fell": 1.0 if self.fell_at_seconds is not None else 0.0,
            **(self.bms.on_tick(self.data, time_seconds) if self.bms else {}),
        }


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
    library = worlds_module.WorldLibrary()
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
            return episode, model, meta

        model, meta, definition = library.load(
            name, on_build_start=lambda: announce_build(name))
        episode = Episode(model, meta, policy, wind_drag, definition, name,
                          seed=arguments.seed,
                          randomise_reset_velocity=arguments.randomise_reset_velocity)
        print(f"[runtime] world={name} ({definition['label']})"
              f"  slope={episode.slope_degrees} deg"
              f"  rope={'ON' if episode.rope_enabled else 'OFF'}"
              f"  control {episode.control_hz:.0f} Hz  physics"
              f" {1.0 / meta['physics_dt_seconds']:.0f} Hz"
              f"  substeps/tick={episode.substeps}", flush=True)
        print(f"[runtime] spawn pelvis {episode.spawn_position_world.round(4).tolist()}"
              f"  palm-on-line error {episode.hand_line_error_meters():.2e} m"
              f"  rope travel {episode.rope_travel_meters:.4f} m"
              f"  grip eq_active"
              f" {int(episode.data.eq_active[episode.grip_equality_id])}",
              flush=True)
        return episode, model, meta

    episode, model, meta = open_world(arguments.world)
    print(f"[runtime] observation noise OFF (training level {meta['noise_level']});"
          f" wind NOT in training; friction knob starts at"
          f" {meta['foot_friction']}", flush=True)
    if server is not None:
        server.knobs["friction"] = meta["foot_friction"]

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
        if server is not None:
            latest = server.latest_input
            keys = set(latest.get("keys", []))
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
                    episode, model, meta = open_world(requested)
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
                wall_start = time.time()
                continue
        else:
            command = np.array([
                arguments.command_speed if arguments.hold_w else 0.0, 0.0, 0.0])

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
        recorder.append(**{k: v for k, v in row.items() if k != "observation"})
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

        elapsed = max(time.time() - wall_start, 1e-9)
        realtime_factor = row["time_seconds"] / elapsed

        if server is not None:
            if jpeg is not None:
                server.broadcast(jpeg)
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
    parser.add_argument("--output-name", default=None)
    parser.add_argument("--port", type=int, default=8765)
    return parser


if __name__ == "__main__":
    run(build_argument_parser().parse_args())
