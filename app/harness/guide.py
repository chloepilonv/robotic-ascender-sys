"""The human guide, the robot's stereo eyes, and the follower that joins them.

THE FEATURE. A human guide walks ahead of the robot along the fixed line. The
robot LOOKS at it with two cameras on its head, measures how far away it is by
stereo, and decides for itself whether to keep walking or to stop and wait. No
new policy is trained: this is a supervisory layer that writes the same
three-number command (`lin_vel_x`, `lin_vel_y`, `ang_vel_yaw`) the walking
policy already takes, in place of the keyboard's.

FOUR PIECES, kept apart so each can be replaced on its own:

    attach_guide(scene)   MODEL SURGERY, once per compiled scene. Adds the
                          guide's body and the two eye cameras to HIS MjSpec and
                          recompiles, refusing if the recompile moved anything
                          structural. THIS IS THE ONE PLACE THE GUIDE'S BODY IS
                          BUILT -- Chloe's animated human mesh replaces
                          `_add_guide_body` and nothing else.
    Guide                 where the human is. A kinematic body: it walks along
                          the rope route's arc length at a fixed speed, sits on
                          the terrain surface, and never falls, slips or is
                          pushed. It is a mocap body, so it has no degrees of
                          freedom and cannot touch the physics.
    StereoEyes            what the robot sees. Renders the two head cameras,
                          runs OpenCV block matching to get a disparity map,
                          finds the guide in the left image, and turns the two
                          together into a distance and a bearing.
    GuideFollower         what the robot does about it. A three-state machine
                          with hysteresis: FOLLOW / WAIT / LOST.

WHAT IS REAL VISION AND WHAT IS A STAND-IN (the same list is in PARITY.md):

  * REAL. The two cameras are rendered images of the scene, 320x240 RGB, from
    two MuJoCo cameras 6 cm apart on the head. The DISTANCE is true passive
    stereo: OpenCV's semi-global block matcher on those two images, then
    depth = focal_pixels * baseline / disparity. Nothing about the distance
    reads the simulator's state.
  * STAND-IN. The DETECTION -- which pixels are the human -- is a colour
    threshold on the guide's deliberately distinctive orange-red, not a person
    detector. A real robot would run a detector network here; the interface
    (`detect_guide` -> a bounding box) is the seam where one drops in.
  * LABELLED CHEAT. `true_distance_meters` is read from the simulator and is
    for the HUD and for grading the stereo only. Nothing in the decision path
    touches it.

DISTANCES ARE RANGES TO THE PERSON, not depths along the optical axis, and the
truth is measured to the guide's CHEST POINT -- a point on its body axis at
`REFERENCE_HEIGHT_METERS`. MEASURED against that reference (flat_0, static
poses): -9.0% at 1 m, -0.7% at 2 m, +4.6% at 4 m, -5.3% at 8 m.

The 1 m figure is not a bug and is worth understanding, because it says what
this measurement IS. A dense matcher's median disparity over a convex body lands
somewhere between the body's near SURFACE and its AXIS: at 8 m a 0.18 m radius
is 2% of the range and the two are the same number, while at 1 m they are 18%
apart and the median sits between them, so the reading comes back short. The
follower's thresholds are set on this quantity as measured, so the behaviour is
calibrated whatever the reading is called.

Inputs  : the built `ClimbScene` (its model, spec, rope route and terrain), and
          `MjData` each tick.
Outputs : `Guide.state()` and `StereoEyes.look()` return named maps; the
          follower returns the (3,) command. See each docstring.
"""
from __future__ import annotations

import math

import numpy as np

# --------------------------------------------------------------- the body
GUIDE_BODY_NAME = "guide"
# Orange-red: chosen to be far from anything else in the scene (snow is a cool
# blue-white, the rope is dark red 0.85/0.08/0.05, the robot is grey/black), so
# the stand-in colour detector has an easy job and its failures are obviously
# its own rather than the scene's.
GUIDE_RGBA = (1.0, 0.27, 0.05, 1.0)
GUIDE_HEAD_RGBA = (0.95, 0.42, 0.12, 1.0)
# The body origin sits at hip height above the terrain; every geom below is an
# offset from the hips, so "snap to the surface" is one number. The figure is
# 1.75 m tall: legs 0.88 m, torso to the shoulders 0.55 m, head on top.
HIP_HEIGHT_METERS = 0.90
TORSO_RADIUS_METERS = 0.18
TORSO_TOP_METERS = 0.55              # shoulders, relative to the hips
LEG_RADIUS_METERS = 0.11
LEG_BOTTOM_METERS = -0.88            # feet, i.e. the terrain surface
HEAD_CENTRE_METERS = 0.72
HEAD_RADIUS_METERS = 0.13
# The point every TRUE range is measured to: chest height on the body axis,
# which is about where the visible pixels' centroid lands. The range itself is
# then this distance less TORSO_RADIUS_METERS, because a camera sees a person's
# front, not their axis.
REFERENCE_HEIGHT_METERS = 0.20

GUIDE_SPEED_METERS_PER_SECOND = 0.5
GUIDE_LATERAL_METERS = 0.6           # left of the rope, looking uphill
GUIDE_LEAD_METERS = 2.5              # where it is placed on reset

# --------------------------------------------------------------- the eyes
EYE_LEFT_CAMERA_NAME = "eye_left"
EYE_RIGHT_CAMERA_NAME = "eye_right"
SOURCE_CAMERA_NAME = "d435i"         # the RealSense site already in the MJCF
EYE_BASELINE_METERS = 0.06
EYE_WIDTH_PIXELS, EYE_HEIGHT_PIXELS = 320, 240
EYE_RENDER_EVERY_N_TICKS = 5         # 10 Hz against the 50 Hz control tick
EYE_JPEG_QUALITY = 70
EYE_MESSAGE_PREFIX = b"EYE0"         # 4 ASCII bytes, then the JPEG

# The colour detector's gate, in OpenCV HSV (hue 0-179, sat/val 0-255).
# MEASURED off a render, not guessed: the guide's lit pixels come back at hue
# 5-12 with saturation ~241 and value ~236, while the ROPE -- the only other
# strongly red thing in the scene -- sits at hue 0-1 and value ~56. A window of
# 4-16 separates them with room on both sides.
GUIDE_HUE_RANGE = (4, 16)
GUIDE_MINIMUM_SATURATION = 120
GUIDE_MINIMUM_VALUE = 80
GUIDE_MINIMUM_PIXELS = 24            # below this it is noise, not a person

# ------------------------------------------------------------ the decision
FOLLOW_RANGE_METERS = 1.3            # start following beyond this
WAIT_RANGE_METERS = 1.0              # stop within this
LOST_AFTER_SECONDS = 1.0
FOLLOW_SPEED_METERS_PER_SECOND = 0.5
BEARING_GAIN_PER_RADIAN = 2.0
MAXIMUM_YAW_RATE_RADIANS_PER_SECOND = 1.0
BEARING_DEADBAND_RADIANS = math.radians(2.0)

# Recorded as a number, because `Recorder.append` stacks float arrays.
GUIDE_MODE_CODES = {"WAIT": 0, "FOLLOW": 1, "LOST": 2}
# hud.json is read by the browser, and `JSON.parse` rejects a bare NaN, so a
# missing measurement is recorded as this rather than as NaN. The websocket
# state uses `null` for the same thing, which JSON does allow.
NO_MEASUREMENT = -1.0

# Everything a recompile of HIS spec must leave exactly where it was. nbody,
# ngeom, nmocap and ncam are DELIBERATELY absent: those are the four the surgery
# is supposed to change.
STRUCTURAL_FIELDS = ("nq", "nv", "nu", "njnt", "neq", "nsite", "nsensor", "nkey")


def _structure(model, body_limit=None) -> dict:
    """The fields a recompile must not move.

    `body_limit` truncates the per-body list, because the surgery APPENDS one
    body: comparing the full `body_mass` arrays would flag the intended change
    and refuse every time. The point of the check is that the bodies that were
    already there did not MOVE or change mass, and truncating to the old count
    asks exactly that -- appended entries are invisible to it, a reordering or a
    mass edit is not.
    """
    signature = {name: int(getattr(model, name)) for name in STRUCTURAL_FIELDS}
    signature["jnt_qposadr"] = model.jnt_qposadr.tolist()
    signature["jnt_dofadr"] = model.jnt_dofadr.tolist()
    signature["actuator_target"] = model.actuator_trnid[:, 0].tolist()
    limit = model.nbody if body_limit is None else int(body_limit)
    signature["body_mass"] = np.round(model.body_mass[:limit], 9).tolist()
    signature["body_names"] = [
        model.body(i).name for i in range(min(limit, model.nbody))]
    return signature


# ------------------------------------------------------------------ surgery
def _add_guide_body(spec) -> None:
    """Build the guide's body on the spec. THE PLACEHOLDER LIVES HERE.

    A capsule torso and a sphere head, in a colour nothing else in the scene
    wears. Both geoms are `contype=0, conaffinity=0` -- no collision channel at
    all -- and the body is a MOCAP body, so it has zero degrees of freedom and
    adds nothing to nq/nv. The robot cannot bump into it and it cannot fall
    over; its pose is written every tick from `Guide`.

    Chloe: to swap in an animated human, replace the two `add_geom` calls with a
    mesh (`spec.add_mesh(...)` for the asset, then `type=mjGEOM_MESH,
    meshname=...`). Keep the body name, keep `mocap = True`, keep the geoms
    collision-free, and keep something orange-red on the outside or the
    stand-in colour detector stops seeing it.
    """
    import mujoco

    body = spec.worldbody.add_body()
    body.name = GUIDE_BODY_NAME
    body.mocap = True
    body.pos = [0.0, 0.0, -50.0]      # parked under the world until placed

    def add(name, geom_type, rgba, **fields):
        geom = body.add_geom()
        geom.name = name
        geom.type = geom_type
        geom.rgba = list(rgba)
        geom.contype = 0              # no collision channel at all: the robot
        geom.conaffinity = 0          # cannot touch it and it cannot touch back
        geom.group = 0                # MuJoCo's default view mask shows group 0
        geom.mass = 1.0               # a mocap body's inertia is never integrated
        for field, value in fields.items():
            setattr(geom, field, value)
        return geom

    add("guide_legs", mujoco.mjtGeom.mjGEOM_CAPSULE, GUIDE_RGBA,
        fromto=[0.0, 0.0, LEG_BOTTOM_METERS, 0.0, 0.0, 0.0],
        size=[LEG_RADIUS_METERS, 0.0, 0.0])
    add("guide_torso", mujoco.mjtGeom.mjGEOM_CAPSULE, GUIDE_RGBA,
        fromto=[0.0, 0.0, 0.0, 0.0, 0.0, TORSO_TOP_METERS],
        size=[TORSO_RADIUS_METERS, 0.0, 0.0])
    add("guide_head", mujoco.mjtGeom.mjGEOM_SPHERE, GUIDE_HEAD_RGBA,
        pos=[0.0, 0.0, HEAD_CENTRE_METERS],
        size=[HEAD_RADIUS_METERS, 0.0, 0.0])


def _add_eye_cameras(spec, model, verbose=True) -> bool:
    """Two cameras either side of the existing `d435i` mount, on the same body.

    The RealSense's pose is not retyped here: the `d435i` camera already in
    `assets/robots/mujoco/g1_unitree_ascender.xml` is looked up and its position,
    orientation and field of view are copied, then the pair is displaced
    +/- baseline/2 along the CAMERA's own x axis (which is image-right). That
    makes them a rectified stereo pair by construction -- same orientation, same
    intrinsics, offset purely sideways -- which is the assumption every line of
    the disparity maths below rests on.

    THE ORIENTATION IS READ OFF THE COMPILED MODEL, NOT THE SPEC. The MJCF
    writes that camera as `xyaxes="0 -1 0  0 0 1"`, and MjSpec keeps an
    alternative orientation like that in the element's `alt` field while leaving
    `quat` at IDENTITY. Copying `source.quat` therefore silently produces two
    cameras pointing straight down -- which is exactly what the first version of
    this function did, and the symptom was a black picture and no detections at
    any range. `model.cam_quat` is the compiler's RESOLVED local orientation, so
    it is the honest source.

    Cameras are visual-only: MuJoCo integrates nothing from them.
    """
    import mujoco

    source = None
    for camera in spec.cameras:
        if camera.name == SOURCE_CAMERA_NAME:
            source = camera
            break
    source_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_CAMERA, SOURCE_CAMERA_NAME)
    if source is None or source_id < 0:
        print(f"[guide] no {SOURCE_CAMERA_NAME!r} camera in the model: the eyes"
              " cannot be mounted, the guide stays off", flush=True)
        return False

    parent = source.parent
    position = np.asarray(model.cam_pos[source_id], dtype=float)
    quaternion = np.asarray(model.cam_quat[source_id], dtype=float)
    fovy = float(model.cam_fovy[source_id])

    # The camera's x axis in the PARENT body's frame -- the first column of the
    # rotation matrix the quaternion stands for.
    rotation = np.zeros(9)
    mujoco.mju_quat2Mat(rotation, quaternion)
    camera_x_in_body = rotation.reshape(3, 3)[:, 0]

    for name, sign in ((EYE_LEFT_CAMERA_NAME, -1.0), (EYE_RIGHT_CAMERA_NAME, +1.0)):
        camera = parent.add_camera()
        camera.name = name
        camera.pos = (position + sign * 0.5 * EYE_BASELINE_METERS
                      * camera_x_in_body).tolist()
        camera.quat = quaternion.tolist()
        camera.fovy = fovy
    if verbose:
        print(f"[guide] eyes mounted on body {parent.name!r} from"
              f" {SOURCE_CAMERA_NAME!r}: baseline"
              f" {EYE_BASELINE_METERS * 100:.0f} cm along camera-x"
              f" {camera_x_in_body.round(3).tolist()}, fovy {fovy:.1f} deg,"
              f" {EYE_WIDTH_PIXELS}x{EYE_HEIGHT_PIXELS}", flush=True)
    return True


def attach_guide(scene, verbose=True) -> bool:
    """Add the guide body and the two eyes to a built `ClimbScene`, in place.

    Returns True if the scene now carries them. Idempotent: a scene is cached
    and re-opened for a second world, so a second call finds the body already
    there and does nothing.

    THE SAFETY RULE, the same one `graphics.add_skybox` uses: recompiling
    someone else's spec is only allowed if it moved nothing the rest of the
    harness addresses. Every joint address, actuator target, sensor and keyframe
    is compared before and after, and the swap is REFUSED -- leaving the scene
    exactly as it was -- if any of them moved. Body and geom counts are expected
    to change; the new ones are appended after the existing tree, so no existing
    id shifts.
    """
    import mujoco

    model = scene.model
    if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, GUIDE_BODY_NAME) >= 0:
        return True

    bodies_before = int(model.nbody)
    before = _structure(model, body_limit=bodies_before)
    if not _add_eye_cameras(scene.spec, model, verbose=verbose):
        return False
    _add_guide_body(scene.spec)

    recompiled = scene.spec.compile()
    after = _structure(recompiled, body_limit=bodies_before)
    moved = [key for key in before if before[key] != after[key]]
    if moved:
        print(f"[guide] REFUSED: recompiling moved {moved}. Keeping the"
              " original model; the guide stays off.", flush=True)
        return False

    # `vis.global_.offwidth/offheight` are set on the COMPILED model by whoever
    # makes a renderer, not in the spec, so a recompile drops them back to
    # MuJoCo's 640x480 default and the next 1920-wide render raises. Carry them.
    recompiled.vis.global_.offwidth = model.vis.global_.offwidth
    recompiled.vis.global_.offheight = model.vis.global_.offheight

    scene.model = recompiled
    scene.data = mujoco.MjData(recompiled)
    scene.ascender.bind(recompiled, mujoco)
    scene.reset()
    if verbose:
        print(f"[guide] attached: bodies {before_after(model, recompiled, 'nbody')},"
              f" geoms {before_after(model, recompiled, 'ngeom')},"
              f" cameras {before_after(model, recompiled, 'ncam')},"
              f" mocap {before_after(model, recompiled, 'nmocap')};"
              f" all {len(before)} structural fields unchanged", flush=True)
    return True


def before_after(old, new, field) -> str:
    return f"{int(getattr(old, field))}->{int(getattr(new, field))}"


# -------------------------------------------------------------- the human
class Guide:
    """Where the human is. Kinematic: an arc length along the rope route.

    The route is the same polyline the ascender rides (`RopeRoute`), so the
    guide walks the line the robot is climbing, offset `GUIDE_LATERAL_METERS`
    to its LEFT looking uphill -- close enough to lead, far enough not to be
    walked into. Height is snapped to the terrain surface every tick, so it
    cannot sink into a slope or float over a dip, and it has no velocity state
    to be disturbed: it either advances or it does not.

    Inputs  : dt seconds, and whether the human was told to walk.
    Outputs : `state()` -- progress along the route in metres, world position
              of the torso centre, and whether it has run out of rope.
    """

    def __init__(self, route, terrain, model=None):
        self.route = route
        self.terrain = terrain
        self.arclength_meters = 0.0
        self.enabled = False
        self.body_id = -1
        self.mocap_id = -1
        if model is not None:
            self.bind(model)

    def bind(self, model) -> None:
        import mujoco
        self.body_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, GUIDE_BODY_NAME)
        self.mocap_id = (int(model.body_mocapid[self.body_id])
                         if self.body_id >= 0 else -1)

    def place_ahead_of(self, position_world, lead_meters=GUIDE_LEAD_METERS) -> None:
        """Put the human `lead_meters` further up the route than a given point."""
        start, _ = self.route.project_arclen(np.asarray(position_world, dtype=float))
        self.arclength_meters = float(np.clip(
            start + lead_meters, 0.0, self.route.length))

    def advance(self, dt_seconds: float, walking: bool) -> None:
        if walking:
            self.arclength_meters = float(min(
                self.arclength_meters
                + GUIDE_SPEED_METERS_PER_SECOND * float(dt_seconds),
                self.route.length))

    @property
    def at_rope_end(self) -> bool:
        return self.arclength_meters >= self.route.length - 1e-6

    def hips_world(self) -> np.ndarray:
        """World position of the body origin: on the route, offset left, on the
        surface. This is the pose written to `mocap_pos`."""
        on_rope = self.route.point_at(self.arclength_meters)
        tangent = self.route.tangent_at(self.arclength_meters)
        # Left of the direction of travel, on the ground plane.
        left = np.array([-tangent[1], tangent[0], 0.0])
        norm = float(np.linalg.norm(left))
        left = left / norm if norm > 1e-9 else np.array([0.0, 1.0, 0.0])
        x, y = (on_rope[:2] + GUIDE_LATERAL_METERS * left[:2])
        z = float(self.terrain.surface_z(x, y)) + HIP_HEIGHT_METERS
        return np.array([float(x), float(y), z])

    def reference_point_world(self) -> np.ndarray:
        """Chest height on the body axis -- what a TRUE range is measured to."""
        hips = self.hips_world()
        hips[2] += REFERENCE_HEIGHT_METERS
        return hips

    def yaw_radians(self) -> float:
        tangent = self.route.tangent_at(self.arclength_meters)
        return math.atan2(float(tangent[1]), float(tangent[0]))

    def write(self, model, data) -> None:
        """Write the pose into `mocap_pos`/`mocap_quat`, or park it out of sight.

        A mocap body has no degrees of freedom, so this is the whole of its
        physics: MuJoCo reads these two arrays during the forward kinematics and
        nothing ever writes back to them. `mj_resetData` parks mocap bodies at
        their model pose, so this runs every tick rather than once.
        """
        if self.mocap_id < 0:
            return
        if not self.enabled:
            data.mocap_pos[self.mocap_id] = (0.0, 0.0, -50.0)
            return
        data.mocap_pos[self.mocap_id] = self.hips_world()
        half = 0.5 * self.yaw_radians()
        data.mocap_quat[self.mocap_id] = (math.cos(half), 0.0, 0.0, math.sin(half))

    def state(self) -> dict:
        return {
            "human_progress_meters": float(self.arclength_meters),
            "at_rope_end": bool(self.at_rope_end),
        }


# ---------------------------------------------------------------- the eyes
class StereoEyes:
    """Two rendered cameras -> one distance and one bearing to the guide.

    THE MEASUREMENT IS REAL STEREO. Both cameras are rendered, OpenCV's
    semi-global block matcher produces a disparity map from the pair, and depth
    comes out of the pinhole relation

        depth_metres = focal_pixels * baseline_metres / disparity_pixels

    with `focal_pixels = (height/2) / tan(fovy/2)`, the same focal length in x
    and y because MuJoCo's pixels are square. Disparity is the SGBM matcher's
    own fixed-point output (sixteenths of a pixel), so sub-pixel precision is
    real rather than rounded to whole pixels -- which matters, because at 4 m
    the whole disparity is only about 3 px.

    WHICH PIXELS ARE THE HUMAN IS A STAND-IN. `detect_guide` thresholds the
    guide's distinctive orange-red in HSV and takes the largest blob. A real
    robot runs a person detector here. The seam is the returned box: swap the
    function, keep everything below it.

    Ranges, not depths. A disparity gives `depth` -- the distance along the
    camera's optical axis. What the follower wants is the RANGE to the human, so
    the box's median depth is turned back into a 3-D point using the pixel's
    offset from the principal point and the norm is taken. On the axis the two
    agree; 20 degrees off they differ by 6%.

    Inputs  : `MjData` after a step.
    Outputs : `look()` -> a named map with `detected`, `range_meters`,
              `bearing_radians` (+ = the human is to the robot's LEFT, the same
              sign as `ang_vel_yaw`), the box, and the annotated left image.
    """

    def __init__(self, model, width=EYE_WIDTH_PIXELS, height=EYE_HEIGHT_PIXELS,
                 verbose=True):
        import cv2
        import mujoco
        from app.harness import graphics as graphics_module

        self.width, self.height = int(width), int(height)
        self.model = model
        self.renderer = None
        self.left_image = None
        self.right_image = None
        self.render_milliseconds = 0.0
        self.match_milliseconds = 0.0
        self.left_camera_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_CAMERA, EYE_LEFT_CAMERA_NAME)
        self.right_camera_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_CAMERA, EYE_RIGHT_CAMERA_NAME)
        self.available = self.left_camera_id >= 0 and self.right_camera_id >= 0
        if not self.available:
            print("[guide] the eye cameras are not in this model; vision off",
                  flush=True)
            return

        # THE OFFSCREEN BUFFER IS SIZED FROM THE MODEL, NOT FROM THE RENDERER.
        # `mujoco.Renderer` builds its `MjrContext` from `model.vis.global_`, so
        # a second small renderer on a model whose offwidth was raised to
        # 1920x1080 for the main view allocates a 1920x1080 8x-MSAA buffer and
        # then reads 320x240 out of it. MEASURED on this machine, per stereo
        # pair: 17.07 ms that way, 13.23 ms with the buffer at 320x240,
        # 11.09 ms with MSAA off as well -- 6 ms a vision tick, for free.
        # The values are restored immediately: they belong to the main
        # renderer, and this only needs them at CONTEXT CREATION.
        saved = (int(model.vis.global_.offwidth), int(model.vis.global_.offheight),
                 int(model.vis.quality.offsamples))
        model.vis.global_.offwidth = self.width
        model.vis.global_.offheight = self.height
        model.vis.quality.offsamples = 0     # a matcher does not want smoothing
        try:
            self.renderer = mujoco.Renderer(model, self.height, self.width)
        finally:
            (model.vis.global_.offwidth, model.vis.global_.offheight,
             model.vis.quality.offsamples) = saved
        # SHADOWS OFF for the eyes, deliberately. The shadow pass costs the same
        # 4096x4096 shadow map whatever the output size, so it is the single
        # most expensive thing in a 320x240 render and it buys the matcher
        # nothing. Fog, haze and the sky stay on: they are what the picture
        # would really look like.
        graphics_module.apply_render_flags(self.renderer, shadows=False)

        self.fovy_degrees = float(model.cam_fovy[self.left_camera_id])
        self.focal_pixels = (self.height / 2.0) / math.tan(
            math.radians(self.fovy_degrees) / 2.0)
        self.principal_x = (self.width - 1) / 2.0
        self.principal_y = (self.height - 1) / 2.0
        # numDisparities must be a multiple of 16. 32 px at this focal length
        # reaches from 0.42 m (disparity 31) out past 40 m; the far limit is set
        # by sub-pixel precision, not by this number.
        self.matcher = cv2.StereoSGBM_create(
            minDisparity=0, numDisparities=32, blockSize=5,
            P1=8 * 3 * 5 * 5, P2=32 * 3 * 5 * 5,
            disp12MaxDiff=1, uniquenessRatio=5,
            speckleWindowSize=64, speckleRange=2,
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)
        if verbose:
            minimum = self.focal_pixels * EYE_BASELINE_METERS / 31.0
            print(f"[guide] stereo: {self.width}x{self.height}, fovy"
                  f" {self.fovy_degrees:.1f} deg -> focal"
                  f" {self.focal_pixels:.1f} px, baseline"
                  f" {EYE_BASELINE_METERS:.3f} m, disparity 0-31 px ="
                  f" {minimum:.2f} m and out; SGBM 3-way, block 5", flush=True)

    def close(self) -> None:
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None

    def render_pair(self, data) -> tuple:
        import time
        started = time.time()
        self.renderer.update_scene(data, camera=self.left_camera_id)
        left = self.renderer.render().copy()
        self.renderer.update_scene(data, camera=self.right_camera_id)
        right = self.renderer.render().copy()
        self.render_milliseconds = (time.time() - started) * 1000.0
        self.left_image, self.right_image = left, right
        return left, right

    def disparity(self, left, right) -> np.ndarray:
        """SGBM disparity for the LEFT image, in pixels. <= 0 means no match."""
        import cv2
        import time
        started = time.time()
        left_grey = cv2.cvtColor(left, cv2.COLOR_RGB2GRAY)
        right_grey = cv2.cvtColor(right, cv2.COLOR_RGB2GRAY)
        raw = self.matcher.compute(left_grey, right_grey)
        self.match_milliseconds = (time.time() - started) * 1000.0
        return raw.astype(np.float32) / 16.0

    def look(self, data) -> dict:
        """One vision tick: render, match, detect, measure."""
        left, right = self.render_pair(data)
        disparity_pixels = self.disparity(left, right)
        box, mask = detect_guide(left)
        result = {
            "detected": box is not None,
            "range_meters": None,
            "bearing_radians": None,
            "box": box,
            "disparity_pixels": None,
            "pixels": 0 if box is None else int(mask.sum() // 255),
            "left_image": left,
        }
        if box is None:
            return result

        x0, y0, x1, y1 = box
        inside = np.zeros(disparity_pixels.shape, dtype=bool)
        inside[y0:y1 + 1, x0:x1 + 1] = True
        # Only pixels the detector called "human" AND the matcher matched. The
        # box always contains some background; a median over the box alone would
        # be pulled toward the snow behind.
        usable = inside & (mask > 0) & (disparity_pixels > 0.25)
        if usable.sum() < GUIDE_MINIMUM_PIXELS:
            result["detected"] = False
            return result

        median_disparity = float(np.median(disparity_pixels[usable]))
        depth_meters = self.focal_pixels * EYE_BASELINE_METERS / median_disparity
        # The centroid of the matched human pixels, not the box centre: a box
        # clipped by the image edge would otherwise bias the bearing.
        rows, columns = np.nonzero(usable)
        centre_x = float(columns.mean())
        centre_y = float(rows.mean())
        offset_x = (centre_x - self.principal_x) / self.focal_pixels
        offset_y = (centre_y - self.principal_y) / self.focal_pixels
        range_meters = depth_meters * math.sqrt(1.0 + offset_x ** 2 + offset_y ** 2)
        # Image +x is the camera's +x, which is the robot's RIGHT (the camera
        # looks down body +x with its own x along body -y). A human to the LEFT
        # therefore sits at a NEGATIVE column offset, and the sign flips here so
        # that a positive bearing means "turn left", matching `ang_vel_yaw`.
        bearing_radians = -math.atan2(centre_x - self.principal_x, self.focal_pixels)
        result.update({
            "range_meters": float(range_meters),
            "depth_meters": float(depth_meters),
            "bearing_radians": float(bearing_radians),
            "disparity_pixels": median_disparity,
            "centre_pixels": (centre_x, centre_y),
        })
        return result


def detect_guide(image) -> tuple:
    """STAND-IN person detector: the guide's colour, in HSV. -> (box, mask).

    `box` is (x0, y0, x1, y1) inclusive, or None. `mask` is the 0/255 pixel
    mask. This is NOT vision in any interesting sense -- it knows the answer's
    colour. It exists so the rest of the pipeline (which is real) can be built
    and measured now, and so the seam a detector network plugs into is a
    function with one job and a two-value return.
    """
    import cv2

    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(
        hsv,
        np.array([GUIDE_HUE_RANGE[0], GUIDE_MINIMUM_SATURATION,
                  GUIDE_MINIMUM_VALUE], dtype=np.uint8),
        np.array([GUIDE_HUE_RANGE[1], 255, 255], dtype=np.uint8))
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if count <= 1:
        return None, mask
    areas = stats[1:, cv2.CC_STAT_AREA]
    best = int(np.argmax(areas)) + 1
    if int(stats[best, cv2.CC_STAT_AREA]) < GUIDE_MINIMUM_PIXELS:
        return None, mask
    mask = np.where(labels == best, np.uint8(255), np.uint8(0))
    x = int(stats[best, cv2.CC_STAT_LEFT])
    y = int(stats[best, cv2.CC_STAT_TOP])
    width = int(stats[best, cv2.CC_STAT_WIDTH])
    height = int(stats[best, cv2.CC_STAT_HEIGHT])
    return (x, y, x + width - 1, y + height - 1), mask


def annotate_eye(image, box, label_text) -> bytes:
    """The left eye as a JPEG with the detection drawn on it. -> jpeg bytes.

    Sent to the page as `EYE0` + these bytes, so the browser can tell an eye
    frame from a main frame by its first four bytes (a main frame is raw JPEG
    and starts 0xFFD8).
    """
    import io
    from PIL import Image, ImageDraw, ImageFont

    picture = Image.fromarray(image)
    draw = ImageDraw.Draw(picture)
    if box is not None:
        draw.rectangle(box, outline=(60, 255, 90), width=2)
    try:
        font = ImageFont.load_default(size=13)
    except TypeError:                      # Pillow < 10.1: fixed-size default
        font = ImageFont.load_default()
    draw.text((6, 5), label_text, fill=(255, 255, 255), font=font,
              stroke_width=2, stroke_fill=(0, 0, 0))
    buffer = io.BytesIO()
    picture.save(buffer, "JPEG", quality=EYE_JPEG_QUALITY)
    return buffer.getvalue()


# ------------------------------------------------------------ the decision
class GuideFollower:
    """FOLLOW / WAIT / LOST, with hysteresis, from a range and a bearing.

    The bands overlap on purpose. A single 1.0 m threshold makes the robot
    chatter between walking and standing at exactly the distance it is trying to
    hold, because each decision changes the very number the next decision reads.
    So the switch out of WAIT is at 1.3 m and the switch into it at 1.0 m, and
    the 30 cm in between belongs to whichever state is already running:

        FOLLOW   range > 1.3 m, or > 1.0 m while already following
        WAIT     range <= 1.0 m, or <= 1.3 m while already waiting
        LOST     nothing detected for a whole second

    LOST is not a third band but a timeout, and it commands zero -- a robot that
    cannot see the person it is following has no business walking toward where
    it last saw them.

    Inputs  : `range_meters` (None if not detected), `bearing_radians`, dt.
    Outputs : `mode` (str) and `command()` -> (3,) [lin_vel_x, lin_vel_y,
              ang_vel_yaw], the same layout the walking policy takes.
    """

    def __init__(self, command_speed=FOLLOW_SPEED_METERS_PER_SECOND):
        self.command_speed = float(command_speed)
        self.mode = "LOST"
        self.seconds_since_detection = LOST_AFTER_SECONDS
        self.range_meters = None
        self.bearing_radians = None

    def reset(self) -> None:
        self.mode = "LOST"
        self.seconds_since_detection = LOST_AFTER_SECONDS
        self.range_meters = None
        self.bearing_radians = None

    def update(self, measurement, dt_seconds: float) -> None:
        """`measurement` is None on a tick with no fresh vision (the eyes run at
        10 Hz, the loop at 50): the state simply persists and the clock runs."""
        if measurement is not None:
            if measurement["detected"]:
                self.seconds_since_detection = 0.0
                self.range_meters = measurement["range_meters"]
                self.bearing_radians = measurement["bearing_radians"]
            else:
                self.seconds_since_detection += EYE_RENDER_EVERY_N_TICKS * dt_seconds
        if self.seconds_since_detection >= LOST_AFTER_SECONDS:
            self.mode = "LOST"
            self.range_meters = None
            self.bearing_radians = None
            return
        distance = self.range_meters
        if distance is None:
            self.mode = "LOST"
        elif self.mode == "FOLLOW":
            self.mode = "FOLLOW" if distance > WAIT_RANGE_METERS else "WAIT"
        else:
            self.mode = "FOLLOW" if distance > FOLLOW_RANGE_METERS else "WAIT"

    def command(self) -> np.ndarray:
        if self.mode != "FOLLOW":
            return np.zeros(3)
        bearing = self.bearing_radians or 0.0
        if abs(bearing) < BEARING_DEADBAND_RADIANS:
            yaw_rate = 0.0
        else:
            yaw_rate = float(np.clip(
                BEARING_GAIN_PER_RADIAN * bearing,
                -MAXIMUM_YAW_RATE_RADIANS_PER_SECOND,
                MAXIMUM_YAW_RATE_RADIANS_PER_SECOND))
        return np.array([self.command_speed, 0.0, yaw_rate])


# ------------------------------------- one detection, for the safety gate too
class GuideVisionDetector:
    """Chloe's `HumanDetector` interface, answered by THIS module's vision.

    WHY THIS EXISTS. `human-safety/human_gate.py` already owns the rule "the
    robot may not climb UP while a human is in front", and it is deliberately
    deterministic and auditable. Its shipped detector is a SIM ORACLE
    (`VirtualFrustumDetector` projects known human positions into the camera
    frustum). If the follower ran its own idea of "a human is there" alongside
    that oracle, the demo would have TWO detectors that could disagree -- the
    gate blocking at 2 m while the follower is happily walking at 1.5 m.

    So while the guide is on, the gate is driven from here instead, and the two
    agree by construction: `seen` is true exactly when the follower's own
    hysteresis says WAIT. The gate then blocks UP over precisely the band in
    which the follower commands zero, and Chloe's file is imported, not edited.

    Her `Detection` is the return type, unchanged: `seen`, `distance_meters`,
    `bearing_radians` (+ left of the optical axis -- the same sign convention
    this module uses), `count`.

    NOTE for the gate's own hysteresis: build the gate that uses this detector
    with `clear_after_seconds=0.0`. The hysteresis lives in `GuideFollower`
    (the 1.0/1.3 m bands and the 1 s LOST timeout); a second, slower one
    stacked on top would keep UP blocked for an extra second after the human
    walked away, and the robot would restart late every time.
    """

    def __init__(self, system: "GuideSystem"):
        self.system = system

    def detect(self, data):
        from human_gate import Detection
        follower = self.system.follower
        if not self.system.enabled or follower.range_meters is None:
            return Detection(seen=False)
        return Detection(
            seen=follower.mode == "WAIT",
            distance_meters=float(follower.range_meters),
            bearing_radians=(None if follower.bearing_radians is None
                             else float(follower.bearing_radians)),
            count=1)


# ------------------------------------------------------- the whole feature
class GuideSystem:
    """Guide + eyes + follower, wired together and driven by one call a tick.

    This is what `runtime.run` holds: it hides the 10 Hz vision rate, the
    annotated eye frame, the state message block and the recorded columns
    behind `update(...)`.
    """

    def __init__(self, scene, model, control_hz, verbose=True):
        import mujoco
        self._mujoco = mujoco
        # A legacy world has no MjSpec to operate on -- their old env hands back
        # a compiled model and nothing else -- so the guide is simply not
        # available there and every entry point below turns into a no-op.
        self.available = scene is not None and attach_guide(scene, verbose=verbose)
        self.model = scene.model if self.available else model
        self.control_hz = float(control_hz)
        self.dt_seconds = 1.0 / self.control_hz
        self.enabled = False
        self.guide = None
        self.eyes = None
        self.follower = GuideFollower()
        self.latest = None                 # the last vision measurement
        self.eye_jpeg = None
        self.true_range_meters = float("nan")
        self.vision_milliseconds = 0.0
        if not self.available:
            return
        self.guide = Guide(scene.route, scene.terrain, self.model)
        self.eyes = StereoEyes(self.model, verbose=verbose)
        self.available = self.eyes.available
        self._eye_camera_ids = (self.eyes.left_camera_id, self.eyes.right_camera_id)

    def close(self) -> None:
        if self.eyes is not None:
            self.eyes.close()

    def place(self, robot_position_world) -> None:
        """Reset the human to its lead position and forget every measurement."""
        if not self.available:
            return
        self.guide.place_ahead_of(robot_position_world)
        self.follower.reset()
        self.latest = None
        self.eye_jpeg = None

    def true_range_to_guide(self, data) -> float:
        """LABELLED CHEAT, HUD and grading only: the true 3-D eye-to-human range.

        Straight-line distance from the midpoint of the two eyes to the guide's
        chest point. Read out of the simulator; nothing in the decision path
        ever sees it.
        """
        if not self.available:
            return float("nan")
        eye = 0.5 * (data.cam_xpos[self._eye_camera_ids[0]]
                     + data.cam_xpos[self._eye_camera_ids[1]])
        return float(np.linalg.norm(self.guide.reference_point_world() - eye))

    def update(self, data, tick: int, enabled: bool, walking: bool) -> np.ndarray | None:
        """One control tick. -> the command to fly, or None if the guide is off.

        Order matters: the human moves, its body is written, and only THEN are
        the cameras rendered, so the robot sees where the human is now rather
        than where it was a tick ago.
        """
        if not self.available:
            return None
        was_enabled, self.enabled = self.enabled, bool(enabled)
        self.guide.enabled = self.enabled
        if not self.enabled:
            self.guide.write(self.model, data)
            self.follower.reset()
            self.latest = None
            return None
        if not was_enabled:
            self.place(np.asarray(data.qpos[0:3]))
            self.guide.enabled = True

        self.guide.advance(self.dt_seconds, walking)
        self.guide.write(self.model, data)
        # `mocap_pos` is an INPUT to the forward kinematics, not a pose: nothing
        # in `data` moves until something recomputes frames from it, and the
        # renderer reads `geom_xpos`. Without these two the eyes would see the
        # human a tick behind where it is. Both are pure functions of qpos and
        # mocap -- they integrate nothing, touch no sensor and no constraint --
        # and `mj_step` re-derives all of it from scratch on the very next line
        # of the control loop, so the physics is untouched. (`mj_kinematics`
        # does bodies, geoms and sites; cameras and lights are `mj_camlight`.)
        self._mujoco.mj_kinematics(self.model, data)
        self._mujoco.mj_camlight(self.model, data)
        self.true_range_meters = self.true_range_to_guide(data)

        measurement = None
        if tick % EYE_RENDER_EVERY_N_TICKS == 0:
            import time
            started = time.time()
            measurement = self.eyes.look(data)
            self.latest = measurement
            self.vision_milliseconds = (time.time() - started) * 1000.0
        self.follower.update(measurement, self.dt_seconds)
        if measurement is not None:
            self.eye_jpeg = EYE_MESSAGE_PREFIX + annotate_eye(
                measurement["left_image"], measurement["box"], self.label_text())
        return self.follower.command()

    def take_eye_jpeg(self):
        """The newest eye frame, ONCE. -> bytes or None.

        The eyes run at 10 Hz and the loop at 50, so four ticks in five have no
        new picture. Returning it once and clearing is what keeps the websocket
        from re-sending the same frame five times.
        """
        jpeg, self.eye_jpeg = self.eye_jpeg, None
        return jpeg

    def label_text(self) -> str:
        if self.follower.range_meters is None:
            return f"-- m · {self.follower.mode}"
        return f"{self.follower.range_meters:.1f} m · {self.follower.mode}"

    def state(self) -> dict:
        """The `guide` block of the websocket state message."""
        if not self.available:
            return {"enabled": False, "mode": "LOST", "distance_meters": None,
                    "bearing_degrees": None, "true_distance_meters": None,
                    "human_progress_meters": 0.0}
        bearing = self.follower.bearing_radians
        return {
            "enabled": bool(self.enabled),
            "mode": self.follower.mode,
            "distance_meters": (None if self.follower.range_meters is None
                                else round(float(self.follower.range_meters), 3)),
            "bearing_degrees": (None if bearing is None
                                else round(math.degrees(bearing), 2)),
            # LABELLED CHEAT: read from the simulator, HUD only.
            "true_distance_meters": (None if not np.isfinite(self.true_range_meters)
                                     else round(float(self.true_range_meters), 3)),
            "human_progress_meters": round(
                float(self.guide.arclength_meters), 3),
        }

    def recorded(self) -> dict:
        """The four columns `Recorder` stacks into frames.npz / hud.json."""
        distance = self.follower.range_meters
        true_range = self.true_range_meters
        return {
            "guide_mode": float(GUIDE_MODE_CODES[self.follower.mode]),
            "guide_distance_meters": (NO_MEASUREMENT if distance is None
                                      else float(distance)),
            "guide_true_distance_meters": (
                NO_MEASUREMENT if not (self.enabled and np.isfinite(true_range))
                else float(true_range)),
            "guide_human_progress_meters": (
                float(self.guide.arclength_meters) if self.available else 0.0),
        }
