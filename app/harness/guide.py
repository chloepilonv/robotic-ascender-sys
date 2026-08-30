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
    threshold on the guide's deliberately distinctive BRIGHT ORANGE BACKPACK,
    not a person detector. A real robot would run a detector network here; the
    interface (`detect_guide` -> a bounding box) is the seam where one drops
    in. The pack is a much smaller target than the whole jacket used to be, so
    the range at which the threshold still fires is shorter, and the pack is
    HIDDEN when she turns to face the robot. Both limits are measured in
    test_guide (A1, A2) rather than left to be discovered in a demo.
  * LABELLED CHEAT. `true_distance_meters` is read from the simulator and is
    for the HUD and for grading the stereo only. Nothing in the decision path
    touches it.

DISTANCES ARE RANGES TO THE PERSON, not depths along the optical axis, and the
truth is measured to the guide's CHEST POINT -- a point on its body axis at
`REFERENCE_HEIGHT_METERS`. RE-MEASURED against that reference with the pack as
the target (flat_0, static poses): -14.8% at 1 m, -7.0% at 2 m, +3.2% at 4 m,
+9.1% at 8 m.

The 1 m figure is not a bug and is worth understanding, because it says what
this measurement IS. A dense matcher's median disparity over the pixels the
detector kept lands somewhere between that surface and the body AXIS the truth
is measured to, and the pack's rear face is 0.30 m out: at 8 m that is 4% of the
range and the two are the same number, while at 1 m they are 30% apart and the
median sits between them, so the reading comes back short. The 8 m row swings
the other way for a different reason -- the pack is 135 pixels there and the
disparity is 1.5 px, so a quarter-pixel of quantisation is metres. The
follower's thresholds are set on this quantity as measured, so the behaviour is
calibrated whatever the reading is called.

TWO LIMITS THE BACKPACK MARKER HAS AND THE JACKET DID NOT, both measured in
test_guide rather than discovered in a demo. She is INVISIBLE FACING THE ROBOT:
0 mask pixels at 2 m and at 5 m, and the follower goes LOST rather than
inventing a range (A2) -- which is exactly the case the EARS exist for
(`hearing.py`). And there is a CLOSE-RANGE HOLE on the approach: she walks
0.6 m to the left of the rope, so as the robot closes inside about 1.5 m the
bearing to her passes the +/-29 deg frame edge and a narrow marker on her back
goes with it, where a whole jacket still filled the picture.

Inputs  : the built `ClimbScene` (its model, spec, rope route and terrain), and
          `MjData` each tick.
Outputs : `Guide.state()` and `StereoEyes.look()` return named maps; the
          follower returns the (3,) command. See each docstring.
"""
from __future__ import annotations

import math
import os

import numpy as np

# --------------------------------------------------------------- the body
GUIDE_BODY_NAME = "guide"
# CHLOE'S HIKER IS THE GUIDE. `assets/humans/human.xml` is loaded, re-parented
# into jointed limbs and baked to a neutral standing pose by `guide_skeleton()`
# below; nothing about the figure is retyped here. Her geom names (`human_*`),
# her group (2) and her collision-free flags are carried through unchanged, so
# `human-safety/human_gate.py`'s segmentation gate still recognises the pixels.
HUMAN_XML_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "assets", "humans", "human.xml")
# Her root frame has z = 0 AT THE GROUND CONTACT (feet), not at the hips, so the
# body origin is snapped straight onto the terrain surface -- less whatever the
# current stride puts below zero (`Guide.root_world`).
# The nearest DETECTED surface to a camera behind her. It used to be the jacket
# capsules (radius 0.17-0.19 about the body axis); with the orange moved to the
# BACKPACK it is the pack's rear face, and the pack sticks out further: its box
# sits at x = -0.20 with a 0.10 half-extent, so that face is 0.30 m behind the
# body axis. The matcher medians over exactly the pixels the detector kept, so
# this is the surface it is looking at. Used only to print the like-for-like
# column in test_guide.
DETECTED_SURFACE_RADIUS_METERS = 0.30
# The point every TRUE range is measured to, on the body axis: where the visible
# ORANGE pixels' centroid lands seen from behind. That is now the pack, whose
# box spans z 0.96-1.44 with its lid to 1.49, so the centroid sits at its middle
# rather than at the old jacket-and-hips average of 1.15.
REFERENCE_HEIGHT_METERS = 1.20

# ---------------------------------------------------------------- her clothes
# THE OUTFIT, APPLIED AT ATTACH TIME. `assets/humans/human.xml` is shared with
# `human-safety/` and is not edited for a wardrobe choice, so the palette is
# overridden where her materials are copied onto the scene spec
# (`_add_guide_body`). That is the compiled model, so the SAME clothes reach the
# GLB export the browser draws and the eye cameras the robot detects with --
# there is no second place a colour could disagree.
#
# ONLY `rgba` IS TOUCHED. Not a geom, not a size, not a mass, not a contact
# flag: a material colour is a rendering property and MuJoCo integrates none of
# it, which is why test D still comes back bit-identical.
#
# WHY THESE COLOURS, AND WHY THEY ARE ONE DESIGN WITH THE DETECTOR. The detector
# is a hue window, so choosing the clothes and choosing the window is a single
# decision: the PACK is the only saturated orange on the person, and every other
# material is pushed away from it in hue, in saturation, or in value -- with
# margin, not by one unit. Their OpenCV hues (0-179) AS RENDERED, read off
# test_guide's A0 table rather than computed from the rgba and hoped for:
#
#   pack     11-12   saturation 246-250      <- THE TARGET
#   jacket  111-113  pants 114-115  boots 113  -- the blues, a third of the
#                    wheel away from the pack and from nothing else in the scene
#   beanie   86-91   (her rolled sleeping mat wears this material too)
#   skin      wraps through 0, BUT saturation 39-74 -- a hundred short of the
#                    window's floor, which is the barrier that matters, since
#                    skin hue is always going to sit near an orange one
#   glove   110-116  value 11-31: black defeats the value floor as well
#
# The boots moved off brown for exactly this reason. Brown renders at hue 12-14,
# which is the pack's own window, and the old jacket window only kept them out
# on the value floor -- a one-barrier margin that a brighter light would break.
GUIDE_OUTFIT_RGBA = {
    "jacket": (0.13, 0.30, 0.72, 1.0),   # cobalt
    "pants": (0.09, 0.14, 0.34, 1.0),    # navy, a second and darker blue
    "pack": (1.00, 0.38, 0.02, 1.0),     # safety orange -- what the robot sees
    "beanie": (0.05, 0.62, 0.55, 1.0),   # teal
    "boots": (0.16, 0.19, 0.28, 1.0),    # slate, off brown on purpose
    "skin": (0.82, 0.63, 0.58, 1.0),
    "glove": (0.10, 0.11, 0.14, 1.0),
}

GUIDE_SPEED_METERS_PER_SECOND = 1.0  # W walks her forward, S walks her back
# HOW FAST A AND D TURN HER, on a world with no rope. A person changing
# direction while walking briskly turns at roughly 60-90 deg/s; 70 is in the
# middle and it is slow enough that the figure does not pirouette.
GUIDE_TURN_RATE_RADIANS_PER_SECOND = math.radians(70.0)
GUIDE_LATERAL_METERS = 0.6           # left of the rope, looking uphill
GUIDE_LEAD_METERS = 2.5              # where it is placed on reset

# ------------------------------------------------------------------ the walk
# SIX HINGES, all about the body's +y axis (her left), on child bodies of the
# mocap root -- and NOT ONE OF THEM IS A JOINT. Each limb is a body with no
# degrees of freedom, welded to its parent, and the "hinge angle" is written
# into `model.body_quat` every control tick. `mj_kinematics` reads that field
# for a welded body, so the figure poses exactly as if it had joints.
#
# WHY, AND IT IS THE WHOLE REASON THIS FEATURE IS SAFE. The first version used
# real hinge joints. They work and they animate, and they also grow `nq` 39->45
# and `nv` 38->44 -- six degrees of freedom that MuJoCo's solver then carries
# through every step. The guide's limbs cannot touch the robot (no contacts, no
# constraints, a mocap root welded to the world), so they exert no force on it,
# but the solver's global convergence test is not per-body, and the walking
# robot is chaotic: MEASURED on flat_0, two 6 s same-seed runs with and without
# the jointed guide came back 1.447 m and 1.675 m of rope travel. That is not a
# force, it is floating-point divergence -- and it is still a run that does not
# reproduce, which is exactly what PARITY.md exists to forbid.
#
# Welded bodies posed through `body_quat` leave `nq`, `nv` and `njnt` untouched,
# so the state vector the solver integrates is byte-for-byte the one it
# integrated before the guide existed, and the same-seed diff is 0.000e+00.
#
#   positive hip      = that leg swings BACKWARD   (rotation about +y takes a
#                       downward segment toward -x, and -x is behind her)
#   positive knee     = flexion, the heel comes up behind
#   positive shoulder = that arm swings BACKWARD
#   negative elbow    = flexion, the hand comes forward -- the one sign that
#                       reads backwards, because the elbow folds the other way
#
# ZERO IS A STANDING POSE, not the mid-stride she is authored in: every limb is
# rotated back to vertical once, at surgery time, and the angles that were
# subtracted are printed (`guide_skeleton()["bake_radians"]`).
# STRIDE IS WHAT SETS THE CADENCE, because the phase is locked to distance and
# the speed is fixed at 1.0 m/s: cadence = 2 * speed / stride. At the 0.70 m
# this was first drawn with, that is 171 steps/min -- a JOG, and the figure
# scurried. A brisk 1.0 m/s walk is about 110-115 steps/min, so the stride is
# set from the cadence rather than the other way round: 1.05 m of ground per
# cycle = 0.525 m per step = 114 steps/min, and the hips swing +/-17 deg
# (asin(stride/4 / 0.885 m leg)), which is where a real walker's are.
GUIDE_STRIDE_METERS = 1.05           # one full cycle = two steps of 0.525 m
GUIDE_KNEE_SWING_RADIANS = 0.90      # peak knee flexion, mid-swing only
GUIDE_SHOULDER_SWING_RADIANS = 0.35
GUIDE_ELBOW_BEND_RADIANS = 0.25      # a walker's elbows are never straight
GUIDE_ELBOW_SWING_RADIANS = 0.25
# Standing still: a slow weight shift, so a stopped guide is not a statue.
GUIDE_IDLE_PERIOD_SECONDS = 4.0
GUIDE_IDLE_HIP_RADIANS = 0.03
GUIDE_IDLE_SHOULDER_RADIANS = 0.05
# How fast the figure crosses between the walk cycle and the idle sway.
GUIDE_MOTION_BLEND_SECONDS = 0.35

# The limbs, parents before children. `anchor_root` is the hinge's position in
# HER OWN root frame (the numbers are read off human.xml's fromto endpoints:
# the thigh tops, the thigh/shin junctions, the left shoulder sphere and the
# left upper-arm/forearm junction). `align` names the capsule whose axis is
# rotated to vertical to define this joint's zero; `level` names geoms whose
# orientation is reset to flat afterwards, which is how the boots end up soles
# down instead of inheriting the stride's ankle tilt.
GUIDE_SEGMENTS = (
    {"body": "guide_thigh_l", "parent": None, "hinge": "hip_l",
     "anchor_root": (0.00, 0.10, 0.92), "align": "human_thigh_l",
     "geoms": ("human_thigh_l",), "level": ()},
    {"body": "guide_shin_l", "parent": "guide_thigh_l", "hinge": "knee_l",
     "anchor_root": (0.16, 0.11, 0.50), "align": "human_shin_l",
     "geoms": ("human_shin_l", "human_boot_l"), "level": ("human_boot_l",)},
    {"body": "guide_thigh_r", "parent": None, "hinge": "hip_r",
     "anchor_root": (0.00, -0.10, 0.92), "align": "human_thigh_r",
     "geoms": ("human_thigh_r",), "level": ()},
    {"body": "guide_shin_r", "parent": "guide_thigh_r", "hinge": "knee_r",
     "anchor_root": (-0.14, -0.11, 0.52), "align": "human_shin_r",
     "geoms": ("human_shin_r", "human_boot_r"), "level": ("human_boot_r",)},
    {"body": "guide_upper_arm_l", "parent": None, "hinge": "shoulder_l",
     "anchor_root": (0.02, 0.22, 1.40), "align": "human_upper_arm_l",
     "geoms": ("human_upper_arm_l",), "level": ()},
    {"body": "guide_forearm_l", "parent": "guide_upper_arm_l", "hinge": "elbow_l",
     "anchor_root": (-0.06, 0.28, 1.12), "align": "human_forearm_l",
     "geoms": ("human_forearm_l", "human_hand_l"), "level": ()},
)
GUIDE_HINGE_NAMES = tuple(segment["hinge"] for segment in GUIDE_SEGMENTS)
# Every hinge turns about +y (her left). The axis is not a settable parameter:
# `_rotation_y`, `_pitch_to_vertical` and the quaternion in `Guide.write` all
# hard-code it, because a sagittal walk has no use for any other axis.
GUIDE_HINGE_AXIS_IS_BODY_Y = True

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
# RE-MEASURED ON THE BACKPACK. The target used to be her jacket; the jacket is
# now BLUE and the orange has moved to the pack, so the window was re-derived
# from scratch rather than nudged. The method is unchanged and it is the point:
# the eye camera is rendered at 1/2/4/8 m in colour AND in SEGMENTATION from the
# same pose, so every pixel is attributed to the geom that painted it before its
# hue is counted (`test_guide.colour_window_table`, table A0). Pooled 1st-99th
# percentiles on flat_0 --
#
#   pack   hue 11-12  sat 246-250  value 51-239   <- THE TARGET, 6,380 px
#   jacket hue 111-113   pants hue 114-115   boots hue 113   beanie hue 86-91
#   skin   hue 0-178 BUT saturation 39-74        glove value 11-31
#   snow/sky/robot ("everything else", 267,471 px) hue 1-111
#   rope   hue 1  sat 233+      ascender carrier (orange, alpha 0.6) hue ~17
#
# -- and the window below takes 100.0% of the pack and 0.0% of every other
# group, measured, with margin on all three axes rather than at one edge:
#
#   hue         2 units either side of the pack's own 11-12; 8 clear of the
#               ROPE at hue 1 and 3 clear of the ASCENDER CARRIER at hue ~17,
#               the translucent orange sphere clipped to the robot's own palm.
#               The carrier is why the high end stops at 14 rather than opening
#               up: it is the only other orange in the world, it rides a metre
#               from the lens, and it is out of frame in these poses rather
#               than reliably absent.
#   saturation  66 clear of the pack's 246. This is the barrier that keeps SKIN
#               out (39-74), because skin hue wraps right through the pack's
#               band and hue alone could never separate the two.
#   value       11 clear of the pack's darkest face (51 -- the one box face
#               turned away from the sun, seen at 1 m). The floor no longer
#               does the discriminating work it did for the jacket; hue and
#               saturation do. It is kept as a guard against near-black pixels,
#               which is why it is low rather than absent.
#
# The BOOTS were moved off brown as part of this: brown renders at hue 12-14,
# inside the pack's own window, and the old jacket window kept boots out on the
# value floor alone. Choosing the clothes and choosing the window is ONE
# decision -- the other half is at GUIDE_OUTFIT_RGBA.
#
# THE OTHER REDS, by arithmetic (OpenCV hue = degrees / 2):
#   rope 0.85/0.08/0.05 -> max 216.8, min 12.8, delta 204, hue 60*(20.4-12.8)
#     /204 = 2.25 deg -> hue 1, saturation 240, value 217. It clears the sat and
#     value floors easily, so HUE IS ITS ONLY BARRIER -- 8 units, and the reason
#     the low end sits at 9 rather than dropping toward the skin.
#   wind pennant 0xc41414 (196,20,20) -> delta 176, hue 60*(20-20)/176 = 0 deg
#     -> hue 0, saturation 229, value 196. Nine units below the window.
#     IT CAN NEVER REACH AN EYE IMAGE ANYWAY: the two pennants are THREE.JS-ONLY
#     decoration drawn by `app/web/three/flag.js` in the browser. They exist in
#     no MuJoCo model, so nothing the renderer sees contains them. The
#     arithmetic is written down regardless, because "it is not in the model" is
#     a fact about today's scene and the hue margin is a fact about the window.
GUIDE_HUE_RANGE = (9, 14)
GUIDE_MINIMUM_SATURATION = 180
GUIDE_MINIMUM_VALUE = 40
GUIDE_MINIMUM_PIXELS = 24            # below this it is noise, not a person

# ------------------------------------------------------------ the decision
FOLLOW_RANGE_METERS = 1.8            # start following beyond this
WAIT_RANGE_METERS = 1.5              # stop within this (user ruling 2026-08-30: follow if > 1.5 m)
LOST_AFTER_SECONDS = 1.0
FOLLOW_SPEED_METERS_PER_SECOND = 0.5
BEARING_GAIN_PER_RADIAN = 2.0
MAXIMUM_YAW_RATE_RADIANS_PER_SECOND = 1.0

# ------------------------------------------------- WALKING THE VECTOR (2026-08-30)
# THE USER'S RULING: "the stock unitree should be able to walk in all directions
# when it hears the voice." The old approach law was scalar -- `[speed, 0, yaw]`
# -- so the ONLY way the robot could close a bearing was to turn, and MEASURED on
# `flat_free` (the mels policy's own training floor, stock playground G1, 10 s per
# cell from standstill) turning is the one thing it barely does:
#
#     command (vx, vy, yaw)   achieved vx    achieved vy    achieved yaw
#     (1.0, 0.0, 0.0)           +0.70 m/s      -0.19 m/s      -0.05 rad/s
#     (0.0, 0.5, 0.0)           +0.21          +0.08          -0.05      (fell 9.8 s)
#     (0.0, 0.0, 1.0)           +0.09          +0.01          +0.07
#     (0.7, 0.0, 1.0)           +0.05          -0.11          -0.11
#     (0.7, 0.5, 0.3)           +0.32          -0.19          -0.27
#
# 70% of the commanded FORWARD speed arrives; about 7% of the commanded YAW does.
# A controller that steers only with `ang_vel_yaw` is therefore spending its one
# strong actuator (forward) on a direction it did not choose and its one weak
# actuator on the whole steering job. The fix is to stop treating the command as
# a speed plus a rudder and start treating it as a VELOCITY VECTOR in the body
# frame, which is what the policy's `lin_vel_x` / `lin_vel_y` ports already are:
#
#     lin_vel_x  = v * cos(beta)
#     lin_vel_y  = clip(v * sin(beta), +/- 0.5)      <- CMD_LIMITS' own lateral cap
#     ang_vel_yaw = clip(k * beta, +/- 1.0)          <- face her WHILE walking
#
# `k` is set so the heading closes over about two seconds, which is gentle enough
# that the turn is a drift rather than a lurch -- the robot is already moving
# toward her on the linear ports, so yaw is now a comfort term, not the engine.
#
# BEHIND HER (|beta| > 90 deg) `cos(beta)` goes negative and the law would ask for
# a backwards walk. It is FLOORED AT A SMALL POSITIVE CREEP instead, and the floor
# is MEASURED, not chosen: a target 6 m away at 135 deg off the nose, `flat_free`,
# 3 seeds, 90 s budget, walking the vector with only the floor varied --
#
#     lin_vel_x floor   arrivals   mean arrival   falls
#         -0.20 m/s       0/3          --           3
#         +0.00           3/3         27.4 s        0
#         +0.15           3/3         26.7 s        0
#         +0.30           3/3         42.6 s        0
#
# -- so asking this walker to step BACKWARDS while crabbing and turning tips it
# over every time, and a small forward creep does not. +0.15 is taken over +0.00
# because at a bearing of exactly 180 deg the lateral term vanishes too, and a
# robot commanded (0, 0, yaw) is standing still: yaw only exists while the gait
# steps (see `hearing.HearingBehaviour._walk_toward`). So the robot walks a
# CURVED approach, crabbing sideways on `lin_vel_y` while the yaw term swings
# the nose around -- and it never stops to pivot, because a stopped robot cannot
# turn at all.
VECTOR_YAW_GAIN_PER_RADIAN = 0.5          # ~2 s to face the target
VECTOR_MAXIMUM_YAW_RATE_RADIANS_PER_SECOND = 1.0   # walk_policy.CMD_LIMITS[2]
VECTOR_MAXIMUM_LATERAL_METERS_PER_SECOND = 0.5     # walk_policy.CMD_LIMITS[1]
VECTOR_MAXIMUM_FORWARD_METERS_PER_SECOND = 1.0     # walk_policy.CMD_LIMITS[0]
# The floor on `lin_vel_x` when she is BEHIND. Positive, not negative and not
# zero: the gait has to keep stepping or the yaw term buys nothing, and stepping
# BACKWARDS while crabbing tips this walker over. See the table above.
VECTOR_MINIMUM_FORWARD_METERS_PER_SECOND = 0.15


def vector_command(bearing_radians, speed_meters_per_second,
                   yaw_gain=VECTOR_YAW_GAIN_PER_RADIAN,
                   deadband_radians=None) -> np.ndarray:
    """Walk TOWARD a body-frame bearing, using all three command ports.

    Inputs : `bearing_radians` -- where the target is in the robot's own frame,
             +ve to its left (the same sign the detector and the ears report);
             `speed_meters_per_second` -- how fast to approach, metres/second.
    Output : (3,) float [lin_vel_x m/s, lin_vel_y m/s, ang_vel_yaw rad/s], the
             walking policy's own layout, already clamped to `CMD_LIMITS`.
             The linear pair is the approach direction resolved into the body
             frame; the yaw term turns the nose onto her over ~2 s WITHOUT
             stopping. See the block comment above for the measured reason.
    """
    if deadband_radians is None:
        # Resolved here, not in the signature: `BEARING_DEADBAND_RADIANS` is
        # assigned further down this module and a default argument is evaluated
        # at def time.
        deadband_radians = BEARING_DEADBAND_RADIANS
    bearing = float(_wrap_to_pi(bearing_radians))
    speed = float(speed_meters_per_second)
    forward = speed * math.cos(bearing)
    lateral = speed * math.sin(bearing)
    forward = max(forward, VECTOR_MINIMUM_FORWARD_METERS_PER_SECOND)
    forward = float(np.clip(forward, VECTOR_MINIMUM_FORWARD_METERS_PER_SECOND,
                            VECTOR_MAXIMUM_FORWARD_METERS_PER_SECOND))
    lateral = float(np.clip(lateral, -VECTOR_MAXIMUM_LATERAL_METERS_PER_SECOND,
                            VECTOR_MAXIMUM_LATERAL_METERS_PER_SECOND))
    if abs(bearing) < deadband_radians:
        yaw_rate = 0.0
    else:
        yaw_rate = float(np.clip(
            yaw_gain * bearing,
            -VECTOR_MAXIMUM_YAW_RATE_RADIANS_PER_SECOND,
            VECTOR_MAXIMUM_YAW_RATE_RADIANS_PER_SECOND))
    return np.array([forward, lateral, yaw_rate])


def _wrap_to_pi(angle_radians: float) -> float:
    """(-pi, pi]. A bearing of +190 deg is a bearing of -170 deg."""
    return (float(angle_radians) + math.pi) % (2.0 * math.pi) - math.pi


# ------------------------------------------------------- looking for her
# THE G1 HAS NO NECK. The stereo pair is the `d435i` mount on `torso_link`, and
# the chain from the pelvis up is
#     pelvis -> waist_yaw_link  (waist_yaw_joint, +z, +/-150 deg, actuator 12)
#            -> waist_roll_link -> torso_link   [the cameras]
# so WAIST YAW is the joint that pans the cameras, and it sits above them in the
# tree. Read off the compiled model, not assumed -- and if a future robot moves
# the mount, `WaistYaw.bind` says so on stdout and the search turns itself off.
WAIST_YAW_JOINT_NAME = "waist_yaw_joint"
# Rate limit on the injected offset. Slow enough that the picture is not a smear
# the block matcher cannot work with, and slow enough not to shove a robot whose
# palm is clipped to a rope.
WAIST_RATE_RADIANS_PER_SECOND = 1.5
# HOW FAR THE "NECK" MAY EVER TURN, and this is a MEASURED number rather than a
# design one. The joint allows +/-150 deg, but a torso twisted that far on a
# robot hanging off a rope by one palm falls over. Panning `flat_0` for 25 s,
# roped, at 1.5 rad/s:
#
#     90 deg -> fell at 9.5 s | 80 -> 8.3 s | 75 -> 5.6 s
#     70 deg -> survived      | 65 -> fell at 21.6 s | 60 -> survived, upright 0.96
#
# So 60 deg is the widest angle this robot can actually hold.
# ASK to Mrinal: a policy that expects a moving waist would lift this.
WAIST_LIMIT_RADIANS = math.radians(60.0)
BEARING_DEADBAND_RADIANS = math.radians(2.0)

# Every `MjData` field `mj_kinematics` and `mj_camlight` write. The guide
# refreshes the frames so its cameras see the human where it is NOW, then puts
# these back so the next physics step reads exactly the frames it would have
# read -- see `GuideSystem.update` for why that is not paranoia.
KINEMATICS_OUTPUT_FIELDS = (
    "xanchor", "xaxis", "xpos", "xquat", "xmat", "xipos", "ximat",
    "geom_xpos", "geom_xmat", "site_xpos", "site_xmat",
    "cam_xpos", "cam_xmat", "light_xpos", "light_xdir")

# Recorded as a number, because `Recorder.append` stacks float arrays.
GUIDE_MODE_CODES = {"WAIT": 0, "FOLLOW": 1, "LOST": 2}
# hud.json is read by the browser, and `JSON.parse` rejects a bare NaN, so a
# missing measurement is recorded as this rather than as NaN. The websocket
# state uses `null` for the same thing, which JSON does allow.
NO_MEASUREMENT = -1.0

# Everything a recompile of HIS spec must leave exactly where it was. Only
# nbody, ngeom, nmocap and ncam are absent, because those four are what the
# surgery adds. `nq`, `nv` and `njnt` are IN the list, and that is the point of
# the jointless limbs: the guide may grow the model's bodies and geoms, but not
# by one number the solver integrates.
STRUCTURAL_FIELDS = ("nq", "nv", "njnt", "nu", "neq", "nsite", "nsensor", "nkey")


def _structure(model, body_limit=None, joint_limit=None) -> dict:
    """The fields a recompile must not move.

    `body_limit` / `joint_limit` truncate the per-body and per-joint lists,
    because the surgery APPENDS bodies and joints: comparing the full arrays
    would flag the intended change and refuse every time. The point of the check
    is that the bodies and joints that were already there did not MOVE, change
    mass, or have their state-vector addresses shifted, and truncating to the
    old counts asks exactly that -- appended entries are invisible to it, a
    reordering or a mass edit is not.
    """
    signature = {name: int(getattr(model, name)) for name in STRUCTURAL_FIELDS}
    joints = model.njnt if joint_limit is None else int(joint_limit)
    joints = min(joints, model.njnt)
    signature["jnt_qposadr"] = model.jnt_qposadr[:joints].tolist()
    signature["jnt_dofadr"] = model.jnt_dofadr[:joints].tolist()
    signature["actuator_target"] = model.actuator_trnid[:, 0].tolist()
    limit = model.nbody if body_limit is None else int(body_limit)
    signature["body_mass"] = np.round(model.body_mass[:limit], 9).tolist()
    signature["body_names"] = [
        model.body(i).name for i in range(min(limit, model.nbody))]
    return signature


# ------------------------------------------------------------------ surgery
def _rotation_y(angle_radians) -> np.ndarray:
    """Rotation about +y -- the axis every one of the guide's hinges turns on."""
    cosine, sine = math.cos(angle_radians), math.sin(angle_radians)
    return np.array([[cosine, 0.0, sine],
                     [0.0, 1.0, 0.0],
                     [-sine, 0.0, cosine]])


def _matrix_from_quaternion(quaternion) -> np.ndarray:
    import mujoco
    matrix = np.zeros(9)
    mujoco.mju_quat2Mat(matrix, np.asarray(quaternion, dtype=float))
    return matrix.reshape(3, 3)


def _quaternion_from_matrix(matrix) -> np.ndarray:
    import mujoco
    quaternion = np.zeros(4)
    mujoco.mju_mat2Quat(quaternion, np.asarray(matrix, dtype=float).reshape(9))
    return quaternion


def _pitch_to_vertical(direction) -> float:
    """The +y rotation that points a segment straight DOWN. -> radians.

    `direction` runs from the hinge toward the body of the limb it drives.
    Rotating by this angle lands it on (0, 0, -length), so it is the angle
    SUBTRACTED from her authored mid-stride pose to make that pose the joints'
    zero -- one number per hinge, printed by `attach_guide`.
    """
    x, _y, z = [float(value) for value in direction]
    return math.atan2(x, -z)


def _compiled_hiker():
    """`assets/humans/human.xml`, compiled. -> (geoms, materials), her frame.

    Read off the COMPILED model rather than the spec, for the reason
    `_add_eye_cameras` learned the hard way: MjSpec keeps an alternative
    orientation (`euler`, `fromto`, `xyaxes`) in a side field and leaves `quat`
    at identity, so a spec-level copy silently drops the right boot's 20 degree
    tilt and every capsule's axis. `geom_pos` / `geom_quat` on the compiled
    model are the compiler's own resolved answer, and because every geom hangs
    off the one body `human` at the origin, they are already in her root frame
    -- z = 0 at the ground contact.
    """
    import mujoco

    model = mujoco.MjSpec.from_file(HUMAN_XML_PATH).compile()
    geoms = {}
    for geom_id in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
        material_id = int(model.geom_matid[geom_id])
        geoms[name] = {
            "name": name,
            "type": int(model.geom_type[geom_id]),
            "size": np.asarray(model.geom_size[geom_id], dtype=float).copy(),
            "pos": np.asarray(model.geom_pos[geom_id], dtype=float).copy(),
            "quat": np.asarray(model.geom_quat[geom_id], dtype=float).copy(),
            "group": int(model.geom_group[geom_id]),
            "material": (mujoco.mj_id2name(
                model, mujoco.mjtObj.mjOBJ_MATERIAL, material_id)
                if material_id >= 0 else None),
        }
    materials = {}
    for material_id in range(model.nmat):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_MATERIAL, material_id)
        materials[name] = {
            "rgba": np.asarray(model.mat_rgba[material_id], dtype=float).copy(),
            "specular": float(model.mat_specular[material_id]),
            "shininess": float(model.mat_shininess[material_id]),
        }
    return geoms, materials


_SKELETON = None


def guide_skeleton() -> dict:
    """Chloe's hiker, re-parented into six hinged limbs and baked to standing.

    Built once and cached: the surgery needs it to write the bodies, and
    `Guide` needs the same numbers at runtime to know where the boots are.

    Output (all lengths metres, all angles radians, all frames HERS -- x
    forward, y left, z up, z = 0 at the ground contact):
      `bake_radians`   joint name -> the angle subtracted from her authored
                       mid-stride pose to make the joints' zero a standing one.
      `root_geoms`     [geom map] the geoms that stay on the mocap root:
                       torso, head, beanie, pack, collar, shoulders and the
                       WHOLE RIGHT ARM, which is posed gripping the rope and
                       must not swing.
      `bodies`         [{name, parent, joint, pos_in_parent, geoms}] parents
                       first, ready to write straight onto an MjSpec. Each
                       geom map's `pos`/`quat` are already in ITS body frame.
      `materials`      name -> {rgba, specular, shininess}, hers verbatim.
      `legs`           side -> the forward kinematics constants the gait needs:
                       `hip_anchor` (root frame), `knee_in_thigh`,
                       `boot_in_shin`, `boot_half_extents`, and
                       (`foot_radius`, `foot_phase`) -- the polar form of the
                       hip-to-boot vector, which turns "put this foot that far
                       forward" into a hip angle with one arcsine.
    """
    global _SKELETON
    if _SKELETON is not None:
        return _SKELETON

    geoms, materials = _compiled_hiker()
    by_body = {segment["body"]: segment for segment in GUIDE_SEGMENTS}

    def chain_below(body_name):
        """That segment and every segment hanging off it, in table order."""
        below = []
        for segment in GUIDE_SEGMENTS:
            node = segment
            while node is not None:
                if node["body"] == body_name:
                    below.append(segment)
                    break
                node = by_body.get(node["parent"]) if node["parent"] else None
        return below

    anchors = {segment["body"]: np.asarray(segment["anchor_root"], dtype=float)
               for segment in GUIDE_SEGMENTS}
    bake_radians, to_level = {}, []
    for segment in GUIDE_SEGMENTS:
        # From the hinge toward the middle of the segment it drives. NOT the
        # capsule's own +z axis: MuJoCo's `fromto` compiler is free to point
        # that either way along the segment, and picking the wrong end bakes
        # the limb 160 degrees the wrong way -- which it did, once.
        angle = _pitch_to_vertical(
            geoms[segment["align"]]["pos"] - anchors[segment["body"]])
        bake_radians[segment["hinge"]] = angle
        rotation = _rotation_y(angle)
        anchor = anchors[segment["body"]].copy()
        for follower in chain_below(segment["body"]):
            for geom_name in follower["geoms"]:
                geom = geoms[geom_name]
                geom["pos"] = anchor + rotation @ (geom["pos"] - anchor)
                geom["quat"] = _quaternion_from_matrix(
                    rotation @ _matrix_from_quaternion(geom["quat"]))
            if follower["body"] != segment["body"]:
                anchors[follower["body"]] = (
                    anchor + rotation @ (anchors[follower["body"]] - anchor))
        to_level.extend(segment["level"])
    # The boots ride the shin rigidly, so straightening the leg would tip them
    # toe-up by the ankle angle her stride was drawn with. A standing pose has
    # flat feet: their orientation is reset once, here, and nothing else about
    # them moves.
    for geom_name in to_level:
        geoms[geom_name]["quat"] = np.array([1.0, 0.0, 0.0, 0.0])

    # SYMMETRISE THE LEGS. She was drawn mid-stride by hand, and the two legs
    # came out 2.7 cm different in length, with the knees 3.2 cm apart along
    # the stride. A standing pose cannot be asymmetric or she limps: the lower
    # foot is the planted one on every step, so one leg would carry the whole
    # walk and the other would paw the air, and the swing clearances came out
    # 7.6 cm against 5.1 cm. Both the knee anchor and the boot are nudged to
    # the mean of the two sides in the sagittal plane -- 1.6 cm and 1.4 cm,
    # half the difference each way, invisible on a 1.75 m figure. Left/right
    # offsets in y are untouched: those are her stance width, not an error.
    thigh_body = {"l": "guide_thigh_l", "r": "guide_thigh_r"}
    shin_body = {"l": "guide_shin_l", "r": "guide_shin_r"}
    boot_geom = {"l": "human_boot_l", "r": "human_boot_r"}

    def sagittal(vector):
        return np.array([vector[0], 0.0, vector[2]])

    knee_target = 0.5 * sum(
        sagittal(anchors[shin_body[side]] - anchors[thigh_body[side]])
        for side in "lr")
    boot_target = 0.5 * sum(
        sagittal(geoms[boot_geom[side]]["pos"] - anchors[shin_body[side]])
        for side in "lr")
    leg_correction = {}
    for side in "lr":
        knee_shift = knee_target - sagittal(
            anchors[shin_body[side]] - anchors[thigh_body[side]])
        anchors[shin_body[side]] = anchors[shin_body[side]] + knee_shift
        for geom_name in by_body[shin_body[side]]["geoms"]:
            geoms[geom_name]["pos"] = geoms[geom_name]["pos"] + knee_shift
        boot_shift = boot_target - sagittal(
            geoms[boot_geom[side]]["pos"] - anchors[shin_body[side]])
        geoms[boot_geom[side]]["pos"] = geoms[boot_geom[side]]["pos"] + boot_shift
        leg_correction[side] = {"knee_meters": knee_shift, "boot_meters": boot_shift}

    limb_geom_names = {name for segment in GUIDE_SEGMENTS
                       for name in segment["geoms"]}
    bodies = []
    for segment in GUIDE_SEGMENTS:
        anchor = anchors[segment["body"]]
        parent_anchor = (anchors[segment["parent"]] if segment["parent"]
                         else np.zeros(3))
        bodies.append({
            "name": segment["body"],
            "parent": segment["parent"],
            "hinge": segment["hinge"],
            "pos_in_parent": anchor - parent_anchor,
            "geoms": [dict(geoms[name], pos=geoms[name]["pos"] - anchor)
                      for name in segment["geoms"]],
        })

    legs = {}
    for side, thigh, shin, boot in (
            ("l", "guide_thigh_l", "guide_shin_l", "human_boot_l"),
            ("r", "guide_thigh_r", "guide_shin_r", "human_boot_r")):
        knee_in_thigh = anchors[shin] - anchors[thigh]
        boot_in_shin = geoms[boot]["pos"] - anchors[shin]
        hip_to_boot = knee_in_thigh + boot_in_shin
        legs[side] = {
            "hip_anchor": anchors[thigh].copy(),
            "knee_in_thigh": knee_in_thigh,
            "boot_in_shin": boot_in_shin,
            "boot_half_extents": geoms[boot]["size"][:3].copy(),
            "foot_radius": float(math.hypot(hip_to_boot[0], hip_to_boot[2])),
            "foot_phase": float(math.atan2(hip_to_boot[0], -hip_to_boot[2])),
        }

    _SKELETON = {
        "bake_radians": bake_radians,
        "leg_correction": leg_correction,
        "root_geoms": [geoms[name] for name in geoms
                       if name not in limb_geom_names],
        "bodies": bodies,
        "materials": materials,
        "legs": legs,
    }
    return _SKELETON


def _add_guide_body(spec) -> None:
    """Build the guide's body on the spec. CHLOE'S HIKER LIVES HERE.

    One mocap root carrying her torso, head, pack and rope-gripping right arm,
    plus six child bodies -- thigh/shin either side and the left upper
    arm/forearm -- each a HINGE IN NAME ONLY: no joint is added, the body is
    welded to its parent, and `Guide.write` turns it by writing
    `model.body_quat` every control tick. The whole figure therefore has ZERO
    degrees of freedom, `nq`, `nv` and `njnt` do not move, and the solver
    integrates exactly the state vector it integrated before she existed. The
    long comment at GUIDE_SEGMENTS says what the jointed version cost.

    Every geom keeps her name (`human_*`), her group (2) and her
    `contype = conaffinity = 0`, so `human-safety/human_gate.py`'s segmentation
    gate sees exactly what it saw before and the robot still cannot touch her.
    Her materials are copied across under a `guide_` prefix so a scene that also
    calls `assets/humans/humans.py::attach_humans` cannot collide with them --
    and this copy is where `GUIDE_OUTFIT_RGBA` dresses her, so her shared XML
    stays untouched while the compiled model, the GLB export and the eye
    cameras all see one set of clothes.
    """
    import mujoco

    skeleton = guide_skeleton()
    for name, material in skeleton["materials"].items():
        added = spec.add_material()
        added.name = f"guide_{name}"
        # The outfit overrides the colour and NOTHING ELSE -- specular and
        # shininess still come from her own XML, and no geometry is touched.
        added.rgba = [float(value) for value in
                      GUIDE_OUTFIT_RGBA.get(name, material["rgba"])]
        added.specular = material["specular"]
        added.shininess = material["shininess"]

    def add_geom(body, part):
        geom = body.add_geom()
        geom.name = part["name"]
        geom.type = mujoco.mjtGeom(part["type"])
        geom.size = [float(value) for value in part["size"]]
        geom.pos = [float(value) for value in part["pos"]]
        geom.quat = [float(value) for value in part["quat"]]
        geom.group = int(part["group"])
        geom.contype = 0
        geom.conaffinity = 0
        if part["material"]:
            geom.material = f"guide_{part['material']}"
        return geom

    root = spec.worldbody.add_body()
    root.name = GUIDE_BODY_NAME
    root.mocap = True
    root.pos = [0.0, 0.0, -50.0]      # parked under the world until placed
    for part in skeleton["root_geoms"]:
        add_geom(root, part)

    bodies = {None: root}
    for description in skeleton["bodies"]:
        body = bodies[description["parent"]].add_body()
        body.name = description["name"]
        body.pos = [float(value) for value in description["pos_in_parent"]]
        # NO `add_joint` HERE, deliberately. See _add_guide_body's docstring.
        bodies[description["name"]] = body
        for part in description["geoms"]:
            add_geom(body, part)


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
        # The BARE playground G1 (menagerie) ships without the RealSense the
        # real robot carries. Mount one at the jacketed model's exact pose
        # (assets/robots/mujoco/g1_unitree_ascender.xml: pos 0.0789635 0 0.386,
        # xyaxes "0 -1 0  0 0 1", fovy 58) so the eyes -- and with them the
        # guide and the hearing demo -- work on the stock robot too
        # (user ruling 2026-08-30: flat_free carries the stock G1). A camera
        # is visual-only: MuJoCo integrates nothing from it.
        torso = None
        for body in spec.bodies:
            if body.name == "torso_link":
                torso = body
                break
        if torso is None:
            print(f"[guide] no {SOURCE_CAMERA_NAME!r} camera and no"
                  " 'torso_link' to mount one on: the guide stays off",
                  flush=True)
            return False
        # xyaxes "0 -1 0  0 0 1": camera x (image right) = -world-y of the
        # torso, y (image up) = +z, so z (out the back) = x cross y = -x.
        rotation_columns = np.array([[0.0, 0.0, -1.0],
                                     [-1.0, 0.0, 0.0],
                                     [0.0, 1.0, 0.0]])
        fallback_quaternion = np.zeros(4)
        mujoco.mju_mat2Quat(fallback_quaternion, rotation_columns.reshape(9))
        source = torso.add_camera(name=SOURCE_CAMERA_NAME)
        source.pos = [0.0789635, 0.0, 0.386]
        source.quat = fallback_quaternion.tolist()
        source.fovy = 58.0
        if verbose:
            print(f"[guide] bare robot: mounted the {SOURCE_CAMERA_NAME!r}"
                  " RealSense at the jacketed model's pose on 'torso_link'"
                  " (the real G1 carries one; the menagerie file omits it)",
                  flush=True)
        parent = torso
        position = np.array([0.0789635, 0.0, 0.386])
        quaternion = fallback_quaternion
        fovy = 58.0
    else:
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
        baked = guide_skeleton()["bake_radians"]
        print(f"[guide] attached: bodies {before_after(model, recompiled, 'nbody')},"
              f" geoms {before_after(model, recompiled, 'ngeom')},"
              f" cameras {before_after(model, recompiled, 'ncam')},"
              f" mocap {before_after(model, recompiled, 'nmocap')},"
              f" joints {before_after(model, recompiled, 'njnt')},"
              f" nq {before_after(model, recompiled, 'nq')},"
              f" nv {before_after(model, recompiled, 'nv')};"
              f" all {len(before)} structural fields unchanged", flush=True)
        print("[guide] neutral pose baked out of her mid-stride, degrees: "
              + ", ".join(f"{name} {math.degrees(angle):+.1f}"
                          for name, angle in baked.items()), flush=True)
    return True


def before_after(old, new, field) -> str:
    return f"{int(getattr(old, field))}->{int(getattr(new, field))}"


# -------------------------------------------------------------- the human
class Guide:
    """Where the human is, and what her legs are doing. Kinematic throughout.

    The route is the same polyline the ascender rides (`RopeRoute`), so the
    guide walks the line the robot is climbing, offset `GUIDE_LATERAL_METERS`
    to its LEFT looking uphill -- close enough to lead, far enough not to be
    walked into. Height is snapped to the terrain surface every tick, so she
    cannot sink into a slope or float over a dip, and she has no velocity state
    to be disturbed: she either advances, retreats, or stands.

    THE WALK IS DISTANCE-LOCKED, which is the whole trick to feet that do not
    skate. The cycle phase is `2 pi * travel / GUIDE_STRIDE_METERS` -- a
    function of how far she has WALKED, never of the clock -- so one stride of
    ground covers exactly one stride of animation whatever the speed, and
    walking backwards (S) runs the same cycle in reverse. Within a cycle the
    stance foot is placed by the same arithmetic in the other direction: the
    foot's offset from the hip is a straight ramp from +stride/4 to -stride/4
    while the root advances stride/2, so the two cancel and the planted boot
    holds still in the world. The swing half returns it on a raised cosine.

    Because the stance leg is straight, the hip drops by
    `leg * (1 - cos(hip))` as it swings out -- the walking bob, about +/-2 cm,
    which the root-z snapping produces for free rather than adding by hand: the
    root is placed so the LOWEST boot corner of either leg sits on the surface.

    Inputs  : dt seconds, and whether she was told to walk forwards or back.
    Outputs : `state()` -- progress along the route in metres and whether she
              has run out of rope; `limb_angles()` -- the six hinge angles;
              `write()` puts both into `MjData`.
    """

    def __init__(self, route, terrain, model=None, free_walk=False):
        self.route = route
        self.terrain = terrain
        # FREE WALK: on a world with no rope there is no line for her to be on,
        # so W/S walk her along her OWN heading and A/D turn it (user's ruling,
        # 2026-08-30). On a roped world she stays on the rope exactly as before
        # and A/D do nothing to her. She is still initialised from the route in
        # both cases -- only the DRIVING changes, so the spawn is unmoved.
        self.free_walk = bool(free_walk)
        self.free_position_world = np.zeros(2)
        self.free_yaw_radians = 0.0
        self.arclength_meters = 0.0
        # HOW FAR SHE HAS WALKED, signed, and the ONLY thing the gait phase
        # reads. On the rope it tracks the arc length; free of it, it tracks
        # ground covered. Keeping them separate is what stops the walk cycle
        # freezing when free-walking past the end of the rope's length.
        self.travel_meters = 0.0
        self.enabled = False
        self.body_id = -1
        self.mocap_id = -1
        # 1 while she is walking, 0 while she stands; crossfades between the
        # walk cycle and the idle sway so a stop is not a freeze-frame.
        self.motion_blend = 0.0
        self.idle_seconds = 0.0
        self.direction = 0.0
        self.limb_body_ids = {}
        self.skeleton = guide_skeleton()
        if model is not None:
            self.bind(model)

    def bind(self, model) -> None:
        """Cache the mocap slot and the six limb BODY ids for this model."""
        import mujoco
        self.body_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, GUIDE_BODY_NAME)
        self.mocap_id = (int(model.body_mocapid[self.body_id])
                         if self.body_id >= 0 else -1)
        self.limb_body_ids = {}
        for segment in GUIDE_SEGMENTS:
            body_id = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_BODY, segment["body"])
            if body_id >= 0:
                self.limb_body_ids[segment["hinge"]] = body_id

    def place_ahead_of(self, position_world, lead_meters=GUIDE_LEAD_METERS) -> None:
        """Put the human `lead_meters` further up the route than a given point."""
        start, _ = self.route.project_arclen(np.asarray(position_world, dtype=float))
        self.arclength_meters = float(np.clip(
            start + lead_meters, 0.0, self.route.length))
        self.travel_meters = self.arclength_meters
        # Seed the free-walk pose from the rope route, so she stands exactly
        # where she always did and only the controls differ.
        on_route = self._route_ground_position()
        self.free_position_world[:] = on_route[:2]
        self.free_yaw_radians = self._route_yaw_radians()

    def advance(self, dt_seconds: float, walking: bool, backing: bool = False,
                turning: float = 0.0) -> None:
        """W walks her forward, S walks her back, both held = stop.

        ON THE ROPE she is on the line: forward is up it, and `turning` is
        ignored because there is nowhere for her to turn to. Backwards is a real
        retreat, not a turn -- the yaw still comes from the route tangent, so she
        keeps facing uphill and steps backwards down the slope, the way a guide
        backs toward the person behind them.

        FREE OF THE ROPE (`free_walk`, i.e. any world with `rope=False`) she
        walks along her OWN heading and `turning` -- A and D -- is what changes
        it. +1 is left, matching every other yaw sign in this harness.

        The gait follows either way, because the phase is distance-locked to
        `travel_meters` and that goes down when she backs up.
        """
        dt_seconds = float(dt_seconds)
        self.direction = (1.0 if walking else 0.0) - (1.0 if backing else 0.0)
        step_meters = GUIDE_SPEED_METERS_PER_SECOND * dt_seconds * self.direction
        if self.free_walk:
            self.free_yaw_radians += (GUIDE_TURN_RATE_RADIANS_PER_SECOND
                                      * dt_seconds * float(turning))
            self.free_position_world[0] += step_meters * math.cos(
                self.free_yaw_radians)
            self.free_position_world[1] += step_meters * math.sin(
                self.free_yaw_radians)
            self.travel_meters += step_meters
        else:
            self.arclength_meters = float(np.clip(
                self.arclength_meters + step_meters, 0.0, self.route.length))
            self.travel_meters = self.arclength_meters
        self.idle_seconds += dt_seconds
        target = 1.0 if self.direction != 0.0 else 0.0
        rate = 1.0 - math.exp(-dt_seconds / GUIDE_MOTION_BLEND_SECONDS)
        self.motion_blend += (target - self.motion_blend) * rate

    @property
    def at_rope_end(self) -> bool:
        return self.arclength_meters >= self.route.length - 1e-6

    @property
    def at_rope_start(self) -> bool:
        return self.arclength_meters <= 1e-6

    # ------------------------------------------------------------- the gait
    def phase_radians(self) -> float:
        """One full cycle -- two steps -- per `GUIDE_STRIDE_METERS` of ground."""
        return 2.0 * math.pi * self.travel_meters / GUIDE_STRIDE_METERS

    def _hip_for_foot_offset(self, side, offset_meters) -> float:
        """The hip angle that puts that boot `offset_meters` in front. -> radians.

        With the knee straight the hip-to-boot vector is rigid, so its x
        component is `radius * sin(phase - hip)` and the inverse is one arcsine.
        This is where the no-skate property is actually enforced.
        """
        leg = self.skeleton["legs"][side]
        ratio = float(np.clip(offset_meters / leg["foot_radius"], -1.0, 1.0))
        return leg["foot_phase"] - math.asin(ratio)

    def limb_angles(self) -> dict:
        """The six hinge angles for the current travel. -> name -> radians."""
        phase = self.phase_radians() % (2.0 * math.pi)
        half_step = 0.25 * GUIDE_STRIDE_METERS

        def foot_offset(leg_phase):
            """Ahead of the hip, metres: a linear stance ramp, cosine swing."""
            leg_phase %= 2.0 * math.pi
            if leg_phase < math.pi:                       # planted: rides back
                return half_step * (1.0 - 2.0 * leg_phase / math.pi)
            return -half_step * math.cos(leg_phase - math.pi)   # swinging home

        walk = {
            "hip_l": self._hip_for_foot_offset("l", foot_offset(phase)),
            "hip_r": self._hip_for_foot_offset("r", foot_offset(phase + math.pi)),
            "knee_l": GUIDE_KNEE_SWING_RADIANS * max(0.0, -math.sin(phase)),
            "knee_r": GUIDE_KNEE_SWING_RADIANS * max(0.0, math.sin(phase)),
            # The left arm answers the left leg: leg forward, arm back.
            "shoulder_l": GUIDE_SHOULDER_SWING_RADIANS * math.cos(phase),
            "elbow_l": -(GUIDE_ELBOW_BEND_RADIANS
                         + GUIDE_ELBOW_SWING_RADIANS * max(0.0, -math.cos(phase))),
        }
        sway = math.sin(2.0 * math.pi * self.idle_seconds / GUIDE_IDLE_PERIOD_SECONDS)
        idle = {
            "hip_l": GUIDE_IDLE_HIP_RADIANS * sway,
            "hip_r": -GUIDE_IDLE_HIP_RADIANS * sway,
            "knee_l": 0.0,
            "knee_r": 0.0,
            "shoulder_l": GUIDE_IDLE_SHOULDER_RADIANS * sway,
            "elbow_l": -GUIDE_ELBOW_BEND_RADIANS,
        }
        blend = float(np.clip(self.motion_blend, 0.0, 1.0))
        return {name: blend * walk[name] + (1.0 - blend) * idle[name]
                for name in GUIDE_HINGE_NAMES}

    def _boot_lowest_offset(self, angles) -> float:
        """How far the lowest boot corner sits below the root. -> metres (<= 0).

        Forward kinematics of two hinges and a box, done here rather than read
        out of `MjData`, because the root's height has to be known BEFORE the
        pose is written -- MuJoCo has not computed anything yet at that point.
        """
        lowest = 0.0
        for side, hip_name, knee_name in (("l", "hip_l", "knee_l"),
                                          ("r", "hip_r", "knee_r")):
            leg = self.skeleton["legs"][side]
            thigh_rotation = _rotation_y(angles[hip_name])
            shin_rotation = _rotation_y(angles[hip_name] + angles[knee_name])
            knee = leg["hip_anchor"] + thigh_rotation @ leg["knee_in_thigh"]
            centre = knee + shin_rotation @ leg["boot_in_shin"]
            half = leg["boot_half_extents"]
            for sign_x in (-1.0, 1.0):
                for sign_y in (-1.0, 1.0):
                    for sign_z in (-1.0, 1.0):
                        corner = centre + shin_rotation @ np.array(
                            [sign_x * half[0], sign_y * half[1], sign_z * half[2]])
                        lowest = min(lowest, float(corner[2]))
        return lowest

    # ------------------------------------------------------------ the pose
    def ground_position_world(self) -> np.ndarray:
        """Where she stands, on the surface. -> (3,) world."""
        if self.free_walk:
            x = float(self.free_position_world[0])
            y = float(self.free_position_world[1])
            return np.array([x, y, float(self.terrain.surface_z(x, y))])
        return self._route_ground_position()

    def _route_ground_position(self) -> np.ndarray:
        """On the route, offset left, on the surface."""
        on_rope = self.route.point_at(self.arclength_meters)
        tangent = self.route.tangent_at(self.arclength_meters)
        # Left of the direction of travel, on the ground plane.
        left = np.array([-tangent[1], tangent[0], 0.0])
        norm = float(np.linalg.norm(left))
        left = left / norm if norm > 1e-9 else np.array([0.0, 1.0, 0.0])
        x, y = (on_rope[:2] + GUIDE_LATERAL_METERS * left[:2])
        return np.array([float(x), float(y),
                         float(self.terrain.surface_z(x, y))])

    def root_world(self, angles=None) -> np.ndarray:
        """World position of the body origin -- the pose written to `mocap_pos`.

        Her root frame has z = 0 at the ground contact, so the origin would sit
        exactly on the surface if her feet were flat. Mid-stride they are not:
        the root is lifted by however far the lowest boot corner hangs below
        zero, which is what keeps the planted foot ON the snow through the whole
        cycle instead of sinking into it at the ends.
        """
        position = self.ground_position_world()
        position[2] -= self._boot_lowest_offset(
            self.limb_angles() if angles is None else angles)
        return position

    def reference_point_world(self) -> np.ndarray:
        """Chest height on the body axis -- what a TRUE range is measured to."""
        point = self.root_world()
        point[2] += REFERENCE_HEIGHT_METERS
        return point

    def yaw_radians(self) -> float:
        if self.free_walk:
            return float(self.free_yaw_radians)
        return self._route_yaw_radians()

    def _route_yaw_radians(self) -> float:
        tangent = self.route.tangent_at(self.arclength_meters)
        return math.atan2(float(tangent[1]), float(tangent[0]))

    def write(self, model, data) -> None:
        """Pose her: `mocap_pos`/`mocap_quat` for the root, `body_quat` for the
        six limbs.

        TWO DIFFERENT ARRAYS, AND NEITHER IS STATE. The root is a MOCAP body:
        MuJoCo reads `mocap_pos`/`mocap_quat` during the forward kinematics and
        nothing ever writes back to them. The six limbs are WELDED child bodies
        with no joints, so their orientation lives in `model.body_quat` --
        a model field, read by `mj_kinematics` every step exactly like a fixed
        offset, and written here every control tick to turn each hinge. Nothing
        the solver integrates is touched: `data.qpos` and `data.qvel` are the
        robot's alone, which is what makes the same-seed diff zero.

        `mj_resetData` parks mocap bodies at their model pose, so this runs
        every tick rather than once.
        """
        if self.mocap_id < 0:
            return
        angles = self.limb_angles()
        for name, body_id in self.limb_body_ids.items():
            half = 0.5 * angles[name]
            # A +y hinge, as a quaternion: (cos, 0, sin, 0).
            model.body_quat[body_id] = (math.cos(half), 0.0, math.sin(half), 0.0)
        if not self.enabled:
            data.mocap_pos[self.mocap_id] = (0.0, 0.0, -50.0)
            return
        data.mocap_pos[self.mocap_id] = self.root_world(angles)
        half = 0.5 * self.yaw_radians()
        data.mocap_quat[self.mocap_id] = (math.cos(half), 0.0, 0.0, math.sin(half))

    def state(self) -> dict:
        return {
            "human_progress_meters": float(self.travel_meters),
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
                 verbose=True, degradation=None):
        import cv2
        import mujoco
        from app.harness import graphics as graphics_module

        self.width, self.height = int(width), int(height)
        self.model = model
        # `degradation(image) -> image`, run on EACH eye between the render and
        # the matcher: `storm.StormVision.degrade` in the live loop, None
        # otherwise. It belongs HERE and not further down because a degradation
        # applied after the measurement is a special effect, not a sensor model.
        self.degradation = degradation
        self.renderer = None
        self.left_image = None
        self.right_image = None
        self.render_milliseconds = 0.0
        self.degrade_milliseconds = 0.0
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
        """Both eyes, degraded if a storm is blowing. -> (left, right) RGB.

        THE DEGRADATION RUNS WHILE THAT EYE'S SCENE IS STILL LOADED, which is
        why it is interleaved with the renders rather than done to both images
        at the end. `storm.StormVision.degrade` composites the white-out from the
        renderer's own DEPTH buffer, and `Renderer.render` draws whatever scene
        was last handed to `update_scene` -- so degrading the left eye after the
        right had been set up would fog the left picture with the right eye's
        depth. Six centimetres of error, and completely avoidable.

        The order also matters for a second reason: the left is degraded before
        the right is even rendered, so a stateful degradation (the storm's
        seeded generator) hands the two eyes DIFFERENT sensor grain. Identical
        grain would sit at zero disparity and the matcher would happily match
        it.
        """
        import time
        started = time.time()
        self.renderer.update_scene(data, camera=self.left_camera_id)
        left = self.renderer.render().copy()
        degrade_started = time.time()
        if self.degradation is not None:
            left = self.degradation(left, self.renderer)
        self.degrade_milliseconds = (time.time() - degrade_started) * 1000.0

        self.renderer.update_scene(data, camera=self.right_camera_id)
        right = self.renderer.render().copy()
        degrade_started = time.time()
        if self.degradation is not None:
            right = self.degradation(right, self.renderer)
        self.degrade_milliseconds += (time.time() - degrade_started) * 1000.0
        self.render_milliseconds = ((time.time() - started) * 1000.0
                                    - self.degrade_milliseconds)
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


# ------------------------------------------------------------- the "neck"
class WaistYaw:
    """Turn the cameras without turning the robot. The G1 has no neck.

    WHAT THIS IS. A single number, `offset_radians`, added to the WALKING
    POLICY'S OWN waist-yaw PD target after the policy has written it and before
    the `mj_step` that acts on it (`ClimbSceneEpisode.control_hooks`). The
    policy is untouched: it is not retrained, not asked for anything, and its
    next observation sees the result as it sees any other disturbance the world
    hands it.

    WHY THE WAIST AND NOT THE HEAD. The stereo pair is the `d435i` mount on
    `torso_link`, and `waist_yaw_joint` is the last joint above it in the
    kinematic chain -- so it is the joint that pans the cameras, and the only
    one. Verified against the compiled model; `bind` refuses and says so on
    stdout if a future robot moves the mount.

    TWO SAFETY PROPERTIES, both enforced here rather than trusted:
      * RATE LIMITED to `WAIST_RATE_RADIANS_PER_SECOND`. A step change in a PD
        target is a kick, and this robot hangs off a rope by one palm.
      * CLAMPED to the actuator's own `ctrlrange` AFTER the addition, so the
        offset can never drive the joint past a limit the policy was keeping it
        inside.

    Inputs  : `target_radians`, set by the follower; `advance(dt)` once a
              control tick; `apply(model, data)` at the control hook.
    Outputs : `offset_radians` -- what is actually being added after the rate
              limit, which is what the HUD reports.
    """

    def __init__(self, model=None):
        self.actuator_index = -1
        self.qpos_address = -1
        self.control_range = (-math.inf, math.inf)
        self._target_radians = 0.0
        self.offset_radians = 0.0
        self.measured_radians = 0.0
        self.available = False
        if model is not None:
            self.bind(model)

    def bind(self, model, verbose=False) -> bool:
        import mujoco
        self.actuator_index = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_ACTUATOR, WAIST_YAW_JOINT_NAME)
        self.available = self.actuator_index >= 0
        if not self.available:
            print(f"[guide] no {WAIST_YAW_JOINT_NAME!r} actuator in this model:"
                  " the robot cannot pan its cameras; the ear layer cannot"
                  " aim them",
                  flush=True)
            return False
        low, high = model.actuator_ctrlrange[self.actuator_index]
        self.control_range = (float(low), float(high))
        joint_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_JOINT, WAIST_YAW_JOINT_NAME)
        self.qpos_address = (int(model.jnt_qposadr[joint_id])
                             if joint_id >= 0 else -1)
        if verbose:
            print(f"[guide] camera pan = {WAIST_YAW_JOINT_NAME} (actuator"
                  f" {self.actuator_index}), range"
                  f" {math.degrees(self.control_range[0]):+.0f} to"
                  f" {math.degrees(self.control_range[1]):+.0f} deg, rate limit"
                  f" {WAIST_RATE_RADIANS_PER_SECOND} rad/s", flush=True)
        return True

    @property
    def target_radians(self) -> float:
        return self._target_radians

    @target_radians.setter
    def target_radians(self, value) -> None:
        """CLAMPED ON ASSIGNMENT, not on use, and that is the point.

        Clamping inside `advance` instead left `target_radians` reading 90 deg
        while the offset could only ever reach 60 -- so a caller's "have I got
        there yet" test never fired, the waist stuck at the limit and the robot
        stopped looking. Clamping here means every reader sees the angle that
        can actually happen, and that whole class of bug cannot recur.
        """
        self._target_radians = float(np.clip(
            float(value), -WAIST_LIMIT_RADIANS, WAIST_LIMIT_RADIANS))

    def reset(self) -> None:
        self.target_radians = 0.0
        self.offset_radians = 0.0

    def measure(self, data) -> float:
        """The waist yaw the robot ACTUALLY has. -> radians.

        NOT the same as `offset_radians`, and the difference is the whole reason
        this method exists. The offset is added to a PD TARGET that the walking
        policy is also writing, so the policy pulls back against it and the joint
        settles somewhere short of what was asked. Feeding the COMMANDED offset
        back into "where is she relative to the body" is then a positive feedback
        loop: the image bearing never closes, so the target grows every vision
        tick. MEASURED on flat_0 before this was read from `qpos`: the waist wound
        up to 168 degrees and the robot fell over at 7.3 s.
        """
        if self.qpos_address < 0:
            return 0.0
        self.measured_radians = float(data.qpos[self.qpos_address])
        return self.measured_radians

    def advance(self, dt_seconds: float) -> None:
        step = WAIST_RATE_RADIANS_PER_SECOND * float(dt_seconds)
        error = self.target_radians - self.offset_radians
        self.offset_radians += float(np.clip(error, -step, step))

    def apply(self, model, data) -> None:
        """The control hook: add the offset to the policy's own PD target."""
        if not self.available or self.offset_radians == 0.0:
            return
        low, high = self.control_range
        data.ctrl[self.actuator_index] = float(np.clip(
            data.ctrl[self.actuator_index] + self.offset_radians, low, high))

    @property
    def degrees(self) -> float:
        return math.degrees(self.offset_radians)


# ------------------------------------------------------------ the decision
class GuideFollower:
    """FOLLOW / WAIT / LOST, from a range and a bearing.

    THE THREE FOLLOWING BANDS overlap on purpose. A single 1.0 m threshold makes
    the robot chatter between walking and standing at exactly the distance it is
    trying to hold, because each decision changes the very number the next
    decision reads. So the switch out of WAIT is at 1.8 m and the switch into it
    at 1.5 m, and the 30 cm in between belongs to whichever state is running:

        FOLLOW   range > 1.8 m, or > 1.5 m while already following
        WAIT     range <= 1.5 m, or <= 1.8 m while already waiting
        LOST     nothing detected for a whole second

    LOST MEANS STAND STILL AND LISTEN (user's ruling, 2026-08-30). There used to
    be a SEARCH state here: on losing her the robot swung its waist through a
    20/60/90 degree ladder hunting for the orange, then ran an acquire/realign
    hand-over back to FOLLOW. It worked and it was measured (`flat_0` roped:
    0.20 s to acquire, hand-over at 0.24 s, 10.2 deg of camera-bearing error;
    `terrain_free_10` rope off: 0.40 s and 3.2 s; the 60 degree clamp above is
    what survived that experiment) -- and it is gone, because the robot now has
    EARS. A machine that has lost the person it is following should not wave its
    torso about hoping; it should hold still and wait to be called, and the next
    shout gives it a direction that a camera sweep never could. That is
    `hearing.HearingBehaviour`'s `LISTENING` and `COMING_BY_EARS`, and this
    class simply reports LOST and commands zero.

    THE WAIST STAYS, because it is still the only thing that can point the
    cameras: `hearing` aims it at the direction a shout came from. What is gone
    is this class's use of it to hunt. In FOLLOW and WAIT the target is driven
    back to zero, because she is in front and there is nothing to aim at; in
    LOST the target is LEFT ALONE, because the ear layer owns it then.

    THE ROBOT TURNS ITS WAIST, NOT ITS BODY, unless told otherwise.
    `yaw_command_available` is False by default and for every climb world: the
    policy was trained with ang_vel_yaw ~ 0 and the palm is clipped to a fixed
    line, so commanding yaw does almost nothing (PARITY.md: +1.0 and -1.0 rad/s
    for 3 s end 10 deg apart). With it False, `ang_vel_yaw` is zero in EVERY
    state.

    Inputs  : `range_meters` (None if not detected), `bearing_radians`, dt, and
              a `WaistYaw` the ear layer may aim.
    Outputs : `mode` (str) and `command()` -> (3,)
              [lin_vel_x, lin_vel_y, ang_vel_yaw], the walking policy's layout.
    """

    def __init__(self, command_speed=FOLLOW_SPEED_METERS_PER_SECOND,
                 waist=None, yaw_command_available=False,
                 vector_steering=False, rope_climb=False):
        self.command_speed = float(command_speed)
        # ROPE MODE (user's ruling, 2026-08-30): on a roped world the robot
        # climbs the line INDEFINITELY -- there is nowhere else to go on a
        # fixed rope -- until the eyes have her inside the WAIT band (1.5 m),
        # which freezes the world. Not seeing her is not a reason to stop.
        self.rope_climb = bool(rope_climb)
        # WALK THE VECTOR, DON'T STEER WITH THE RUDDER (user's ruling,
        # 2026-08-30). True on rope-off worlds, where the body is free to move
        # sideways and `lin_vel_y` is a real actuator; False on every roped
        # world, where the palm is clipped to a fixed line and a lateral command
        # only fights the rope. See `vector_command` for the measured authority
        # table that motivates it.
        self.vector_steering = bool(vector_steering)
        # THE ROBOT'S ONLY WAY TO POINT ITS CAMERAS WITHOUT TURNING. None on a
        # model with no waist-yaw actuator; the ear layer then simply cannot aim
        # and waits for the body to happen to face the right way.
        self.waist = waist
        # WHETHER `ang_vel_yaw` IS WORTH COMMANDING AT ALL. Mrinal's climb policy
        # was trained with ang_vel_yaw ~ 0 and the palm is clipped to a fixed
        # line, so the measured yaw authority is nil: PARITY.md records +1.0 and
        # -1.0 rad/s for 3 s ending 10 deg apart. With this False -- the DEFAULT,
        # and what every climb world gets -- `ang_vel_yaw` is held at zero in
        # EVERY guide state. It is True only for a policy trained to turn.
        self.yaw_command_available = bool(yaw_command_available)
        self.mode = "LOST"
        self.seconds_since_detection = LOST_AFTER_SECONDS
        self.range_meters = None
        self.bearing_radians = None

    def reset(self) -> None:
        self.mode = "LOST"
        self.seconds_since_detection = LOST_AFTER_SECONDS
        self.range_meters = None
        self.bearing_radians = None
        if self.waist is not None:
            self.waist.reset()

    # ------------------------------------------------------------- geometry
    def body_bearing_radians(self):
        """Where she is relative to the BODY, not the camera. -> radians or None.

        The detector reports a bearing in the IMAGE, which is relative to
        wherever the cameras happen to be pointing -- and the ear layer may have
        pointed them somewhere. The direction to her in the robot's own frame is
        therefore `theta_waist + beta`. Using the image bearing alone would be
        60 degrees wrong whenever the waist was 60 degrees out.
        """
        if self.bearing_radians is None:
            return None
        # THE MEASURED waist angle, not the commanded offset -- see
        # `WaistYaw.measure`. Using the command here is a windup loop.
        waist_angle = self.waist.measured_radians if self.waist is not None else 0.0
        return waist_angle + self.bearing_radians

    # --------------------------------------------------------- the machine
    def update(self, measurement, dt_seconds: float) -> None:
        """`measurement` is None on a tick with no fresh vision (the eyes run at
        10 Hz, the loop at 50): the state persists and the clocks run."""
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
            # THE WAIST IS NOT TOUCHED HERE. It belongs to the ear layer while
            # she is lost, and driving it to zero every tick would undo the aim
            # a shout just gave it.
        else:
            self._follow_or_wait()

        if self.waist is not None:
            self.waist.advance(dt_seconds)

    def _follow_or_wait(self) -> None:
        distance = self.range_meters
        if distance is None:
            self.mode = "LOST"
            return
        if self.mode == "FOLLOW":
            self.mode = "FOLLOW" if distance > WAIT_RANGE_METERS else "WAIT"
        else:
            self.mode = "FOLLOW" if distance > FOLLOW_RANGE_METERS else "WAIT"
        # SHE IS IN FRONT, SO THE WAIST UNWINDS. It must NOT keep tracking her
        # here, tempting as that is. Tried: with the palm clipped to the rope,
        # twisting the waist counter-rotates the PELVIS, so the image bearing
        # never closes and the waist chases it straight to the limit. On flat_0
        # that is a fall at 1.9 s.
        if self.waist is not None:
            self.waist.target_radians = 0.0

    # ------------------------------------------------------------ the output
    def command(self) -> np.ndarray:
        """-> (3,) [lin_vel_x, lin_vel_y, ang_vel_yaw], the policy's own layout.

        LOST and WAIT command ZERO. A robot that has lost the person it is
        following has no business walking off on its own -- the ear layer is
        what gives it somewhere to go.
        """
        if self.mode != "FOLLOW":
            if self.rope_climb and self.mode == "LOST":
                return np.array([self.command_speed, 0.0, 0.0])
            return np.zeros(3)
        if self.vector_steering:
            # The BODY bearing, not the image bearing: the ear layer may have
            # panned the waist, and the legs live in the body frame.
            bearing = self.body_bearing_radians()
            if bearing is None:
                bearing = 0.0
            return vector_command(bearing, self.command_speed)
        return np.array([self.command_speed, 0.0,
                         self._yaw_rate(self.bearing_radians)])

    def _yaw_rate(self, bearing) -> float:
        """Zero unless the policy can actually turn. See `yaw_command_available`."""
        if not self.yaw_command_available or bearing is None:
            return 0.0
        if abs(bearing) < BEARING_DEADBAND_RADIANS:
            return 0.0
        return float(np.clip(BEARING_GAIN_PER_RADIAN * bearing,
                             -MAXIMUM_YAW_RATE_RADIANS_PER_SECOND,
                             MAXIMUM_YAW_RATE_RADIANS_PER_SECOND))


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

    def __init__(self, scene, model, control_hz, verbose=True, enable=True,
                 degradation=None, yaw_command_available=False,
                 free_walk=False, vector_steering=True):
        import mujoco
        self._mujoco = mujoco
        # A legacy world has no MjSpec to operate on -- their old env hands back
        # a compiled model and nothing else -- so the guide is simply not
        # available there and every entry point below turns into a no-op.
        # `enable=False` is the same no-op on purpose, for the parity runs that
        # need a model with no guide bodies in it at all.
        self.available = (enable and scene is not None
                          and attach_guide(scene, verbose=verbose))
        self.model = scene.model if self.available else model
        self.control_hz = float(control_hz)
        self.dt_seconds = 1.0 / self.control_hz
        self.enabled = False
        self.guide = None
        self.eyes = None
        # The "neck": a waist-yaw offset injected into the policy's own PD
        # target so the robot can look around without turning. Bound below,
        # once the model is known.
        self.waist = WaistYaw()
        self.follower = GuideFollower(
            waist=self.waist, yaw_command_available=yaw_command_available,
            vector_steering=vector_steering,
            rope_climb=not free_walk)
        self.latest = None                 # the last vision measurement
        self.eye_jpeg = None
        self.true_range_meters = float("nan")
        self.vision_milliseconds = 0.0
        if not self.available:
            return
        self.guide = Guide(scene.route, scene.terrain, self.model,
                           free_walk=free_walk)
        self.waist.bind(self.model, verbose=verbose)
        self.eyes = StereoEyes(self.model, verbose=verbose,
                               degradation=degradation)
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

    def update(self, data, tick: int, enabled: bool, walking: bool,
               backing: bool = False, turning: float = 0.0,
               eyes_enabled: bool = True) -> np.ndarray | None:
        """One control tick. -> the command to fly, or None if the guide is off.

        Order matters: the human moves, its body is written, and only THEN are
        the cameras rendered, so the robot sees where the human is now rather
        than where it was a tick ago.

        `walking` is W, `backing` is S and `turning` is A/D, and they are all
        the HUMAN's keys, not the robot's: W walks her, S backs her up, and on a
        rope-off world A and D turn her. What the robot does about any of it is
        the follower's business.
        """
        if not self.available:
            return None
        was_enabled, self.enabled = self.enabled, bool(enabled)
        self.guide.enabled = self.enabled
        if not self.enabled:
            self.guide.write(self.model, data)
            # UNWIND THE WAIST, and keep advancing it. Switching the guide off
            # mid-aim would otherwise leave the robot standing permanently
            # twisted with nothing driving it back -- the offset is added to the
            # policy's target every substep whether the guide is on or not.
            self.follower.reset()
            self.waist.target_radians = 0.0
            self.waist.advance(self.dt_seconds)
            self.latest = None
            return None
        if not was_enabled:
            self.place(np.asarray(data.qpos[0:3]))
            self.guide.enabled = True

        # Read the waist BEFORE anything decides anything: every bearing the
        # follower is about to compute is relative to where the cameras really
        # are, which is this number.
        self.waist.measure(data)
        self.guide.advance(self.dt_seconds, walking, backing, turning)
        self.guide.write(self.model, data)
        # `mocap_pos` is an INPUT to the forward kinematics, not a pose: nothing
        # in `data` moves until something recomputes frames from it, and the
        # renderer reads `geom_xpos`. Without the two calls below the eyes would
        # see the human a tick behind where it is.
        #
        # AND THEY ARE UNDONE AGAIN, which is the whole reason `_restore` exists
        # and is not defensive programming. `mj_step` is forward-then-integrate,
        # so when it returns, `data.qpos` is the NEW state while `data.xpos` and
        # friends still describe the OLD one -- and the next control tick reads
        # those stale frames (the ascender's carrier projection and the
        # observation both do). Refreshing them for the cameras therefore hands
        # the next step a different, fresher world than it would have had, and
        # MEASURED on flat_0 that moved the robot 0.95 rad of joint angle in six
        # seconds. So the frames are refreshed for the cameras and then put back
        # exactly as they were, and the physics sees precisely what it saw
        # before the guide existed. `test_guide`'s section D is this claim.
        frozen = self._freeze(data)
        self._mujoco.mj_kinematics(self.model, data)
        self._mujoco.mj_camlight(self.model, data)
        self.true_range_meters = self.true_range_to_guide(data)

        # EYES OFF (user's ruling, 2026-08-30: an option to disable eyes): the
        # stereo pair is not even rendered -- the robot is genuinely blind, not
        # ignoring what it saw -- and the follower is snapped to LOST at once so
        # the ear layer owns the approach. The stale-range trap this avoids:
        # with no fresh non-detection arriving, `seconds_since_detection` would
        # freeze and the follower would FOLLOW a remembered range forever.
        self.eyes_enabled = bool(eyes_enabled)
        if not self.eyes_enabled:
            self.latest = None
            self.follower.seconds_since_detection = max(
                self.follower.seconds_since_detection, LOST_AFTER_SECONDS)
        measurement = None
        if self.eyes_enabled and tick % EYE_RENDER_EVERY_N_TICKS == 0:
            import time
            started = time.time()
            measurement = self.eyes.look(data)
            self.latest = measurement
            self.vision_milliseconds = (time.time() - started) * 1000.0
        self._restore(data, frozen)
        self.follower.update(measurement, self.dt_seconds)
        if measurement is not None:
            self.eye_jpeg = EYE_MESSAGE_PREFIX + annotate_eye(
                measurement["left_image"], measurement["box"], self.label_text())
        return self.follower.command()

    @staticmethod
    def _freeze(data) -> dict:
        """Copy every array `mj_kinematics`/`mj_camlight` write. -> name -> array.

        Exactly the two functions' output fields and nothing else: joint anchors
        and axes, body/inertial/geom/site frames, then camera and light frames.
        `data.qpos`, `qvel`, `ctrl` and every force are deliberately absent --
        the guide never writes them, so there is nothing to put back.
        """
        return {name: np.array(getattr(data, name), copy=True)
                for name in KINEMATICS_OUTPUT_FIELDS}

    @staticmethod
    def _restore(data, frozen) -> None:
        for name, values in frozen.items():
            getattr(data, name)[:] = values

    def take_eye_jpeg(self):
        """The newest eye frame, ONCE. -> bytes or None.

        The eyes run at 10 Hz and the loop at 50, so four ticks in five have no
        new picture. Returning it once and clearing is what keeps the websocket
        from re-sending the same frame five times.
        """
        jpeg, self.eye_jpeg = self.eye_jpeg, None
        return jpeg

    def label_text(self) -> str:
        """The line drawn on the eye frame. Carries the waist angle whenever it
        is off centre, because a robot looking sideways is unreadable
        without it."""
        mode = self.follower.mode
        if abs(self.waist.degrees) > 1.0:
            mode += f" {self.waist.degrees:+.0f}°"
        if self.follower.range_meters is None:
            return f"-- m · {mode}"
        return f"{self.follower.range_meters:.1f} m · {mode}"

    def state(self) -> dict:
        """The `guide` block of the websocket state message."""
        if not self.available:
            return {"enabled": False, "mode": "LOST",
                    "waist_yaw_degrees": 0.0,
                    "distance_meters": None, "bearing_degrees": None,
                    "true_distance_meters": None, "human_progress_meters": 0.0,
                    "free_walk": False, "eyes": False}
        bearing = self.follower.bearing_radians
        return {
            "enabled": bool(self.enabled),
            "mode": self.follower.mode,
            # What the waist is actually doing, AFTER the rate limit -- the
            # number the HUD should believe, not the target. The ear layer is
            # what aims it now.
            "waist_yaw_degrees": round(self.waist.degrees, 2),
            "distance_meters": (None if self.follower.range_meters is None
                                else round(float(self.follower.range_meters), 3)),
            "bearing_degrees": (None if bearing is None
                                else round(math.degrees(bearing), 2)),
            # LABELLED CHEAT: read from the simulator, HUD only.
            "true_distance_meters": (None if not np.isfinite(self.true_range_meters)
                                     else round(float(self.true_range_meters), 3)),
            "human_progress_meters": round(
                float(self.guide.travel_meters), 3),
            # Which keys drive HER: on a rope-off world A and D turn her.
            "free_walk": bool(self.available and self.guide.free_walk),
            "eyes": bool(getattr(self, "eyes_enabled", True)),
        }

    def recorded(self) -> dict:
        """The columns `Recorder` stacks into frames.npz / hud.json.

        All floats, because `Recorder.append` stacks float arrays -- hence the
        code tables rather than the strings the websocket carries.
        """
        distance = self.follower.range_meters
        true_range = self.true_range_meters
        return {
            "guide_mode": float(GUIDE_MODE_CODES[self.follower.mode]),
            "guide_waist_yaw_degrees": float(self.waist.degrees),
            "guide_realign_body_yaw": (
                1.0 if self.follower.yaw_command_available else 0.0),
            "guide_distance_meters": (NO_MEASUREMENT if distance is None
                                      else float(distance)),
            "guide_true_distance_meters": (
                NO_MEASUREMENT if not (self.enabled and np.isfinite(true_range))
                else float(true_range)),
            "guide_human_progress_meters": (
                float(self.guide.travel_meters) if self.available else 0.0),
        }
