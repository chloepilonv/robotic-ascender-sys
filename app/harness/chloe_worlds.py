"""Chloe's rope-ascender worlds: the plant her policy was trained in.

    Chloe v1 · 20° · rope   chloe_v1_20     (the whole ladder: chloe_v1_{0..40})

These two are the ONLY worlds in the harness where the walking policy does not
fly the robot. Picking one of them loads
`chloe_policy.AscenderController` -- her mjlab-trained network -- and the plant
underneath it is hers too, rebuilt here from plain MuJoCo:

  robot       assets/robots/mujoco/g1_unitree_ascender.xml, its own actuators
              and keyframes DELETED and replaced with mjlab's articulation
              (kp/kd/armature per motor group, `chloe_policy.G1_ARTICULATION`)
  rope        assets/robots/mujoco/rope_rail.py -- ONE straight non-colliding
              line 0.60 m above the ground, an ascender WELDED (mjEQ_WELD) to a
              1-DoF `rope_carriage` slide, and a ratchet that is the slide's
              moving lower limit
  ground      one plane, no roughness, no heightfield
  rates       physics 0.005 s, decimation 4, control 50 Hz

NONE OF THAT IS NEGOTIABLE and none of it is ours. Measured, both directions:
give her policy the harness's walking gains (kp 75 / kd 2) and it falls in
under two seconds; give it a flat 0.5 action scale instead of her per-joint
table and it falls in under two seconds. The usable slope band is 10-30
degrees, and these worlds sit at 20 and 25.

THE FRAME, and why there are two of them
----------------------------------------
She trained on a FLAT plane with GRAVITY TILTED by the slope: world +x is
"uphill" because that is the direction gravity does not pull, and the robot's
spawn attitude is a rotation about +y by the slope angle, which puts its
projected gravity at exactly (0, 0, -1). That is `frame="tilted_gravity"`, and
it reproduces `rl/scripts/sim2sim.py` exactly.

It also LOOKS wrong: on the 3-D page the ground is level, the rope is
horizontal, the robot leans 20 degrees backwards, and "height gained" reads
0.00 m forever while the robot visibly climbs nothing.

So the shipped worlds use `frame="tilted_plane"`: the WHOLE WORLD -- floor,
rope, carriage, robot -- rigidly rotated by -slope about +y, and gravity put
back to (0, 0, -9.81). This is a change of coordinates and nothing else, and
every quantity the policy reads is invariant under it:

    pelvis angular velocity   qvel[3:6] is already in the BODY frame
    projected gravity         R(q_R q)^T (R g) = R(q)^T g
    joint angles / velocities  untouched by a world rotation
    carriage - pelvis, in the pelvis frame   same cancellation

The weld and the slide survive it for free: the carriage BODY is rotated with
everything else, so the weld's relative pose (expressed in the carriage frame)
and the slide's axis (expressed in the carriage frame) are unchanged numbers.
Only floating-point rounding differs, and `python -m app.harness.chloe_worlds
--equivalence` runs both frames and prints both 15-second numbers so the claim
is measured rather than asserted.

WHAT THE PAGE GETS OUT OF IT: a slope that rises, a rope that climbs it, a
robot standing upright on it, honest metres of height gained, and the snow /
sky / flag / hiker / storm layers working exactly as they do on the Lhotse
worlds.

STOP AND GO (user's ruling 2026-08-30). Her network has no command input, so
W does not steer -- it gates. W held runs the network; W released freezes the
PD targets where the policy left them and lets the ratchet hold the body on
the line. A and D and the mouse heading do nothing at all on these worlds, and
the guide follower, when it is on, may drive the same go/stop gate but can
never steer. See `chloe_policy.AscenderController.go`.

Inputs  : a world name from `CHLOE_WORLD_DEFINITIONS`.
Outputs : `ChloeSceneLibrary.load(name)` -> (scene, meta, definition);
          `ChloeAscenderEpisode` presents the interface `runtime.run` drives --
          `step(command, wind)` -> a row dict, `reset()`, the same attributes
          `ClimbSceneEpisode` carries.
"""

from __future__ import annotations

import math
import os

import numpy as np

from app.harness import chloe_policy
from app.harness import scripted_ascender

_HARNESS_DIRECTORY = os.path.dirname(os.path.abspath(__file__))
REPOSITORY_ROOT = os.path.dirname(os.path.dirname(_HARNESS_DIRECTORY))

ROBOT_XML = os.path.join(REPOSITORY_ROOT, "assets", "robots", "mujoco",
                         "g1_unitree_ascender.xml")

# Her sim2sim numbers: physics 5 ms, decimation 4, solver 10/20.
PHYSICS_TIMESTEP_SECONDS = 0.005
SOLVER_ITERATIONS = 10
SOLVER_LINE_SEARCH_ITERATIONS = 20
GRAVITY_MAGNITUDE = 9.81
CONTROL_DT_SECONDS = chloe_policy.CONTROL_DT_SECONDS

# Her reset pose (rl/task/robot.py BASE_JOINT_POS), as the mjlab
# regex->radians map `rope_rail.set_pose` consumes.
BASE_JOINT_POSITIONS_RADIANS = {
    ".*_hip_pitch_joint": -0.312,
    ".*_knee_joint": 0.669,
    ".*_ankle_pitch_joint": -0.363,
    ".*_elbow_joint": 0.6,
    "left_shoulder_roll_joint": 0.2,
    "left_shoulder_pitch_joint": 0.2,
    "right_shoulder_roll_joint": -0.2,
    "right_shoulder_pitch_joint": 0.2,
}
# The ankles pitch further with the slope so the soles lie flat on it, floored
# at -0.85 rad by the joint's own range.
ANKLE_PITCH_FLOOR_RADIANS = -0.85
SPAWN_CLEARANCE_METERS = 0.005     # lift the lowest foot sphere this far clear
NOMINAL_SPAWN_HEIGHT_METERS = 0.8  # the height the fit starts from

# The ground. Physics is an infinite half-space whatever this says -- a MuJoCo
# plane's size is a RENDER extent only -- so these numbers buy the page a
# finite snowfield and the exporter a finite quad, and cost the solver nothing.
GROUND_HALF_LENGTH_METERS = 40.0   # along the rope
GROUND_HALF_WIDTH_METERS = 20.0
GROUND_FRICTION = 0.8              # the XML's own foot friction

FOOT_GEOM_CLASS = "foot"
ROPE_RENDER_GEOM_NAME = "ropeseg0"  # `export_scene.rope_polyline` finds the
                                    # rope by this prefix, and the page draws
                                    # its polyline from it. `rope_rail` calls
                                    # it `rope_geom`; renaming it in the spec
                                    # is cheaper than teaching the exporter a
                                    # second convention, and nothing reads the
                                    # old name.

SLOPE_BAND_DEGREES = (10.0, 30.0)   # measured: outside this she does not climb


def _definition(name, slope_degrees, label, description):
    return name, {
        "kind": "chloe_ascender",
        "label": label,
        "patch": None,
        "robot": "ascender",
        "rope": True,
        "slope_degrees": float(slope_degrees),
        "slope_provenance": "her training slope, exact",
        "description": description,
        "terrain_factory": None,
        "autonomous": True,
    }


# Her policies, as the APP numbers them (user ruling 2026-08-30: "classify
# this as v1 of Chloe's policy"). Her own file names carry her training-run
# counter (this one is her `..._v3_...`); the app counts the checkpoints it
# has actually shipped, starting at v1. Add a row per new checkpoint; the
# world keys and labels pick the version up automatically.
CHLOE_POLICY_VERSIONS = {
    "v1": {
        "relative_path": os.path.join(
            "rl", "policies",
            "g1_ascender_slope20_v3_2026-08-30_04-35-59.onnx"),
        "trained_slope_degrees": 20.0,
        "note": "her run v3 of 2026-08-30 04:35; mjlab PPO, 96-d obs, no command",
    },
    "m1": {
        "relative_path": os.path.join(
            "rl", "policies",
            "g1_ascender_slope20_mrinal_working.onnx"),
        "trained_slope_degrees": 20.0,
        "display": "Mrinal",
        "note": "Mrinal's checkpoint (branch rl-training-2, iter 7998, trained"
                " WITHOUT the climb-mode bit -> 96-d obs; his training rope did"
                " not collide, ours does -- transfer measured anyway: 15 s"
                " uphill 5.73/4.43/1.88 m at 10/20/30 deg, upright ~0.96,"
                " wind-tolerant). ONNX derived from the .pt with plain torch"
                " (normaliser baked in), verified vs torch to 3.8e-06.",
        "slopes": (10, 20, 30),
    },
    "m2": {
        "relative_path": os.path.join(
            "rl", "policies",
            "g1_ascender_slope20_final_2026-08-30_13-55-14.onnx"),
        "trained_slope_degrees": 20.0,
        "display": "Mrinal 2",
        "note": "Mrinal's FINAL checkpoint (main, iter 12997, trained with the"
                " domain-randomisation guardrails of PR #20; same 96-d obs and"
                " 512/256/128 ELU actor as m1). ONNX derived from the .pt with"
                " plain torch (normaliser baked in), verified vs torch to"
                " 5.7e-06. MEASURED 2026-08-30 (15 s, calm, tilted_plane):"
                " uphill 6.06/6.19/5.93/5.42/4.87 m at 20/25/30/35/40 deg,"
                " STANDING at every rung, up_z 0.87-0.96, ratchet gap"
                " <= 0.11 cm -- the first checkpoint that climbs 35 and 40.",
        "slopes": (20, 25, 30, 35, 40),
    },
    "v2": {
        "relative_path": os.path.join(
            "rl", "policies",
            "g1_ascender_slope20_v7_2026-08-30_17-16-35.onnx"),
        "trained_slope_degrees": 20.0,
        "note": "her run v7 of 2026-08-30 17:16; 97-d obs = v3's + the WALK/SLIDE"
                " mode bit driven by the climb-mode FSM; final rope_rail."
                " MEASURED 2026-08-30: stands still at 10/20/30 deg, either"
                " start mode -- the loophole her v8 run is fixing. Shipped so"
                " the team can see it; not the default.",
        "slopes": (0, 20),
    },
}
CURRENT_CHLOE_VERSION = "v1"

# The slope ladder (user ruling 2026-08-30: "run Chloe's policy with varying
# degrees of slope, but no uneven ground, just a flat slope"). Every rung is
# the same plant -- flat plane, one straight rope 0.60 m up, her gains and
# scales -- with only the slope changed. She trained at 20; the measured band
# she climbs in is 10-30 (SLOPE_BAND_DEGREES); rungs outside it are kept ON
# PURPOSE so the failure is visible in the app, not just in a table.
CHLOE_SLOPE_LADDER_DEGREES = (0, 20, 30)   # trimmed to the demo rungs (user, 2026-08-30)


def _display_name(version):
    return CHLOE_POLICY_VERSIONS[version].get("display", f"Chloe {version}")


def _ladder_description(version, slope):
    trained = CHLOE_POLICY_VERSIONS[version]["trained_slope_degrees"]
    display = _display_name(version)
    if slope == trained:
        return (f"{display}'s mjlab rope-ascender policy on the plant it"
                " was trained in: one straight line 0.60 m up, an ascender"
                f" welded to it, and a {slope:g} degree slope. W gates it;"
                " nothing steers it.")
    low, high = SLOPE_BAND_DEGREES
    where = ("inside" if low <= slope <= high else "OUTSIDE")
    return (f"The same {display} policy and the same flat plant at {slope:g}"
            f" degrees -- {where} the measured {low:g}-{high:g} degree band"
            f" (trained at {trained:g}).")


def _versioned_definition(version, slope):
    name, definition = _definition(
        f"chloe_{version}_{slope:g}", float(slope),
        f"{_display_name(version)} · {slope:g}° · rope",
        _ladder_description(version, slope))
    definition["policy_version"] = version
    definition["policy_relative_path"] = CHLOE_POLICY_VERSIONS[version]["relative_path"]
    return name, definition


CHLOE_WORLD_DEFINITIONS = dict([
    _versioned_definition(version, slope)
    for version in CHLOE_POLICY_VERSIONS
    for slope in CHLOE_POLICY_VERSIONS[version].get(
        "slopes", CHLOE_SLOPE_LADDER_DEGREES)
])

DEFAULT_CHLOE_WORLD = f"chloe_{CURRENT_CHLOE_VERSION}_20"


# --------------------------------------------------- the scripted-gait ladder
# THE SAME PLANT, NO NETWORK (user ruling 2026-08-30: "a default deterministic
# non-ML walking policy on the rope ... right foot, left foot, right hand
# slides up"). Same flat slope, same one straight rope 0.60 m up, same weld and
# same ratchet -- the only differences are the controller
# (`scripted_ascender.ScriptedAscenderController`) and stiffer leg gains, which
# a script may have because no network was trained around the soft ones.
SCRIPTED_SLOPE_LADDER_DEGREES = (0, 10, 20, 30)
SCRIPTED_POLICY_VERSION = "scripted"


def _scripted_measured(slope):
    """What the gait actually did there, 15 s, calm, friction 0.8.

    Measured 2026-08-30 by `scripted_ascender._matrix`, and quoted rather than
    promised: the rungs that do not climb are kept in the app ON PURPOSE, the
    same way Chloe's out-of-band slopes are, so the failure is visible on the
    page and not only in a table.
    """
    return {
        0.0: "climbs 1.48 m in 15 s (1.58 m of rope)",
        10.0: "climbs 0.68 m in 15 s (0.80 m of rope)",
        20.0: "does NOT climb -- holds the line but slides 0.09 m downhill",
        30.0: "does NOT climb -- holds the line but slides 0.39 m downhill",
    }[float(slope)]


def _scripted_definition(slope):
    name, definition = _definition(
        f"scripted_{slope:g}", float(slope),
        f"Scripted · {slope:g}° · rope",
        "A hand-written quasi-static climbing gait -- no network anywhere."
        " The ascender is welded to the rope, so the right wrist is pinned to a"
        " straight line and the weld carries most of the weight; the feet are"
        " lightly loaded and skate, which is why the left foot only DRAGS"
        " (2 cm of clearance) while the right one steps."
        f" Right foot steps, left foot drags, the hand slides up. At {slope:g}"
        f" degrees it {_scripted_measured(slope)}. W gates it; nothing steers"
        " it.")
    definition["policy_version"] = SCRIPTED_POLICY_VERSION
    definition["policy_relative_path"] = None
    definition["slope_provenance"] = "chosen for the demo (nothing was trained)"
    return name, definition


SCRIPTED_WORLD_DEFINITIONS = dict([
    _scripted_definition(slope) for slope in SCRIPTED_SLOPE_LADDER_DEGREES])

CHLOE_WORLD_DEFINITIONS.update(SCRIPTED_WORLD_DEFINITIONS)

DEFAULT_SCRIPTED_WORLD = "scripted_20"


# --------------------------------------------------------------- the frames
def slope_quaternion(slope_degrees) -> np.ndarray:
    """Her spawn attitude: a rotation about world +y by the slope angle."""
    half = math.radians(float(slope_degrees)) / 2.0
    return np.array([math.cos(half), 0.0, math.sin(half), 0.0])


def gravity_for_slope(slope_degrees) -> np.ndarray:
    """Her tilted gravity: down-slope is -x, so uphill is +x."""
    slope = math.radians(float(slope_degrees))
    return np.array([-GRAVITY_MAGNITUDE * math.sin(slope), 0.0,
                     -GRAVITY_MAGNITUDE * math.cos(slope)])


def world_rotation_quaternion(slope_degrees, frame: str) -> np.ndarray:
    """The rigid rotation applied to HER build to get the shipped world.

    `tilted_gravity` is identity -- her build, untouched. `tilted_plane` is a
    rotation about +y by MINUS the slope, which stands the robot up, tips the
    floor into a slope rising toward +x, and turns her tilted gravity back into
    plain (0, 0, -9.81).
    """
    if frame == "tilted_gravity":
        return np.array([1.0, 0.0, 0.0, 0.0])
    if frame != "tilted_plane":
        raise ValueError(f"frame must be tilted_plane or tilted_gravity, not {frame!r}")
    half = math.radians(float(slope_degrees)) / 2.0
    return np.array([math.cos(half), 0.0, -math.sin(half), 0.0])


def rotation_matrix(quaternion_wxyz) -> np.ndarray:
    import mujoco
    matrix = np.zeros(9)
    mujoco.mju_quat2Mat(matrix, np.asarray(quaternion_wxyz, dtype=float))
    return matrix.reshape(3, 3)


def quaternion_product(left_wxyz, right_wxyz) -> np.ndarray:
    import mujoco
    out = np.zeros(4)
    mujoco.mju_mulQuat(out, np.asarray(left_wxyz, dtype=float),
                       np.asarray(right_wxyz, dtype=float))
    return out


# ------------------------------------------------------------- the ingredients
def body_joint_positions(slope_degrees) -> dict:
    """Her reset pose for one slope: the ankles pitch with the ground."""
    positions = dict(BASE_JOINT_POSITIONS_RADIANS)
    positions[".*_ankle_pitch_joint"] = max(
        ANKLE_PITCH_FLOOR_RADIANS, -0.363 - math.radians(float(slope_degrees)))
    return positions


def _spec_body(spec, name):
    """`MjSpec.body(name)` by hand.

    `spec.body(name)` resolves names the COMPILED model knows about, and the
    rope and the carriage are added to the spec after the last compile -- it
    hands back None for both. Walking `spec.bodies` is the answer that does not
    depend on when the last compile happened.
    """
    for body in spec.bodies:
        if body.name == name:
            return body
    raise KeyError(f"no body {name!r} in the spec;"
                   f" have {[b.name for b in spec.bodies][:8]}...")


def base_spec():
    """Her robot with its own actuators and keyframes stripped out.

    The XML ships PLAYGROUND-style actuators and a keyframe; mjlab supplies its
    own, so hers go and the mjlab table goes on in `build_plant`. The foot
    spheres are NAMED on the way past, because the friction knob, the footstep
    detector and the pose stream all address feet by geom id and there is
    nothing else to look them up by.
    """
    import mujoco
    spec = mujoco.MjSpec.from_file(ROBOT_XML)
    for actuator in list(spec.actuators):
        spec.delete(actuator)
    for keyframe in list(spec.keys):
        spec.delete(keyframe)
    for side in ("left", "right"):
        index = 0
        for geom in spec.body(f"{side}_ankle_roll_link").geoms:
            if geom.classname is not None and geom.classname.name == FOOT_GEOM_CLASS:
                geom.name = f"{side}_foot_{index}"
                index += 1
    return spec


def fit_spawn_height(slope_degrees) -> np.ndarray:
    """Her `robot.init_pos`: drop the base until the lowest foot sphere grazes.

    Done on a THROWAWAY compile in her flat frame, exactly as she does it: the
    floor is at z = 0 there, so the correction is one subtraction rather than a
    search.
    """
    import mujoco
    rope_rail = chloe_policy.rope_rail_module()
    model = base_spec().compile()
    data = mujoco.MjData(model)
    rope_rail.set_pose(model, data, (0.0, 0.0, NOMINAL_SPAWN_HEIGHT_METERS),
                       slope_quaternion(slope_degrees),
                       body_joint_positions(slope_degrees))
    foot_geoms = [i for i in range(model.ngeom)
                  if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i) or "")
                  .startswith(("left_foot_", "right_foot_"))]
    if not foot_geoms:
        raise RuntimeError("no named foot spheres; base_spec did not name them")
    lowest = min(float(data.geom_xpos[i][2] - model.geom_size[i][0])
                 for i in foot_geoms)
    height = NOMINAL_SPAWN_HEIGHT_METERS - lowest + SPAWN_CLEARANCE_METERS
    return np.array([0.0, 0.0, height]), len(foot_geoms)


# ------------------------------------------------------- terrain / route stubs
class SlopeGround:
    """A flat slope, presenting the two things the harness asks a terrain for.

    `guide.Guide` snaps the hiker's boots to `surface_z(x, y)`, and `snow` /
    `graphics` size their texture and their fog from `size_xy`. There is no
    heightfield here and no roughness: it is one plane, and the honest answer
    to "what is the ground under (x, y)" is a plane's answer.
    """

    def __init__(self, slope_degrees, frame):
        self.slope_deg = float(slope_degrees)
        self.frame = frame
        self.name = f"chloe_plane_slope{self.slope_deg:g}_{frame}"
        self.source = "plane (no DEM, no roughness) -- her training ground"
        self.size_xy = (2.0 * GROUND_HALF_LENGTH_METERS,
                        2.0 * GROUND_HALF_WIDTH_METERS)
        self.res = 1.0
        self.rough = np.zeros((2, 2))
        self.shape = self.rough.shape
        # In her own frame the ground IS level and gravity is tilted, so the
        # surface is z = 0; in the shipped frame the world was rotated and the
        # same plane rises toward +x at exactly the slope angle.
        self._gradient = (0.0 if frame == "tilted_gravity"
                          else math.tan(math.radians(self.slope_deg)))

    def surface_z(self, x, y):
        return np.asarray(x, dtype=float) * self._gradient

    @property
    def slope_rad(self) -> float:
        return math.radians(self.slope_deg)


class SlideAscender:
    """The ratchet's own state, read off the `rope_slide` joint.

    `ClimbScene` rides a mocap bead along a draped polyline and has to project
    it back on every substep. This rope is ONE STRAIGHT LINE and the carriage
    is a real 1-DoF slide welded to the hand, so there is nothing to project:
    the joint coordinate IS the arc length, and the ratchet is its moving lower
    limit. `bind` exists because `graphics.add_skybox`, `snow.attach_snow` and
    `guide.attach_guide` each recompile the spec and hand the scene a new
    model.
    """

    def __init__(self, model, mujoco_module, rope_tail_meters):
        self.rope_tail_meters = float(rope_tail_meters)
        self.ratchet = True
        self.s0 = 0.0
        self.bind(model, mujoco_module)

    def bind(self, model, mujoco_module) -> None:
        rope_rail = chloe_policy.rope_rail_module()
        self.model = model
        self.joint_id = int(model.joint(rope_rail.SLIDE_JOINT).id)
        self.qpos_address = int(model.jnt_qposadr[self.joint_id])
        self.dof_address = int(model.jnt_dofadr[self.joint_id])

    def slide_meters(self, data) -> float:
        return float(data.qpos[self.qpos_address])

    def arclength_meters(self, data) -> float:
        """Distance from the rope's BOTTOM anchor, so it matches `route`."""
        return self.rope_tail_meters + self.slide_meters(data)

    def progress_meters(self, data) -> float:
        return self.slide_meters(data) - self.s0


# ------------------------------------------------------------------ the plant
class ChloeScene:
    """Her compiled plant, with the handles the harness's decorators expect.

    Duck-compatible with `rl.environment.climb_scene.ClimbScene` in exactly the
    places the app touches: `.spec`, `.model`, `.data`, `.terrain`, `.route`,
    `.ascender.bind(model, mujoco)`, `.reset()`, `.step(wind)`. That is the
    whole contract `graphics.add_skybox`, `snow.attach_snow` and
    `guide.attach_guide` use, and it is why the snow, the sky, the wind flag,
    the storm and the walking hiker all work here with no special case.
    """

    def __init__(self, slope_degrees, frame="tilted_plane", verbose=True,
                 gain_scale=None, warn_outside_band=True):
        import mujoco
        rope_rail = chloe_policy.rope_rail_module()

        self._mujoco = mujoco
        self._rope_rail = rope_rail
        self.slope_degrees = float(slope_degrees)
        self.frame = frame
        # OPTIONAL STIFFER ACTUATORS, for a controller that is not a network.
        # `gain_scale` maps a joint-name suffix (the `G1_ARTICULATION` keys) to
        # (stiffness multiplier, damping multiplier). None -- the default --
        # leaves the mjlab row exactly as her policy was trained with, so every
        # existing world is bit-identical to before this argument existed.
        self.gain_scale = dict(gain_scale) if gain_scale else {}
        low, high = SLOPE_BAND_DEGREES
        if warn_outside_band and not low - 1e-9 <= self.slope_degrees <= high + 1e-9:
            print(f"[chloe] WARNING slope {self.slope_degrees} deg is outside the"
                  f" measured {low:g}-{high:g} deg band this policy climbs in",
                  flush=True)

        her_spawn_position, foot_sphere_count = fit_spawn_height(self.slope_degrees)
        her_spawn_quaternion = slope_quaternion(self.slope_degrees)
        joint_positions = body_joint_positions(self.slope_degrees)

        # --- the rope rail, solved in HER frame ----------------------------
        # `add_rope_rail` assumes the rope runs along +x at `rope_height` above
        # a floor at z = 0 and asserts the solved channel is parallel to it, so
        # the IK has to happen before any world rotation. Everything it builds
        # is rotated afterwards, rigidly.
        spec = base_spec()
        self.default_joint_positions = rope_rail.add_rope_rail(
            spec, her_spawn_position, her_spawn_quaternion, joint_positions)

        # --- mjlab's articulation, replacing the XML's own -----------------
        hinges = [j for j in spec.joints
                  if j.type == mujoco.mjtJoint.mjJNT_HINGE]
        for joint in hinges:
            stiffness, damping, armature, _scale = chloe_policy.articulation_for(joint.name)
            for suffix, (stiffness_multiplier, damping_multiplier) in self.gain_scale.items():
                if joint.name.endswith(suffix + "_joint"):
                    stiffness *= float(stiffness_multiplier)
                    damping *= float(damping_multiplier)
                    break
            joint.armature = armature
            joint.damping = [0.0, 0.0, 0.0]
            joint.frictionloss = 0.0
            actuator = spec.add_actuator(name=joint.name, target=joint.name,
                                         trntype=mujoco.mjtTrn.mjTRN_JOINT)
            actuator.gaintype = mujoco.mjtGain.mjGAIN_FIXED
            actuator.biastype = mujoco.mjtBias.mjBIAS_AFFINE
            actuator.gainprm = [stiffness] + [0.0] * 9
            actuator.biasprm = [0.0, -stiffness, -damping] + [0.0] * 7
            actuator.ctrlrange = [-1e6, 1e6]
            actuator.ctrllimited = 0

        # --- the world rotation --------------------------------------------
        rotation_quaternion = world_rotation_quaternion(self.slope_degrees, frame)
        rotation = rotation_matrix(rotation_quaternion)
        self.spawn = rotation @ her_spawn_position
        self.spawn_quaternion = quaternion_product(rotation_quaternion,
                                                   her_spawn_quaternion)
        # The rope and the carriage were built in her frame with world-aligned
        # body frames. Rotating the BODY -- pose and orientation together --
        # carries the slide axis (body-local +x) and the weld's relative pose
        # (expressed in the carriage frame) with it, unchanged.
        for body_name in (rope_rail.ROPE_BODY, rope_rail.CARRIER_BODY):
            body = _spec_body(spec, body_name)
            body.pos = (rotation @ np.asarray(body.pos, dtype=float)).tolist()
            body.quat = quaternion_product(rotation_quaternion, body.quat).tolist()
        for geom in _spec_body(spec, rope_rail.ROPE_BODY).geoms:
            geom.name = ROPE_RENDER_GEOM_NAME

        floor = spec.worldbody.add_geom(
            name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE,
            size=[GROUND_HALF_LENGTH_METERS, GROUND_HALF_WIDTH_METERS, 0.05],
            quat=rotation_quaternion.tolist(),
            rgba=[0.86, 0.90, 0.95, 1.0])
        floor.friction = [GROUND_FRICTION, 0.005, 0.0001]

        # --- options on the SPEC, never on the compiled model --------------
        # `add_skybox`, `attach_snow` and `attach_guide` each recompile this
        # spec. Anything written to the compiled model's `opt` would be thrown
        # away by the first of them, and the run would silently continue at
        # MuJoCo's default 2 ms timestep with vertical gravity.
        spec.option.timestep = PHYSICS_TIMESTEP_SECONDS
        spec.option.iterations = SOLVER_ITERATIONS
        spec.option.ls_iterations = SOLVER_LINE_SEARCH_ITERATIONS
        spec.option.gravity = (rotation @ gravity_for_slope(self.slope_degrees)).tolist()

        self.spec = spec
        self.model = spec.compile()
        self.data = mujoco.MjData(self.model)

        # --- the rope as a route, for the hiker and the telemetry ----------
        from rl.environment.ascender import RopeRoute
        grip_world_her_frame = np.asarray(spec.body(rope_rail.ROPE_BODY).pos,
                                          dtype=float)
        # `body.pos` has already been rotated, so walk back out along the
        # rotated axis rather than rotating twice.
        along = rotation @ np.array([1.0, 0.0, 0.0])
        self.route = RopeRoute(np.stack([
            grip_world_her_frame - rope_rail.ROPE_TAIL * along,
            grip_world_her_frame + rope_rail.ROPE_LENGTH * along]))
        self.uphill_direction_world = along
        self.terrain = SlopeGround(self.slope_degrees, frame)
        self.ascender = SlideAscender(self.model, mujoco, rope_rail.ROPE_TAIL)
        self.friction_applied = GROUND_FRICTION
        self.adapt_report = {
            "source": "app/harness/chloe_worlds.build",
            "actuators": "mjlab G1_ARTICULATION (kp/kd/armature per motor group)",
            "feet": f"{foot_sphere_count} spheres, unchanged from the XML",
            "policy_compat": False,
            "frame": frame,
        }
        self.reset()

        if verbose:
            gravity = np.asarray(self.model.opt.gravity)
            print(f"[chloe] plant slope {self.slope_degrees:.1f} deg frame={frame}:"
                  f" nq={self.model.nq} nv={self.model.nv} nu={self.model.nu}"
                  f" nbody={self.model.nbody} neq={self.model.neq}"
                  f"  dt={self.model.opt.timestep * 1000:.1f} ms"
                  f"  gravity {np.round(gravity, 3).tolist()}", flush=True)
            print(f"[chloe] spawn pelvis {np.round(self.spawn, 4).tolist()}"
                  f"  quat {np.round(self.spawn_quaternion, 4).tolist()}"
                  f"  projected gravity"
                  f" {np.round(self.projected_gravity_body, 4).tolist()}"
                  f"  hand-rope gap {self.hand_rope_distance() * 100:.2f} cm",
                  flush=True)

    # ------------------------------------------------------------- binding
    def _bind(self) -> None:
        """Re-derive every id from `self.model`. Called on every reset.

        The three decorators (`attach_guide`, `attach_snow`, `add_skybox`) each
        recompile the spec, swap `self.model` / `self.data`, call
        `self.ascender.bind(...)` and then `self.reset()` -- so this is the
        hook that keeps the ids honest across all three.
        """
        model = self.model
        mujoco = self._mujoco
        self.torso_body_id = int(model.body("torso_link").id)
        self.pelvis_body_id = int(model.body("pelvis").id)
        self.carriage_body_id = int(model.body(self._rope_rail.CARRIER_BODY).id)
        self.imu_torso_site_id = int(model.site("imu_in_torso").id)
        self.palm_site_id = int(model.site("ascender_anchor").id)
        self.grip_equality_id = int(model.equality("ascender_grip").id)
        self.floor_geom_id = int(model.geom("floor").id)
        self.foot_geom_ids = [
            i for i in range(model.ngeom)
            if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i) or "")
            .startswith(("left_foot_", "right_foot_"))]
        names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
                 for j in range(model.njnt)]
        self.joint_names = [n for n in names
                            if n and n.endswith("_joint") and n != "floating_base_joint"]
        self.joint_qpos_addresses = np.array(
            [int(model.jnt_qposadr[model.joint(n).id]) for n in self.joint_names])

    # --------------------------------------------------------------- state
    def reset(self) -> None:
        """Her reset: the pose written by hand, forward, then the cam re-seated."""
        mujoco = self._mujoco
        self._bind()
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[0:3] = self.spawn
        self.data.qpos[3:7] = self.spawn_quaternion
        for name, address in zip(self.joint_names, self.joint_qpos_addresses):
            self.data.qpos[address] = chloe_policy._matching_value(
                self.default_joint_positions, name)
        slide_address = int(self.model.jnt_qposadr[
            self.model.joint(self._rope_rail.SLIDE_JOINT).id])
        self.data.qpos[slide_address] = 0.0
        self.data.qvel[:] = 0.0
        self.data.ctrl[:] = 0.0
        mujoco.mj_forward(self.model, self.data)
        self._rope_rail.ratchet_reset(self.model, self.data)
        self.ascender.s0 = self.ascender.slide_meters(self.data)

    @property
    def projected_gravity_body(self) -> np.ndarray:
        """(0, 0, -1) at reset in both frames -- the equivalence in one line."""
        gravity = np.asarray(self.model.opt.gravity, dtype=float)
        return chloe_policy.quaternion_inverse_rotate(
            self.data.qpos[3:7], gravity / np.linalg.norm(gravity))

    @property
    def palm_xyz(self) -> np.ndarray:
        return np.asarray(self.data.site_xpos[self.palm_site_id], dtype=float).copy()

    def hand_rope_distance(self) -> float:
        """Perpendicular slack between the ascender channel and the line."""
        return float(self.route.project_arclen(self.palm_xyz)[1])

    def set_friction(self, friction: float) -> None:
        for geom_id in self.foot_geom_ids:
            self.model.geom_friction[geom_id, 0] = float(friction)
        self.model.geom_friction[self.floor_geom_id, 0] = float(friction)
        self.friction_applied = float(friction)

    # ----------------------------------------------------------- one substep
    def apply_wind(self, wind) -> None:
        """Quadratic drag on the torso, `climb_scene.ClimbScene.apply_wind`.

        The same law and the same constants, on the same body, so the wind dial
        means the same thing here as on every other world. `mj_step` does not
        clear `xfrc_applied`, so a zero-speed wind has to write the zeros.
        """
        if wind is None or wind.speed == 0.0:
            self.data.xfrc_applied[self.torso_body_id, :] = 0.0
            return
        torso_velocity = self.data.cvel[self.torso_body_id, 3:5]
        relative = wind.velocity - torso_velocity
        force = wind.drag_coeff * float(np.linalg.norm(relative)) * relative
        self.data.xfrc_applied[self.torso_body_id, :] = 0.0
        self.data.xfrc_applied[self.torso_body_id, :2] = force

    def step(self, wind=None) -> float:
        """One physics substep. The cam is raised BEFORE the solver runs."""
        if wind is not None:
            self.apply_wind(wind)
        self._rope_rail.ratchet(self.model, self.data)
        self._mujoco.mj_step(self.model, self.data)
        return self.ascender.slide_meters(self.data)


# ------------------------------------------------------------------- the meta
def describe_chloe_scene(scene, definition) -> dict:
    """Everything the loop, the recorder and the exporter read. One place."""
    rope_rail = chloe_policy.rope_rail_module()
    model = scene.model
    return {
        "kind": "chloe_ascender",
        "robot": definition["robot"],
        "patch": "plane",
        "autonomous": True,
        # --- control contract (hers) ---------------------------------------
        "default_pose_radians": np.array([
            chloe_policy._matching_value(scene.default_joint_positions, name)
            for name in scene.joint_names]),
        "action_scale": "per joint (mjlab G1_ARTICULATION)",
        "action_size": int(model.nu),
        "control_dt_seconds": CONTROL_DT_SECONDS,
        "physics_dt_seconds": float(model.opt.timestep),
        "substeps_per_control_step": int(round(
            CONTROL_DT_SECONDS / float(model.opt.timestep))),
        "observation_size": chloe_policy.OBSERVATION_SIZE,
        "joint_qpos_end": 7 + len(scene.joint_names),
        "joint_qvel_end": 6 + len(scene.joint_names),
        "slide_qpos_address": 7 + len(scene.joint_names),
        "slide_dof_address": 6 + len(scene.joint_names),
        # --- the world -------------------------------------------------------
        "slope_degrees": float(scene.slope_degrees),
        "slope_provenance": definition["slope_provenance"],
        "frame": scene.frame,
        "rope_length_meters": float(scene.route.length),
        "rope_waypoints": int(len(scene.route.points)),
        "rope_radius_meters": float(rope_rail.ROPE_RADIUS),
        "rope_height_meters": float(rope_rail.ROPE_HEIGHT),
        "line_point_world": np.asarray(scene.route.points[0], dtype=float),
        "slope_axis_world": np.asarray(scene.uphill_direction_world, dtype=float),
        "spawn_world": np.asarray(scene.spawn, dtype=float),
        # Reported for the header's sake; this plant leans the WHOLE ROBOT with
        # the slope rather than leaning a base on a level keyframe, so the
        # walking worlds' "lean" and "ankle" numbers do not exist here. The
        # ankle pitch that DOES vary with the slope is the honest analogue.
        "lean_degrees": 0.0,
        "ankle_degrees": float(math.degrees(
            body_joint_positions(scene.slope_degrees)[".*_ankle_pitch_joint"])),
        "foot_friction": float(scene.friction_applied),
        "terrain_friction": float(scene.friction_applied),
        "friction_clamped": False,
        # --- ids --------------------------------------------------------------
        "palm_site_id": int(scene.palm_site_id),
        "torso_body_id": int(scene.torso_body_id),
        "pelvis_body_id": int(scene.pelvis_body_id),
        "carrier_body_id": int(scene.carriage_body_id),
        "imu_torso_site_id": int(scene.imu_torso_site_id),
        "grip_equality_id": int(scene.grip_equality_id),
        "foot_geom_ids": list(scene.foot_geom_ids),
        "terrain_geom_id": int(scene.floor_geom_id),
        "keyframe_id": -1,
        "joint_names": list(scene.joint_names),
        "adapt_report": dict(scene.adapt_report),
        # --- training-time facts ----------------------------------------------
        # Unlike the walking worlds, the SLOPE here IS what she trained on --
        # that is the whole point of these two entries. Wind and the storm are
        # still demo-only.
        "noise_level": 0.0,
        "wind_in_training": False,
        "terrain_in_training": True,
    }


class ChloeSceneLibrary:
    """One compiled plant per (slope, frame), built lazily and cached."""

    def __init__(self, frame="tilted_plane", verbose=True):
        self.frame = frame
        self.verbose = verbose
        self._scenes = {}

    def load(self, name, on_build_start=None):
        import time
        definition = CHLOE_WORLD_DEFINITIONS[name]
        # THE GAINS ARE PART OF THE PLANT, so they are part of the cache key: a
        # scripted world and one of hers at the same slope are two different
        # compiled models, and handing one the other's would be silent.
        scripted = definition.get("policy_version") == SCRIPTED_POLICY_VERSION
        key = (definition["slope_degrees"], self.frame, scripted)
        if key not in self._scenes:
            if on_build_start is not None:
                on_build_start()
            started = time.time()
            print(f"[chloe] building {name}: slope"
                  f" {definition['slope_degrees']:.1f} deg, frame {self.frame}",
                  flush=True)
            scene = ChloeScene(
                definition["slope_degrees"], frame=self.frame,
                verbose=self.verbose,
                gain_scale=(scripted_ascender.SCRIPTED_GAIN_SCALE
                            if scripted else None),
                warn_outside_band=not scripted)
            self._scenes[key] = (scene, describe_chloe_scene(scene, definition))
            print(f"[chloe] built {name} in {time.time() - started:.2f} s",
                  flush=True)
        else:
            print(f"[chloe] {name}: plant already built, no rebuild", flush=True)
        scene, meta = self._scenes[key]
        return scene, meta, definition


# ---------------------------------------------------------------- the episode
class ChloeAscenderEpisode:
    """One spawn-to-outcome run of HER policy on HER plant.

    The same interface `ClimbSceneEpisode` presents, so `runtime.run`, the
    recorder, the pose stream and the websocket drive it without knowing which
    kind of world they have. What is different, and stated everywhere it shows:

      * the policy has no command port. `command[0] > 0` is read as GO and
        nothing else in the 3-vector is read at all. A and D and the mouse
        heading do nothing.
      * stopping is a HELD POSE, not a paused simulation
        (`chloe_policy.AscenderController.go`).
      * `climb_meters` is the ratchet's own coordinate -- the slide joint --
        which on a straight line is exactly the distance climbed.
    """

    autonomous = True

    def __init__(self, scene, meta, definition, world_name, policy_path=None,
                 hold_blend_seconds=0.0, seed=0):
        import mujoco

        self.scene = scene
        self.model = scene.model
        self.data = scene.data
        self.meta = meta
        self.definition = definition
        self.world_name = world_name
        self.rope_enabled = True          # there is no rope-off Chloe world:
                                          # her policy holds the line by weld
        self.slope_degrees = meta["slope_degrees"]
        self.random = np.random.default_rng(seed)

        self.substeps = meta["substeps_per_control_step"]
        self.control_hz = 1.0 / meta["control_dt_seconds"]
        self.palm_site_id = meta["palm_site_id"]
        self.torso_body_id = meta["torso_body_id"]
        self.pelvis_body_id = meta["pelvis_body_id"]
        self.imu_torso_site_id = meta["imu_torso_site_id"]
        self.grip_equality_id = meta["grip_equality_id"]

        # ONE EPISODE CLASS, TWO CONTROLLERS. The plant, the rope, the ratchet,
        # the telemetry and the go/stop gate are identical; the only thing a
        # `scripted_*` world changes is who writes `data.ctrl`.
        self.policy_version = definition.get("policy_version")
        self.scripted = self.policy_version == SCRIPTED_POLICY_VERSION
        if self.scripted:
            self.controller = scripted_ascender.ScriptedAscenderController(
                self.model, scene.default_joint_positions,
                slope_degrees=meta["slope_degrees"],
                control_dt_seconds=meta["control_dt_seconds"])
        else:
            if policy_path is None and definition.get("policy_relative_path"):
                policy_path = os.path.join(
                    chloe_policy.REPOSITORY_ROOT,
                    definition["policy_relative_path"])
            self.controller = chloe_policy.AscenderController(
                self.model, scene.default_joint_positions,
                policy_path=policy_path,
                control_dt_seconds=meta["control_dt_seconds"],
                hold_blend_seconds=hold_blend_seconds)
        self.applied_friction = meta["foot_friction"]

        self.physics_step_hooks = []
        # The control seam exists so the interface matches, and it is honoured:
        # a hook still runs between the policy writing `data.ctrl` and the
        # `mj_step`. `runtime` does NOT register the guide's waist-yaw hook on
        # these worlds, because steering this policy is exactly what we promise
        # not to do.
        self.control_hooks = []
        self.latest_bms = None
        from app.harness.runtime import make_battery_plugin
        self.bms = make_battery_plugin(self.model, self.substeps)

        self.wind_velocity_world = np.zeros(2)
        self.wind_force_world_newtons = np.zeros(3)
        self._mujoco = mujoco
        self.reset()

    # ------------------------------------------------------------- state
    def reset(self) -> None:
        self.scene.reset()
        self.controller.reset()
        self.controller.go = False
        self.spawn_position_world = self.data.qpos[0:3].copy()
        self.spawn_arclength_meters = self.scene.ascender.arclength_meters(self.data)
        self.fell_at_seconds = None
        self.fall_reason = None
        self.maximum_rope_force_newtons = 0.0
        self.latest_bms = None
        self.tick = 0
        if getattr(self, "bms", None) is not None:
            self.bms.reset()

    def set_foot_friction(self, friction: float) -> None:
        self.scene.set_friction(friction)
        self.applied_friction = float(friction)

    @property
    def pelvis_position_world(self) -> np.ndarray:
        return self.data.qpos[0:3].copy()

    @property
    def rope_travel_meters(self) -> float:
        """Metres of rope taken up since the spawn -- the ratchet's own state."""
        return float(self.scene.ascender.progress_meters(self.data))

    @property
    def arclength_meters(self) -> float:
        return float(self.scene.ascender.arclength_meters(self.data))

    @property
    def uphill_distance_meters(self) -> float:
        """Displacement projected on the slope's uphill direction, metres."""
        return float(np.dot(self.pelvis_position_world - self.spawn_position_world,
                            self.scene.uphill_direction_world))

    @property
    def height_gained_meters(self) -> float:
        return float(self.pelvis_position_world[2] - self.spawn_position_world[2])

    @property
    def torso_upright(self) -> float:
        """World z of the torso IMU site's z-axis. The other worlds' fall test."""
        return float(self.data.site_xmat[self.imu_torso_site_id].reshape(3, 3)[2, 2])

    @property
    def rope_force_newtons(self) -> float:
        """Magnitude of the weld constraint's force, newtons."""
        if self.data.nefc == 0:
            return 0.0
        rows = np.where(
            (np.asarray(self.data.efc_type[:self.data.nefc])
             == int(self._mujoco.mjtConstraint.mjCNSTR_EQUALITY))
            & (np.asarray(self.data.efc_id[:self.data.nefc]) == self.grip_equality_id)
        )[0]
        if rows.size == 0:
            return 0.0
        return float(np.linalg.norm(np.asarray(self.data.efc_force)[rows]))

    def hand_line_error_meters(self) -> float:
        return self.scene.hand_rope_distance()

    # ------------------------------------------------------- one control tick
    def step(self, command, wind_velocity_world) -> dict:
        from rl.environment import climb_scene as climb_scene_module

        # THE ONLY THING READ OUT OF THE COMMAND. There is no lin_vel_y and no
        # ang_vel_yaw port on this network; the 3-vector is carried through to
        # the telemetry unchanged so the page's "climbing / holding" badge and
        # the recorder both keep working, but only element 0 does anything.
        command = np.asarray(command, dtype=np.float64)
        self.controller.go = bool(command[0] > 0.0)
        self.controller.command = command.copy()

        self.wind_velocity_world[:] = wind_velocity_world
        speed = float(np.linalg.norm(self.wind_velocity_world))
        wind = None
        if speed > 0.0:
            wind = climb_scene_module.WindParams(
                speed=speed,
                heading=math.atan2(self.wind_velocity_world[1],
                                   self.wind_velocity_world[0]))

        for _ in range(self.substeps):
            self.controller.substep(self.data)
            for hook in self.control_hooks:
                hook(self.model, self.data)
            self.scene.step(wind)
            for hook in self.physics_step_hooks:
                value = hook(self.model, self.data)
                if value is not None:
                    self.latest_bms = value

        self.wind_force_world_newtons[:] = self.data.xfrc_applied[
            self.torso_body_id, :3]

        self.tick += 1
        time_seconds = self.tick / self.control_hz
        rope_force = self.rope_force_newtons
        self.maximum_rope_force_newtons = max(self.maximum_rope_force_newtons,
                                              rope_force)
        upright = self.torso_upright
        finite = bool(np.isfinite(self.data.qpos).all()
                      and np.isfinite(self.data.qvel).all())
        if self.fell_at_seconds is None and (upright < 0.0 or not finite):
            self.fell_at_seconds = time_seconds
            self.fall_reason = "not_finite" if not finite else "tipped_over"
            print(f"[runtime] FELL at t={time_seconds:.2f}s"
                  f" reason={self.fall_reason} upright={upright:+.3f}", flush=True)

        return {
            "time_seconds": time_seconds,
            "root_position_world": self.pelvis_position_world,
            "root_quaternion_world_wxyz": self.data.qpos[3:7].copy(),
            "root_velocity_world": self.data.qvel[0:3].copy(),
            "joint_positions_radians":
                self.data.qpos[7:self.meta["joint_qpos_end"]].copy(),
            "joint_velocities_radians_per_second":
                self.data.qvel[6:self.meta["joint_qvel_end"]].copy(),
            "action": np.asarray(self.controller.last_action, dtype=float).copy(),
            "target_positions_radians": self.data.ctrl.copy(),
            "command": command.copy(),
            "policy_running": 1.0 if self.controller.go else 0.0,
            "wind_velocity_world_meters_per_second": self.wind_velocity_world.copy(),
            "wind_force_world_newtons": self.wind_force_world_newtons.copy(),
            "rope_travel_meters": self.rope_travel_meters,
            "climb_meters": self.rope_travel_meters,
            "arclength_meters": self.arclength_meters,
            "hand_height_on_line_meters": self.arclength_meters,
            "hand_line_error_meters": self.hand_line_error_meters(),
            "height_gained_meters": self.height_gained_meters,
            "uphill_distance_meters": self.uphill_distance_meters,
            "rope_force_newtons": rope_force,
            "torso_upvector_z": upright,
            "fell": 1.0 if self.fell_at_seconds is not None else 0.0,
            **(self.bms.on_tick(self.data, time_seconds) if self.bms else {}),
        }


# ------------------------------------------------------------------ the gates
def run_headless(slope_degrees=20.0, seconds=15.0, frame="tilted_plane",
                 wind_speed=0.0, wind_heading_degrees=180.0,
                 stop_at_seconds=None, resume_at_seconds=None,
                 hold_blend_seconds=0.0, policy_path=None, verbose=True,
                 print_every_seconds=1.0):
    """Her loop with no server, no guide and no graphics. -> a report dict.

    This is the measuring instrument for every number in the README: it builds
    the plant, runs the policy, and optionally releases and re-presses the
    go gate so the hold-pose stop can be measured rather than believed.

    `wind_heading_degrees` defaults to 180 -- straight DOWN the slope, which is
    the direction that actually fights the climb.
    """
    from rl.environment import climb_scene as climb_scene_module

    scene = ChloeScene(slope_degrees, frame=frame, verbose=verbose)
    definition = dict(_definition("headless", slope_degrees, "headless",
                                  "headless")[1])
    meta = describe_chloe_scene(scene, definition)
    controller = chloe_policy.AscenderController(
        scene.model, scene.default_joint_positions, policy_path=policy_path,
        control_dt_seconds=meta["control_dt_seconds"],
        hold_blend_seconds=hold_blend_seconds, verbose=verbose)
    substeps = meta["substeps_per_control_step"]
    control_hz = 1.0 / meta["control_dt_seconds"]

    wind = None
    if wind_speed > 0.0:
        wind = climb_scene_module.WindParams(
            speed=float(wind_speed),
            heading=math.radians(float(wind_heading_degrees)))

    spawn = scene.data.qpos[0:3].copy()
    uphill = scene.uphill_direction_world
    slide_address = scene.ascender.qpos_address
    imu_site = scene.imu_torso_site_id

    marks = {}
    stop_state = None
    report = {"slope_degrees": float(slope_degrees), "frame": frame,
              "wind_speed_mps": float(wind_speed), "fell_at_seconds": None}
    if verbose:
        print(f"{'t':>6} {'uphill':>8} {'rope':>8} {'pelvis_z':>9} {'up_z':>6}"
              f" {'gap_cm':>7} {'go':>3}")
    for tick in range(int(round(seconds * control_hz))):
        time_seconds = tick / control_hz
        controller.go = not (stop_at_seconds is not None
                             and stop_at_seconds <= time_seconds
                             and (resume_at_seconds is None
                                  or time_seconds < resume_at_seconds))
        for _ in range(substeps):
            controller.substep(scene.data)
            scene.step(wind)
        upright = float(scene.data.site_xmat[imu_site].reshape(3, 3)[2, 2])
        if report["fell_at_seconds"] is None and (
                upright < 0.0 or not np.isfinite(scene.data.qpos).all()):
            report["fell_at_seconds"] = time_seconds

        now = time_seconds + 1.0 / control_hz
        if stop_at_seconds is not None and stop_state is None \
                and now >= stop_at_seconds:
            stop_state = {
                "uphill_meters": float(np.dot(scene.data.qpos[0:3] - spawn, uphill)),
                "rope_meters": float(scene.data.qpos[slide_address]),
                "pelvis_height_meters": float(scene.data.qpos[2]),
                "upright": upright}
        if resume_at_seconds is not None and "at_resume" not in marks \
                and now >= resume_at_seconds:
            marks["at_resume"] = {
                "uphill_meters": float(np.dot(scene.data.qpos[0:3] - spawn, uphill)),
                "rope_meters": float(scene.data.qpos[slide_address]),
                "pelvis_height_meters": float(scene.data.qpos[2]),
                "upright": upright}
        if verbose and tick % max(1, int(round(print_every_seconds * control_hz))) == 0:
            print(f"{time_seconds:6.1f}"
                  f" {np.dot(scene.data.qpos[0:3] - spawn, uphill):+8.2f}"
                  f" {scene.data.qpos[slide_address]:+8.2f}"
                  f" {scene.data.qpos[2]:9.3f} {upright:+6.2f}"
                  f" {scene.hand_rope_distance() * 100:7.2f}"
                  f" {'GO' if controller.go else 'HOLD':>3}")

    report.update({
        "uphill_meters": float(np.dot(scene.data.qpos[0:3] - spawn, uphill)),
        "rope_meters": float(scene.data.qpos[slide_address]),
        "pelvis_height_meters": float(scene.data.qpos[2]),
        "upright": float(scene.data.site_xmat[imu_site].reshape(3, 3)[2, 2]),
        "hand_line_error_meters": scene.hand_rope_distance(),
        "standing": bool(report["fell_at_seconds"] is None),
        "at_stop": stop_state,
        "at_resume": marks.get("at_resume"),
        "policy_evaluations": controller.evaluations,
    })
    if stop_state is not None and marks.get("at_resume") is not None:
        report["hold_slide_meters"] = (marks["at_resume"]["rope_meters"]
                                       - stop_state["rope_meters"])
        report["hold_uphill_drift_meters"] = (marks["at_resume"]["uphill_meters"]
                                              - stop_state["uphill_meters"])
        report["hold_sag_meters"] = (marks["at_resume"]["pelvis_height_meters"]
                                     - stop_state["pelvis_height_meters"])
        report["resume_uphill_meters"] = (report["uphill_meters"]
                                          - marks["at_resume"]["uphill_meters"])
        report["stood_through_hold"] = bool(marks["at_resume"]["upright"] > 0.0)
    if verbose:
        print(f"[chloe] end: {'STANDING' if report['standing'] else 'FELL'}"
              f"  uphill={report['uphill_meters']:+.2f} m"
              f"  rope={report['rope_meters']:+.2f} m"
              f"  pelvis_z={report['pelvis_height_meters']:.3f} m"
              f"  gap={report['hand_line_error_meters'] * 100:.2f} cm", flush=True)
    return report


def _equivalence(slope_degrees, seconds, policy_path=None):
    """Both frames, same seed, same everything. The rotation is a rotation."""
    print(f"\n=== EQUIVALENCE at {slope_degrees:g} deg, {seconds:g} s ===")
    rows = []
    for frame in ("tilted_gravity", "tilted_plane"):
        report = run_headless(slope_degrees, seconds, frame=frame,
                              policy_path=policy_path, verbose=False)
        rows.append((frame, report))
        print(f"  {frame:<16} uphill {report['uphill_meters']:+7.3f} m"
              f"  rope {report['rope_meters']:+7.3f} m"
              f"  pelvis_z {report['pelvis_height_meters']:.3f} m"
              f"  {'STANDING' if report['standing'] else 'FELL'}")
    difference = abs(rows[0][1]["uphill_meters"] - rows[1][1]["uphill_meters"])
    print(f"  |difference| in uphill distance: {difference:.4f} m"
          f"  ({difference / max(1e-9, abs(rows[0][1]['uphill_meters'])) * 100:.2f} %)")
    return rows


MATRIX_SLOPES_DEGREES = (20.0, 25.0)   # the stop/go matrix; the full ladder is --ladder
MATRIX_WIND_SPEEDS_MPS = (0.0, 6.0, 12.0)
MATRIX_CLIMB_SECONDS = 5.0     # climb, then release the gate
MATRIX_HOLD_SECONDS = 10.0     # stand there, held pose, ratchet engaged
MATRIX_RESUME_SECONDS = 10.0   # gate back on: does she pick the climb up again


def _matrix(hold_blend_seconds=0.0, frame="tilted_plane", policy_path=None):
    """The two tables the README quotes. Prints; returns the rows.

    Table 1 -- STRAIGHT CLIMB, 15 s, no stop. `--wind` blows straight DOWN the
    slope, which is the heading that actually fights the climb.

    Table 2 -- STOP AND GO. 5 s of climbing, 10 s with the gate released
    (`AscenderController.go = False`: PD targets frozen, physics still
    running), then 10 s with it back on. What is measured while stopped: does
    she stay upright, how far does the ratchet let her slide back (it should be
    zero, that is what a cam is for), and how far does the pelvis sag. What is
    measured after: does she climb again, and how far.
    """
    print(f"\n=== TABLE 1  straight climb, 15 s, frame={frame} ===")
    print(f"{'slope':>6} {'wind':>6} {'uphill_m':>9} {'rope_m':>8}"
          f" {'height_m':>9} {'up_z':>6}  outcome")
    straight = []
    for slope in MATRIX_SLOPES_DEGREES:
        spawn_height = None
        for wind in MATRIX_WIND_SPEEDS_MPS:
            report = run_headless(slope, 15.0, frame=frame, wind_speed=wind,
                                  policy_path=policy_path, verbose=False)
            if spawn_height is None:
                spawn_height = report["pelvis_height_meters"]
            straight.append(report)
            print(f"{slope:6.0f} {wind:6.1f} {report['uphill_meters']:9.2f}"
                  f" {report['rope_meters']:8.2f}"
                  f" {report['uphill_meters'] * math.sin(math.radians(slope)) if frame == 'tilted_plane' else 0.0:9.2f}"
                  f" {report['upright']:+6.2f}"
                  f"  {'STANDING' if report['standing'] else 'FELL at %.1f s' % report['fell_at_seconds']}")

    total = MATRIX_CLIMB_SECONDS + MATRIX_HOLD_SECONDS + MATRIX_RESUME_SECONDS
    print(f"\n=== TABLE 2  stop and go, {MATRIX_CLIMB_SECONDS:.0f} s climb /"
          f" {MATRIX_HOLD_SECONDS:.0f} s HELD POSE /"
          f" {MATRIX_RESUME_SECONDS:.0f} s climb,"
          f" hold-blend {hold_blend_seconds:.2f} s ===")
    print(f"{'slope':>6} {'wind':>6} {'stood':>6} {'slide_m':>8}"
          f" {'drift_m':>8} {'sag_m':>7} {'resume_m':>9} {'total_m':>8}  outcome")
    stopgo = []
    for slope in MATRIX_SLOPES_DEGREES:
        for wind in MATRIX_WIND_SPEEDS_MPS:
            report = run_headless(
                slope, total, frame=frame, wind_speed=wind,
                stop_at_seconds=MATRIX_CLIMB_SECONDS,
                resume_at_seconds=MATRIX_CLIMB_SECONDS + MATRIX_HOLD_SECONDS,
                hold_blend_seconds=hold_blend_seconds,
                policy_path=policy_path, verbose=False)
            stopgo.append(report)
            print(f"{slope:6.0f} {wind:6.1f}"
                  f" {('yes' if report['stood_through_hold'] else 'NO'):>6}"
                  f" {report['hold_slide_meters']:+8.3f}"
                  f" {report['hold_uphill_drift_meters']:+8.3f}"
                  f" {report['hold_sag_meters']:+7.3f}"
                  f" {report['resume_uphill_meters']:+9.2f}"
                  f" {report['uphill_meters']:8.2f}"
                  f"  {'STANDING' if report['standing'] else 'FELL at %.1f s' % report['fell_at_seconds']}")
    print("  slide_m = rope taken up while stopped (the cam should give 0);"
          " drift_m = uphill displacement while stopped;"
          " sag_m = pelvis height change while stopped;"
          " resume_m = uphill distance in the 10 s after the gate came back on.")
    return straight, stopgo


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--slope", type=float, default=20.0)
    parser.add_argument("--seconds", type=float, default=15.0)
    parser.add_argument("--frame", default="tilted_plane",
                        choices=("tilted_plane", "tilted_gravity"))
    parser.add_argument("--wind", type=float, default=0.0,
                        help="wind speed m/s, blowing straight down the slope")
    parser.add_argument("--stop-at", type=float, default=None)
    parser.add_argument("--resume-at", type=float, default=None)
    parser.add_argument("--hold-blend", type=float, default=0.0,
                        help="seconds over which a held pose eases toward the"
                             " reset pose; 0 is a pure freeze")
    parser.add_argument("--policy", default=None)
    parser.add_argument("--equivalence", action="store_true",
                        help="run both frames and print both numbers")
    parser.add_argument("--matrix", action="store_true",
                        help="slope x wind, straight climb and stop/go")
    arguments = parser.parse_args()

    if arguments.matrix:
        _matrix(hold_blend_seconds=arguments.hold_blend, frame=arguments.frame,
                policy_path=arguments.policy)
    elif arguments.equivalence:
        _equivalence(arguments.slope, arguments.seconds, arguments.policy)
    else:
        run_headless(arguments.slope, arguments.seconds, frame=arguments.frame,
                     wind_speed=arguments.wind,
                     stop_at_seconds=arguments.stop_at,
                     resume_at_seconds=arguments.resume_at,
                     hold_blend_seconds=arguments.hold_blend,
                     policy_path=arguments.policy)
