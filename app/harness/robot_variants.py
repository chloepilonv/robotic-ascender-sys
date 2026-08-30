"""Robot variants: run THEIR climb env on the actual demo robot.

`rl/environment/climb_env.py::_build_model` starts from
`consts.task_to_xml(task)` -- Playground's `scene_mjx_feetonly_flat_terrain.xml`
-- and then does all the interesting work itself: tilts the floor, adds the rope
and the carrier, connects `right_palm` to `carrier_site`, sets foot
condim/friction, rebuilds the keyframes with the slide coordinate.

This module changes ONE thing: which scene that line starts from. We generate a
Playground-compatible scene that wraps the team's real robot
(`assets/robots/mujoco/g1_unitree_ascender.xml` -- jacket, snow boots, ascender
end-effector in place of the right hand) and then point `task_to_xml` at it for
the duration of one env construction. Their builder does everything else,
unchanged and unaware.

WHAT THE GENERATED SCENE MUST PROVIDE, and why (each item is something their
code looks up BY NAME and would crash without):

  floor geom `floor`, type plane      climb_env.py:140-148 tilts it for the slope
  keyframe `knees_bent`               climb_env.py:159, :237, :247 -- the reset
  site `right_palm`                   climb_env.py:162, :216 -- the grip endpoint
  geoms `left_foot` / `right_foot`    climb_env.py:151-154 (condim/friction) and
                                      joystick.py:209 `_feet_geom_id`
  sites `left_foot` / `right_foot`    joystick.py:202 `_feet_site_id`
  sites `imu_in_pelvis`/`imu_in_torso` joystick.py:199-200
  22 state sensors                    the whole `_get_obs` + `_get_termination`
  7 `..._found` contact sensors       joystick.py:235-247

WHAT THE REAL ROBOT ALREADY HAS: all 29 joints and actuators, in Playground's
exact order (verified at generation time -- see `verify_joint_parity`); the two
IMU sites; the two foot sites; `pelvis` and `torso_link`.

WHAT IT DOES NOT HAVE, and what we do about it:

  `right_palm` site        The right HAND IS GONE -- the ascender replaces it.
                           We add the site at the ascender's own centre (the
                           bounding-box centre of `ascender_collision` mapped
                           into `right_wrist_yaw_link`), which is where the rope
                           runs through the device. Measured offset from
                           Playground's palm site (0.08, 0, 0): about 3.9 mm.
                           So the grip point barely moves -- the ascender sits
                           essentially where the palm was.
  named foot geoms         The real robot's feet are FOUR unnamed spheres each
                           (class `foot`, size 0.005, condim 3, friction 0.8),
                           not Playground's one named box. We NAME one sphere
                           per foot so their lookups resolve. This changes no
                           physics: their `_build_model` sets condim 3 and
                           friction 0.8 on that geom, which is what all four
                           spheres already are.
  shin/thigh/hand geoms    Playground's self-collision sensors reference geoms
                           (`left_shin`, `right_thigh`, `right_hand_collision`,
                           ...) the real robot does not define. Rather than
                           inventing collision geometry -- which WOULD change
                           the physics -- the generated sensors are BODY-based
                           (`body1="left_ankle_roll_link"` etc.). Same question
                           asked of the same links, and on the right arm it asks
                           it of the ascender, which is the correct body now
                           that the hand is gone.
  Playground's sensors     The real robot ships 5 IMU sensors with different
                           names. We ADD Playground's 22 (they only reference
                           sites that exist) and leave the robot's own alone.

Everything is generated into `app/harness/generated/` at load time and
gitignored: the file carries absolute mesh paths, so it is machine-specific and
must never be committed as if it were a source asset.

Inputs  : nothing (paths are derived from the repository root).
Outputs : `build_pemba_scene()` -> absolute path to the generated scene xml.
          `pemba_task_to_xml()` -> a context manager that redirects
          `g1_constants.task_to_xml` at that file.
"""

import os
import re
import tempfile
import xml.etree.ElementTree as ElementTree
from contextlib import contextmanager

import numpy as np

_HARNESS_DIRECTORY = os.path.dirname(os.path.abspath(__file__))
_REPOSITORY_ROOT = os.path.dirname(os.path.dirname(_HARNESS_DIRECTORY))
GENERATED_DIRECTORY = os.path.join(_HARNESS_DIRECTORY, "generated")
ROBOT_DIRECTORY = os.path.join(_REPOSITORY_ROOT, "assets", "robots", "mujoco")
PEMBA_ROBOT_XML = os.path.join(ROBOT_DIRECTORY, "g1_unitree_ascender.xml")
GENERATED_SCENE_PATH = os.path.join(
    GENERATED_DIRECTORY, "pemba_scene_mjx_feetonly_flat_terrain.xml")

# The body the ascender is mounted on (the right hand's old link).
RIGHT_TOOL_BODY = "right_wrist_yaw_link"
# Playground's own palm sites (g1_mjx_feetonly.xml:340 and :394), both at
# (0.08, 0, 0) on their wrist-yaw link. `consts.HAND_SITES` wants BOTH -- only
# the RIGHT hand was replaced by the ascender, so the left palm keeps
# Playground's exact position on an unchanged left arm.
PLAYGROUND_RIGHT_PALM_POSITION = np.array([0.08, 0.0, 0.0])
LEFT_HAND_BODY = "left_wrist_yaw_link"
PLAYGROUND_LEFT_PALM_POSITION = np.array([0.08, 0.0, 0.0])

# Playground's foot links, used to name one collision sphere per foot and to
# aim the body-based contact sensors.
FOOT_BODIES = {"left_foot": "left_ankle_roll_link",
               "right_foot": "right_ankle_roll_link"}
# Which link stands in for each Playground collision geom in the contact
# sensors. `*_hand` maps to the wrist link, which on this robot carries the
# ascender rather than a hand.
SHIN_BODIES = {"left": "left_knee_link", "right": "right_knee_link"}
# Playground names eight collision geoms that its rewards and its
# `_post_init` look up (joystick.py:224-231). The real robot has collision
# geometry in all the same places, just unnamed, so we NAME the existing geom
# rather than add one: naming changes nothing physical, adding would.
# geom name -> (body, required mesh or None for "the body's collision mesh")
COLLISION_GEOM_NAMES = {
    "left_thigh": ("left_hip_roll_link", None),
    "right_thigh": ("right_hip_roll_link", None),
    "left_shin": ("left_knee_link", None),
    "right_shin": ("right_knee_link", None),
    # The left rubber hand is VISUAL-only on this robot, so the wrist link's
    # own collision mesh is what could actually strike the thigh.
    "left_hand_collision": ("left_wrist_yaw_link", None),
    # On the right there is no hand at all: the ASCENDER is what is out there.
    "right_hand_collision": ("right_wrist_yaw_link", "ascender_collision"),
}
THIGH_BODIES = {"left": "left_hip_roll_link", "right": "right_hip_roll_link"}
HAND_BODIES = {"left": "left_wrist_yaw_link", "right": "right_wrist_yaw_link"}

# Playground's `knees_bent` keyframe, verbatim from
# scene_mjx_feetonly_flat_terrain.xml. Legal to copy only because the joint
# order matches -- `verify_joint_parity` proves that before we use it.
KNEES_BENT_QPOS = (
    "0 0 0.755  1 0 0 0  "
    "-0.312 0 0 0.669 -0.363 0  -0.312 0 0 0.669 -0.363 0  0 0 0.073  "
    "0.2 0.2 0 0.6 0 0 0  0.2 -0.2 0 0.6 0 0 0")
KNEES_BENT_CTRL = (
    "-0.312 0 0 0.669 -0.363 0  -0.312 0 0 0.669 -0.363 0  0 0 0.073  "
    "0.2 0.2 0 0.6 0 0 0  0.2 -0.2 0 0.6 0 0 0")

# Playground's 22 state sensors, verbatim from g1_mjx_feetonly.xml. Every one
# references a site the real robot already has.
PLAYGROUND_STATE_SENSORS = """
    <framezaxis objtype="site" objname="imu_in_torso" name="upvector_torso"/>
    <velocimeter site="imu_in_torso" name="local_linvel_torso"/>
    <velocimeter site="imu_in_pelvis" name="local_linvel_pelvis"/>
    <accelerometer site="imu_in_torso" name="accelerometer_torso"/>
    <accelerometer site="imu_in_pelvis" name="accelerometer_pelvis"/>
    <gyro site="imu_in_torso" name="gyro_torso"/>
    <gyro site="imu_in_pelvis" name="gyro_pelvis"/>
    <framezaxis objtype="site" objname="imu_in_pelvis" name="upvector_pelvis"/>
    <framexaxis objtype="site" objname="imu_in_torso" name="forwardvector_torso"/>
    <framexaxis objtype="site" objname="imu_in_pelvis" name="forwardvector_pelvis"/>
    <framequat objtype="site" objname="imu_in_torso" name="orientation_torso"/>
    <framequat objtype="site" objname="imu_in_pelvis" name="orientation_pelvis"/>
    <framelinvel objtype="site" objname="imu_in_torso" name="global_linvel_torso"/>
    <framelinvel objtype="site" objname="imu_in_pelvis" name="global_linvel_pelvis"/>
    <frameangvel objtype="site" objname="imu_in_torso" name="global_angvel_torso"/>
    <frameangvel objtype="site" objname="imu_in_pelvis" name="global_angvel_pelvis"/>
    <framelinvel objtype="site" objname="left_foot" name="left_foot_global_linvel"/>
    <framelinvel objtype="site" objname="right_foot" name="right_foot_global_linvel"/>
    <framezaxis objtype="site" objname="left_foot" name="left_foot_upvector"/>
    <framezaxis objtype="site" objname="right_foot" name="right_foot_upvector"/>
    <force name="left_foot_force" site="left_foot"/>
    <force name="right_foot_force" site="right_foot"/>
"""


def _contact_sensors_xml():
    """Playground's 7 `..._found` sensors, re-aimed at BODIES.

    Playground names collision geoms this robot does not have. Asking the same
    question of the whole link is both possible and more faithful than
    inventing geometry: it covers all four foot spheres instead of one, and on
    the right arm it asks about the ascender, which is what is there now.
    """
    return f"""
    <contact name="left_foot_floor_found" body1="{FOOT_BODIES['left_foot']}" geom2="floor" reduce="mindist" num="1" data="found"/>
    <contact name="right_foot_floor_found" body1="{FOOT_BODIES['right_foot']}" geom2="floor" reduce="mindist" num="1" data="found"/>
    <contact name="right_foot_left_foot_found" body1="{FOOT_BODIES['right_foot']}" body2="{FOOT_BODIES['left_foot']}" reduce="mindist" num="1" data="found"/>
    <contact name="left_foot_right_shin_found" body1="{FOOT_BODIES['left_foot']}" body2="{SHIN_BODIES['right']}" reduce="mindist" num="1" data="found"/>
    <contact name="right_foot_left_shin_found" body1="{FOOT_BODIES['right_foot']}" body2="{SHIN_BODIES['left']}" reduce="mindist" num="1" data="found"/>
    <contact name="left_hand_left_thigh_found" body1="{HAND_BODIES['left']}" body2="{THIGH_BODIES['left']}" reduce="mindist" num="1" data="found"/>
    <contact name="right_hand_right_thigh_found" body1="{HAND_BODIES['right']}" body2="{THIGH_BODIES['right']}" reduce="mindist" num="1" data="found"/>
"""


FLOOR_AND_VISUAL_XML = """
  <statistic center="0 0 0.7" extent="1.2" meansize="0.04"/>
  <visual>
    <headlight diffuse=".8 .8 .8" ambient=".2 .2 .2" specular="1 1 1"/>
    <rgba force="1 0 0 1"/>
    <global azimuth="140" elevation="-20"/>
    <map force="0.01"/>
    <scale forcewidth="0.3" contactwidth="0.5" contactheight="0.2"/>
    <quality shadowsize="8192"/>
  </visual>
  <asset>
    <texture type="skybox" builtin="gradient" rgb1="1 1 1" rgb2="1 1 1" width="800" height="800"/>
    <texture type="2d" name="groundplane" builtin="checker" mark="edge" rgb1="1 1 1" rgb2="1 1 1"
      markrgb="0 0 0" width="300" height="300"/>
    <material name="groundplane" texture="groundplane" texuniform="true" texrepeat="5 5" reflectance="0"/>
  </asset>
  <worldbody>
    <geom name="floor" size="0 0 0.01" type="plane" material="groundplane"/>
  </worldbody>
"""


def ascender_grip_position():
    """Where `right_palm` goes on this robot, in `right_wrist_yaw_link` coords.

    The rope runs through the middle of the ascender, so the grip point is the
    bounding-box centre of the ascender's collision mesh mapped through its
    mount transform. Returns (position, offset_from_playground_palm_metres).
    """
    import trimesh
    tree = ElementTree.parse(PEMBA_ROBOT_XML)
    mount = None
    for body in tree.iter("body"):
        if body.get("name") != RIGHT_TOOL_BODY:
            continue
        for geom in body.findall("geom"):
            if geom.get("mesh") == "ascender_collision":
                mount = geom
    if mount is None:
        raise LookupError(
            f"no ascender_collision geom on {RIGHT_TOOL_BODY} in"
            f" {PEMBA_ROBOT_XML}: the mount moved, re-read the robot README")
    position = np.fromstring(mount.get("pos", "0 0 0"), sep=" ")
    w, x, y, z = np.fromstring(mount.get("quat", "1 0 0 0"), sep=" ")
    rotation = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
    mesh = trimesh.load(
        os.path.join(ROBOT_DIRECTORY, "meshes", "ascender_collision.obj"),
        process=False)
    vertices = np.asarray(mesh.vertices) @ rotation.T + position
    centre = (vertices.min(axis=0) + vertices.max(axis=0)) / 2.0
    offset = float(np.linalg.norm(centre - PLAYGROUND_RIGHT_PALM_POSITION))
    return centre, offset


def verify_joint_parity(verbose=True):
    """Prove the real robot's joints ARE Playground's, in order. Raises if not.

    This is the gate the whole variant rests on: we copy Playground's
    `knees_bent` qpos and hand the policy's 29 actions straight to this robot's
    actuators. Both are only legal if the joint lists match name-for-name in
    order. Returns (playground_joint_names, pemba_joint_names).
    """
    import mujoco
    from mujoco_playground._src.locomotion.g1 import g1_constants as constants

    def actuator_joint_names(path, assets=None):
        if assets is None:
            model = mujoco.MjModel.from_xml_path(path)
        else:
            with open(path) as handle:
                model = mujoco.MjModel.from_xml_string(handle.read(), assets)
        return [mujoco.mj_id2name(
            model, mujoco.mjtObj.mjOBJ_JOINT,
            int(model.actuator_trnid[actuator_id, 0]))
            for actuator_id in range(model.nu)]

    from mujoco_playground._src.locomotion.g1 import base as g1_base
    playground_names = actuator_joint_names(
        constants.task_to_xml("flat_terrain").as_posix(), g1_base.get_assets())
    pemba_names = actuator_joint_names(_rewritten_robot_path())

    if verbose:
        print(f"[variant] actuated joints: playground {len(playground_names)},"
              f" pemba {len(pemba_names)}", flush=True)
    if playground_names != pemba_names:
        only_playground = [n for n in playground_names if n not in pemba_names]
        only_pemba = [n for n in pemba_names if n not in playground_names]
        order_changed = (sorted(playground_names) == sorted(pemba_names))
        raise RuntimeError(
            "JOINT PARITY FAILED -- the demo robot is not the Playground G1's"
            " actuator set.\n"
            f"  playground ({len(playground_names)}): {playground_names}\n"
            f"  pemba      ({len(pemba_names)}): {pemba_names}\n"
            f"  only in playground: {only_playground}\n"
            f"  only in pemba: {only_pemba}\n"
            f"  same set, different ORDER: {order_changed}\n"
            "Copying Playground's knees_bent keyframe and feeding the policy's"
            " 29 actions to these actuators would silently drive the wrong"
            " joints. STOP and re-map before going further.")
    return playground_names, pemba_names


def _write_atomically(path, text):
    """Write via a unique temp file + os.replace.

    Two harness processes started together (a sweep across worlds, say) both
    regenerate these files, and a plain open()/write() lets one read the other's
    half-written file -- which surfaced as `ParseXML: empty file`. os.replace is
    atomic within a directory, so a reader sees either the old file or the new
    one, never a partial one.
    """
    directory = os.path.dirname(path)
    os.makedirs(directory, exist_ok=True)
    handle, temporary = tempfile.mkstemp(
        dir=directory, prefix=os.path.basename(path) + ".", suffix=".tmp")
    try:
        with os.fdopen(handle, "w") as stream:
            stream.write(text)
        os.replace(temporary, path)
    except BaseException:
        if os.path.exists(temporary):
            os.unlink(temporary)
        raise
    return path


def _rewritten_robot_path():
    """The robot xml with every asset path made absolute, written to generated/.

    The robot file points at `../g1/_menagerie/unitree_g1/assets/*.STL`, which
    `build.py --fetch` is supposed to populate (34 MB, gitignored). That fetch
    imports `pxr` and fails without usd-core -- but the identical menagerie
    STLs are already vendored inside mujoco_playground, so we point at those
    instead and skip the fetch entirely. This is exactly what their own
    `climb_env._rewrite_mesh_paths` (climb_env.py:83-95) does, done earlier
    because the model is compiled once before their surgery runs.
    """
    from mujoco_playground._src import mjx_env
    menagerie = mjx_env.MENAGERIE_PATH / "unitree_g1" / "assets"
    os.makedirs(GENERATED_DIRECTORY, exist_ok=True)
    with open(PEMBA_ROBOT_XML) as handle:
        text = handle.read()

    def absolute(match):
        attribute, path = match.group(1), match.group(2)
        basename = os.path.basename(path)
        candidate = menagerie / basename
        resolved = (candidate.as_posix() if candidate.exists()
                    else os.path.normpath(os.path.join(ROBOT_DIRECTORY, path)))
        return f'{attribute}="{resolved}"'

    text = re.sub(r'(file)="([^"]+)"', absolute, text)
    text = text.replace('meshdir="." texturedir="."', 'meshdir="" texturedir=""')
    return _write_atomically(
        os.path.join(GENERATED_DIRECTORY, "pemba_robot_absolute.xml"), text)


def build_pemba_scene(verbose=True, playground_gains=False):
    """Generate the Playground-compatible scene for the real robot. -> path.

    `playground_gains=True` is a DIAGNOSTIC, not the robot. The demo robot's
    actuators are `kp=500 dampratio=1` for all 29 joints; Playground's G1 uses
    per-joint gains from 2 to 75, and the mels policy was trained against
    those. Setting this copies Playground's gains onto the demo robot so a run
    can separate "the policy cannot handle the new gear" from "the policy
    cannot handle 500". Never use it for anything shipped -- it is not the
    robot the team built.
    """
    import mujoco

    verify_joint_parity(verbose=verbose)
    grip_position, grip_offset = ascender_grip_position()

    spec = mujoco.MjSpec.from_file(_rewritten_robot_path())

    # 1. The palms. `consts.HAND_SITES` (joystick.py:205-207) wants both.
    #    RIGHT: the hand is gone, so the site goes at the ascender's centre --
    #    where the rope runs through the device. LEFT: the arm is untouched, so
    #    it keeps Playground's exact position.
    for body_name, site_name, position in (
            (RIGHT_TOOL_BODY, "right_palm", grip_position),
            (LEFT_HAND_BODY, "left_palm", PLAYGROUND_LEFT_PALM_POSITION)):
        body = spec.body(body_name)
        if body is None:
            raise LookupError(f"no body {body_name!r} on the demo robot")
        site = body.add_site()
        site.name = site_name
        site.pos = position
        site.size = [0.01, 0.0, 0.0]

    # 2. Name one collision sphere per foot so their lookups resolve. No
    #    physics change: their surgery sets condim 3 / friction 0.8, which is
    #    what all four spheres on each foot already are.
    named_feet = {}
    for geom_name, body_name in FOOT_BODIES.items():
        spheres = [g for g in spec.body(body_name).geoms
                   if g.classname is not None and g.classname.name == "foot"]
        if not spheres:
            raise LookupError(
                f"no class-'foot' geoms on {body_name}: the foot build changed")
        spheres[0].name = geom_name
        named_feet[geom_name] = (body_name, len(spheres))

    # 3. Name the collision geoms Playground looks up. No geometry is added.
    named_collisions = {}
    for geom_name, (body_name, required_mesh) in COLLISION_GEOM_NAMES.items():
        candidates = [
            g for g in spec.body(body_name).geoms
            if g.classname is not None and g.classname.name == "collision"
            and (required_mesh is None or g.meshname == required_mesh)]
        if not candidates:
            raise LookupError(
                f"no class-'collision' geom"
                f"{'' if required_mesh is None else f' with mesh {required_mesh!r}'}"
                f" on {body_name}: the robot build changed, re-read"
                " assets/robots/mujoco/README.md before patching this map")
        candidates[0].name = geom_name
        named_collisions[geom_name] = (body_name, candidates[0].meshname)

    # 4. DIAGNOSTIC ONLY: swap in Playground's per-joint actuator gains.
    if playground_gains:
        gains = _playground_actuator_gains()
        for actuator in spec.actuators:
            if actuator.name in gains:
                kp, kv = gains[actuator.name]
                actuator.gainprm = [kp] + [0.0] * 9
                actuator.biasprm = [0.0, -kp, -kv] + [0.0] * 7
        if verbose:
            print(f"[variant] DIAGNOSTIC: Playground actuator gains copied onto"
                  f" {len(gains)} actuators (this is NOT the team's robot)",
                  flush=True)

    text = spec.to_xml()

    # 3. Floor, ground material, visual settings -- Playground's scene, verbatim.
    text = text.replace("</mujoco>", FLOOR_AND_VISUAL_XML + "</mujoco>")
    # 4. Playground's state sensors + the re-aimed contact sensors, merged into
    #    the robot's own <sensor> block (MJCF merges repeated sections, but
    #    appending in place keeps the generated file readable).
    text = text.replace(
        "</sensor>", PLAYGROUND_STATE_SENSORS + _contact_sensors_xml() + "  </sensor>")
    # 5. The reset pose. Legal because verify_joint_parity passed.
    text = text.replace(
        "</keyframe>",
        f'    <key name="knees_bent" qpos="{KNEES_BENT_QPOS}"'
        f' ctrl="{KNEES_BENT_CTRL}"/>\n  </keyframe>')

    # The gains diagnostic gets its OWN file so it can never be confused with,
    # or overwrite, the real robot's scene.
    scene_path = (GENERATED_SCENE_PATH if not playground_gains
                  else GENERATED_SCENE_PATH.replace(".xml", "_playground_gains.xml"))
    _write_atomically(scene_path, text)

    model = mujoco.MjModel.from_xml_path(scene_path)  # must compile
    if verbose:
        print(f"[variant] pemba scene generated: {scene_path}", flush=True)
        print(f"[variant] compiles: nq {model.nq} nv {model.nv} nu {model.nu}"
              f"  nsensor {model.nsensor}  mass {model.body_subtreemass[0]:.4f} kg",
              flush=True)
        print(f"[variant] right_palm placed at the ascender centre"
              f" {np.round(grip_position, 5).tolist()} on {RIGHT_TOOL_BODY};"
              f" {grip_offset * 1000:.1f} mm from Playground's palm site"
              f" {PLAYGROUND_RIGHT_PALM_POSITION.tolist()}", flush=True)
        for geom_name, (body_name, count) in named_feet.items():
            print(f"[variant] named 1 of {count} class-'foot' spheres on"
                  f" {body_name} as {geom_name!r}", flush=True)
        for geom_name, (body_name, mesh_name) in named_collisions.items():
            print(f"[variant] named the {mesh_name!r} collision geom on"
                  f" {body_name} as {geom_name!r}", flush=True)
    return scene_path


def _playground_actuator_gains():
    """{actuator name: (kp, kv)} read off Playground's own compiled model."""
    import mujoco
    from mujoco_playground._src.locomotion.g1 import base as g1_base
    from mujoco_playground._src.locomotion.g1 import g1_constants as constants
    with open(constants.task_to_xml("flat_terrain").as_posix()) as handle:
        model = mujoco.MjModel.from_xml_string(handle.read(), g1_base.get_assets())
    return {
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_id):
            (float(model.actuator_gainprm[actuator_id, 0]),
             float(-model.actuator_biasprm[actuator_id, 2]))
        for actuator_id in range(model.nu)
    }


@contextmanager
def pemba_task_to_xml(scene_path=None):
    """Point `g1_constants.task_to_xml` at the generated scene, scoped.

    Their `_build_model` calls `consts.task_to_xml(self._task)` (climb_env.py:136
    and :231) and `Joystick.__init__` calls it once more (joystick.py:121). All
    three land here for the duration of one env construction, and nothing else
    in the process is affected.
    """
    from etils import epath
    from mujoco_playground._src.locomotion.g1 import g1_constants as constants

    path = scene_path or build_pemba_scene()
    # A gains-diagnostic scene is written to its own file so it can never be
    # confused with, or overwrite, the real robot's scene.
    original = constants.task_to_xml

    def task_to_xml(task: str):
        if task != "flat_terrain":
            raise ValueError(
                f"the pemba variant only has a flat_terrain scene, got {task!r}")
        return epath.Path(path)

    constants.task_to_xml = task_to_xml
    try:
        yield path
    finally:
        constants.task_to_xml = original


if __name__ == "__main__":
    build_pemba_scene()


# ---------------------------------------------------------------- the diff
def format_robot_diff(bare_fingerprint, pemba_fingerprint) -> str:
    """What actually changed between the stock G1 and the demo robot.

    Reads the two fingerprint dicts `team_env.build_fingerprint` produces, so
    it compares the COMPILED models -- not the XML, not the intent. Anything
    the harness would silently inherit shows up here as a row.
    """
    lines = ["=" * 74,
             "ROBOT DIFF   bare Playground G1   ->   Pemba G1 (jacket, boots, ascender)",
             "=" * 74]
    bare_tables = bare_fingerprint["robot_tables"]
    pemba_tables = pemba_fingerprint["robot_tables"]

    bare_mass = bare_tables["total_mass_kilograms"]
    pemba_mass = pemba_tables["total_mass_kilograms"]
    lines.append(f"total mass        {bare_mass:.4f} kg -> {pemba_mass:.4f} kg"
                 f"   ({pemba_mass - bare_mass:+.4f} kg)")

    for label, key in (("bodies", "bodies"), ("geoms", "geoms"),
                       ("sites", "sites"), ("joints", "joints"),
                       ("actuators", "actuators"), ("sensors", "sensors")):
        lines.append(f"{label:<17} {len(bare_tables[key]):>4} -> "
                     f"{len(pemba_tables[key]):>4}")

    # --- per-body mass deltas -------------------------------------------
    bare_body_mass = {b["name"]: b["mass_kilograms"] for b in bare_tables["bodies"]}
    pemba_body_mass = {b["name"]: b["mass_kilograms"] for b in pemba_tables["bodies"]}
    changed = [(name, bare_body_mass.get(name), pemba_body_mass.get(name))
               for name in sorted(set(bare_body_mass) | set(pemba_body_mass))
               if abs((pemba_body_mass.get(name) or 0.0)
                      - (bare_body_mass.get(name) or 0.0)) > 1e-9]
    lines.append("")
    lines.append(f"per-body mass changes ({len(changed)} of"
                 f" {len(set(bare_body_mass) | set(pemba_body_mass))} bodies):")
    if not changed:
        lines.append("  none -- every link keeps its stock mass")
    for name, before, after in changed:
        before_text = "absent" if before is None else f"{before:.4f} kg"
        after_text = "absent" if after is None else f"{after:.4f} kg"
        lines.append(f"  {name:<28} {before_text:>12} -> {after_text:>12}")

    # --- bodies that exist on only one side ------------------------------
    for label, missing in (
            ("bodies only on the BARE robot", set(bare_body_mass) - set(pemba_body_mass)),
            ("bodies only on the PEMBA robot", set(pemba_body_mass) - set(bare_body_mass))):
        if missing:
            lines.append(f"{label}: {sorted(missing)}")

    # --- feet -------------------------------------------------------------
    lines.append("")
    lines.append("feet:")
    for label, fingerprint in (("bare ", bare_fingerprint), ("pemba", pemba_fingerprint)):
        feet = fingerprint["feet"]
        lines.append(f"  {label}  geoms {feet['geom_names']} on {feet['body_names']}")
        lines.append(f"         condim {feet['condim']}  friction {feet['friction']}"
                     f"  body mass {[round(v, 4) for v in feet['body_mass_kilograms']]} kg")
    for label, tables in (("bare ", bare_tables), ("pemba", pemba_tables)):
        foot_geoms = [g for g in tables["geoms"]
                      if g["body"] in ("left_ankle_roll_link", "right_ankle_roll_link")
                      and (g["contype"] != 0 or g["conaffinity"] != 0)]
        lines.append(f"  {label}  {len(foot_geoms)} colliding geoms on the two foot links: "
                     + ", ".join(f"{g['name'] or '<unnamed>'}"
                                 f"({g['type'].split('_')[-1].lower()},"
                                 f" mu {g['friction'][0]}, size {[round(v, 4) for v in g['size'][:3]]})"
                                 for g in foot_geoms))

    # --- joints / actuators ------------------------------------------------
    bare_joints = [j["name"] for j in bare_tables["joints"]]
    pemba_joints = [j["name"] for j in pemba_tables["joints"]]
    bare_actuators = [a["name"] for a in bare_tables["actuators"]]
    pemba_actuators = [a["name"] for a in pemba_tables["actuators"]]
    lines.append("")
    lines.append(f"joints    identical order: {bare_joints == pemba_joints}"
                 f"   ({len(bare_joints)} vs {len(pemba_joints)})")
    if bare_joints != pemba_joints:
        lines.append(f"  only bare : {[j for j in bare_joints if j not in pemba_joints]}")
        lines.append(f"  only pemba: {[j for j in pemba_joints if j not in bare_joints]}")
    lines.append(f"actuators identical order: {bare_actuators == pemba_actuators}"
                 f"   ({len(bare_actuators)} vs {len(pemba_actuators)})")
    bare_kp = bare_fingerprint["control"]["actuator_kp"]
    pemba_kp = pemba_fingerprint["control"]["actuator_kp"]
    lines.append(f"actuator kp   bare min/max {min(bare_kp):.0f}/{max(bare_kp):.0f}"
                 f"   pemba min/max {min(pemba_kp):.0f}/{max(pemba_kp):.0f}"
                 f"   identical: {bare_kp == pemba_kp}")

    # --- the grip and the line ---------------------------------------------
    lines.append("")
    lines.append("grip and line:")
    for label, fingerprint in (("bare ", bare_fingerprint), ("pemba", pemba_fingerprint)):
        geometry = fingerprint["geometry"]
        equality = fingerprint["grip_equality"]
        lines.append(f"  {label}  palm at reset"
                     f" {[round(v, 5) for v in geometry['palm_world_position_at_reset']]}"
                     f"   line point {[round(v, 5) for v in geometry['line_point_world']]}")
        lines.append(f"         grip {equality['type_name'].split('_')[-1]}"
                     f" {equality['site_1']} <-> {equality['site_2']}"
                     f"   |palm - carrier| {geometry['palm_minus_carrier_meters']:.2e} m"
                     f"   pelvis {[round(v, 4) for v in geometry['pelvis_world_position_at_reset']]}")
    bare_palm = np.array(bare_fingerprint["geometry"]["palm_world_position_at_reset"])
    pemba_palm = np.array(pemba_fingerprint["geometry"]["palm_world_position_at_reset"])
    lines.append(f"  the line therefore moves {np.linalg.norm(pemba_palm - bare_palm) * 1000:.1f} mm"
                 f"  (delta {np.round(pemba_palm - bare_palm, 5).tolist()} m)")
    lines.append("=" * 74)
    return "\n".join(lines)
