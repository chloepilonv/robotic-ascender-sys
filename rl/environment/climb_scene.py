"""Assemble the G1, the Lhotse Face terrain and the fixed rope into one model.

This is the merge point of the three assets that used to live on separate
branches:

  robot    mujoco_playground's G1 (`rl/tools/fetch_reference_model.py` pins a
           mesh-free copy under `.reference/`, so the scene builds and runs on
           a laptop with no menagerie download and no jax)
  terrain  the Lhotse Face patches, via `rl.environment.terrain`
  rope     a polyline fixed line draped on that terrain, ridden by the mocap
           ascender in `rl.environment.ascender`

WHAT REPLACES WHAT
The upstream playground scene has a flat plane geom named `floor`. That geom is
mutated in place into the terrain heightfield rather than deleted and re-added,
because the G1's foot-contact sensors (`left_foot_floor`, `right_foot_floor`)
resolve their collision pair by that name -- rename it and the env's contact
observations silently go dead.

THE FOUR RANDOMISATION AXES
    slope             terrain geom quat            (see terrain.py)
    surface variation hfield data: rms and seed     (see terrain.py)
    icyness           geom friction, terrain + feet (IceParams below)
    wind              xfrc_applied on the torso     (apply_wind below)

All four are per-geom or per-body fields rather than model topology, so a
single compiled model covers the whole distribution and MJX can vmap them
across environments without recompiling.
"""
from __future__ import annotations

import hashlib
import math
import os
import struct
from dataclasses import dataclass, field

import mujoco
import numpy as np

from rl.environment import ascender as asc_mod
from rl.environment import robot as robot_mod
from rl.environment import terrain as terrain_mod

_HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
DEFAULT_ROBOT_SCENE = robot_mod.PLAYGROUND_SCENE

# Geom groups. The rope and carrier carry no collision (contype/conaffinity 0),
# but mj_ray is purely geometric and ignores that, so they need a group of their
# own to stay out of terrain raycasts.
#
# Group 1 specifically: the G1 uses group 2 for visual meshes and group 3 for
# collision primitives, and MuJoCo's default view mask is [1,1,1,0,0,0] -- so
# anything parked in group 3 is invisible in the viewer and in renders. The rope
# is the point of the scene; it belongs in a group that shows up by default.
GROUP_TERRAIN = 0
GROUP_ROPE = 1

# Collision bitmasks. MuJoCo pairs two geoms when
# (contype1 & conaffinity2) or (contype2 & conaffinity1), so separate bits give
# the rope its own channel:
#
#   bit 0  the world  -- terrain, and everything the robot normally touches
#   bit 1  the rope   -- carried only by geoms allowed to hit the line
#
# The gripping hand is deliberately left off bit 1. It is held *on* the rope by
# the ascender equality, so it would otherwise be in permanent contact with it,
# fighting the constraint that put it there.
BIT_WORLD = 1
BIT_ROPE = 2

# Bodies exempt from rope collision: the whole gripping arm.
#
# The palm is pinned to the line, so the entire right arm necessarily lies
# alongside it and snags on the rope it is holding. The elbow alone accounted
# for 1215 contacts over a 10 s roped run -- the only body touching the rope --
# and exempting it merely moved the problem to the shoulder. Excluding a held
# object from the limb holding it is the normal treatment.
#
# Torso, pelvis, legs, head and the left arm still collide, so the robot cannot
# walk through the line -- pushed at it, it travels 0.16 m instead of 0.66 m.
GRIP_BODIES = (
    "right_shoulder_pitch_link",
    "right_shoulder_roll_link",
    "right_shoulder_yaw_link",
    "right_elbow_link",
    "right_wrist_roll_link",
    "right_wrist_pitch_link",
    "right_wrist_yaw_link",
)
GROUP_GEAR = 2   # alongside the G1's own visual meshes


@dataclass
class RopeParams:
    """Geometry of the fixed line."""

    n_waypoints: int = 9
    radius: float = 0.025
    lateral_amp: float = 0.35     # sideways weave, metres
    lateral_waves: float = 2.2
    lateral_phase: float = 0.0
    margin: float = 1.0           # keep the anchors this far inside the patch
    standoff: float | None = None  # None => match the robot's rest palm height
    rgba: tuple = (0.85, 0.08, 0.05, 1.0)
    collide: bool = True          # solid rope; limbs cannot pass through it
    # Firm normally, slippery tangentially. The rope has to stop a body walking
    # into it, and a soft contact simply does not: at solref 0.03 or above the
    # robot pushes straight through (0.67 m under a 250 N shove, against 0.66 m
    # with no rope at all), while 0.02 holds it to 0.13 m.
    #
    # "Too solid" was never the normal stiffness -- it was tangential grab by the
    # gripping arm, fixed by GRIP_BODIES. Low friction then keeps limbs sliding
    # along the line instead of catching on it: 0.15 gives 121 contacts over a
    # 10 s roped run where 0.4 gives 305, and blocks marginally better.
    friction: float = 0.15
    solref: tuple = (0.02, 1.0)
    solimp: tuple = (0.9, 0.95, 0.001, 0.5, 2.0)


# A condim=3 contact with friction exactly 0 has a zero-width friction cone,
# which is degenerate: the solver diverges within ~30 ms and the NaN surfaces at
# whichever DOF is lightest (here the wrist, which reads as a rope-attachment
# fault -- it is not; it blows up with the grip disabled, on flat ground, with
# no policy). Anything above zero is fine, and zero is not a physical surface
# anyway: rubber or steel on wet ice is 0.02-0.10.
MIN_FRICTION = 0.01

# Below roughly this, the G1 cannot stand on LEVEL ground -- it slips out from
# under itself in about a second whatever the policy does. Measured on the flat
# patch over 20 s: mu 0.20 and up stay upright indefinitely; 0.15 falls at 4.4 s,
# 0.10 at 1.1 s, 0.01 at 0.6 s. So a low-friction run tells you nothing about a
# policy: it is a physics limit, not a controller failure.
STAND_FRICTION = 0.20


@dataclass
class FrictionParams:
    """Sliding friction of the foot-terrain contact.

    This is friction, not "icyness", and the two run opposite ways: LOW
    friction is icy, high friction is dry rock. The knob used to be called
    `ice`, which invited `ice=0` meaning "no ice" when it actually meant zero
    friction -- a frictionless rink, the most slippery setting there is.

    Both sides are set together. Which one actually decides the contact depends
    on the robot -- see `_set_foot_friction` -- so setting only one silently
    does nothing on one of the two models.

    0.9 dry rock, 0.3 the least the robot can stand on, 0.1 verglas, 0.02 bare
    ice. Values are floored at MIN_FRICTION; `clamped` records whether that
    happened so callers can report it.
    """

    terrain: float = 0.9
    foot: float = 0.8
    torsional: float = 0.02
    rolling: float = 0.001
    clamped: bool = False

    def __post_init__(self):
        requested = (self.terrain, self.foot)
        self.terrain = max(float(self.terrain), MIN_FRICTION)
        self.foot = max(float(self.foot), MIN_FRICTION)
        self.clamped = (self.terrain, self.foot) != requested

    @classmethod
    def from_scalar(cls, mu: float) -> "FrictionParams":
        """One knob for a curriculum: mu=0.9 dry rock, mu=0.1 verglas."""
        return cls(terrain=float(mu), foot=float(mu))


@dataclass
class WindParams:
    """The wind axis. Same quadratic-drag law as rl/environment/wind_env.py."""

    speed: float = 0.0     # m/s
    heading: float = 0.0   # rad, world XY
    rho: float = 1.225
    cd_torso: float = 1.2
    area_torso: float = 0.5

    @property
    def drag_coeff(self) -> float:
        return 0.5 * self.rho * self.cd_torso * self.area_torso

    @property
    def velocity(self) -> np.ndarray:
        return self.speed * np.array([np.cos(self.heading), np.sin(self.heading)])


@dataclass
class ClimbScene:
    """A compiled merged model plus the handles needed to drive it."""

    model: mujoco.MjModel
    data: mujoco.MjData
    terrain: terrain_mod.Terrain
    route: asc_mod.RopeRoute
    ascender: asc_mod.MocapAscender
    spec: mujoco.MjSpec
    friction: FrictionParams
    adapt_report: dict
    lean_rad: float
    ankle_rad: float
    spawn: np.ndarray
    palm_site_id: int
    torso_body_id: int
    key_id: int

    def reset(self) -> None:
        mujoco.mj_resetDataKeyframe(self.model, self.data, self.key_id)
        # mj_resetDataKeyframe writes qpos but leaves site_xpos stale, and the
        # carrier is placed relative to the palm. Without this forward pass the
        # ascender grabs the previous pose's palm, opening a multi-metre
        # equality error that detonates the solver on step one.
        mujoco.mj_forward(self.model, self.data)
        self.ascender.reset()
        self.ascender.place(self.data)
        mujoco.mj_forward(self.model, self.data)

    @property
    def palm_xyz(self) -> np.ndarray:
        return self.data.site_xpos[self.palm_site_id].copy()

    def sync_carrier(self) -> float:
        """Hold the carrier on the rope and apply the ratchet."""
        return self.ascender.constrain(self.data)

    def apply_wind(self, wind: WindParams) -> None:
        """Quadratic drag on the torso, written into xfrc_applied.

        mj_step does not clear xfrc_applied, so this persists until overwritten.
        """
        if wind.speed == 0.0:
            self.data.xfrc_applied[self.torso_body_id, :] = 0.0
            return
        v_torso = self.data.cvel[self.torso_body_id, 3:5]
        rel = wind.velocity - v_torso
        f = wind.drag_coeff * np.linalg.norm(rel) * rel
        self.data.xfrc_applied[self.torso_body_id, :] = 0.0
        self.data.xfrc_applied[self.torso_body_id, :2] = f

    def step(self, wind: WindParams | None = None) -> float:
        """One physics substep, with the carrier held on the rope afterwards.

        The projection runs AFTER mj_step: the hand pulls the carrier during the
        step, and the perpendicular part of that motion is then removed. Doing it
        beforehand would discard the very displacement the hand just produced.
        """
        if wind is not None:
            self.apply_wind(wind)
        mujoco.mj_step(self.model, self.data)
        return self.sync_carrier()

    def hand_rope_distance(self) -> float:
        """Perpendicular slack between the palm and the rope. 0 = gripping."""
        return self.route.project_arclen(self.palm_xyz)[1]

    def export(self, xml_path: str) -> str:
        """Write the merged model as XML plus a float32 heightfield sidecar."""
        return export_scene(self, xml_path)


def spawn_quat_of(lean: float) -> np.ndarray:
    """Base orientation for a given lean angle (rotation about world -y)."""
    return np.array([math.cos(lean / 2), 0.0, -math.sin(lean / 2), 0.0])


def _fit_spawn_height(model, data, key_id, xy, z0, quat=None, ankle=None,
                      iters: int = 24) -> float:
    """Raise/lower the base until the deepest foot contact just grazes the surface.

    The stock keyframe height is calibrated for a flat floor at z=0. Dropped
    straight onto a 39 deg heightfield it buries one foot by a couple of
    centimetres, and the resulting contact impulse fights the grip constraint.
    A few bisection-free correction passes remove it.
    """
    # Identify the feet by geom id, not by name. The himalaya robot's foot
    # contacts are four unnamed priority-1 spheres per ankle; a name-based
    # match finds nothing, takes the "airborne" branch every pass, and buries
    # the robot several centimetres into the terrain with its knees in the
    # ground -- which then reads as the walking policy failing.
    foot_ids = set(robot_mod.foot_contact_geoms(model))
    z = float(z0)
    for _ in range(iters):
        mujoco.mj_resetDataKeyframe(model, data, key_id)
        data.qpos[0:2] = xy
        data.qpos[2] = z
        if quat is not None:
            data.qpos[3:7] = quat
        if ankle is not None:
            data.qpos[7 + ANKLE_PITCH_IDX] = ankle
        mujoco.mj_forward(model, data)
        feet = [
            data.contact[i].dist
            for i in range(data.ncon)
            if data.contact[i].geom1 in foot_ids or data.contact[i].geom2 in foot_ids
        ]
        if not feet:
            z -= 0.005  # airborne: settle down toward the surface
            continue
        deepest = min(feet)
        if abs(deepest) < 5e-4:
            break
        z -= deepest  # deepest < 0 means penetrating, so this lifts the robot
    return z


ANKLE_PITCH_IDX = np.array([4, 10])   # left/right ankle pitch within qpos[7:]
ANKLE_PITCH_MIN = np.radians(-50.0)   # G1 ankle pitch lower stop
KNEES_BENT_ANKLE = -0.363             # ankle pitch in the training reset pose
ANKLE_MARGIN = np.radians(3.0)        # keep off the hard stop


def _slope_reset_pose(slope_deg: float, lean_frac: float | None):
    """(base lean rad, ankle pitch rad) that puts the soles flat on the slope.

    The ankle takes as much of the slope as its travel allows; the base leans by
    whatever is left over. `lean_frac` overrides the split with an explicit
    fraction of the slope angle. On flat ground both come out at their
    training-pose values, so nothing changes.
    """
    slope = float(np.radians(slope_deg))
    headroom = (KNEES_BENT_ANKLE - ANKLE_PITCH_MIN) - ANKLE_MARGIN
    lean = (
        float(lean_frac) * slope if lean_frac is not None
        else max(0.0, slope - headroom)
    )
    ankle = float(np.clip(KNEES_BENT_ANKLE - (slope - lean),
                          ANKLE_PITCH_MIN + ANKLE_MARGIN, KNEES_BENT_ANKLE))
    return lean, ankle


def _pose_qpos(base_qpos, spawn, quat, ankle):
    q = np.asarray(base_qpos, dtype=float).copy()
    q[:3] = spawn
    q[3:7] = quat
    q[7 + ANKLE_PITCH_IDX] = ankle
    return q


def _palm_at_pose(robot_scene, terrain, friction, hf_path, policy_compat, keyframe,
                  spawn, quat, ankle):
    """World position of `right_palm` with the robot in its reset pose."""
    spec = mujoco.MjSpec.from_file(robot_scene)
    robot_mod.adapt(spec, policy_compat)
    _add_terrain(spec, terrain, friction, hf_path)
    m = spec.compile()
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, m.key(keyframe).id)
    d.qpos[:] = _pose_qpos(d.qpos, spawn, quat, ankle)
    mujoco.mj_forward(m, d)
    return d.site_xpos[m.site("right_palm").id].copy()


def _probe_rest_pose(scene_xml: str, key: str, policy_compat: bool = True):
    """Return (pelvis_xyz, palm_xyz) for `key`, robot alone on flat ground."""
    spec = mujoco.MjSpec.from_file(scene_xml)
    robot_mod.adapt(spec, policy_compat)
    m = spec.compile()
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, m.key(key).id)
    mujoco.mj_forward(m, d)
    return d.qpos[:3].copy(), d.site_xpos[m.site("right_palm").id].copy()


def _write_hfield_bin(path: str, grid: np.ndarray) -> None:
    """MuJoCo's binary heightfield: int32 nrow, int32 ncol, then float32 data.

    Used in preference to a PNG because an 8-bit PNG quantises a 20 m span into
    8 cm steps -- coarser than the 12 cm roughness the terrain is meant to carry.
    """
    ny, nx = grid.shape
    with open(path, "wb") as f:
        f.write(struct.pack("<ii", ny, nx))
        f.write(np.ascontiguousarray(grid, dtype="<f4").tobytes())


def _open_rope_channel(spec) -> int:
    """Let the robot collide with the rope -- except the hand holding it.

    Without this the rope is scenery and limbs pass straight through it. Every
    colliding robot geom gains the rope bit, apart from the wrist chain carrying
    the ascender: that hand is pinned to the line by the grip equality, so giving
    it rope collision puts it in permanent self-contradiction with the constraint
    holding it there.
    """
    n = 0
    for body in spec.bodies:
        if body.name in GRIP_BODIES:
            continue
        for g in body.geoms:
            if g.contype == 0 and g.conaffinity == 0:
                continue  # visual-only: jacket, boots, logos
            g.contype = int(g.contype) | BIT_ROPE
            g.conaffinity = int(g.conaffinity) | BIT_ROPE
            n += 1
    return n


def _set_foot_friction(spec, friction) -> int:
    """Put `friction.foot` on whatever actually carries foot-ground contact.

    The two robots express feet differently and both have a trap:

      playground  two geoms named `left_foot`/`right_foot`, condim=1
                  (frictionless), PLUS explicit <pair> elements that pin
                  foot-floor friction to 0.6 and override the geoms entirely.
      himalaya    four unnamed priority-1 spheres per ankle_roll link. Priority
                  wins outright, so terrain friction never reaches these
                  contacts and setting the terrain alone does nothing.

    Miss either and the icyness axis is silently inert.
    """
    n = 0
    for body in spec.bodies:
        on_foot = body.name in ("left_ankle_roll_link", "right_ankle_roll_link")
        for g in body.geoms:
            named_foot = g.name in ("left_foot", "right_foot")
            if not (named_foot or (on_foot and g.contype != 0)):
                continue
            g.condim = 3
            g.friction = [friction.foot, friction.torsional, friction.rolling]
            n += 1
    for pair in spec.pairs:
        if pair.name in ("left_foot_floor", "right_foot_floor"):
            pair.condim = 3
            pair.friction = [
                friction.foot, friction.foot, friction.torsional, friction.rolling, friction.rolling
            ]
            n += 1
    return n


def _add_gear(spec, gear_dir, items) -> int:
    """Attach the project's jacket / boots / ascender shells as visual geoms.

    These come out of `assets/robots/g1_unitree.usd` via `rl.tools.usd_gear`.
    They are visual-only by construction (each is a link's convex hull inflated
    along its normals), so they go on with contype/conaffinity 0 and zero mass
    and physics is bit-identical with or without them.

    Bodies missing from the model are skipped rather than raising: the
    mesh-stripped `.reference/` fixture keeps every body, but a future trimmed
    model might not, and losing a jacket panel should not break the scene.
    """
    have = {b.name for b in spec.bodies}
    n = 0
    for it in items:
        if it["body"] not in have:
            continue
        asset = f"gear_{it['body']}_{it['name']}"
        spec.add_mesh(name=asset, file=os.path.join(gear_dir, it["obj"]))
        body = spec.body(it["body"])
        body.add_geom(
            name=asset,
            type=mujoco.mjtGeom.mjGEOM_MESH,
            meshname=asset,
            rgba=list(it["rgba"]),
            contype=0,
            conaffinity=0,
            mass=0.0,
            group=GROUP_GEAR,
        )
        n += 1
    return n


def _add_terrain(spec, terrain, friction, hf_path) -> None:
    """Mutate the scene's flat `floor` plane into the terrain heightfield.

    The geom is edited in place rather than replaced because the G1's foot
    contact sensors (`left_foot_floor`, `right_foot_floor`) resolve their
    collision pair by that name; renaming it makes them silently go dead.
    """
    spec.add_hfield(name="lhotse", file=hf_path, size=list(terrain.hfield_size()))
    if "floor" in {g.name for g in spec.geoms}:
        floor = spec.geom("floor")
    else:
        # `assets/robots/mujoco/*.xml` ships as a bare robot with no ground, by
        # design -- the terrain belongs to the scene. Create the geom, keeping
        # the name `floor` so the playground foot-contact sensors still resolve.
        floor = spec.worldbody.add_geom(name="floor")
    floor.type = mujoco.mjtGeom.mjGEOM_HFIELD
    floor.hfieldname = "lhotse"
    floor.pos = terrain.geom_pos()
    floor.quat = terrain.geom_quat()
    floor.condim = 3
    floor.friction = [friction.terrain, friction.torsional, friction.rolling]
    floor.size = [0, 0, 0]
    floor.group = GROUP_TERRAIN
    _set_foot_friction(spec, friction)

    # The G1 XML declares EXPLICIT contact pairs for the feet:
    #     <pair name="left_foot_floor" ... friction="0.6 0.6"/>
    # An explicit pair overrides both geoms' friction, so setting geom_friction
    # alone leaves the real foot-ground coefficient pinned at 0.6 and the
    # icyness axis does nothing at all. The pairs must be retuned too.
    # (rl/environment/climb_env.py's `foot_friction` config knob is inert for
    # exactly this reason.)
    for pair in spec.pairs:
        if pair.name in ("left_foot_floor", "right_foot_floor"):
            pair.condim = 3
            pair.friction = [
                friction.foot, friction.foot, friction.torsional, friction.rolling, friction.rolling
            ]


def build_scene(
    terrain: terrain_mod.Terrain | None = None,
    *,
    robot_scene: str = DEFAULT_ROBOT_SCENE,
    keyframe: str = "knees_bent",
    spawn_frac: float = 0.12,
    rope: RopeParams | None = None,
    friction: FrictionParams | None = None,
    ratchet: bool = True,
    grip_solref: tuple = (0.004, 1.0),
    grip_solimp: tuple = (0.95, 0.99, 0.001, 0.5, 2.0),
    hfield_dir: str | None = None,
    fit_spawn: bool = True,
    gear: bool = False,
    policy_compat: bool = True,
    lean_frac: float | None = None,
    carrier_mass: float = 1.0,
    carrier_damping: float = 1.0,
    slide_friction: float = 8.0,
) -> ClimbScene:
    """Compile G1 + terrain + rope + ascender into one MuJoCo model.

    `spawn_frac` places the robot that fraction of the way up the patch, so it
    starts near the bottom anchor with rope above it to climb.

    Built in two passes: the terrain goes down first and the base height is
    fitted to it, then the rope is draped through the palm's *corrected* rest
    position. Doing it the other way round leaves the hand a couple of
    centimetres off the line at reset, which the stiff grip then yanks out.
    """
    terrain = terrain if terrain is not None else terrain_mod.load_patch("B")
    rope = rope or RopeParams()
    friction = friction or FrictionParams()

    pelvis0, palm0 = _probe_rest_pose(robot_scene, keyframe, policy_compat)

    # Standing gravity-upright on a slope forces the ankle to absorb the whole
    # slope angle. knees_bent already sits at -20.8 deg and the G1's ankle pitch
    # stops at -50 deg, so past ~29 deg the sole cannot lie flat and the robot
    # balances on a foot edge -- a kinematic limit no policy can learn around,
    # which reads as "it falls instantly" on every steep patch. So the reset
    # pose leans the base into the hill and pitches the ankles to match; between
    # them the soles go back on the ground.
    lean, ankle = _slope_reset_pose(terrain.slope_deg, lean_frac)
    spawn_quat = spawn_quat_of(lean)

    lx, _ = terrain.size_xy
    half = min(lx / 2, terrain.world_extent_x) - rope.margin
    if half <= 0:
        raise ValueError(f"rope margin {rope.margin} m too large for this patch")
    spawn_x = float(-half + spawn_frac * 2 * half)
    spawn_y = 0.0
    ground_z = float(terrain.surface_z(spawn_x, spawn_y))

    hf_dir = hfield_dir or os.path.join(REPO_ROOT, ".reference")
    os.makedirs(hf_dir, exist_ok=True)
    # Name the file after a hash of its contents. MuJoCo caches assets by path,
    # so two terrains sharing a filename (e.g. patch "B" loaded separable and
    # baked) would otherwise both get whichever grid compiled first -- silently,
    # and off by the whole macro slope.
    grid = terrain.hfield_data()
    digest = hashlib.sha1(
        np.ascontiguousarray(grid, dtype="<f4").tobytes()
    ).hexdigest()[:12]
    hf_path = os.path.join(hf_dir, f"terrain_{digest}.hfield")
    _write_hfield_bin(hf_path, grid)

    # -- pass 1: terrain only, to fit the base height to the surface ----
    spawn_z = ground_z + float(pelvis0[2])
    if fit_spawn:
        probe = mujoco.MjSpec.from_file(robot_scene)
        robot_mod.adapt(probe, policy_compat)
        _add_terrain(probe, terrain, friction, hf_path)
        pm = probe.compile()
        spawn_z = _fit_spawn_height(
            pm, mujoco.MjData(pm), pm.key(keyframe).id, (spawn_x, spawn_y),
            spawn_z, quat=spawn_quat, ankle=ankle,
        )
    spawn = np.array([spawn_x, spawn_y, spawn_z])
    # Read the palm off the actual model in the actual reset pose. Deriving it
    # as spawn + R @ offset drifts once the ankles are pitched, and the rope is
    # then laid a centimetre or two off the hand.
    palm_world = _palm_at_pose(
        robot_scene, terrain, friction, hf_path, policy_compat, keyframe,
        spawn, spawn_quat, ankle,
    )

    # -- rope, draped at the palm's rest height and shifted onto it ------
    standoff = (
        rope.standoff if rope.standoff is not None else float(palm_world[2] - ground_z)
    )
    route = asc_mod.drape_route(
        terrain,
        n_waypoints=rope.n_waypoints,
        standoff=standoff,
        margin=rope.margin,
        lateral_amp=rope.lateral_amp,
        lateral_waves=rope.lateral_waves,
        lateral_phase=rope.lateral_phase,
    ).shifted_through(palm_world)
    s_start, _ = route.project_arclen(palm_world)

    # -- pass 2: the full scene -----------------------------------------
    spec = mujoco.MjSpec.from_file(robot_scene)
    adapt_report = robot_mod.adapt(spec, policy_compat)
    _add_terrain(spec, terrain, friction, hf_path)
    if gear:
        from rl.tools import usd_gear

        gear_dir, items = usd_gear.load_manifest()
        if not items:
            raise SystemExit(
                "no gear extracted yet. Run:\n    python -m rl.tools.usd_gear"
            )
        _add_gear(spec, gear_dir, items)
    wb = spec.worldbody

    for i in range(len(route.points) - 1):
        seg = wb.add_geom(
            name=f"ropeseg{i}",
            type=mujoco.mjtGeom.mjGEOM_CAPSULE,
            size=[rope.radius, 0, 0],
            rgba=list(rope.rgba),
            contype=BIT_ROPE if rope.collide else 0,
            conaffinity=BIT_ROPE if rope.collide else 0,
            condim=3,
            friction=[rope.friction, 0.005, 0.0001],
            solref=list(rope.solref),
            solimp=list(rope.solimp),
            priority=1,          # so the rope's softness wins over the robot's
            mass=0.0,
            group=GROUP_ROPE,
        )
        seg.fromto = list(route.points[i]) + list(route.points[i + 1])

    # `carrier_mass` is a lumped grip inertia, not the tool's catalogue mass --
    # the ascender's real 0.1 kg is already folded into the wrist inertial by
    # assets/robots/mujoco. Holding the carrier on the rope by projection fights
    # the grip equality, and a light carrier gets yanked off the line: worst
    # palm-to-carrier error over 8 s is 59 mm at 0.1 kg, 14 mm at 0.5, 9 mm at
    # 1.0, 3.5 mm at 2.0. 1.0 kg keeps the hand inside the 25 mm rope radius
    # while still sliding freely.
    # A real body with three slide joints, not a mocap body: see RopeCarrier.
    # A zero-DOF carrier cannot slide, because `connect` pins the hand to it and
    # the hand's projection is then what decides where it goes -- a deadlock that
    # welds the hand to one point. Its joints are appended after the robot's, so
    # qpos[7:7+29] still addresses the robot alone.
    carrier = wb.add_body(name="rope_carrier", pos=list(route.point_at(s_start)))
    for axis_name, axis in (("carrier_x", [1, 0, 0]),
                            ("carrier_y", [0, 1, 0]),
                            ("carrier_z", [0, 0, 1])):
        carrier.add_joint(
            name=axis_name,
            type=mujoco.mjtJoint.mjJNT_SLIDE,
            axis=axis,
            damping=float(carrier_damping),
        )
    if rope.collide:
        _open_rope_channel(spec)
    carrier.add_site(name="carrier_site", size=[0.02, 0, 0])
    carrier.add_geom(
        name="carrier_geom",
        type=mujoco.mjtGeom.mjGEOM_SPHERE,
        size=[0.035, 0, 0],
        rgba=[0.95, 0.55, 0.05, 0.6],
        contype=0,
        conaffinity=0,
        mass=float(carrier_mass),
        group=GROUP_ROPE,
    )
    eq = spec.add_equality(
        name="ascender_grip",
        type=mujoco.mjtEq.mjEQ_CONNECT,
        objtype=mujoco.mjtObj.mjOBJ_SITE,
        name1="right_palm",
        name2="carrier_site",
    )
    eq.solref = list(grip_solref)
    eq.solimp = list(grip_solimp)

    for k in spec.keys:
        if k.name == keyframe:
            q = _pose_qpos(np.asarray(k.qpos, dtype=float), spawn, spawn_quat, ankle)
            k.qpos = q
            c = np.asarray(k.ctrl, dtype=float).copy()
            c[ANKLE_PITCH_IDX] = q[7 + ANKLE_PITCH_IDX]
            k.ctrl = c


    model = spec.compile()
    model.vis.global_.offwidth, model.vis.global_.offheight = 1920, 1080
    data = mujoco.MjData(model)

    scene = ClimbScene(
        model=model,
        data=data,
        terrain=terrain,
        route=route,
        ascender=asc_mod.RopeCarrier(route, s0=s_start, ratchet=ratchet,
                                     slide_friction=slide_friction),
        spec=spec,
        friction=friction,
        adapt_report=adapt_report,
        lean_rad=lean,
        ankle_rad=ankle,
        spawn=spawn,
        palm_site_id=model.site("right_palm").id,
        torso_body_id=model.body("torso_link").id,
        key_id=model.key(keyframe).id,
    )
    scene.ascender.bind(model, mujoco)
    scene.reset()
    return scene


def export_scene(scene: ClimbScene, xml_path: str) -> str:
    """Write the merged scene to `xml_path` with its heightfield beside it."""
    xml_path = os.path.abspath(xml_path)
    out_dir = os.path.dirname(xml_path)
    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(xml_path))[0]
    hf_name = f"{stem}.hfield"
    _write_hfield_bin(os.path.join(out_dir, hf_name), scene.terrain.hfield_data())

    # to_xml() re-resolves asset paths, so meshdir has to point at the export
    # directory before the heightfield is renamed -- otherwise it looks for the
    # new name next to the source model and refuses to serialise.
    prev_meshdir = scene.spec.meshdir
    scene.spec.meshdir = out_dir
    try:
        for hf in scene.spec.hfields:
            hf.file = hf_name
        xml = scene.spec.to_xml()
    finally:
        scene.spec.meshdir = prev_meshdir
    with open(xml_path, "w") as f:
        f.write(xml)
    return xml_path


# The old name, kept so existing callers keep working. It reads backwards --
# "ice" going up as friction goes down -- which is why it was renamed.
IceParams = FrictionParams
