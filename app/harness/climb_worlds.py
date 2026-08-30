"""The merged ClimbScene worlds: G1 + Lhotse terrain + draped rope, in the browser.

`rl/environment/climb_scene.py::build_scene` builds the whole thing -- terrain
heightfield, rope polyline, bead-on-a-wire carrier, grip equality, spawn pose
fitted to the slope. This module calls it and drives it. It builds NOTHING.

The physics step is HIS: `ClimbScene.step(wind)` is one `mj_step` plus the
carrier projection plus the arc-length ratchet, with the wind drag written into
`xfrc_applied` first. We call it ten times per 50 Hz control tick and put our
app layer (W, mouse-look heading, wind dial, friction knob, recorder,
websocket) around it.

THE POLICY IS ALSO HIS. `rl/environment/walk_policy.py::WalkController` is now
the authoritative 103-dim observation builder, and using it rather than our own
removes a whole class of drift. One thing about it is load-bearing and easy to
get wrong:

    default_pose = robot.KNEES_BENT_QPOS[7:36]     NOT the scene's keyframe

On a slope `build_scene` leans the base and pitches the ankles so the soles lie
flat, so the scene's `knees_bent` keyframe is NOT the pose the policy's action
deltas are defined about. Reading `default_pose` off the compiled model -- which
is what our legacy `PlaygroundObservation` does, correctly, for the old flat
`climb_env` -- would silently move the policy's operating point with the
terrain. `app/harness/test_parity.py` measures our builder against his to show
they agree at the same state.

WHAT THIS SCENE DOES NOT HAVE, and what we do instead:

  `upvector_torso` sensor      absent (`robot.adapt` adds only
                               `local_linvel_pelvis` and `gyro_pelvis`). The
                               fall check reads the same quantity straight off
                               `site_xmat[imu_in_torso]` column z -- that IS
                               what a `framezaxis` sensor on that site reports.
  the 7 `..._found` sensors    absent, so the foot/shin self-collision half of
                               Playground's termination cannot be evaluated
                               here. Our fall test is upright < 0 or non-finite
                               state: NARROWER than the training env's. Stated
                               in PARITY.md.
  a rope-off switch            `build_scene` has no such argument, so the free
                               worlds deactivate the `ascender_grip` equality
                               through `data.eq_active` exactly as the legacy
                               worlds do. The carrier still rides the rope; the
                               hand is simply no longer tied to it.

Inputs  : a world name from `CLIMB_WORLD_DEFINITIONS`.
Outputs : `ClimbSceneLibrary.load(name)` -> (scene, meta, definition);
          `ClimbSceneEpisode` presents the same interface `runtime.Episode`
          does, so the control loop in `runtime.run` drives either kind.
"""

import math
import os

import numpy as np

_HARNESS_DIRECTORY = os.path.dirname(os.path.abspath(__file__))
REPOSITORY_ROOT = os.path.dirname(os.path.dirname(_HARNESS_DIRECTORY))

# Patches `python -m rl.scripts.climb_scene --list` reports, with the mean slope
# it measured. Real patches are Copernicus GLO-30 slopes; the `B_slope*` family
# has its slope SYNTHETICALLY overridden (his doc is emphatic: `B_slope45` is
# not the Lhotse Face at 45 degrees).
TERRAIN_PATCHES = {
    "B": (38.60, "real"), "A": (33.70, "real"),
    "C": (36.18, "real"), "D": (35.47, "real"),
    "B_slope0": (0.42, "synthetic"), "B_slope25": (25.02, "synthetic"),
    "B_slope30": (29.88, "synthetic"), "B_slope35": (35.36, "synthetic"),
    "B_slope45": (44.98, "synthetic"), "B_slope50": (49.95, "synthetic"),
}

DEFAULT_CLIMB_WORLD = "lhotse_B"

# --- the sandbox ---------------------------------------------------------
# The shipped DEM patches are FIXED at 25 x 15 m: `terrain.load_patch` reads a
# whole `.npz` and there is no crop or window argument anywhere in that module.
# So a bigger map cannot come from the DEM without new code, and writing a new
# terrain pipeline was explicitly out of scope.
#
# `terrain.make_terrain(slope_deg, rough_rms, seed, length_m, width_m, res)`
# DOES take an arbitrary size, and builds it from the same octave recipe and
# the same `synth_roughness` the patches use. So the sandbox is big and it is
# the same noise family -- but it is SYNTHETIC, not measured. Nothing about it
# is Lhotse beyond the roughness statistics, and it must never be quoted as
# terrain evidence. (The shipped patches are only measured above ~30 m anyway:
# one 25 x 15 m patch covers 0.447 of a single DEM cell.)
#
# Size chosen by measurement, at 1920x1080 with an idle robot:
#     25 x 15 m  res 0.05   300 x 500   60.9 fps   0.6 MB   (the patch size)
#     60 x 60 m  res 0.10   600 x 600   54.4 fps   1.4 MB
#    120 x 120 m res 0.15   800 x 800   39.0 fps   2.6 MB
#    120 x 120 m res 0.20   600 x 600   48.8 fps   1.4 MB  <- chosen
#    200 x 200 m res 0.25   800 x 800   37.9 fps   2.6 MB
#    200 x 200 m res 0.30   667 x 667   43.7 fps   1.8 MB
# Physics was 20-29x realtime at EVERY size -- the heightfield never bound the
# solver. Render cost is what buys area, and it tracks the GRID, not the metres:
# 200 x 200 m is affordable only by making cells so coarse (0.30 m) that the
# finest roughness octave (0.6 m correlation) spans two cells and stops being
# resolved. 120 x 120 m at 0.20 m keeps three cells per finest octave, leaves
# ~10 fps of headroom for the graphics pass, and is still 38x the area of a
# patch (14400 m2 against 375).
SANDBOX_LENGTH_METERS = 120.0
SANDBOX_WIDTH_METERS = 120.0
SANDBOX_RESOLUTION_METERS = 0.20
SANDBOX_SLOPE_DEGREES = 12.0   # gentle enough that the walking policy stands
SANDBOX_ROUGHNESS_RMS = 0.12   # terrain.DEFAULT_ROUGH_RMS
SANDBOX_SEED = 7


# --- Ines's uneven terrain at arbitrary slope --------------------------------
# `terrain.load_patch` runs a least-squares DE-PLANE: it returns the patch's
# real micro-roughness as a mean-zero grid (patch B: RMS 0.1138 m) and hands the
# macro tilt to the terrain geom's quaternion. Slope and roughness are therefore
# separable, and a `dataclasses.replace` on the slope field is the SAME
# mechanism the shipped `B_slope*` family uses -- the surface keeps every
# centimetre of the measured 12 cm roughness and only the macro tilt moves.
# Verified: replace(slope_deg=10) leaves `rough` bit-identical to patch B's.
#
# This is used instead of the `B_slope*` files because those only exist at 0,
# 25, 30, 35, 45 and 50 degrees, and each carries its OWN noise seed (roughness
# correlation between B and B_slope25 is -0.06, i.e. unrelated draws). Reusing
# patch B's actual roughness for all six means slope is the only variable that
# changes across the ladder.
UNEVEN_SLOPE_DEGREES = (0, 5, 10, 15, 20, 25, 30)   # 0: flat Himalaya ground, no rope (user, 2026-08-30)


def make_uneven_terrain(slope_degrees):
    """Patch B's measured roughness, re-tilted. -> a Terrain."""
    import dataclasses
    from rl.environment import terrain as terrain_module
    patch = terrain_module.load_patch("B")
    return dataclasses.replace(
        patch, slope_deg=float(slope_degrees),
        name=f"B_rough_slope{slope_degrees:g}",
        source=f"patch:real/B re-tilted to {slope_degrees:g} deg")


def make_sandbox_terrain():
    """The free-roam map, through THEIR randomisation entry point."""
    from rl.environment import terrain as terrain_module
    return terrain_module.make_terrain(
        slope_deg=SANDBOX_SLOPE_DEGREES,
        rough_rms=SANDBOX_ROUGHNESS_RMS,
        seed=SANDBOX_SEED,
        length_m=SANDBOX_LENGTH_METERS,
        width_m=SANDBOX_WIDTH_METERS,
        res=SANDBOX_RESOLUTION_METERS,
    )


def _definition(name, patch, label, description, robot="himalaya", rope=True,
                terrain_factory=None, slope=None, provenance=None):
    if patch is not None:
        slope, provenance = TERRAIN_PATCHES[patch]
    return name, {
        "kind": "climb_scene",
        "label": label,
        "patch": patch,
        "robot": robot,
        "rope": rope,
        "slope_degrees": slope,
        "slope_provenance": provenance,
        "description": description,
        "terrain_factory": terrain_factory,
    }


CLIMB_WORLD_DEFINITIONS = dict([
    _definition("lhotse_B", "B", "Lhotse B · 38.6° · rope",
                "The real Lhotse Face patch the scene defaults to, jacketed"
                " robot on the fixed line. This is the task."),
    _definition("lhotse_A", "A", "Lhotse A · 33.7° · rope",
                "A second measured patch, a little gentler than B."),
    _definition("lhotse_D", "D", "Lhotse D · 35.5° · rope",
                "A third measured patch."),
    _definition("lhotse_C", "C", "Lhotse C · 36.2° · rope",
                "A fourth measured patch."),
    _definition("flat_0", "B_slope0", "Flat 0.4° · rope",
                "B's roughness with the macro slope removed: the flat"
                " reference the walking policy can actually stand on."),
    _definition("slope_25", "B_slope25", "Curriculum 25° · rope",
                "Synthetic slope override on patch B. Curriculum rung."),
    _definition("slope_30", "B_slope30", "Curriculum 30° · rope",
                "Synthetic slope override on patch B. Curriculum rung."),
    _definition("slope_35", "B_slope35", "Curriculum 35° · rope",
                "Synthetic slope override on patch B. Curriculum rung."),
    _definition("slope_45", "B_slope45", "Curriculum 45° · rope",
                "Synthetic slope override on patch B. Steeper than any"
                " measured patch."),
    _definition("slope_50", "B_slope50", "Curriculum 50° · rope",
                "Synthetic slope override on patch B. The top rung."),
    _definition("lhotse_B_free", "B", "Lhotse B · 38.6° · NO rope",
                "Patch B with the grip equality deactivated: what the face"
                " does to the walker with nothing to hold.", rope=False),
    _definition("lhotse_B_playground", "B", "Lhotse B · 38.6° · Playground G1",
                "Patch B with the bare mujoco_playground G1 instead of the"
                " jacketed demo robot -- the mels policy's own training body,"
                " for a like-for-like comparison.", robot="playground"),
] + [
    _definition(f"terrain_free_{degrees}", None,
                f"Lhotse terrain {degrees}° · no rope",
                "Patch B's measured micro-roughness (RMS 0.114 m) re-tilted to"
                f" {degrees} degrees, rope off. Where does the walker give up"
                " on rough ground? Spawns at the bottom facing uphill.",
                rope=False,
                terrain_factory=(lambda d=degrees: make_uneven_terrain(d)),
                slope=float(degrees), provenance="real roughness, set slope")
    for degrees in UNEVEN_SLOPE_DEGREES
] + [
    _definition("sandbox_free", None, "Sandbox · 120 x 120 m · 12° · no rope",
                "Free roam. 1.44 hectares of synthetic Himalaya at a walkable"
                " 12 degrees -- 38x the area of a measured patch. SYNTHETIC:"
                " the same roughness family as the patches, but no DEM.",
                rope=False, terrain_factory=make_sandbox_terrain,
                slope=SANDBOX_SLOPE_DEGREES, provenance="synthetic"),
    _definition("sandbox_rope", None, "Sandbox · 120 x 120 m · 12° · rope",
                "The same free-roam map with a rope laid across it by the"
                " scene's own route builder.",
                rope=True, terrain_factory=make_sandbox_terrain,
                slope=SANDBOX_SLOPE_DEGREES, provenance="synthetic"),
])


def robot_scene_path(robot: str) -> str:
    """`--robot` by name, resolved through THEIR module, not a path we typed."""
    from rl.environment import robot as robot_module
    if robot == "himalaya":
        return robot_module.HIMALAYA_ROBOT
    if robot == "himalaya-bare":
        return robot_module.HIMALAYA_ROBOT_BARE
    if robot == "playground":
        return robot_module.PLAYGROUND_SCENE
    raise ValueError(f"unknown robot {robot!r}; have himalaya, himalaya-bare,"
                     " playground")


class ClimbSceneLibrary:
    """Lazy, cached `build_scene` calls -- one compiled scene per world."""

    def __init__(self, verbose=True):
        self._scenes = {}
        self.verbose = verbose

    @staticmethod
    def _key(definition):
        if definition["terrain_factory"] is not None:
            # Synthesised terrains have no patch name; the label's first field
            # plus the slope identifies the build uniquely, and rope-on/rope-off
            # twins still share one compiled scene.
            return (definition["label"].split(" · ")[0],
                    definition["slope_degrees"], definition["robot"])
        # Patch AND robot. The rope flag is a data-level switch on one compiled
        # scene, so it is deliberately NOT in the key: lhotse_B and
        # lhotse_B_free share a build.
        return (definition["patch"], definition["robot"])

    def load(self, name, on_build_start=None):
        import time
        from app.harness import provision_assets
        from rl.environment import climb_scene as climb_scene_module
        from rl.environment import terrain as terrain_module

        definition = CLIMB_WORLD_DEFINITIONS[name]
        key = self._key(definition)
        if key not in self._scenes:
            if on_build_start is not None:
                on_build_start()
            provision_assets.ensure_all(verbose=self.verbose)
            started = time.time()
            print(f"[climb] building {name}:"
                  f" {definition['patch'] or 'synthesised terrain'},"
                  f" robot {definition['robot']}", flush=True)
            factory = definition["terrain_factory"]
            terrain = (factory() if factory is not None
                       else terrain_module.load_patch(definition["patch"]))
            print(f"[climb] terrain {terrain.name}: {terrain.size_xy[0]:.1f} x"
                  f" {terrain.size_xy[1]:.1f} m, grid {terrain.shape[0]}x"
                  f"{terrain.shape[1]} at {terrain.res:g} m,"
                  f" slope {terrain.slope_deg:.2f} deg,"
                  f" roughness rms {terrain.rough.std():.3f} m", flush=True)
            scene = climb_scene_module.build_scene(
                terrain, robot_scene=robot_scene_path(definition["robot"]),
            )
            self._scenes[key] = (scene, describe_climb_scene(scene, definition))
            print(f"[climb] built {name} in {time.time() - started:.2f} s"
                  f"  (adapt: {scene.adapt_report})", flush=True)
        else:
            print(f"[climb] {name}: scene already built, no rebuild", flush=True)
        scene, meta = self._scenes[key]
        return scene, meta, definition


def describe_climb_scene(scene, definition) -> dict:
    """Everything the loop needs, read off HIS scene object. Nothing restated."""
    import mujoco
    from rl.environment import robot as robot_module
    from rl.environment import walk_policy

    model = scene.model
    # The three carrier slide joints are appended AFTER the robot's, so the
    # robot's joints end here. BOUNDED slices only -- an open-ended qpos[7:]
    # picks the carrier up as three phantom joints, which is the same trap the
    # old slide joint set and which his doc calls out.
    joint_qpos_end = 7 + walk_policy.N_JOINTS
    joint_qvel_end = 6 + walk_policy.N_JOINTS

    joint_names = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        for joint_id in range(model.njnt)
        if 7 <= model.jnt_qposadr[joint_id] < joint_qpos_end
    ]
    grip_equality_id = model.equality("ascender_grip").id
    return {
        "kind": "climb_scene",
        "robot": definition["robot"],
        "patch": definition["patch"] or "synthetic",
        # --- control contract (his walk_policy's constants, not ours) -------
        "default_pose_radians": robot_module.KNEES_BENT_QPOS[7:joint_qpos_end].copy(),
        "action_scale": walk_policy.ACTION_SCALE,
        "action_size": int(model.nu),
        "control_dt_seconds": walk_policy.CTRL_DT,
        "physics_dt_seconds": float(model.opt.timestep),
        "substeps_per_control_step": int(round(
            walk_policy.CTRL_DT / model.opt.timestep)),
        "observation_size": walk_policy.OBS_DIM,
        "joint_qpos_end": joint_qpos_end,
        "joint_qvel_end": joint_qvel_end,
        # `slide_qpos_address` is the legacy meta name the recorder header and
        # the observation builder use for "where the robot's joints stop".
        "slide_qpos_address": joint_qpos_end,
        "slide_dof_address": joint_qvel_end,
        # --- the world ------------------------------------------------------
        "slope_degrees": float(scene.terrain.slope_deg),
        "slope_provenance": definition["slope_provenance"],
        "rope_length_meters": float(scene.route.length),
        "rope_waypoints": int(len(scene.route.points)),
        "line_point_world": np.asarray(scene.route.points[0], dtype=float),
        "slope_axis_world": _rope_mean_direction(scene.route),
        "spawn_world": np.asarray(scene.spawn, dtype=float),
        "lean_degrees": float(np.degrees(scene.lean_rad)),
        "ankle_degrees": float(np.degrees(scene.ankle_rad)),
        "foot_friction": float(scene.friction.foot),
        "terrain_friction": float(scene.friction.terrain),
        "friction_clamped": bool(scene.friction.clamped),
        # --- ids -------------------------------------------------------------
        "palm_site_id": int(scene.palm_site_id),
        "torso_body_id": int(scene.torso_body_id),
        "pelvis_body_id": int(model.body("pelvis").id),
        "imu_torso_site_id": int(model.site("imu_in_torso").id),
        "grip_equality_id": int(grip_equality_id),
        "foot_geom_ids": list(robot_module.foot_contact_geoms(model)),
        "terrain_geom_id": int(model.geom("floor").id),
        "keyframe_id": int(scene.key_id),
        "joint_names": joint_names,
        "adapt_report": dict(scene.adapt_report),
        # --- training-time facts, for the header ------------------------------
        # The trainer (rl/environment/the joystick env) still uses the OLD flat
        # tilted plane and slide joint -- his doc's "Not yet done: training".
        # So wind and this terrain are both demo-only, and the state message
        # keeps saying so.
        "noise_level": 0.0,
        "wind_in_training": False,
        "terrain_in_training": False,
    }


def _rope_mean_direction(route) -> np.ndarray:
    """Unit vector from the rope's bottom anchor to its top. For the header."""
    points = np.asarray(route.points, dtype=float)
    delta = points[-1] - points[0]
    norm = float(np.linalg.norm(delta))
    return delta / norm if norm > 1e-9 else np.array([1.0, 0.0, 0.0])


class ClimbSceneEpisode:
    """One spawn-to-outcome run on HIS merged scene.

    Presents the same interface `runtime.Episode` does -- `step(command, wind)`
    returning a row dict, `reset()`, the same attribute names -- so the control
    loop, recorder, renderer and websocket in `runtime.run` drive either kind
    without knowing which they have.

    The physics is entirely his: `ClimbScene.step(wind)` per substep. The
    policy is entirely his: `WalkController.substep(data)` writes `data.ctrl`
    and evaluates the network once per decimation. What is ours is the command,
    the wind vector, the friction knob, the telemetry and the bookkeeping.
    """

    def __init__(self, scene, meta, definition, world_name, wind_drag=None,
                 seed=0):
        import mujoco
        from rl.environment import walk_policy

        self.scene = scene
        self.model = scene.model
        self.data = scene.data
        self.meta = meta
        self.definition = definition
        self.world_name = world_name
        self.rope_enabled = bool(definition["rope"])
        self.slope_degrees = meta["slope_degrees"]
        self.random = np.random.default_rng(seed)

        self.substeps = meta["substeps_per_control_step"]
        self.control_hz = 1.0 / meta["control_dt_seconds"]
        self.palm_site_id = meta["palm_site_id"]
        self.torso_body_id = meta["torso_body_id"]
        self.pelvis_body_id = meta["pelvis_body_id"]
        self.imu_torso_site_id = meta["imu_torso_site_id"]
        self.grip_equality_id = meta["grip_equality_id"]

        # His controller owns the observation, the gait clock, `last_act` and
        # the decimation. We only ever write `.command`.
        self.controller = walk_policy.WalkController(self.model)
        self.applied_friction = meta["foot_friction"]

        # The BMS seam, unchanged: callable(model, data) -> dict | None, called
        # after every substep.
        self.physics_step_hooks = []
        # THE CONTROL SEAM: callable(model, data) -> None, called after the
        # policy has written `data.ctrl` and BEFORE the `mj_step` that acts on
        # it. That is the only place a supervisory layer can bend one joint's
        # PD target without touching the policy, the observation, or anything
        # the policy will see next tick. The guide's SEARCH waist-yaw offset is
        # the only user (`guide.WaistYaw.apply`). Empty by default, so a run
        # with no hooks is exactly the run that was there before.
        self.control_hooks = []
        self.latest_bms = None
        # Chloe's BMS, always on. Once per CONTROL tick -- her plugin
        # integrates with dt = timestep * substeps, so a per-substep call would
        # run it ten times too often on a dt ten times too long.
        from app.harness.runtime import make_battery_plugin
        self.bms = make_battery_plugin(self.model, self.substeps)

        self.wind_velocity_world = np.zeros(2)
        self.wind_force_world_newtons = np.zeros(3)
        self._mujoco = mujoco
        if not self.rope_enabled:
            self._hide_rope_apparatus()
        self.reset()

    def _hide_rope_apparatus(self) -> None:
        """Make the rope and carrier invisible on a rope-off world.

        Visual only -- alpha, not contype. A rope-off world still has the rope's
        collider in the model, and leaving it there is deliberate: hiding a
        thing the robot can still bump into would be a lie of a different kind.
        What goes is the picture of an apparatus the robot is demonstrably not
        using.
        """
        from rl.environment import climb_scene as climb_scene_module
        hidden = 0
        for geom_id in range(self.model.ngeom):
            if self.model.geom_group[geom_id] == climb_scene_module.GROUP_ROPE:
                self.model.geom_rgba[geom_id, 3] = 0.0
                hidden += 1
        print(f"[climb] {self.world_name}: rope off, {hidden} rope/carrier geoms"
              " hidden (alpha only -- the collider stays)", flush=True)


    # ------------------------------------------------------------- state
    def reset(self) -> None:
        """His `ClimbScene.reset()` -- keyframe, forward, carrier placed."""
        self.scene.reset()
        # Rope off = the grip equality deactivated. His builder has no such
        # argument, so this is the same data-level switch the legacy worlds
        # use. `mj_resetDataKeyframe` restores eq_active from the model, so it
        # has to be re-applied after every reset.
        self.data.eq_active[self.grip_equality_id] = 1 if self.rope_enabled else 0
        self._mujoco.mj_forward(self.model, self.data)
        self.controller.reset()
        self.spawn_position_world = self.data.qpos[0:3].copy()
        self.spawn_arclength_meters = float(self.scene.ascender.s)
        self.fell_at_seconds = None
        self.fall_reason = None
        self.maximum_rope_force_newtons = 0.0
        self.latest_bms = None
        self.tick = 0
        if getattr(self, "bms", None) is not None:
            self.bms.reset()

    def set_foot_friction(self, friction: float) -> None:
        """Live friction knob -> foot and terrain geoms.

        Safe to write directly here: this merged scene compiles with `npair 0`,
        so no explicit contact pair overrides the geoms. That is exactly the
        trap his doc records for `the joystick env`, whose `<pair>` elements pin the
        real coefficient at 0.6 no matter what `geom_friction` says. Verified
        on the built model, not assumed.
        """
        for geom_id in self.meta["foot_geom_ids"]:
            self.model.geom_friction[geom_id, 0] = float(friction)
        self.model.geom_friction[self.meta["terrain_geom_id"], 0] = float(friction)
        self.applied_friction = float(friction)

    @property
    def pelvis_position_world(self) -> np.ndarray:
        return self.data.qpos[0:3].copy()

    @property
    def rope_travel_meters(self) -> float:
        """Arc length climbed along the rope since the spawn, metres.

        His ratchet's own state (`RopeCarrier.progress` = s - s0), which is the
        honest climb metric on a draped polyline: straight-line height would
        count sliding sideways along the face as progress.
        """
        return float(self.scene.ascender.progress)

    @property
    def arclength_meters(self) -> float:
        """Absolute position along the rope, metres from its bottom anchor."""
        return float(self.scene.ascender.s)

    @property
    def height_gained_meters(self) -> float:
        return float(self.pelvis_position_world[2] - self.spawn_position_world[2])

    @property
    def torso_upright(self) -> float:
        """World z of the torso IMU site's z-axis. +1 upright, <0 fallen.

        This is precisely what a `framezaxis` sensor on `imu_in_torso` reports
        -- Playground's `upvector_torso`, which `robot.adapt` does not add to
        this scene -- read straight off the kinematics instead.
        """
        return float(self.data.site_xmat[self.imu_torso_site_id].reshape(3, 3)[2, 2])

    @property
    def rope_force_newtons(self) -> float:
        """Magnitude of the grip equality's constraint force, newtons."""
        if self.data.nefc == 0 or not self.rope_enabled:
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
        """His `hand_rope_distance()`: perpendicular slack, 0 = gripping."""
        return float(self.scene.hand_rope_distance())

    # ------------------------------------------------------- one control tick
    def step(self, command, wind_velocity_world) -> dict:
        from rl.environment import climb_scene as climb_scene_module

        from rl.environment import walk_policy

        # Clip to the ranges the policy was trained over -- HIS constant, not a
        # number we chose (walk_policy.CMD_LIMITS).
        self.controller.command = np.clip(
            np.asarray(command, dtype=np.float64),
            walk_policy.CMD_LIMITS[:, 0], walk_policy.CMD_LIMITS[:, 1])

        self.wind_velocity_world[:] = wind_velocity_world
        speed = float(np.linalg.norm(self.wind_velocity_world))
        wind = None
        if speed > 0.0:
            wind = climb_scene_module.WindParams(
                speed=speed,
                heading=math.atan2(self.wind_velocity_world[1],
                                   self.wind_velocity_world[0]))

        for _ in range(self.substeps):
            self.controller.substep(self.data)   # writes data.ctrl
            for hook in self.control_hooks:      # bends a PD target, see above
                hook(self.model, self.data)
            self.scene.step(wind)                # mj_step + carrier + ratchet
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
            "action": self.controller.last_action.copy(),
            "target_positions_radians": self.data.ctrl.copy(),
            "command": self.controller.command.copy(),
            "wind_velocity_world_meters_per_second": self.wind_velocity_world.copy(),
            "wind_force_world_newtons": self.wind_force_world_newtons.copy(),
            "rope_travel_meters": self.rope_travel_meters,
            "climb_meters": self.rope_travel_meters,
            "arclength_meters": self.arclength_meters,
            "hand_height_on_line_meters": self.arclength_meters,
            "hand_line_error_meters": self.hand_line_error_meters(),
            "height_gained_meters": self.height_gained_meters,
            "rope_force_newtons": rope_force,
            "torso_upvector_z": upright,
            "fell": 1.0 if self.fell_at_seconds is not None else 0.0,
            **(self.bms.on_tick(self.data, time_seconds) if self.bms else {}),
        }


# ------------------------------------------------------------------ gates
def verify_joint_parity(scene, meta, verbose=True):
    """The 29 actuated joints, in order, against Playground's own model.

    The same gate `robot_variants.verify_joint_parity` applies to the old env,
    re-run here because the merged scene is a different build path: his
    `robot.adapt` retunes gains and swaps feet, and a policy trained on
    Playground's joint ORDER is only transferable if that order survived.
    """
    import mujoco
    from mujoco_playground._src.locomotion.g1 import base as g1_base
    from mujoco_playground._src.locomotion.g1 import g1_constants as constants

    with open(constants.task_to_xml("flat_terrain").as_posix()) as handle:
        reference = mujoco.MjModel.from_xml_string(handle.read(), g1_base.get_assets())
    reference_names = [
        mujoco.mj_id2name(reference, mujoco.mjtObj.mjOBJ_JOINT,
                          int(reference.actuator_trnid[i, 0]))
        for i in range(reference.nu)]
    scene_names = [
        mujoco.mj_id2name(scene.model, mujoco.mjtObj.mjOBJ_JOINT,
                          int(scene.model.actuator_trnid[i, 0]))
        for i in range(scene.model.nu)]
    if verbose:
        print(f"[gate] actuated joints: playground {len(reference_names)},"
              f" scene {len(scene_names)}  identical order:"
              f" {reference_names == scene_names}", flush=True)
    if reference_names != scene_names:
        raise RuntimeError(
            "JOINT PARITY FAILED on the merged scene.\n"
            f"  playground: {reference_names}\n  scene: {scene_names}\n"
            "The policy's 29 actions would drive the wrong joints.")
    return reference_names


def verify_observation_parity(scene, meta, verbose=True):
    """Our 103-obs builder vs HIS `WalkController.observe()`, same state.

    His is the contract now -- `runtime` drives the climb worlds through his
    controller, not ours. This measures that the two implementations agree, so
    that `test_parity.py`'s result against the JAX env still transfers.

    The one difference that is NOT a bug: `default_pose`. His is pinned to
    `robot.KNEES_BENT_QPOS`; ours reads the compiled keyframe, which on a slope
    `build_scene` has leaned and re-pitched. We hand ours his value, because on
    a slope the keyframe is the wrong answer and his is the right one.
    """
    import mujoco
    from app.harness.playground_policy import GaitPhase, PlaygroundObservation

    model = scene.model
    address = {
        "pelvis_local_linvel": [
            int(model.sensor_adr[model.sensor("local_linvel_pelvis").id]),
            int(model.sensor_adr[model.sensor("local_linvel_pelvis").id]) + 3],
        "pelvis_gyro": [
            int(model.sensor_adr[model.sensor("gyro_pelvis").id]),
            int(model.sensor_adr[model.sensor("gyro_pelvis").id]) + 3],
    }
    our_meta = {
        "default_pose_radians": meta["default_pose_radians"],
        "slide_qpos_address": meta["joint_qpos_end"],
        "slide_dof_address": meta["joint_qvel_end"],
        "pelvis_imu_site_id": int(model.site("imu_in_pelvis").id),
        "sensor_addresses": address,
        "noise_scales": {"linvel": 0.0, "gyro": 0.0, "gravity": 0.0,
                         "joint_pos": 0.0, "joint_vel": 0.0},
    }
    builder = PlaygroundObservation(model, our_meta, noise_level=0.0)

    scene.reset()
    random = np.random.default_rng(3)
    worst = 0.0
    for label, disturb in (("reset state", False), ("perturbed state", True)):
        if disturb:
            scene.data.qpos[7:meta["joint_qpos_end"]] += random.uniform(-0.2, 0.2, 29)
            scene.data.qvel[6:meta["joint_qvel_end"]] = random.uniform(-0.5, 0.5, 29)
            mujoco.mj_forward(model, scene.data)
        command = np.array([0.5, -0.1, 0.2])
        last_action = random.uniform(-1.0, 1.0, 29)
        phase = np.array([0.7, -1.9])

        scene.controller = getattr(scene, "controller", None)
        from rl.environment import walk_policy
        controller = walk_policy.WalkController(model, command=command)
        controller.last_action = last_action.copy()
        controller.phase = phase.copy()
        theirs = np.asarray(controller.observe(scene.data), dtype=np.float64)

        our_phase = GaitPhase(meta["control_dt_seconds"])
        our_phase.phase_radians = phase.copy()
        ours = builder.build(scene.data, command, last_action, our_phase)

        difference = float(np.abs(ours - theirs).max())
        worst = max(worst, difference)
        if verbose:
            print(f"[gate] obs parity ({label}): max |ours - his|"
                  f" {difference:.3e}   his norm {np.linalg.norm(theirs):.4f}",
                  flush=True)
    if verbose:
        verdict = "PASS" if worst < 1e-9 else "DIFFERS"
        print(f"[gate] observation parity worst {worst:.3e}  {verdict}", flush=True)
    return worst


def fingerprint_climb_scene(scene, meta, definition) -> dict:
    """Model + terrain + rope evidence for one built world."""
    import mujoco
    model = scene.model
    points = np.asarray(scene.route.points, dtype=float)
    rise = float(points[-1][2] - points[0][2])
    run = float(np.linalg.norm(points[-1][:2] - points[0][:2]))
    return {
        "world": definition["label"],
        "patch": definition["patch"] or "synthetic",
        "robot": definition["robot"],
        "robot_scene_file": robot_scene_path(definition["robot"]),
        "source": "rl.environment.climb_scene.build_scene",
        "model": {
            "nq": int(model.nq), "nv": int(model.nv), "nu": int(model.nu),
            "nbody": int(model.nbody), "njnt": int(model.njnt),
            "neq": int(model.neq), "nsensor": int(model.nsensor),
            "ngeom": int(model.ngeom), "nhfield": int(model.nhfield),
            "timestep_seconds": float(model.opt.timestep),
            "integrator": str(mujoco.mjtIntegrator(model.opt.integrator)),
            "total_mass_kilograms": float(model.body_subtreemass[0]),
        },
        "terrain": {
            "patch": definition["patch"] or "synthetic",
            "slope_degrees": float(scene.terrain.slope_deg),
            "slope_provenance": definition["slope_provenance"],
            "hfield_size": model.hfield_size[0].tolist() if model.nhfield else None,
            "hfield_nrow": int(model.hfield_nrow[0]) if model.nhfield else None,
            "hfield_ncol": int(model.hfield_ncol[0]) if model.nhfield else None,
            "terrain_geom_friction":
                model.geom_friction[meta["terrain_geom_id"]].tolist(),
        },
        "rope": {
            "length_meters": float(scene.route.length),
            "waypoints": int(len(points)),
            "first_point": points[0].tolist(),
            "last_point": points[-1].tolist(),
            "rise_meters": rise, "run_meters": run,
            "mean_slope_degrees": float(np.degrees(np.arctan2(rise, run))),
            "polyline": points.tolist(),
        },
        "ascender": {
            "carrier_mass_kilograms": float(sum(
                model.body_mass[i] for i in range(model.nbody)
                if "carrier" in (mujoco.mj_id2name(
                    model, mujoco.mjtObj.mjOBJ_BODY, i) or ""))),
            "arclength_at_spawn": float(scene.ascender.s0),
            "ratchet": bool(scene.ascender.ratchet),
            "grip_equality": mujoco.mj_id2name(
                model, mujoco.mjtObj.mjOBJ_EQUALITY, meta["grip_equality_id"]),
            "grip_solref": model.eq_solref[meta["grip_equality_id"]].tolist(),
            "grip_solimp": model.eq_solimp[meta["grip_equality_id"]].tolist(),
        },
        "spawn": {
            "world": np.asarray(scene.spawn).tolist(),
            "lean_degrees": meta["lean_degrees"],
            "ankle_degrees": meta["ankle_degrees"],
            "palm_world": scene.palm_xyz.tolist(),
            "hand_rope_distance_meters": float(scene.hand_rope_distance()),
        },
        "control": {
            "action_scale": meta["action_scale"],
            "control_dt_seconds": meta["control_dt_seconds"],
            "substeps_per_control_step": meta["substeps_per_control_step"],
            "actuator_kp_min": float(model.actuator_gainprm[:, 0].min()),
            "actuator_kp_max": float(model.actuator_gainprm[:, 0].max()),
            "foot_geom_names": [
                mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g)
                for g in meta["foot_geom_ids"]],
            "foot_friction": meta["foot_friction"],
        },
        "adapt_report": meta["adapt_report"],
        "joint_names": meta["joint_names"],
    }


if __name__ == "__main__":
    import argparse, json
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--world", default=DEFAULT_CLIMB_WORLD)
    parser.add_argument("--fingerprint", default=None,
                        help="write the fingerprint json here")
    arguments = parser.parse_args()

    library = ClimbSceneLibrary()
    scene, meta, definition = library.load(arguments.world)
    verify_joint_parity(scene, meta)
    verify_observation_parity(scene, meta)
    # The parity gate perturbs joints to make its second comparison; fingerprint
    # the RESET state or the spawn numbers describe a pose nothing spawns in.
    scene.reset()
    fingerprint = fingerprint_climb_scene(scene, meta, definition)
    path = arguments.fingerprint or os.path.join(
        _HARNESS_DIRECTORY, f"fingerprint_{arguments.world}.json")
    with open(path, "w") as handle:
        json.dump(fingerprint, handle, indent=2)
    print(f"[gate] fingerprint written to {path}", flush=True)
    for key in ("terrain", "rope", "ascender", "spawn"):
        print(f"  {key}: {json.dumps({k: v for k, v in fingerprint[key].items() if k != 'polyline'})}")
