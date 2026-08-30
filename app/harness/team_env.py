"""Load the TEAM's climbing environment and hand back a plain MuJoCo model.

THE RULE this file exists to enforce: the harness never rebuilds the physics.
`rl/environment/climb_env.py::G1ClimbAscender` is the single source of truth for
the model the policy will be trained against -- the 30 degree tilted plane, the
visual rope cylinder, the `ascender_carrier` body with its `ascender_slide`
joint, the `connect` equality between the `right_palm` site and `carrier_site`,
the condim-3 feet at friction 0.8, the `knees_bent` reset keyframe. We
instantiate their class, take the `mujoco.MjModel` it compiled
(`env._mj_model`), and read every number the interactive loop needs off the
env/config rather than restating it.

Inputs
------
config_overrides : dict | None
    Passed straight through to `G1ClimbAscender` (playground's dotted-key
    form), e.g. {"climb_config.slope_deg": 30.0, "noise_config.level": 0.0}.
    None means their defaults.
slope_degrees : float | None
    Convenience for the one override the demo actually changes; folded into
    config_overrides as "climb_config.slope_deg". None leaves their default
    (30.0).

Outputs
-------
(model, meta)
    model : mujoco.MjModel
        THEIR compiled model. nq 37 / nv 36 / nu 29; the appended
        `ascender_slide` coordinate is the LAST qpos and the LAST qvel.
        opt.timestep = sim_dt = 0.002 s.
    meta : dict, every field in SI units, world frame unless the name says
        otherwise. See METADATA_FIELDS at the bottom of this docstring block
        for the contract; the loud ones:
        default_pose_radians   (29,) knees_bent qpos[7:36]; the pose the
                               policy's action is a delta from.
        action_scale           float, 0.5. ctrl = default_pose + scale*action.
        control_dt_seconds     0.02 (50 Hz), physics_dt_seconds 0.002 (500 Hz),
                               substeps_per_control_step 10.
        slope_radians          tilt of the floor plane about world +y.
        slope_axis_world       (3,) unit uphill direction (cos s, 0, sin s);
                               the ascender slide joint's axis.
        line_point_world       (3,) world position of the rope body origin =
                               the right palm in the reset pose. Slide qpos 0
                               puts the carrier exactly here.
        slide_qpos_address     int, index into qpos of the ascender travel (m).
        slide_dof_address      int, index into qvel of the ascender rate (m/s).
        keyframe_qpos          (37,) the deterministic reset pose.
        joint_names            list[str] length 29, actuator/qpos order.
        ...ids for pelvis body, torso body, palm site, carrier site, imu site,
        the sensors the obs builder reads, and the termination sensors.

Labelled cheats / notes for the real robot: none of this is sensing -- it is
model metadata read at load time. The obs builder in playground_policy.py is
where the sensing contract lives.
"""

import json
import math
import os
import sys

import numpy as np

_REPOSITORY_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
if _REPOSITORY_ROOT not in sys.path:
    sys.path.insert(0, _REPOSITORY_ROOT)

import mujoco  # noqa: E402

FINGERPRINT_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "fingerprint.json"
)

# Sensors their `_get_obs` / `_get_termination` read. The NAMES are built from
# playground's own constants ("<sensor kind>_<frame>", g1_constants.py:53-58)
# rather than typed out here, so a rename on their side follows through.
# The frame strings ("pelvis", "torso") are literals in their code too
# (climb_env.py:422/431/462, joystick.py:427) so they are literals here.
# role the harness needs  ->  the sensor's name in THEIR model.
def sensor_names_by_role(constants):
    return {
        # role                       name                      what it is
        "pelvis_local_linvel": f"{constants.LOCAL_LINVEL_SENSOR}_pelvis",
        # pelvis linear velocity, pelvis frame, m/s -- obs[0:3]
        "pelvis_gyro": f"{constants.GYRO_SENSOR}_pelvis",
        # pelvis angular velocity, pelvis frame, rad/s -- obs[3:6]
        "torso_global_linvel": f"{constants.GLOBAL_LINVEL_SENSOR}_torso",
        # torso linear velocity, world frame, m/s -- the wind law's v_torso
        "torso_upvector": f"{constants.GRAVITY_SENSOR}_torso",
        # torso up-vector in world; z < 0 means fallen over -- termination
    }


def load_team_environment(config_overrides=None, slope_degrees=None):
    """Instantiate their `G1ClimbAscender` (JAX/MJX object).

    Costs a full __init__ -- JAX import plus `mjx.put_model` -- even when only
    the plain MjModel is wanted. Measured warm: 1.64 s for the first in a
    process, 0.21 s for each additional. Slower on a cold venv (the first run
    also clones mujoco_menagerie).
    """
    import rl.environment  # noqa: F401  registers G1ClimbAscender
    from mujoco_playground import registry

    overrides = dict(config_overrides or {})
    if slope_degrees is not None:
        overrides["climb_config.slope_deg"] = float(slope_degrees)
    return registry.load("G1ClimbAscender", config_overrides=overrides)


def load_team_model(config_overrides=None, slope_degrees=None,
                    write_fingerprint=True, print_fingerprint=True,
                    fingerprint_path=None):
    """(mujoco.MjModel, meta dict). See the module docstring for the contract.

    `fingerprint_path` lets a caller that builds SEVERAL models (the world
    library) give each one its own file instead of having the last build
    overwrite the evidence for the others. Defaults to FINGERPRINT_PATH.
    """
    environment = load_team_environment(config_overrides, slope_degrees)
    model, meta = describe_team_environment(environment)
    fingerprint = build_fingerprint(environment, model, meta)
    if print_fingerprint:
        print(format_fingerprint(fingerprint), flush=True)
    if write_fingerprint:
        path = fingerprint_path or FINGERPRINT_PATH
        with open(path, "w") as handle:
            json.dump(fingerprint, handle, indent=2, default=_jsonable)
        print(f"[team_env] fingerprint written to {path}", flush=True)
    return model, meta


def describe_team_environment(environment):
    """Read the loop's every constant off THEIR env object. No restating.

    NOTHING about the robot is named here. Foot geoms, the palm and carrier
    sites, the torso and pelvis bodies, the ascender joint and the grip
    equality are all DERIVED -- from the ids their `_post_init`/`_build_model`
    already computed, or from the model's own topology. That is deliberate: the
    training robot is expected to change (snow boots with new foot geoms and
    masses, an ascender end-effector body in place of the right hand, a
    jacket), and the harness has to inherit those changes by reloading, not by
    being edited to match.
    """
    from mujoco_playground._src.locomotion.g1 import g1_constants as constants

    model = environment.mj_model
    config = environment._config
    climb_config = config.climb_config

    slide_qpos_address = int(environment._slide_qposadr)
    slide_dof_address = int(environment._slide_dofadr)
    slope_radians = math.radians(float(climb_config.slope_deg))

    # The ascender joint, found by its qpos address rather than its name.
    slide_joint_id = int(
        np.where(model.jnt_qposadr == slide_qpos_address)[0][0]
    )
    carrier_body_id = int(model.jnt_bodyid[slide_joint_id])
    # The grip: the one equality wired to the palm site. Found by its endpoints,
    # so renaming "ascender_grip" -- or swapping the palm for an ascender
    # end-effector site -- costs nothing here.
    palm_site_id = int(environment._palm_site_id)
    carrier_site_id = int(environment._carrier_site_id)
    grip_equality_id = _find_equality(model, palm_site_id, carrier_site_id)

    joint_names = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        for joint_id in range(model.njnt)
    ]
    # The actuated robot joints, in qpos/ctrl order: every joint whose qpos
    # address lands in [7, slide_qpos_address). Derived, not sliced by a
    # remembered count -- a robot with a different joint set still works.
    actuated_joint_names = [
        joint_names[joint_id] for joint_id in range(model.njnt)
        if 7 <= model.jnt_qposadr[joint_id] < slide_qpos_address
    ]
    actuator_names = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_id)
        for actuator_id in range(model.nu)
    ]
    # Feet: THEIR ids (g1_constants.FEET_GEOMS via _post_init). Snow boots that
    # arrive as new geom names land here automatically as long as their
    # constants list them.
    foot_geom_ids = [int(v) for v in np.asarray(environment._feet_geom_id).ravel()]
    # The pelvis: the body the pelvis IMU site is attached to. No name.
    pelvis_body_id = int(model.site_bodyid[int(environment._pelvis_imu_site_id)])

    meta = {
        # --- the control contract -------------------------------------
        "default_pose_radians": np.array(environment._default_pose, dtype=np.float64),
        "action_scale": float(config.action_scale),
        "action_size": int(model.nu),
        "control_dt_seconds": float(environment.dt),
        "physics_dt_seconds": float(environment.sim_dt),
        "substeps_per_control_step": int(environment._n_substeps),
        "observation_size": 103,
        # --- the ascender ---------------------------------------------
        "slide_joint_id": slide_joint_id,
        "slide_joint_name": joint_names[slide_joint_id],
        "slide_qpos_address": slide_qpos_address,
        "slide_dof_address": slide_dof_address,
        "slide_range_meters": [float(v) for v in model.jnt_range[slide_joint_id]],
        "slide_damping": float(model.dof_damping[slide_dof_address]),
        "slide_frictionloss": float(model.dof_frictionloss[slide_dof_address]),
        "carrier_body_id": carrier_body_id,
        # --- the world -------------------------------------------------
        "slope_degrees": float(climb_config.slope_deg),
        "slope_radians": slope_radians,
        "slope_axis_world": np.array(environment._slope_axis, dtype=np.float64),
        "slope_normal_world": np.array(
            [-math.sin(slope_radians), 0.0, math.cos(slope_radians)]
        ),
        "line_point_world": np.array(environment._line_pt, dtype=np.float64),
        "rope_length_meters": float(climb_config.rope_length),
        "rope_tail_meters": float(climb_config.rope_tail),
        "foot_friction": float(climb_config.foot_friction),
        # --- reset ------------------------------------------------------
        "keyframe_name": "knees_bent",
        "keyframe_qpos": np.array(model.keyframe("knees_bent").qpos, dtype=np.float64),
        "reset_base_velocity_range": [-0.5, 0.5],
        # --- ids (every one DERIVED; see the function docstring) -----------
        "pelvis_body_id": pelvis_body_id,
        "pelvis_body_name": mujoco.mj_id2name(
            model, mujoco.mjtObj.mjOBJ_BODY, pelvis_body_id),
        "torso_body_id": int(environment._torso_body_id),
        "torso_body_name": mujoco.mj_id2name(
            model, mujoco.mjtObj.mjOBJ_BODY, int(environment._torso_body_id)),
        "palm_site_id": palm_site_id,
        "palm_site_name": mujoco.mj_id2name(
            model, mujoco.mjtObj.mjOBJ_SITE, palm_site_id),
        "carrier_site_id": carrier_site_id,
        "pelvis_imu_site_id": int(environment._pelvis_imu_site_id),
        "feet_site_ids": [int(v) for v in np.asarray(environment._feet_site_id).ravel()],
        "foot_geom_ids": foot_geom_ids,
        "foot_geom_names": [
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
            for geom_id in foot_geom_ids
        ],
        "grip_equality_id": grip_equality_id,
        "total_mass_kilograms": float(model.body_subtreemass[0]),
        # --- naming -------------------------------------------------------
        "joint_names": actuated_joint_names,
        "actuator_names": actuator_names,
        # --- noise (training-time; the harness runs with it OFF) ----------
        "noise_level": float(config.noise_config.level),
        "noise_scales": {
            k: float(v) for k, v in config.noise_config.scales.items()
        },
        "push_enabled": bool(config.push_config.enable),
        "command_limits": {
            "lin_vel_x": [float(v) for v in config.lin_vel_x],
            "lin_vel_y": [float(v) for v in config.lin_vel_y],
            "ang_vel_yaw": [float(v) for v in config.ang_vel_yaw],
        },
        "gait_frequency_range_hz": [1.25, 1.5],
    }
    meta["sensor_names_by_role"] = sensor_names_by_role(constants)
    # role -> [start, stop) into data.sensordata.
    meta["sensor_addresses"] = {
        role: _sensor_slice(model, name)
        for role, name in meta["sensor_names_by_role"].items()
    }
    # The self-collision termination sensors: THEIR ids, so a change to which
    # pairs count as a fall follows through without an edit here.
    meta["self_collision_sensor_addresses"] = [
        _sensor_slice_by_id(model, sensor_id) for sensor_id in (
            environment._right_foot_left_foot_found_sensor,
            environment._left_foot_right_shin_found_sensor,
            environment._right_foot_left_shin_found_sensor,
        )
    ]
    meta["foot_floor_sensor_addresses"] = [
        _sensor_slice_by_id(model, sensor_id)
        for sensor_id in environment._feet_floor_found_sensor
    ]
    return model, meta


def _find_equality(model, site_a_id, site_b_id):
    """The equality wired between two sites, whatever it is called."""
    for equality_id in range(model.neq):
        if model.eq_objtype[equality_id] != int(mujoco.mjtObj.mjOBJ_SITE):
            continue
        endpoints = {int(model.eq_obj1id[equality_id]), int(model.eq_obj2id[equality_id])}
        if endpoints == {int(site_a_id), int(site_b_id)}:
            return equality_id
    raise LookupError(
        f"no equality connects sites {site_a_id} and {site_b_id}; the grip has"
        " moved or been renamed -- reload and re-fingerprint, do not patch ids"
    )


def _sensor_slice(model, name):
    return _sensor_slice_by_id(model, model.sensor(name).id)


def _sensor_slice_by_id(model, sensor_id):
    sensor_id = int(sensor_id)
    start = int(model.sensor_adr[sensor_id])
    return [start, start + int(model.sensor_dim[sensor_id])]


# ---------------------------------------------------------------- fingerprint
def build_fingerprint(environment, model, meta):
    """Every number a reader would need to prove our model IS their model.

    Includes the FULL body / geom / actuator tables, not a hand-picked subset,
    precisely so that a change to the training robot -- snow boots, an
    ascender end-effector body in place of the right hand, a jacket's added
    mass -- shows up as a diff in this file the next time it is written, rather
    than silently changing the physics under a harness that never noticed.
    """
    data = mujoco.MjData(model)
    data.qpos[:] = meta["keyframe_qpos"]
    mujoco.mj_forward(model, data)

    equality_id = meta["grip_equality_id"]
    slide_joint_id = meta["slide_joint_id"]
    foot_geom_ids = meta["foot_geom_ids"]

    return {
        "robot_tables": _robot_tables(model, meta),
        "source": {
            "class": "rl.environment.climb_env.G1ClimbAscender",
            "registry_name": "G1ClimbAscender",
            "base_xml": environment.xml_path,
            "menagerie_assets": "mujoco_playground external_deps/mujoco_menagerie",
        },
        "model": {
            "nq": int(model.nq), "nv": int(model.nv), "nu": int(model.nu),
            "nbody": int(model.nbody), "njnt": int(model.njnt),
            "nsite": int(model.nsite), "ngeom": int(model.ngeom),
            "neq": int(model.neq), "nsensor": int(model.nsensor),
            "timestep_seconds": float(model.opt.timestep),
            "integrator": int(model.opt.integrator),
            "integrator_name": str(mujoco.mjtIntegrator(model.opt.integrator)),
            "solver": str(mujoco.mjtSolver(model.opt.solver)),
            "iterations": int(model.opt.iterations),
            "gravity": model.opt.gravity.tolist(),
            "njmax_config": int(environment._config.njmax),
        },
        "control": {
            "control_dt_seconds": meta["control_dt_seconds"],
            "physics_dt_seconds": meta["physics_dt_seconds"],
            "substeps_per_control_step": meta["substeps_per_control_step"],
            "action_scale": meta["action_scale"],
            "actuator_kp": model.actuator_gainprm[:, 0].tolist(),
            "actuator_kv": (-model.actuator_biasprm[:, 2]).tolist(),
            "actuator_ctrlrange": model.actuator_ctrlrange.tolist(),
            "actuator_names": meta["actuator_names"],
        },
        "feet": {
            "geom_names": meta["foot_geom_names"],
            "condim": [int(model.geom_condim[g]) for g in foot_geom_ids],
            "friction": [model.geom_friction[g].tolist() for g in foot_geom_ids],
            "body_names": [
                mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY,
                                  int(model.geom_bodyid[g]))
                for g in foot_geom_ids
            ],
            "body_mass_kilograms": [
                float(model.body_mass[int(model.geom_bodyid[g])]) for g in foot_geom_ids
            ],
        },
        "grip_equality": {
            "name": mujoco.mj_id2name(
                model, mujoco.mjtObj.mjOBJ_EQUALITY, equality_id),
            "type": int(model.eq_type[equality_id]),
            "type_name": str(mujoco.mjtEq(model.eq_type[equality_id])),
            "site_1": mujoco.mj_id2name(
                model, mujoco.mjtObj.mjOBJ_SITE, int(model.eq_obj1id[equality_id])
            ),
            "site_2": mujoco.mj_id2name(
                model, mujoco.mjtObj.mjOBJ_SITE, int(model.eq_obj2id[equality_id])
            ),
            "solref": model.eq_solref[equality_id].tolist(),
            "solimp": model.eq_solimp[equality_id].tolist(),
            "active": bool(model.eq_active0[equality_id]),
            "note": "CONNECT = a point constraint. The wrist stays free to"
                    " rotate; that is intended, not a missing weld.",
        },
        "ascender_slide": {
            "joint_name": meta["slide_joint_name"],
            "qpos_address": meta["slide_qpos_address"],
            "dof_address": meta["slide_dof_address"],
            "axis_world": model.jnt_axis[slide_joint_id].tolist(),
            "range_meters": meta["slide_range_meters"],
            "damping": meta["slide_damping"],
            "frictionloss": meta["slide_frictionloss"],
            "carrier_body_name": mujoco.mj_id2name(
                model, mujoco.mjtObj.mjOBJ_BODY, meta["carrier_body_id"]),
            "carrier_body_position": model.body_pos[meta["carrier_body_id"]].tolist(),
            "carrier_mass": float(model.body_mass[meta["carrier_body_id"]]),
        },
        "geometry": {
            "slope_degrees": meta["slope_degrees"],
            "slope_axis_world": meta["slope_axis_world"].tolist(),
            "floor_normal_world": data.geom_xmat[int(environment._floor_geom_id)]
            .reshape(3, 3)[:, 2].tolist(),
            "line_point_world": meta["line_point_world"].tolist(),
            "rope_length_meters": meta["rope_length_meters"],
            "rope_tail_meters": meta["rope_tail_meters"],
            "palm_world_position_at_reset": data.site_xpos[
                meta["palm_site_id"]
            ].tolist(),
            "carrier_world_position_at_reset": data.site_xpos[
                meta["carrier_site_id"]
            ].tolist(),
            "palm_minus_carrier_meters": float(
                np.linalg.norm(
                    data.site_xpos[meta["palm_site_id"]]
                    - data.site_xpos[meta["carrier_site_id"]]
                )
            ),
            "pelvis_world_position_at_reset": data.xpos[
                meta["pelvis_body_id"]
            ].tolist(),
        },
        "reset": {
            "keyframe": "knees_bent",
            "keyframe_qpos": meta["keyframe_qpos"].tolist(),
            "default_pose_radians": meta["default_pose_radians"].tolist(),
            "joint_names": meta["joint_names"],
            "base_velocity_randomisation": meta["reset_base_velocity_range"],
        },
        "training_config": {
            "noise_level": meta["noise_level"],
            "noise_scales": meta["noise_scales"],
            "push_enabled": meta["push_enabled"],
            "command_limits": meta["command_limits"],
            "gait_frequency_range_hz": meta["gait_frequency_range_hz"],
            "episode_length": int(environment._config.episode_length),
            "soft_joint_pos_limit_factor": float(
                environment._config.soft_joint_pos_limit_factor
            ),
        },
        "sensor_addresses": meta["sensor_addresses"],
        "versions": _versions(),
    }


def _robot_tables(model, meta):
    """Full body / geom / site / actuator / sensor tables.

    The point is DIFFABILITY. When the training robot gains snow boots or an
    ascender end-effector, the change lands here as a visible row -- a new geom
    with its own friction and condim, a body whose mass moved, a site that
    appeared -- instead of quietly changing the physics under a harness that
    matched by name and therefore never noticed.
    """
    def name_of(object_type, index):
        return mujoco.mj_id2name(model, object_type, index)

    bodies = [{
        "id": body_id,
        "name": name_of(mujoco.mjtObj.mjOBJ_BODY, body_id),
        "parent": name_of(mujoco.mjtObj.mjOBJ_BODY, int(model.body_parentid[body_id])),
        "mass_kilograms": float(model.body_mass[body_id]),
        "subtree_mass_kilograms": float(model.body_subtreemass[body_id]),
        "inertia_diagonal": model.body_inertia[body_id].tolist(),
        "position_in_parent": model.body_pos[body_id].tolist(),
    } for body_id in range(model.nbody)]

    geoms = [{
        "id": geom_id,
        "name": name_of(mujoco.mjtObj.mjOBJ_GEOM, geom_id),
        "body": name_of(mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[geom_id])),
        "type": str(mujoco.mjtGeom(int(model.geom_type[geom_id]))),
        "condim": int(model.geom_condim[geom_id]),
        "friction": model.geom_friction[geom_id].tolist(),
        "size": model.geom_size[geom_id].tolist(),
        "contype": int(model.geom_contype[geom_id]),
        "conaffinity": int(model.geom_conaffinity[geom_id]),
        "solref": model.geom_solref[geom_id].tolist(),
    } for geom_id in range(model.ngeom)]

    sites = [{
        "id": site_id,
        "name": name_of(mujoco.mjtObj.mjOBJ_SITE, site_id),
        "body": name_of(mujoco.mjtObj.mjOBJ_BODY, int(model.site_bodyid[site_id])),
        "position_in_body": model.site_pos[site_id].tolist(),
    } for site_id in range(model.nsite)]

    joints = [{
        "id": joint_id,
        "name": name_of(mujoco.mjtObj.mjOBJ_JOINT, joint_id),
        "type": str(mujoco.mjtJoint(int(model.jnt_type[joint_id]))),
        "body": name_of(mujoco.mjtObj.mjOBJ_BODY, int(model.jnt_bodyid[joint_id])),
        "qpos_address": int(model.jnt_qposadr[joint_id]),
        "dof_address": int(model.jnt_dofadr[joint_id]),
        "range": model.jnt_range[joint_id].tolist(),
        "axis": model.jnt_axis[joint_id].tolist(),
    } for joint_id in range(model.njnt)]

    actuators = [{
        "id": actuator_id,
        "name": name_of(mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_id),
        "kp": float(model.actuator_gainprm[actuator_id, 0]),
        "kv": float(-model.actuator_biasprm[actuator_id, 2]),
        "ctrlrange": model.actuator_ctrlrange[actuator_id].tolist(),
        "forcerange": model.actuator_forcerange[actuator_id].tolist(),
    } for actuator_id in range(model.nu)]

    sensors = [{
        "id": sensor_id,
        "name": name_of(mujoco.mjtObj.mjOBJ_SENSOR, sensor_id),
        "address": int(model.sensor_adr[sensor_id]),
        "dimension": int(model.sensor_dim[sensor_id]),
    } for sensor_id in range(model.nsensor)]

    return {
        "total_mass_kilograms": meta["total_mass_kilograms"],
        "bodies": bodies, "geoms": geoms, "sites": sites,
        "joints": joints, "actuators": actuators, "sensors": sensors,
    }


def _versions():
    import importlib
    out = {"python": sys.version.split()[0]}
    for name in ("mujoco", "jax", "brax", "mujoco_playground", "numpy"):
        try:
            module = importlib.import_module(name)
            out[name] = getattr(module, "__version__", "unknown")
        except Exception as error:  # pragma: no cover - reporting only
            out[name] = f"unavailable ({error})"
    return out


def format_fingerprint(fingerprint):
    m, c, g, e, a = (
        fingerprint["model"], fingerprint["control"], fingerprint["geometry"],
        fingerprint["grip_equality"], fingerprint["ascender_slide"],
    )
    kp = c["actuator_kp"]
    return "\n".join([
        "================ TEAM MODEL FINGERPRINT ================",
        f"source            {fingerprint['source']['class']}",
        f"base xml          {fingerprint['source']['base_xml']}",
        f"nq/nv/nu          {m['nq']} / {m['nv']} / {m['nu']}   neq {m['neq']}"
        f"  nsensor {m['nsensor']}",
        f"timestep          {m['timestep_seconds']} s   integrator"
        f" {m['integrator_name']}   gravity {m['gravity']}",
        f"control           ctrl_dt {c['control_dt_seconds']} s  x"
        f"{c['substeps_per_control_step']} substeps  action_scale"
        f" {c['action_scale']}",
        f"actuator kp       min {min(kp):.1f}  max {max(kp):.1f}"
        f"  ({len(kp)} position actuators)",
        f"robot mass        {fingerprint['robot_tables']['total_mass_kilograms']:.4f} kg"
        f" total  ({len(fingerprint['robot_tables']['bodies'])} bodies,"
        f" {len(fingerprint['robot_tables']['geoms'])} geoms,"
        f" {len(fingerprint['robot_tables']['sites'])} sites -- full tables in"
        " the json)",
        f"feet              {fingerprint['feet']['geom_names']} on"
        f" {fingerprint['feet']['body_names']}",
        f"                  condim {fingerprint['feet']['condim']}  friction"
        f" {fingerprint['feet']['friction']}  body mass"
        f" {[round(v, 4) for v in fingerprint['feet']['body_mass_kilograms']]} kg",
        f"grip equality     {e['type_name']} ({e['name']}) {e['site_1']} <->"
        f" {e['site_2']}   [point constraint: wrist free to rotate, intended]",
        f"                  solref {e['solref'][:2]}  solimp {e['solimp'][:5]}",
        f"ascender_slide    {a['joint_name']} qpos[{a['qpos_address']}]"
        f" dof[{a['dof_address']}]  axis {[round(v, 6) for v in a['axis_world']]}",
        f"                  range {a['range_meters']}  damping {a['damping']}"
        f"  frictionloss {a['frictionloss']}  carrier ({a['carrier_body_name']})"
        f" mass {a['carrier_mass']}",
        f"slope             {g['slope_degrees']} deg   floor normal"
        f" {[round(v, 6) for v in g['floor_normal_world']]}",
        f"line point        {[round(v, 5) for v in g['line_point_world']]}"
        f"  rope length {g['rope_length_meters']} m (+{g['rope_tail_meters']} tail)",
        f"palm at reset     {[round(v, 5) for v in g['palm_world_position_at_reset']]}"
        f"   |palm - carrier| {g['palm_minus_carrier_meters']:.2e} m",
        f"pelvis at reset   {[round(v, 5) for v in g['pelvis_world_position_at_reset']]}",
        f"training noise    level {fingerprint['training_config']['noise_level']}"
        f"  scales {fingerprint['training_config']['noise_scales']}",
        f"pushes            enabled={fingerprint['training_config']['push_enabled']}",
        f"versions          {fingerprint['versions']}",
        "========================================================",
    ])


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return str(value)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--slope", type=float, default=None)
    arguments = parser.parse_args()
    load_team_model(slope_degrees=arguments.slope)
