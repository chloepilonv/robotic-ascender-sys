"""Isaac Lab configs for the Himalaya G1 (assets/g1/g1_himalaya.usd).

Usage in an env cfg:
    from assets.g1.g1_himalaya_cfg import G1_HIMALAYA_CFG, G1_SENSORS
    robot = G1_HIMALAYA_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    imu_pelvis = G1_SENSORS["imu_pelvis"]; contact = G1_SENSORS["contact_feet"]; ...
"""
import os
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.sensors import CameraCfg, ContactSensorCfg, ImuCfg, RayCasterCfg, patterns

USD_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "g1_himalaya.usd")

G1_HIMALAYA_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path=USD_PATH,
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False, retain_accelerations=False, linear_damping=0.0, angular_damping=0.0,
            max_linear_velocity=1000.0, max_angular_velocity=1000.0, max_depenetration_velocity=1.0),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False, solver_position_iteration_count=8, solver_velocity_iteration_count=4),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.79),
        joint_pos={".*_hip_pitch_joint": -0.20, ".*_knee_joint": 0.42, ".*_ankle_pitch_joint": -0.23,
                   ".*_elbow_joint": 0.87, "left_shoulder_roll_joint": 0.16, "left_shoulder_pitch_joint": 0.35,
                   "right_shoulder_roll_joint": -0.16, "right_shoulder_pitch_joint": 0.35},
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    # Gains mirror isaaclab_assets G1_CFG; torque limits = MJCF actuatorfrcrange.
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[".*_hip_yaw_joint", ".*_hip_roll_joint", ".*_hip_pitch_joint", ".*_knee_joint", "waist_.*_joint"],
            effort_limit_sim=139.0, velocity_limit_sim=32.0,
            stiffness={".*_hip_yaw_joint": 150.0, ".*_hip_roll_joint": 150.0, ".*_hip_pitch_joint": 200.0,
                       ".*_knee_joint": 200.0, "waist_.*_joint": 200.0},
            damping={".*_hip_yaw_joint": 5.0, ".*_hip_roll_joint": 5.0, ".*_hip_pitch_joint": 5.0,
                     ".*_knee_joint": 5.0, "waist_.*_joint": 5.0},
            armature=0.01),
        "feet": ImplicitActuatorCfg(
            joint_names_expr=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"],
            effort_limit_sim=50.0, velocity_limit_sim=37.0, stiffness=20.0, damping=2.0, armature=0.01),
        "arms": ImplicitActuatorCfg(
            joint_names_expr=[".*_shoulder_pitch_joint", ".*_shoulder_roll_joint", ".*_shoulder_yaw_joint",
                              ".*_elbow_joint", ".*_wrist_.*_joint"],
            effort_limit_sim=25.0, velocity_limit_sim=37.0, stiffness=40.0, damping=10.0, armature=0.01),
    },
)

G1_SENSORS = {
    # 2 IMUs (gyro + accel), same frames as the MJCF sites / real robot
    "imu_pelvis": ImuCfg(prim_path="{ENV_REGEX_NS}/Robot/pelvis/imu_in_pelvis", update_period=0.0,
                         gravity_bias=(0.0, 0.0, 0.0)),
    "imu_torso": ImuCfg(prim_path="{ENV_REGEX_NS}/Robot/torso_link/imu_in_torso", update_period=0.0,
                        gravity_bias=(0.0, 0.0, 0.0)),
    # foot contacts (snow/ice reward terms)
    "contact_feet": ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*_ankle_roll_link", history_length=3,
                                     track_air_time=True),
    # Intel RealSense D435i on the face (RGB + depth), 87x58 deg
    "camera_d435i": CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/torso_link/head_sensors/d435i_camera",
        update_period=1 / 30, height=240, width=424, data_types=["rgb", "distance_to_image_plane"],
        spawn=None),  # camera prim already exists in the USD
    # Livox Mid-360 on top of the head: 360 deg x (-7..52 deg), ray-cast against the terrain
    "lidar_mid360": RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/torso_link/head_sensors/mid360_lidar",
        update_period=0.1, max_distance=40.0, mesh_prim_paths=["/World/ground"],
        pattern_cfg=patterns.LidarPatternCfg(channels=16, vertical_fov_range=(-7.0, 52.0),
                                             horizontal_fov_range=(-180.0, 180.0), horizontal_res=2.0)),
}
