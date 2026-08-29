# Unitree G1 — Himalaya edition (Isaac Sim USD)

**Shared file: `assets/robots/g1_unitree.usd`** (identical copy of `assets/robots/g1/g1_himalaya.usd`, kept in sync by `build_g1_usd.py`).

`g1_himalaya.usd` = the 29-DoF Unitree G1 from
[mujoco_menagerie/unitree_g1](https://github.com/google-deepmind/mujoco_menagerie/tree/main/unitree_g1)
converted to USD (PhysX articulation) + expedition gear:

- **Jacket** (blue): inflated convex hulls on `torso_link`, shoulder/elbow links.
- **Plastic boots** (yellow): inflated hulls on `*_ankle_roll_link` / `*_ankle_pitch_link` + shin cuffs on `*_knee_link`.

Gear lives under `/G1/<link>/gear/*`, is **visual only** (no collision, no mass), so physics = stock menagerie G1.

## What's converted
| MJCF | USD |
|---|---|
| bodies (30) | `Xform` + `RigidBodyAPI` + `MassAPI` (mass, CoM, inertia, principal axes) |
| hinge joints (29) | `RevoluteJoint` (limits in deg) + `DriveAPI angular` (kp=500, kv from MJCF), `physxJoint:armature/jointFriction` |
| visual meshes | `Mesh` under `<link>/visuals`, metal/black `UsdPreviewSurface` |
| collision meshes | `Mesh` under `<link>/collisions`, convex hull, purpose=guide |
| foot spheres | `Sphere` colliders, friction `BOOT_FRICTION` = 0.8 (stock 0.6; `--no-gear` keeps 0.6). **Changing it alters slip dynamics → retrain/re-eval the policy.** |
| freejoint | floating base: `ArticulationRootAPI` on `/G1`, no fixed joint |

Joint prim names == MJCF joint names (`left_hip_pitch_joint`, …) → Isaac Lab `joint_names_expr` regexes from the stock G1 cfg work unchanged.

## Sensors
| Prim | Type | Real hardware |
|---|---|---|
| `/G1/pelvis/imu_in_pelvis` | Xform (MJCF site) | pelvis IMU (gyro+accel) |
| `/G1/torso_link/imu_in_torso` | Xform (MJCF site) | torso IMU |
| `/G1/torso_link/head_sensors/d435i_camera` | `Camera` (87x58 deg, looks +X, Z up) | Intel RealSense D435i |
| `/G1/torso_link/head_sensors/mid360_lidar` | Xform on top of head | Livox Mid-360 |
| `/G1/*_ankle_roll_link` | rigid body w/ contact reporting | foot contact |

`g1_himalaya_cfg.py` = Isaac Lab `ArticulationCfg` + `ImuCfg` x2 + `ContactSensorCfg` + `CameraCfg` + `RayCasterCfg` for all of the above.

## Rebuild
```bash
pip install mujoco usd-core trimesh scipy
python build_g1_usd.py            # clones menagerie (unitree_g1 only) into _menagerie/, writes g1_himalaya.usd
python build_g1_usd.py --no-gear  # plain G1
```

## Use in Isaac Lab
```python
from isaaclab.assets import ArticulationCfg
import isaaclab.sim as sim_utils
G1_HIMALAYA_CFG = ArticulationCfg(
    ...)  # see g1_himalaya_cfg.py for the full, ready-to-use config
```

## Ascender variant
`assets/robots/g1_unitree_ascender.usd` = same robot holding the ascender (`assets/ascender/ascender.usd`) in the right hand.
Built by `attach_tool.py`: tool baked into `right_wrist_yaw_link` (visual + convex collision, +0.165 kg folded into the link mass),
so the articulation is unchanged (29 DoF) and the Isaac Lab cfg works as-is. Grip = handle vertical in front of the palm, cam head up.
Tune `TOOL_POS` / `TOOL_ROT` in `attach_tool.py` to change the grip. Scene: `himalaya_scene.py --robot g1_unitree_ascender`.
