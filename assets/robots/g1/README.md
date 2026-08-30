# Unitree G1 — Himalaya edition (Isaac Sim USD)

**Main robot: `assets/robots/g1_unitree.usd`** = dressed G1 **with the ascender end-effector** (references `g1_unitree_ascender.usd`).
`assets/robots/g1_unitree_bare.usd` = same robot with both rubber hands (references `g1/g1_himalaya.usd`). All three are thin reference stages; keep the `assets/` tree together.

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

## Ascender variant (ascender = right end-effector)
`assets/robots/g1_unitree_ascender.usd` = same robot with the **right rubber hand removed and a handleless (Petzl Basic-style, 110 x 73 mm) ascender bolted on in its place**
(handle along the forearm, base 2 cm inside the wrist link, flat face in the wrist X-Y plane). Built by `attach_tool.py`: the visual
**references** `assets/ascender/ascender.usd` (textured, so keep the `assets/` tree together), convex collision baked into
`right_wrist_yaw_link`, +0.165 kg folded into the link mass, so the articulation is unchanged (29 DoF)
and the Isaac Lab cfg works as-is. The end-effector frame is `/G1/right_wrist_yaw_link/tool_ascender` (origin = tool base, +Z = along tool to the cam head).
A black mounting flange (`tool_flange`, cylinder r=26 mm) bridges the wrist link and the cam head. Tune `TOOL_POS` / `_R` / `FLANGE_*` in `attach_tool.py`. Scene: `himalaya_scene.py --robot g1_unitree_ascender`.

### Policy note — wrist offset
The ascender is fixed to `right_wrist_yaw_link`, so its pose is entirely set by the 3 wrist joints
(`right_wrist_roll/pitch/yaw_joint`). Any policy or IK that targeted the rubber hand must be re-targeted to the
tool frame `/G1/right_wrist_yaw_link/tool_ascender` (+Z = up through the cam head, rope runs along it):
the "hand" now points +Z of the wrist instead of +X, i.e. roughly a **-90° wrist-pitch offset** vs. the stock hand,
plus the +0.11 kg on the link. Add that offset to the wrist default / target joint positions in the env cfg.

## MJCF (MuJoCo)
See `../mujoco/README.md` — `g1_unitree.xml` / `g1_unitree_ascender.xml`, built by `../mujoco/build.py` from the same tables.
