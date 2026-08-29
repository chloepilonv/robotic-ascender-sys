# Unitree G1 — Himalaya edition (Isaac Sim USD)

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
| foot spheres | `Sphere` colliders with friction 0.6 |
| freejoint | floating base: `ArticulationRootAPI` on `/G1`, no fixed joint |

Joint prim names == MJCF joint names (`left_hip_pitch_joint`, …) → Isaac Lab `joint_names_expr` regexes from the stock G1 cfg work unchanged.

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
    spawn=sim_utils.UsdFileCfg(usd_path="assets/g1/g1_himalaya.usd", activate_contact_sensors=True),
    init_state=ArticulationCfg.InitialStateCfg(pos=(0, 0, 0.8)),
    actuators={...},  # copy from isaaclab_assets.robots.unitree.G1_CFG
)
```
