# assets/robots/mujoco — the Himalaya G1 for MuJoCo (MJCF)

Same robot as `../g1_unitree.usd` / `../g1_unitree_ascender.usd` (Isaac), built from the same tables.

| File | What |
|---|---|
| `g1_unitree.xml` | G1 29-DoF + jacket, boots, logos, boot friction 0.8, D435i camera, Mid-360 site |
| `g1_unitree_ascender.xml` | **the full robot**: above + ascender on the right wrist (hand removed, +0.1 kg) |
| `ascender.xml` | the tool alone as a free body |
| `meshes/` | generated jacket/boot hulls, logo patches, ascender mesh + PNG textures (tracked) |
| `build.py` | regenerates everything above (only needed if you change the gear, the mount, or menagerie) |

## Use (already built — 2 steps)
1. Fetch the stock Unitree link meshes once per clone (34 MB, git-ignored, from google-deepmind/mujoco_menagerie):
       python assets/robots/mujoco/build.py --fetch
2. Load it inside YOUR scene (the robot file has no floor on purpose — terrain lives in `assets/environments/`):
   ```xml
   <mujoco>
     <include file="assets/robots/mujoco/g1_unitree_ascender.xml"/>
     <worldbody><light pos="0 0 3"/><geom type="plane" size="0 0 0.05"/></worldbody>
   </mujoco>
   ```
   Quick look without physics: `python -m mujoco.viewer --mjcf $PWD/assets/robots/mujoco/g1_unitree_ascender.xml` (press Tab for joint sliders).

Actuators are position servos in SDK joint order (`ctrl` = target angle, rad). Keyframe 0 = standing pose.

## Rebuild
    pip install mujoco trimesh usd-core pillow
    python assets/robots/mujoco/build.py
Sources: `../g1/build_g1_usd.py` tables (GEAR, GEAR_PRIMS, BOOT_FRICTION), `../g1_unitree_ascender.usd` (mount pose),
`../../ascender/ascender.usd` + textures, `../g1/textures/everest_logo.png`.

## Physics
Stock menagerie dynamics (masses, inertias, armature 0.01, friction loss 0.3, torque limits, 2 ms step).
Ours: boot friction 0.8 (stock 0.6), ascender mass 0.1 kg folded into the wrist inertial, ascender contact = convex hull.
Jacket/boots/logos are visual only: no contact, no mass. Not ported: metallic/roughness/normal maps (MuJoCo has no PBR).
