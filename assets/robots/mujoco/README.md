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

## Rope + ascender rail — `rope_rail.py` (use this, do not rebuild your own rope)

The ascender on the rope, as a mechanism: `rope` (visual cylinder along +x) and a `rope_carriage`
with ONE **slide** joint along the rope (`rope_slide`, a prismatic joint), **welded** to
`right_wrist_yaw_link` so the rope always runs through the ascender's channel (green cylinder =
the channel). The tool slides along the rope, nothing else.

```python
import mujoco, sys; sys.path.insert(0, "assets/robots/mujoco")
import rope_rail as rail

spec = mujoco.MjSpec.from_file("assets/robots/mujoco/g1_unitree_ascender.xml")
joint_pos = rail.add_rope_rail(spec, root_pos=(0, 0, 0.8), root_quat=(1, 0, 0, 0),
                               joint_pos={".*_knee_joint": 0.669, ".*_hip_pitch_joint": -0.312,
                                          ".*_ankle_pitch_joint": -0.363})  # your reset pose
model = spec.compile(); data = mujoco.MjData(model)
rail.set_pose(model, data, (0, 0, 0.8), (1, 0, 0, 0), joint_pos)   # wrist angles solved for you

rail.ratchet_reset(model, data)          # after any reset / teleport
rail.ratchet(model, data)                # the cam: up only — call BEFORE every mj_step
mujoco.mj_step(model, data)
print(rail.rope_state(model, data))      # {"rope_progress_m", "tension_N", "engaged"}
```

- `add_rope_rail` sizes the rope from the reset pose you give it and returns the joint dict
  **including the solved wrist angles** (channel parallel to the rope) — use that dict for the reset.
- Slope: tilt gravity (`model.opt.gravity = (-g sin s, 0, -g cos s)`, +x uphill) rather than the floor;
  the rope stays a +x line. This is what the RL task does (`rl/chloe/task/robot.py`).
- Inside mjlab the names are prefixed: `robot/rope_slide` etc. (`prefix="robot/"` in `ratchet`/`rope_state`).
- The cam is a **moving lower joint limit** (never overwrite qpos: that fights the solver and the
  weld drifts by centimetres). Cam friction 3 N, rope Ø11 mm; MuJoCo `njmax` ≥ ~1000 or the weld drops.
