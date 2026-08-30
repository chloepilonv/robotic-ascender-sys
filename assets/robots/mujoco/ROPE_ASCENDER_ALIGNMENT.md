# Rope ↔ ascender alignment — the contract

**One file defines how the ascender sits on the rope: `rope_rail.py`. Every MuJoCo scene, every
policy (training *and* deployment) must go through it. Do not hand-place a rope.**

## The relation (tool frame = the ascender mesh frame, mounted on `right_wrist_yaw_link`)

| Quantity | Value | Where |
|---|---|---|
| Rope channel centre | `CHANNEL_CENTRE_TOOL = (-0.0213, -0.0012, 0.055)` m | measured on `meshes/ascender_visual.obj` (Ø16 mm bore under the cam) |
| Channel axis | tool Z pitched `CHANNEL_PITCH_DEG = -6.4°` about tool y | measured from 3 mesh slices — **tune by eye, then freeze** |
| Tool mount on the wrist | `pos 0.0386 0 -0.0514`, `quat 0 0.2245 0 0.9745` | `g1_unitree_ascender.xml` (do not move) |
| Rope | visual cylinder, Ø11 mm, along world **+x** (= uphill), through the channel at reset | `add_rope_rail` |
| Rope height at reset | `ROPE_HEIGHT = 0.60` m | arm IK solves shoulder/elbow/wrist for it |
| Mechanism | `rope_carriage`: ONE slide joint `rope_slide` along +x, **welded** to the wrist so the channel axis = rope axis | `add_rope_rail` |
| Cam (up only) | slide's lower limit = highest point reached (`ratchet()` before every `mj_step`) | never overwrite qpos |
| Cam drag | `CAM_FRICTION_N = 3` N | |
| Slope | tilt **gravity** `(-g sin s, 0, -g cos s)`, keep the floor flat and the rope on +x | `rl/chloe/task/robot.py` |

## Rules
1. A policy is valid only with the `rope_rail.py` values it was trained with. Change a value → retrain.
2. Inside mjlab the names are prefixed: `robot/rope_slide`, `robot/rope_carriage`, `robot/ascender_anchor`.
3. Check before you use it: `python assets/robots/mujoco/rope_rail_check.py` (prints the numbers, fails if misaligned).
4. To tune the bend: `mjpython -m rl.chloe.scripts.sim2sim <policy.onnx> --channel-pitch <deg>`, then write the value into `CHANNEL_PITCH_DEG`.

## Frames, in words
Rope = x axis. The tool's channel (green cylinder in the viewer) is welded onto that axis at the
carriage. The carriage moves in +x only. The arm joints do the rest. The tool does **not** rotate
around the rope (pure prismatic), by decision — keep it simple and static.
