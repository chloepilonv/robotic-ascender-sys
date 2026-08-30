---
name: real-to-sim
description: Turn a unitree_sdk2py script written for the real G1 into something you can run in MuJoCo (or Isaac) before touching the robot. Covers the backend split, the SDK-to-sim concept map, measuring geometry/signs from the MJCF, and a metrics report. Invoke with /real-to-sim <script>.
---

# real-to-sim

Goal: run the SAME sequencing/IK code against a simulated robot, so the only thing left to
trust on the real G1 is the transport (DDS) and the parts sim cannot model.

Worked example: `deploy/rope_walk.py` (`--dry-run` / `--sim` / real). Everything lives in that
one file: the scene is built in code with `mujoco.MjSpec`, the gait is the mels walking policy.

## 1. Split the script into "brain" and "body"

The brain (sequencer, IK, timing) must never import the SDK. The body is a small class
with ~8 methods; write one per target and pick it on the CLI:

| method          | real (`G1`)                                   | sim (`Sim`)                                  |
|-----------------|-----------------------------------------------|----------------------------------------------|
| `start()`       | `LocoClient.Start()`                          | load `knees_bent` keyframe, walking policy on |
| `send_arm(pose)`| write `LowCmd_` on `rt/arm_sdk` + CRC         | set `d.ctrl[j]` (position actuators)         |
| `set_arm_weight`| `motor_cmd[29].q = w`                         | `ctrl = w*target + (1-w)*keyframe`           |
| `move(vx)`      | `LocoClient.Move(vx,0,0)`                     | policy command vx (+ goal x advances)        |
| `stop_move()`   | `LocoClient.StopMove()`                       | `vx = 0`, position hold on goal x,y          |
| `tilt_ok()`     | `lowstate.imu_state.rpy`                      | pelvis quaternion `qpos[3:7]` -> roll/pitch  |
| `current_arm()` | `lowstate.motor_state[j].q`                   | `d.qpos[qposadr[j]]` or last `ctrl`          |
| `sleep(s)`      | `time.sleep(s)`                               | `mj_step` for `s/dt` steps, then log/render  |

Rule: if a method needs an SDK type, it lives in the body. If the brain needs a value the
SDK gives for free (IMU, joint q), the body exposes it as a plain float/dict.

## 2. SDK concept -> MuJoCo concept

| unitree_sdk2py                              | MuJoCo                                                          |
|---------------------------------------------|-----------------------------------------------------------------|
| motor index 0..28 (`rt/lowcmd`, `rt/lowstate`) | actuator index 0..28 in `g1_unitree.xml` — **same order** (verified) |
| `LowCmd_.motor_cmd[j].q/kp/kd`              | `<position kp=...>` actuator, `d.ctrl[j] = q`                   |
| `rt/arm_sdk` weight (motor 29)              | linear blend between your target and the keyframe ctrl          |
| `LocoClient.Move`                           | a walking policy: `rl/environment/walk_policy.WalkController` (mels G1 joystick MLP, NumPy) on the model patched by `rl/environment/robot.adapt()`, plus an outer x/y goal + yaw hold |
| `LocoClient.StopMove` / standing            | the policy cannot stand (it marches and drifts). STAND mode = wait for double support, freeze the *actual* joint pose with stiff gains (kp 500), plus an ankle-strategy CoM feedback (a stiff statue topples in ~1 s). Freezing the keyframe pose, or blending from the policy's targets, both fall over |
| `rt/lowstate.imu_state.rpy`                 | `d.xquat[pelvis]` or the `imu_in_pelvis` site                   |
| `rt/lowstate.motor_state[j].q`              | `d.qpos[m.jnt_qposadr[joint_id]]` (skip the 7 free-joint dofs!)  |
| `CRC()`                                     | nothing                                                         |
| `ChannelFactoryInitialize(0, iface)`        | `MjModel.from_xml_path(scene.xml)`                              |
| 500 Hz `rt/lowstate`                        | `m.opt.timestep = 0.002`                                        |

## 3. Never guess geometry or signs — measure them in the MJCF

The MJCF is generated from the same URDF the robot runs, so it is the ground truth.

```python
m = mujoco.MjModel.from_xml_path("assets/robots/mujoco/g1_unitree.xml"); d = mujoco.MjData(m)
mujoco.mj_resetDataKeyframe(m, d, m.key("stand").id); mujoco.mj_forward(m, d)
d.xpos[m.body("right_shoulder_pitch_link").id]   # -> shoulder height/offset (1.085, -0.10)
# sign of a joint: set qpos for that joint only, mj_forward, watch where the wrist goes
```

Findings for the G1 that broke the first drafts of `rope_walk.py`:
- a mocap-dragged robot "flies"; it must actually walk (policy) or the test is meaningless
- shoulder z is 1.085 m, not ~0.95 → a 0.60 m rope is 0.10 m out of reach with a vertical arm
- elbow `q=0` is already bent 90° (forearm horizontal); `q≈+π/2` is a straight arm
- shoulder pitch `<0` = forward, right shoulder roll `<0` = outward
- analytic 2-link IK was 0.15-0.3 m off (roll is applied after pitch). Use numerical IK on the
  MJCF (`MjIK`: `mj_jacSite` + damped least squares on 4 joints) — same URDF as the robot, so
  the joint targets transfer as-is

## 4. Build the scene in code (MjSpec)

```python
spec = mujoco.MjSpec.from_file(robot_mod.HIMALAYA_ROBOT_BARE)
robot_mod.adapt(spec)                 # sensors, knees_bent keyframe, RL gains, right_palm site
spec.worldbody.add_geom(name="floor", type=mjGEOM_PLANE, ...)
spec.worldbody.add_geom(name="rope", type=mjGEOM_CAPSULE, fromto=[...], contype=0, conaffinity=0)
spec.visual.global_.offwidth = 1280   # needed for headless video
m = spec.compile()
```
No separate XML: the rope height comes from the same constant the real script uses.

Arm override on top of the policy (this IS what arm_sdk's weight does on the robot):
`d.ctrl[j] = w*target + (1-w)*policy_ctrl[j]` for the right-arm joints only, every substep.

## 5. Make the sim tell you numbers, not just pictures

Log the task-relevant world-frame quantity every `sleep()` (here: palm xyz), tag it with
the sequencer phase, print a per-phase table at the end. Ask each phase one question:
- GRIP/REGRIP: `hand z - rope z ≈ 0`? `hand y - rope y ≈ 0`?
- WALK: `hand x drift ≈ 0` while the arm tracks (hand fixed on the rope)?
Render `--video out.mp4` (offscreen `mujoco.Renderer` + `imageio`) to sanity-check the pose.

## 5b. Lessons from the rope walk (things that cost hours)

- **Never open-loop the hand.** Re-solve the arm every tick from base odometry
  (`base_pose()` → `hand_at_world()`); the report's "hand x drift ≈ 0 during WALK" proves it.
- **Reach check must be 3-D** (dx, dy, dz from the shoulder). A 2-D check said 0.70 m was fine;
  the lateral 0.2 m made it unreachable, the clamp lifted the hand off the rope, and the next
  walk started with a stretched arm and fell.
- **IK branch flips.** 4 joints for a 3-D point let DLS jump to a wild branch mid-walk. Use 3
  joints (pitch/roll/elbow), and rate-limit joint targets (`MAX_DQ_TICK`).
- **Isolate before tuning.** `walk→stand→walk` alone worked 4×; only the sequencer's REGRIP broke
  walk 2. Test the suspect phase standalone before touching gains.
- **The mels policy steers badly** (yaw command sensitivity 0.23 vs 1.65 for vx, veers −30°/3 s
  even on its training model). Documented sim hack: virtual yaw/lateral springs on the pelvis,
  plus x/roll/pitch springs while standing (`GUIDE_*`, `STAND_*` in `Sim`). Never a z force.
- **Ascender = one-way cam** (slides up freely, bites under load): the grip point only advances.

## 6. What sim will NOT catch — check these on the robot with the e-stop in hand

1. Onboard loco controller ≠ the mels policy (different gait, real StopMove, different reaction to an arm pushing on a rope).
2. DDS plumbing: `--iface`, topic names, CRC, `arm_sdk` weight actually taking over.
3. Rope contact forces / friction on the palm.
4. Real motor kp/kd (sim uses the MJCF `kp=500`; real uses your `KP_ARM`).

## Workflow

1. `python script.py --dry-run` — sequence prints, no robot, no sim.
2. `../himalayas-rl/.venv-mjlab/bin/python script.py --sim --video out.mp4` — read the report, watch the mp4.
3. Fix geometry/signs in the script until the report is green.
4. Real: `--speed 0.2 --cycles 1`, someone on the e-stop.
