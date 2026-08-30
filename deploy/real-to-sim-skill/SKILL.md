---
name: real-to-sim
description: Turn a unitree_sdk2py script written for the real G1 into something you can run in MuJoCo (or Isaac) before touching the robot. Covers the backend split, the SDK-to-sim concept map, measuring geometry/signs from the MJCF, and a metrics report. Invoke with /real-to-sim <script>.
---

# real-to-sim

Goal: run the SAME sequencing/IK code against a simulated robot, so the only thing left to
trust on the real G1 is the transport (DDS) and the parts sim cannot model.

Worked example: `deploy/rope_walk.py` (`--dry-run` / `--sim` / real) + `deploy/sim/scene.xml`.

## 1. Split the script into "brain" and "body"

The brain (sequencer, IK, timing) must never import the SDK. The body is a small class
with ~8 methods; write one per target and pick it on the CLI:

| method          | real (`G1`)                                   | sim (`Sim`)                                  |
|-----------------|-----------------------------------------------|----------------------------------------------|
| `start()`       | `LocoClient.Start()`                          | weld pelvis, load `stand` keyframe           |
| `send_arm(pose)`| write `LowCmd_` on `rt/arm_sdk` + CRC         | set `d.ctrl[j]` (position actuators)         |
| `set_arm_weight`| `motor_cmd[29].q = w`                         | `ctrl = w*target + (1-w)*keyframe`           |
| `move(vx)`      | `LocoClient.Move(vx,0,0)`                     | drag mocap anchor `x += vx*dt`               |
| `stop_move()`   | `LocoClient.StopMove()`                       | `vx = 0`                                     |
| `tilt_ok()`     | `lowstate.imu_state.rpy`                      | pelvis `xquat` -> rpy (or `True` if welded)  |
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
| `LocoClient` (balance + gait, onboard)      | NOT simulable. Weld pelvis to a mocap body, drag it forward. Or plug your own RL walk policy (mjlab worktree) |
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

Findings for the G1 that broke the first draft of `rope_walk.py`:
- shoulder z is 1.085 m, not ~0.95 → a 0.60 m rope is 0.10 m out of reach with a vertical arm
- elbow `q=0` is already bent 90° (forearm horizontal); `q≈+π/2` is a straight arm
- shoulder pitch `<0` = forward, right shoulder roll `<0` = outward

## 4. Build the scene

`deploy/sim/scene.xml`: `<include>` the robot MJCF, set `meshdir` so its relative mesh
paths still resolve, add the environment object (rope capsule), a mocap `anchor` body and
`<equality><weld body1="anchor" body2="pelvis"/>`. Give `<visual><global offwidth offheight>`
so headless rendering works.

## 5. Make the sim tell you numbers, not just pictures

Log the task-relevant world-frame quantity every `sleep()` (here: palm xyz), tag it with
the sequencer phase, print a per-phase table at the end. Ask each phase one question:
- GRIP/REGRIP: `hand z - rope z ≈ 0`? `hand y - rope y ≈ 0`?
- WALK: `hand x drift ≈ 0` while the arm tracks (hand fixed on the rope)?
Render `--video out.mp4` (offscreen `mujoco.Renderer` + `imageio`) to sanity-check the pose.

## 6. What sim will NOT catch — check these on the robot with the e-stop in hand

1. Onboard loco controller behaviour (gait, how it reacts to an arm pushing on a rope).
2. DDS plumbing: `--iface`, topic names, CRC, `arm_sdk` weight actually taking over.
3. Rope contact forces / friction on the palm.
4. Real motor kp/kd (sim uses the MJCF `kp=500`; real uses your `KP_ARM`).

## Workflow

1. `python script.py --dry-run` — sequence prints, no robot, no sim.
2. `../himalayas-rl/.venv-mjlab/bin/python script.py --sim --video out.mp4` — read the report, watch the mp4.
3. Fix geometry/signs in the script until the report is green.
4. Real: `--speed 0.2 --cycles 1`, someone on the e-stop.
