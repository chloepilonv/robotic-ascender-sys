# Hardware diagnostics

Written during the first G1 gantry session, in the order they were needed. Each
answers one question with a measurement rather than an impression — the session
lost about an hour to reasoning about `|q-tgt|max`, a single number that hides
*which* joint is failing, and every tool here reports per-joint or per-dimension.

Run with the **system** interpreter (`/usr/bin/python3`), not a conda env — conda
shadows `python3` and the Unitree SDK is only on the system one. Interface is
`enP8p1s0` on this robot, not `eth0`.

| script | question it answers | publishes? |
|---|---|---|
| `motorchk.py` | are motors faulted / enabled? per-joint mode, temp, bus voltage, torque | no |
| `crcbench.py` | what does the publish path cost? CRC vs DDS write, feasible rates | no |
| `shadow_policy.py` | is the policy sane on real data? runs it, logs the action, **applies nothing**; reports per-dimension distance from the training distribution | holds a pose |
| `tracktest.py` | which joints can reach their targets? per-joint error and torque | yes |
| `ratetest.py` | does command *rate* gate motor enable? holds current pose at 50 vs 500 Hz | yes |
| `fsm_test.py` | will the loco FSM accept a walk-ready transition? | state change |
| `stand_test.py` | can the API put the robot into an active stance? | real motion |
| `estop.sh` | **emergency stop** — SIGTERM to the controller (its handler damps), SIGKILL only if that fails, then `robot normal` | — |

## estop.sh

Keep a terminal open with this ready before commanding anything. It targets our
own low-level controllers. It is the wrong tool for `rope_walk.py`, which hands
the arm back to the loco controller instead — Ctrl-C there.

Never `kill -9` first: without the damping handler the last command stays
**latched** and the robot does not go limp, which is worse than doing nothing.

## What these found

- `tracktest.py`: torque = kp x error on every joint, so the motors were fine —
  and `kp=2` wrists sit ~0.45 rad off target, 5 sigma outside training.
- `shadow_policy.py`: policy jitter p99 0.160 open-loop vs 0.658 closed-loop, the
  measurement that showed the instability comes from feedback, not the network.
- `ratetest.py`: motors enable at 50 Hz, killing the command-rate hypothesis.
- `fsm_test.py` / `stand_test.py`: every loco call returns success and does
  nothing while the robot is not actively standing.
