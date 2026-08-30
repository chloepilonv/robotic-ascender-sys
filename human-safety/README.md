# human-safety — the human-detection gate

A standalone safety program, not part of the RL policy and not part of the
harness: **the robot may not climb UP while a human is in front of it.**

- `human_gate.py` — `HumanWorld` (hikers from `assets/humans/`, or virtual
  ones), `VirtualFrustumDetector` (d435i camera frustum oracle for MuJoCo),
  `HumanGate` (state machine with 1 s hysteresis; `mask()` clamps
  `lin_vel_x <= 0`).
- `test_human_gate.py` — `python -m pytest human-safety -q` (needs only mujoco).

The harness (`app/harness/runtime.py --human <m>`) is one consumer. On the real
robot the same `HumanGate` runs with a RealSense + YOLO detector implementing
`HumanDetector.detect()`.

Import: the folder name has a hyphen (it is a program, not a Python package),
so consumers do `sys.path.insert(0, "<repo>/human-safety"); import human_gate`.
