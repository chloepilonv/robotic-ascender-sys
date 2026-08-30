#!/bin/bash
# EMERGENCY STOP -- run from any terminal, at any time.
#
#   ~/estop.sh
#
# 1. SIGTERM the policy/controller process. Its handler writes kp=0, kd=8
#    damping to all 29 joints, then exits. Robot goes slack, harness holds it.
# 2. Only if that fails, SIGKILL. SIGKILL SKIPS the damping handler and leaves
#    the last command latched -- that is why it is step 2, never step 1.
# 3. Hand control back to the built-in motion service, which holds the robot.
#
# Pattern matches the python interpreter running a session script, so it cannot
# match an ssh command line that merely mentions the path.
set +e
PAT='python3?[^|]*session/0[0-9]_'

echo "[estop] targets:"; pgrep -af "$PAT" || echo "  (nothing running)"
echo "[estop] SIGTERM -> damping handlers"
pkill -TERM -f "$PAT"
for _ in $(seq 1 15); do pgrep -f "$PAT" >/dev/null || break; sleep 0.2; done

if pgrep -f "$PAT" >/dev/null; then
  echo "[estop] STILL ALIVE -> SIGKILL (damping handler will NOT run)"
  pkill -KILL -f "$PAT"; sleep 0.5
fi
echo "[estop] survivors:"; pgrep -af "$PAT" || echo "  none"

echo "[estop] handing back to the built-in motion service"
robot normal
robot status
