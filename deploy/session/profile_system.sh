#!/bin/bash
# System-level profiling alongside a running controller: CPU, RAM, thermals.
# Run in a THIRD terminal while 05_step.py is going. Read-only, touches nothing.
#   ~/session/profile_system.sh 30
SECS=${1:-30}
echo "sampling ${SECS}s -- tegrastats + per-process"
( timeout "$SECS" tegrastats --interval 1000 2>/dev/null || echo "tegrastats unavailable" ) &
TG=$!
END=$((SECONDS+SECS))
while [ $SECONDS -lt $END ]; do
  PID=$(pgrep -f "python3?[^|]*session/0[0-9]_" | head -1)
  if [ -n "$PID" ]; then
    read -r _ _ _ _ _ _ _ _ _ _ _ _ _ UT ST _ < /proc/$PID/stat
    RSS=$(awk '/VmRSS/{print $2/1024" MB"}' /proc/$PID/status)
    echo "  pid=$PID rss=$RSS utime=$UT stime=$ST"
  else
    echo "  (no controller running)"
  fi
  sleep 2
done
wait $TG 2>/dev/null
