"""Shutdown and watchdog. Read this before running anything on the robot.

Per the operations page: once you are in developer mode the built-in motion
service has been released, so **there is no separate stop command left to
call**. Terminating your own process IS the emergency stop. That makes the
signal handler below a safety device, not housekeeping.

Two independent ways the robot must end up limp:
  1. SIGINT / SIGTERM  -> damping command to all 29 joints, then exit.
  2. Watchdog          -> no fresh lowstate, or a stalled control loop, or a
                          dropped tunnel, all damp for the same reason.

The tunnel to the robot is transcontinental and will drop mid-session; the
required behaviour on loss of contact is to go limp, not to keep executing the
last command.
"""
import signal
import time

import numpy as np

from . import constants as C

# Zero position/velocity gain, small damping: joints go slack and the harness
# takes the weight. This is the state `robot dev-mode` leaves the robot in.
DAMP_KD = 8.0          # per the access doc: kp=0, kd~8 on all 29 joints.

# Damp if lowstate goes stale or a control step overruns badly.
STALE_TIMEOUT_S = 0.10   # 5 missed policy steps at 50 Hz
OVERRUN_FACTOR = 3.0     # a step taking >3x the period is a stall


class DampingExit(Exception):
    """Raised to unwind the control loop into the damping path."""


class SafetyMonitor:
    """Owns the damping path. Install once, before the policy ever runs.

    `send_damping` must be a callable that writes a zero-torque / damped
    command to all joints and flushes it to the robot.
    """

    def __init__(self, send_damping, logger=print):
        self._send_damping = send_damping
        self._log = logger
        self._damped = False
        self._last_state_t = None
        self._overruns = 0
        self._installed = False

    def install(self) -> None:
        for sig in (signal.SIGINT, signal.SIGTERM):
            signal.signal(sig, self._on_signal)
        self._installed = True
        self._log("safety: SIGINT/SIGTERM -> damping installed")

    def _on_signal(self, signum, _frame):
        self._log(f"safety: signal {signum} -> damping")
        self.damp()
        raise DampingExit(f"signal {signum}")

    def damp(self) -> None:
        """Idempotent. Safe to call from a handler or a finally block."""
        if self._damped:
            return
        try:
            self._send_damping()
        finally:
            self._damped = True
            self._log("safety: damping command sent, robot should be slack")

    def note_state(self, t=None) -> None:
        """Call on every lowstate message."""
        self._last_state_t = time.monotonic() if t is None else t

    def check(self, step_duration=None) -> None:
        """Call once per control step. Raises DampingExit if unsafe."""
        now = time.monotonic()
        if self._last_state_t is None:
            self.damp()
            raise DampingExit("no lowstate ever received")
        age = now - self._last_state_t
        if age > STALE_TIMEOUT_S:
            self.damp()
            raise DampingExit(f"lowstate stale by {age*1e3:.0f} ms")
        if step_duration is not None and step_duration > OVERRUN_FACTOR * C.CTRL_DT:
            self._overruns += 1
            self._log(f"safety: step overrun {step_duration*1e3:.1f} ms "
                      f"(count {self._overruns})")
            if self._overruns >= 5:
                self.damp()
                raise DampingExit("repeated control-loop overruns")


def sanity_check_targets(targets, lowers=None, uppers=None) -> np.ndarray:
    """Last line of defence before a command leaves the process."""
    t = np.asarray(targets, dtype=np.float32)
    if t.shape != (C.N_JOINTS,):
        raise ValueError(f"expected {C.N_JOINTS} targets, got {t.shape}")
    if not np.all(np.isfinite(t)):
        raise ValueError("non-finite joint target")
    if lowers is not None and uppers is not None:
        t = np.clip(t, lowers, uppers)
    return t
