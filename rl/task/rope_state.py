"""Compat shim: the telemetry helper now lives in assets/robots/mujoco/rope_rail.py."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "assets/robots/mujoco"))
from rope_rail import rope_state  # noqa: E402,F401
