"""Always-on episode recorder. One row per control tick.

Copied from pemba_bench/bench/recorder.py; the only changes are the HUD field
list (the ascender readouts this env actually has) and the control rate coming
from the caller rather than a module constant, because the team env's control
rate is a property of THEIR config (ctrl_dt 0.02 -> 50 Hz), not ours.

Writes, per episode folder:
  header.json  the run's constants: model fingerprint summary, slope, policy,
               command/wind change log, outcome, n_ticks.
  frames.npz   every per-tick array the runtime appended, float32, stacked.
  hud.json     the subset the web player's overlay reads, as plain lists.
  episode.mp4  the per-tick JPEGs muxed at the control rate (needs ffmpeg;
               skipped with a printed warning if ffmpeg is missing).
"""
import json
import os
import shutil
import subprocess

import numpy as np

# Which per-tick arrays the web player's overlay reads. A name in here that the
# runtime actually appended lands in hud.json; everything else lives only in
# frames.npz. Keeping it a list means the backend can add a readout without
# touching the player's contract.
HUD_FIELD_NAMES = [
    "time_seconds", "command", "wind_velocity_world_meters_per_second",
    "wind_force_world_newtons", "root_position_world", "fell",
    "rope_travel_meters", "climb_meters", "height_gained_meters",
    "rope_force_newtons", "hand_height_on_line_meters",
    # The INSTANTANEOUS wind, which differs from the dial once natural wind is
    # on -- a replay that only stored the dial could not reproduce the gust
    # that knocked the robot over.
    "wind_speed_mps", "wind_heading_degrees", "wind_gain", "wind_gust",
    "wind_natural",
    # The guide follower (app/harness/guide.py). `guide_mode` is a CODE, not a
    # string, because these rows are stacked into float arrays: 0 WAIT, 1
    # FOLLOW, 2 LOST (guide.GUIDE_MODE_CODES is the authority). The two distance
    # columns carry -1.0 when there is nothing to report -- not NaN, which
    # `JSON.parse` in the browser rejects outright.
    "guide_mode", "guide_distance_meters", "guide_true_distance_meters",
    "guide_human_progress_meters",
]


class Recorder:
    def __init__(self, output_directory: str, header: dict, control_hz: float = 50.0):
        self.output_directory = output_directory
        self.header = header
        self.control_hz = float(control_hz)
        self.rows = {}
        # One entry per control tick, each a dict or None: the battery/thermal
        # readout is a NAMED MAP, not a number, so it cannot ride in
        # frames.npz's stacked float arrays and gets its own list.
        self.bms_rows = []
        self.jpeg_frames = []
        os.makedirs(output_directory, exist_ok=True)

    def append(self, **fields) -> None:
        for name, value in fields.items():
            self.rows.setdefault(name, []).append(np.asarray(value, dtype=np.float32))

    def append_bms(self, value) -> None:
        """One battery reading per control tick (the last of that tick), or None."""
        self.bms_rows.append(value)

    def append_frame(self, jpeg_bytes: bytes) -> None:
        self.jpeg_frames.append(jpeg_bytes)

    def finalize(self, outcome: dict) -> str:
        if not self.rows:
            print(f"[recorder] {self.output_directory}: nothing recorded, skipping")
            return self.output_directory
        arrays = {name: np.stack(values) for name, values in self.rows.items()}
        np.savez_compressed(os.path.join(self.output_directory, "frames.npz"), **arrays)
        self.header["outcome"] = outcome
        self.header["n_ticks"] = int(len(arrays["time_seconds"]))
        with open(os.path.join(self.output_directory, "header.json"), "w") as handle:
            json.dump(self.header, handle, indent=2, default=_jsonable)
        hud = {name: arrays[name].tolist() for name in HUD_FIELD_NAMES if name in arrays}
        if "fell" in hud:
            hud["fell"] = [bool(value) for value in hud["fell"]]
        if any(value is not None for value in self.bms_rows):
            hud["bms"] = self.bms_rows
        hud["outcome"] = outcome
        hud["slope_degrees"] = self.header.get("slope_degrees")
        hud["control_hz"] = self.control_hz
        with open(os.path.join(self.output_directory, "hud.json"), "w") as handle:
            json.dump(hud, handle, default=_jsonable)
        if self.jpeg_frames:
            self._write_video()
        print(f"[recorder] {self.output_directory}: {self.header['n_ticks']} ticks, "
              f"{len(self.jpeg_frames)} frames, outcome={outcome}", flush=True)
        return self.output_directory

    def _write_video(self) -> None:
        if shutil.which("ffmpeg") is None:
            print("[recorder] ffmpeg not found: episode.mp4 skipped "
                  "(frames.npz and hud.json are still complete)", flush=True)
            return
        path = os.path.join(self.output_directory, "episode.mp4")
        process = subprocess.Popen(
            ["ffmpeg", "-y", "-loglevel", "error", "-f", "image2pipe",
             "-vcodec", "mjpeg", "-framerate", str(self.control_hz), "-i", "-",
             "-c:v", "libx264", "-pix_fmt", "yuv420p", path],
            stdin=subprocess.PIPE)
        for frame in self.jpeg_frames:
            process.stdin.write(frame)
        process.stdin.close()
        process.wait()


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return str(value)
