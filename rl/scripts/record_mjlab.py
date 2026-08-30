"""Roll out a trained ascender policy and save an mp4 — no interactive viewer.

Reuses mjlab's play infrastructure (checkpoint loading, VideoRecorder wrapping)
but swaps the viewer for a headless stepper that runs as fast as possible.

Requires --video (enables rgb_array rendering + VideoRecorder wrapper).
Optionally --video-length, --video-height, --video-width, --camera.

    python -m rl.scripts.record_mjlab Himalayas-Ascender-Slope20-G1 \
        --checkpoint-file rl/policies/g1_ascender_slope20_v1_2026-08-30_02-47-06.pt \
        --num-envs 16 --video True --video-length 500

Output: <checkpoint_parent>/videos/play/rl-video-step-0.mp4
  (e.g. rl/policies/videos/play/rl-video-step-0.mp4)
"""

import time

import mjlab.scripts.play as mjlab_play
from mjlab.viewer.base import BaseViewer, VerbosityLevel

import rl.task as ascender  # noqa: F401

mjlab_play.ManagerBasedRlEnv = ascender.RatchetEnv


class HeadlessRecorder(BaseViewer):
  """Steps the env without a window; VideoRecorder captures frames during step()."""

  def __init__(self, env, policy, **kwargs):
    super().__init__(env, policy, frame_rate=30.0, verbosity=VerbosityLevel.INFO)
    # VideoRecorder sits between RslRlVecEnvWrapper and the base env.
    inner = getattr(env, "env", None)
    self._video_length = getattr(inner, "video_length", None)
    if self._video_length is None:
      print("[WARN] No VideoRecorder detected — pass --video to record mp4 output.")
      self._video_length = 200

  def setup(self) -> None:
    pass  # No viewer window to launch.

  def is_running(self) -> bool:
    return self._step_count < self._video_length

  def sync_env_to_viewer(self) -> None:
    pass  # VideoRecorder captures frames inside step().

  def sync_viewer_to_env(self) -> None:
    pass  # No interactive perturbations.

  def close(self) -> None:
    pass  # env.close() in run_play finalizes the VideoRecorder.

  def run(self, num_steps=None, catch_sigint=True) -> None:
    n = num_steps if num_steps is not None else self._video_length
    start = time.perf_counter()
    for _ in range(n):
      if not self._execute_step():
        break
    elapsed = time.perf_counter() - start
    print(f"[INFO] Recorded {self._step_count} steps in {elapsed:.1f}s")
    self.close()


# Swap both viewer backends so run_play uses ours regardless of auto-selection.
mjlab_play.NativeMujocoViewer = HeadlessRecorder
mjlab_play.ViserPlayViewer = HeadlessRecorder

if __name__ == "__main__":
  mjlab_play.main()
