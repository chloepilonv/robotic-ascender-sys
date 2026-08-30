"""Step-by-step policy viewer for inspecting the rope/ascender joint.

Starts paused. Press RIGHT ARROW to advance one policy step (50 Hz).
Press SPACE to resume/pose continuous playback. ENTER to reset.

    python -m rl.scripts.step_viewer Himalayas-Ascender-Slope20-G1 \
        --checkpoint-file rl/policies/g1_ascender_slope20_v3_2026-08-30_04-35-59.pt \
        --num-envs 1
"""

import mjlab.scripts.play as mjlab_play
from mjlab.viewer import NativeMujocoViewer

import rl.task as ascender  # noqa: F401

mjlab_play.ManagerBasedRlEnv = ascender.RatchetEnv


class PausedViewer(NativeMujocoViewer):
  """NativeMujocoViewer that starts paused — step with Right arrow."""

  def setup(self) -> None:
    super().setup()
    self.pause()  # start frozen so the user controls every step


# Swap the viewer class so mjlab's run_play uses ours.
mjlab_play.NativeMujocoViewer = PausedViewer

if __name__ == "__main__":
  mjlab_play.main()
