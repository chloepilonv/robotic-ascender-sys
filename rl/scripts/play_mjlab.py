"""Roll out a trained ascender policy in the MuJoCo viewer.

    python -m rl.scripts.play_mjlab Himalayas-Ascender-Slope20-G1 \
        --checkpoint-file logs/rsl_rl/g1_ascender_slope20/<run>/model_5000.pt

`--agent zero` shows the ratchet mechanics with no policy.
"""

import mjlab.scripts.play as mjlab_play

import rl.task as ascender  # noqa: F401

mjlab_play.ManagerBasedRlEnv = ascender.RatchetEnv

if __name__ == "__main__":
  mjlab_play.main()
