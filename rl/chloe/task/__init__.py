"""Registers `Himalayas-Ascender-Slope{N}-G1` tasks (N in SLOPES) with mjlab.

Gravity is a global MuJoCo option, so the slope is fixed per run and
randomised across runs / a curriculum (train slope 10 -> 20 -> 30 -> 40 by
resuming from the previous checkpoint). Wind and ice friction are randomised
per env inside each run.
"""

from mjlab.tasks.registry import register_mjlab_task

from .env_cfg import RatchetEnv, make_env_cfg, make_ppo_cfg

SLOPES = (0, 10, 20, 30, 40)


def task_id(slope_deg: int) -> str:
  return f"Himalayas-Ascender-Slope{slope_deg}-G1"


for _s in SLOPES:
  register_mjlab_task(
    task_id(_s),
    env_cfg=make_env_cfg(slope_deg=_s),
    play_env_cfg=make_env_cfg(slope_deg=_s, play=True),
    rl_cfg=make_ppo_cfg(_s),
  )

# v8-lite: v3 recipe + face-uphill + per-env slope DR + rope collider; no rhythm FSM.
register_mjlab_task(
  "Himalayas-Ascender-Lite-G1",
  env_cfg=make_env_cfg(slope_deg=20, rhythm=False),
  play_env_cfg=make_env_cfg(slope_deg=20, play=True, rhythm=False),
  rl_cfg=make_ppo_cfg(20),
)

__all__ = ["RatchetEnv", "make_env_cfg", "make_ppo_cfg", "task_id", "SLOPES"]
