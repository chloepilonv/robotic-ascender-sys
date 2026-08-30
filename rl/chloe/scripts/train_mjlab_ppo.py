"""Train the G1 ascender-climb policy with mjlab + rsl_rl PPO.

    python -m rl.chloe.scripts.train_mjlab_ppo Himalayas-Ascender-Slope20-G1 \
        --env.scene.num-envs 4096 --agent.max-iterations 5000

Resume from a checkpoint by passing the .pt path directly:

    python -m rl.chloe.scripts.train_mjlab_ppo Himalayas-Ascender-Slope30-G1 \
        --checkpoint rl/chloe/policies/g1_ascender_slope20_v3_2026-08-30_04-35-59.pt \
        --env.scene.num-envs 512 --agent.max-iterations 5000

Any mjlab `train` flag works after the task id (`--help` lists them).
Checkpoints: logs/rsl_rl/g1_ascender_slope<N>/<date>/model_<iter>.pt
"""

import sys
from dataclasses import replace
from pathlib import Path

import mjlab
import mjlab.scripts.train as mjlab_train
import tyro

import rl.chloe.task as ascender  # noqa: F401  (registers the tasks)

# mjlab's trainer instantiates ManagerBasedRlEnv by name; swap in the ratchet env.
mjlab_train.ManagerBasedRlEnv = ascender.RatchetEnv


def _extract_checkpoint() -> str | None:
  """Pull --checkpoint <path> out of sys.argv, returning the path and mutating argv."""
  for i, arg in enumerate(sys.argv):
    if arg == "--checkpoint" and i + 1 < len(sys.argv):
      path = sys.argv[i + 1]
      del sys.argv[i:i + 2]
      return path
  return None


def main():
  checkpoint = _extract_checkpoint()

  if checkpoint:
    ckpt = Path(checkpoint)
    if not ckpt.exists():
      raise FileNotFoundError(f"Checkpoint not found: {ckpt}")
    # Bypass the regex-based checkpoint discovery — return the direct path.
    mjlab_train.get_checkpoint_path = lambda *a, **kw: ckpt

  mjlab_train.maybe_print_top_level_help("train")
  all_tasks = mjlab_train.list_tasks()
  chosen_task, rest = tyro.cli(
    tyro.extras.literal_type_from_choices(all_tasks),
    add_help=False, return_unknown_args=True, config=mjlab.TYRO_FLAGS,
  )
  args = tyro.cli(
    mjlab_train.TrainConfig, args=rest,
    default=mjlab_train.TrainConfig.from_task(chosen_task),
    prog=sys.argv[0] + f" {chosen_task}", config=mjlab.TYRO_FLAGS,
  )

  if checkpoint:
    args = replace(args, agent=replace(args.agent, resume=True))

  mjlab_train.launch_training(task_id=chosen_task, args=args)


if __name__ == "__main__":
  main()
