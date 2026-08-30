"""Train the G1 ascender-climb policy with mjlab + rsl_rl PPO.

    python -m rl.chloe.scripts.train_mjlab_ppo Himalayas-Ascender-Slope20-G1 \
        --env.scene.num-envs 4096 --agent.max-iterations 5000

Any mjlab `train` flag works after the task id (`--help` lists them). Resume
on a steeper slope for a curriculum:

    python -m rl.chloe.scripts.train_mjlab_ppo Himalayas-Ascender-Slope30-G1 \
        --agent.resume --agent.load-run <run_dir_of_slope20>

Checkpoints: logs/rsl_rl/g1_ascender_slope<N>/<date>/model_<iter>.pt
"""

import mjlab.scripts.train as mjlab_train

import rl.chloe.task as ascender  # noqa: F401  (registers the tasks)

# mjlab's trainer instantiates ManagerBasedRlEnv by name; swap in the ratchet env.
mjlab_train.ManagerBasedRlEnv = ascender.RatchetEnv

if __name__ == "__main__":
  mjlab_train.main()
