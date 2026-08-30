# Copyright 2025 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Train a PPO agent using JAX on the specified environment."""

import datetime
import functools
import json
import os
import time
import warnings
from typing import Any, Tuple

def _bootstrap_pip_cuda():
  """Preload the pip NVIDIA wheels before jax initializes its CUDA plugin.

  On systems with a system CUDA toolkit (e.g. /usr/local/cuda 12.6) on the
  ldconfig cache, pip's libcusparse resolves the SYSTEM libnvJitLink, which
  lacks symbols the pip cuSPARSE 12.9 needs
  (__nvJitLinkGetErrorLogSize_12_9), and the jax plugin falls back to CPU.
  Loading pip's libnvJitLink first fixes the resolution. No-op on systems
  where jax already sees a GPU or the pip wheels are absent.
  """
  import ctypes
  import glob
  try:
    import jaxlib  # noqa: F401
    # jaxlib sits in site-packages; the NVIDIA wheels install to ./nvidia.
    nvidia = os.path.join(
        os.path.dirname(os.path.dirname(jaxlib.__file__)), "nvidia"
    )
  except ImportError:
    return
  nvjitlink = sorted(
      glob.glob(os.path.join(nvidia, "nvjitlink", "lib", "libnvJitLink.so.*"))
  )
  if not nvjitlink:
    return
  try:
    ctypes.CDLL(nvjitlink[-1], mode=ctypes.RTLD_GLOBAL)
  except OSError:
    pass  # Let jax fall back to its own resolution.


_bootstrap_pip_cuda()


from absl import app
from absl import flags
from absl import logging
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import networks_vision as ppo_networks_vision
from brax.training.agents.ppo import train as ppo
from etils import epath
import jax
import jax.numpy as jp
import mediapy as media
from ml_collections import config_dict
import mujoco
import mujoco_playground
from mujoco_playground import registry
from mujoco_playground import wrapper
from mujoco_playground.config import dm_control_suite_params
from mujoco_playground.config import locomotion_params
from mujoco_playground.config import manipulation_params
try:
  import tensorboardX
except ImportError:
  tensorboardX = None

try:
  import wandb
except ImportError:
  wandb = None


xla_flags = os.environ.get("XLA_FLAGS", "")
xla_flags += " --xla_gpu_triton_gemm_any=True"
os.environ["XLA_FLAGS"] = xla_flags
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
# Keep the desktop compositor alive on a shared laptop GPU: cap XLA's
# allocation fraction (MJX training still works well within ~80%).
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.80")
os.environ["MUJOCO_GL"] = "egl"

# Ignore the info logs from brax
logging.set_verbosity(logging.WARNING)

# Suppress warnings

# Suppress RuntimeWarnings from JAX
warnings.filterwarnings("ignore", category=RuntimeWarning, module="jax")
# Suppress DeprecationWarnings from JAX
warnings.filterwarnings("ignore", category=DeprecationWarning, module="jax")
# Suppress UserWarnings from absl (used by JAX and TensorFlow)
warnings.filterwarnings("ignore", category=UserWarning, module="absl")


_ENV_NAME = flags.DEFINE_string(
    "env_name",
    "LeapCubeReorient",
    f"Name of the environment. One of {', '.join(registry.ALL_ENVS)}",
)
_IMPL = flags.DEFINE_enum("impl", "jax", ["jax", "warp"], "MJX implementation")
_PLAYGROUND_CONFIG_OVERRIDES = flags.DEFINE_string(
    "playground_config_overrides",
    None,
    "Overrides for the playground env config.",
)
_VISION = flags.DEFINE_boolean("vision", False, "Use vision input")
_LOAD_CHECKPOINT_PATH = flags.DEFINE_string(
    "load_checkpoint_path", None, "Path to load checkpoint from"
)
_INIT_FROM_POLICY = flags.DEFINE_string(
    "init_from_policy",
    None,
    "Initialize (fine-tune from) a policy npz in brax param layout;"
    " 'mels' resolves to rl/policies/mels_g1_joystick.npz.",
)
_SUFFIX = flags.DEFINE_string("suffix", None, "Suffix for the experiment name")
_PLAY_ONLY = flags.DEFINE_boolean(
    "play_only", False, "If true, only play with the model and do not train"
)
_USE_WANDB = flags.DEFINE_boolean(
    "use_wandb",
    False,
    "Use Weights & Biases for logging (ignored in play-only mode)",
)
_WANDB_ENTITY = flags.DEFINE_string(
    "wandb_entity", None, "Weights & Biases entity (team/user name)"
)
_WANDB_PROJECT = flags.DEFINE_string(
    "wandb_project", "mjxrl", "Weights & Biases project name"
)
_WANDB_EVAL_VIDEOS = flags.DEFINE_integer(
    "wandb_eval_videos", 0,
    "Log N eval-rollout videos to W&B at every eval point",
)

_USE_TB = flags.DEFINE_boolean(
    "use_tb", False, "Use TensorBoard for logging (ignored in play-only mode)"
)
_DOMAIN_RANDOMIZATION = flags.DEFINE_boolean(
    "domain_randomization", False, "Use domain randomization"
)
_SEED = flags.DEFINE_integer("seed", 1, "Random seed")
_NUM_TIMESTEPS = flags.DEFINE_integer(
    "num_timesteps", 1_000_000, "Number of timesteps"
)
_NUM_VIDEOS = flags.DEFINE_integer(
    "num_videos", 1, "Number of videos to record after training."
)
_NUM_EVALS = flags.DEFINE_integer("num_evals", 5, "Number of evaluations")
_REWARD_SCALING = flags.DEFINE_float("reward_scaling", 0.1, "Reward scaling")
_EPISODE_LENGTH = flags.DEFINE_integer("episode_length", 1000, "Episode length")
_NORMALIZE_OBSERVATIONS = flags.DEFINE_boolean(
    "normalize_observations", True, "Normalize observations"
)
_ACTION_REPEAT = flags.DEFINE_integer("action_repeat", 1, "Action repeat")
_UNROLL_LENGTH = flags.DEFINE_integer("unroll_length", 10, "Unroll length")
_NUM_MINIBATCHES = flags.DEFINE_integer(
    "num_minibatches", 8, "Number of minibatches"
)
_NUM_UPDATES_PER_BATCH = flags.DEFINE_integer(
    "num_updates_per_batch", 8, "Number of updates per batch"
)
_DISCOUNTING = flags.DEFINE_float("discounting", 0.97, "Discounting")
_LEARNING_RATE = flags.DEFINE_float("learning_rate", 5e-4, "Learning rate")
_ENTROPY_COST = flags.DEFINE_float("entropy_cost", 5e-3, "Entropy cost")
_NUM_ENVS = flags.DEFINE_integer("num_envs", 1024, "Number of environments")
_NUM_EVAL_ENVS = flags.DEFINE_integer(
    "num_eval_envs", 128, "Number of evaluation environments"
)
_BATCH_SIZE = flags.DEFINE_integer("batch_size", 256, "Batch size")
_MAX_GRAD_NORM = flags.DEFINE_float("max_grad_norm", 1.0, "Max grad norm")
_CLIPPING_EPSILON = flags.DEFINE_float(
    "clipping_epsilon", 0.3, "Clipping epsilon for PPO"
)
_POLICY_HIDDEN_LAYER_SIZES = flags.DEFINE_list(
    "policy_hidden_layer_sizes",
    [64, 64, 64],
    "Policy hidden layer sizes",
)
_VALUE_HIDDEN_LAYER_SIZES = flags.DEFINE_list(
    "value_hidden_layer_sizes",
    [64, 64, 64],
    "Value hidden layer sizes",
)
_POLICY_OBS_KEY = flags.DEFINE_string(
    "policy_obs_key", "state", "Policy obs key"
)
_VALUE_OBS_KEY = flags.DEFINE_string("value_obs_key", "state", "Value obs key")
_RSCOPE_ENVS = flags.DEFINE_integer(
    "rscope_envs",
    None,
    "Number of parallel environment rollouts to save for the rscope viewer",
)
_DETERMINISTIC_RSCOPE = flags.DEFINE_boolean(
    "deterministic_rscope",
    True,
    "Run deterministic rollouts for the rscope viewer",
)
_RUN_EVALS = flags.DEFINE_boolean(
    "run_evals",
    True,
    "Run evaluation rollouts between policy updates.",
)
_LOG_TRAINING_METRICS = flags.DEFINE_boolean(
    "log_training_metrics",
    False,
    "Whether to log training metrics and callback to progress_fn. Significantly"
    " slows down training if too frequent.",
)
_TRAINING_METRICS_STEPS = flags.DEFINE_integer(
    "training_metrics_steps",
    1_000_000,
    "Number of steps between logging training metrics. Increase if training"
    " experiences slowdown.",
)
_WARP_KERNEL_CACHE_DIR = flags.DEFINE_string(
    "warp_kernel_cache_dir", None,
    "Directory for caching compiled Warp kernels.",
)
_LOGDIR = flags.DEFINE_string(
    "logdir", None, "Directory for logging."
)


# G1JoystickWind* maps to the tuned G1Joystick PPO params.
_RL_ENV_ALIASES = {
    "G1JoystickWindFlatTerrain": "G1JoystickFlatTerrain",
    "G1JoystickWindRoughTerrain": "G1JoystickRoughTerrain",
    # Same task/obs as upstream G1Joystick (fine-tune compatible).
    "G1JoystickWalkDR": "G1JoystickFlatTerrain",
    # 103/216-dim obs, 29 actions: same recipe as G1Joystick.
    "G1ClimbTerrain": "G1JoystickFlatTerrain",
}

def get_rl_config(env_name: str) -> config_dict.ConfigDict:
  # G1JoystickWind* envs share the G1Joystick PPO recipe.
  env_name = _RL_ENV_ALIASES.get(env_name, env_name)
  if env_name in mujoco_playground.manipulation._envs:
    if _VISION.value:
      return manipulation_params.brax_vision_ppo_config(env_name, _IMPL.value)
    return manipulation_params.brax_ppo_config(env_name, _IMPL.value)
  elif env_name in mujoco_playground.locomotion._envs:
    return locomotion_params.brax_ppo_config(env_name, _IMPL.value)
  elif env_name in mujoco_playground.dm_control_suite._envs:
    if _VISION.value:
      return dm_control_suite_params.brax_vision_ppo_config(
          env_name, _IMPL.value
      )
    return dm_control_suite_params.brax_ppo_config(env_name, _IMPL.value)

  raise ValueError(f"Env {env_name} not found in {registry.ALL_ENVS}.")

def _restore_params_from_npz(
    npz_path: str, env: Any
) -> Tuple[Any, Any, Any]:
  """Convert a brax-layout policy npz (mels export) to ppo.train
  `restore_params`.

  The npz holds a flax MLP with `hidden_{i}_kernel/bias` entries (last
  layer = 2*action_size distribution params) plus `obs_mean`/`obs_std`
  normalizer stats. The value network is absent from the export, so the
  caller must pass `restore_value_fn=False` (fresh value init is
  correct for fine-tuning: the value function must be re-learned for
  the new domain anyway).

  Returns:
    (normalizer_params, policy_params, None) — the value slot is None
    and is ignored when brax restores with restore_value_fn=False.
  """
  import numpy as _np  # pylint: disable=import-outside-toplevel

  z = _np.load(npz_path)
  n_layers = 0
  while f"hidden_{n_layers}_kernel" in z.files:
    n_layers += 1
  if n_layers == 0:
    raise ValueError(f"no hidden_* layers found in {npz_path}")
  last_out = z[f"hidden_{n_layers - 1}_kernel"].shape[1]
  if last_out != 2 * env.action_size:
    raise ValueError(
        f"policy output {last_out} != 2*action_size"
        f" {2 * env.action_size}: {npz_path} is not compatible with"
        f" {type(env).__name__}"
    )

  policy_params = {
      "params": {
          f"hidden_{i}": {
              "kernel": jp.asarray(z[f"hidden_{i}_kernel"]),
              "bias": jp.asarray(z[f"hidden_{i}_bias"]),
          }
          for i in range(n_layers)
      }
  }

  # Seed the running-statistics normalizer with the npz stats so the
  # restored policy normalizes observations exactly as it was trained.
  # The state nest is built from a real reset observation (correct
  # structure/dtype); Welford's summed_variance is chosen so the stored
  # std recompute holds: sv = count * (std^2), with a large count so
  # the first fine-tune batches nudge rather than jump the statistics.
  from brax.training.acme import running_statistics  # pylint: disable=import-outside-toplevel
  from brax.training import types as brax_types  # pylint: disable=import-outside-toplevel

  obs_nest = jax.jit(env.reset)(jax.random.PRNGKey(0)).obs
  zeros_state = running_statistics.init_state(obs_nest)
  count = brax_types.UInt64(hi=0, lo=1_000_000)
  obs_size = env.observation_size
  state_mean = jp.asarray(z["obs_mean"])
  if state_mean.shape != tuple(obs_size["state"]):
    raise ValueError(
        f"obs_mean shape {state_mean.shape} != env state obs"
        f" {tuple(obs_size['state'])}: {npz_path} incompatible with"
        f" {type(env).__name__}"
    )
  mean = {
      "state": state_mean,
      "privileged_state": jp.zeros(obs_size["privileged_state"]),
  }
  # std is stored (not recomputed from summed_variance in-place), so set
  # it directly; summed_variance is kept consistent (sv = count*std^2)
  # so any later Welford update stays numerically sane.
  std = {
      "state": jp.asarray(z["obs_std"]),
      "privileged_state": jp.ones(obs_size["privileged_state"]),
  }
  summed_variance = {
      "state": 1e6 * jp.square(jp.asarray(z["obs_std"])),
      "privileged_state": 1e6 * jp.ones(obs_size["privileged_state"]),
  }
  normalizer_params = zeros_state.replace(
      count=count, mean=mean, std=std, summed_variance=summed_variance
  )

  return normalizer_params, policy_params, None


def rscope_fn(full_states, obs, rew, done):
  """
  All arrays are of shape (unroll_length, rscope_envs, ...)
  full_states: dict with keys 'qpos', 'qvel', 'time', 'metrics'
  obs: nd.array or dict obs based on env configuration
  rew: nd.array rewards
  done: nd.array done flags
  """
  # Calculate cumulative rewards per episode, stopping at first done flag
  done_mask = jp.cumsum(done, axis=0)
  valid_rewards = rew * (done_mask == 0)
  episode_rewards = jp.sum(valid_rewards, axis=0)
  print(
      "Collected rscope rollouts with reward"
      f" {episode_rewards.mean():.3f} +- {episode_rewards.std():.3f}"
  )


def main(argv):
  """Run training and evaluation for the specified environment."""

  del argv

  # Load environment configuration
  if _ENV_NAME.value.startswith(("G1JoystickWind", "G1JoystickWalkDR", "G1Climb")):
    import sys
    sys.path.insert(0, str(epath.Path(__file__).parent.parent.parent.resolve()))
    import rl.environment  # noqa: F401  registers the rl envs in the registry

  env_cfg = registry.get_default_config(_ENV_NAME.value)

  ppo_params = get_rl_config(_ENV_NAME.value)

  if _NUM_TIMESTEPS.present:
    ppo_params.num_timesteps = _NUM_TIMESTEPS.value
  if _PLAY_ONLY.present:
    ppo_params.num_timesteps = 0
  if _NUM_EVALS.present:
    ppo_params.num_evals = _NUM_EVALS.value
  if _REWARD_SCALING.present:
    ppo_params.reward_scaling = _REWARD_SCALING.value
  if _EPISODE_LENGTH.present:
    ppo_params.episode_length = _EPISODE_LENGTH.value
  if _NORMALIZE_OBSERVATIONS.present:
    ppo_params.normalize_observations = _NORMALIZE_OBSERVATIONS.value
  if _ACTION_REPEAT.present:
    ppo_params.action_repeat = _ACTION_REPEAT.value
  if _UNROLL_LENGTH.present:
    ppo_params.unroll_length = _UNROLL_LENGTH.value
  if _NUM_MINIBATCHES.present:
    ppo_params.num_minibatches = _NUM_MINIBATCHES.value
  if _NUM_UPDATES_PER_BATCH.present:
    ppo_params.num_updates_per_batch = _NUM_UPDATES_PER_BATCH.value
  if _DISCOUNTING.present:
    ppo_params.discounting = _DISCOUNTING.value
  if _LEARNING_RATE.present:
    ppo_params.learning_rate = _LEARNING_RATE.value
  if _ENTROPY_COST.present:
    ppo_params.entropy_cost = _ENTROPY_COST.value
  if _NUM_ENVS.present:
    ppo_params.num_envs = _NUM_ENVS.value
  if _NUM_EVAL_ENVS.present:
    ppo_params.num_eval_envs = _NUM_EVAL_ENVS.value
  if _BATCH_SIZE.present:
    ppo_params.batch_size = _BATCH_SIZE.value
  if _MAX_GRAD_NORM.present:
    ppo_params.max_grad_norm = _MAX_GRAD_NORM.value
  if _CLIPPING_EPSILON.present:
    ppo_params.clipping_epsilon = _CLIPPING_EPSILON.value
  if _POLICY_HIDDEN_LAYER_SIZES.present:
    ppo_params.network_factory.policy_hidden_layer_sizes = list(
        map(int, _POLICY_HIDDEN_LAYER_SIZES.value)
    )
  if _VALUE_HIDDEN_LAYER_SIZES.present:
    ppo_params.network_factory.value_hidden_layer_sizes = list(
        map(int, _VALUE_HIDDEN_LAYER_SIZES.value)
    )
  if _POLICY_OBS_KEY.present:
    ppo_params.network_factory.policy_obs_key = _POLICY_OBS_KEY.value
  if _VALUE_OBS_KEY.present:
    ppo_params.network_factory.value_obs_key = _VALUE_OBS_KEY.value

  env_cfg_overrides = {"impl": _IMPL.value}
  if _VISION.value:
    env_cfg_overrides["vision"] = True
    env_cfg_overrides["vision_config.nworld"] = ppo_params.num_envs
  if _PLAYGROUND_CONFIG_OVERRIDES.value is not None:
    env_cfg_overrides.update(json.loads(_PLAYGROUND_CONFIG_OVERRIDES.value))

  env = registry.load(
      _ENV_NAME.value, config=env_cfg, config_overrides=env_cfg_overrides
  )
  if _RUN_EVALS.present:
    ppo_params.run_evals = _RUN_EVALS.value
  if _LOG_TRAINING_METRICS.present:
    ppo_params.log_training_metrics = _LOG_TRAINING_METRICS.value
  if _TRAINING_METRICS_STEPS.present:
    ppo_params.training_metrics_steps = _TRAINING_METRICS_STEPS.value

  print(f"Environment Config:\n{env_cfg}")
  if env_cfg_overrides:
    print(f"Environment Config Overrides:\n{env_cfg_overrides}\n")
  print(f"PPO Training Parameters:\n{ppo_params}")

  # Generate unique experiment name
  now = datetime.datetime.now()
  timestamp = now.strftime("%Y%m%d-%H%M%S")
  exp_name = f"{_ENV_NAME.value}-{timestamp}"
  if _SUFFIX.value is not None:
    exp_name += f"-{_SUFFIX.value}"
  print(f"Experiment name: {exp_name}")

  # Set up logging directory
  logdir = epath.Path(_LOGDIR.value or "logs").resolve() / exp_name
  logdir.mkdir(parents=True, exist_ok=True)
  print(f"Logs are being stored in: {logdir}")

  # Initialize Weights & Biases if required
  if _USE_WANDB.value and not _PLAY_ONLY.value:
    if wandb is None:
      raise ImportError(
          "wandb is required for --use_wandb. "
          "Install via: pip install wandb"
      )
    wandb.init(
        project=_WANDB_PROJECT.value,
        entity=_WANDB_ENTITY.value,
        name=exp_name,
    )
    wandb.config.update(env_cfg.to_dict())
    wandb.config.update({"env_name": _ENV_NAME.value})

  # Initialize TensorBoard if required
  writer = None
  if _USE_TB.value and not _PLAY_ONLY.value and tensorboardX is not None:
    writer = tensorboardX.SummaryWriter(logdir)

  # Handle checkpoint loading
  if _LOAD_CHECKPOINT_PATH.value is not None:
    # Convert to absolute path
    ckpt_path = epath.Path(_LOAD_CHECKPOINT_PATH.value).resolve()
    if ckpt_path.is_dir():
      latest_ckpts = list(ckpt_path.glob("*"))
      latest_ckpts = [ckpt for ckpt in latest_ckpts if ckpt.is_dir()]
      latest_ckpts.sort(key=lambda x: int(x.name))
      latest_ckpt = latest_ckpts[-1]
      restore_checkpoint_path = latest_ckpt
      print(f"Restoring from: {restore_checkpoint_path}")
    else:
      restore_checkpoint_path = ckpt_path
      print(f"Restoring from checkpoint: {restore_checkpoint_path}")
  else:
    print("No checkpoint path provided, not restoring from checkpoint")
    restore_checkpoint_path = None

  # Fine-tune initialization from a brax-layout policy npz (e.g. the
  # mels export). Mutually exclusive with --load_checkpoint_path; the
  # value network is freshly initialized (the export has no value head,
  # and the value function must be re-learned for the randomized domain).
  restore_params = None
  if _INIT_FROM_POLICY.value is not None:
    if restore_checkpoint_path is not None:
      raise ValueError(
          "--init_from_policy and --load_checkpoint_path are mutually"
          " exclusive."
      )
    if _INIT_FROM_POLICY.value == "mels":
      npz_path = (
          epath.Path(__file__).parent.parent / "policies" / "mels_g1_joystick.npz"
      )
    else:
      npz_path = epath.Path(_INIT_FROM_POLICY.value)
    npz_path = npz_path.resolve()
    if not npz_path.exists():
      raise FileNotFoundError(f"policy npz not found: {npz_path}")
    restore_params = _restore_params_from_npz(npz_path.as_posix(), env)
    print(f"Fine-tuning: policy initialized from {npz_path}")

  # Set up checkpoint directory
  ckpt_path = logdir / "checkpoints"
  ckpt_path.mkdir(parents=True, exist_ok=True)
  print(f"Checkpoint path: {ckpt_path}")

  # Save environment configuration
  with open(ckpt_path / "config.json", "w", encoding="utf-8") as fp:
    json.dump(env_cfg.to_dict(), fp, indent=4)

  training_params = dict(ppo_params)
  if "network_factory" in training_params:
    del training_params["network_factory"]

  network_fn = (
      ppo_networks_vision.make_ppo_networks_vision
      if _VISION.value
      else ppo_networks.make_ppo_networks
  )
  if hasattr(ppo_params, "network_factory"):
    network_factory = functools.partial(
        network_fn, **ppo_params.network_factory
    )
  else:
    network_factory = network_fn

  if _DOMAIN_RANDOMIZATION.value:
    randomizer = registry.get_domain_randomizer(_ENV_NAME.value)
    if randomizer is None:
      raise ValueError(
          f"--domain_randomization: env {_ENV_NAME.value} has no"
          " registered domain randomizer."
      )
    if hasattr(env_cfg, "dr_config"):
      # Bind the randomizer to the LOADED env config so
      # --config_overrides on dr_config.* fields (slope/wind ranges)
      # reach the randomizer.
      from rl.environment import walk_dr_env as _walk_dr_env  # pylint: disable=import-outside-toplevel

      randomizer = functools.partial(
          _walk_dr_env.domain_randomize, dr_cfg=env_cfg.dr_config
      )
    training_params["randomization_fn"] = randomizer

  num_eval_envs = ppo_params.get("num_eval_envs", 128)

  if "num_eval_envs" in training_params:
    del training_params["num_eval_envs"]

  train_fn = functools.partial(
      ppo.train,
      **training_params,
      network_factory=network_factory,
      seed=_SEED.value,
      restore_checkpoint_path=restore_checkpoint_path,
      restore_params=restore_params,
      # The npz export has no value head: keep the fresh value init.
      restore_value_fn=restore_params is None,
      save_checkpoint_path=ckpt_path,
      wrap_env_fn=wrapper.wrap_for_brax_training,
      num_eval_envs=num_eval_envs,
      vision=_VISION.value,
  )

  times = [time.monotonic()]

  # Progress function for logging
  eval_video_env = None
  eval_video_rollout = None
  def log_eval_video(num_steps, metrics, rollout_env):
    """Render a short rollout of the current policy and log it to W&B.

    Uses the latest policy snapshot (set by `policy_params_fn` at every
    eval point) on a dedicated rollout env; renders EGL frames and ships
    them as a wandb.Video under "eval/video". Failures are contained by
    the caller.
    """
    make_policy = _latest_policy["make_policy"]
    snap_params = _latest_policy["params"]
    if make_policy is None or snap_params is None:
      return  # first progress call happens before any snapshot
    inference_fn = make_policy(
        snap_params,
        deterministic=True,
    )
    jit_inference_fn = jax.jit(inference_fn)
    rollout_env = wrapper.wrap_for_brax_training(
        rollout_env,
        episode_length=ppo_params.episode_length,
        action_repeat=ppo_params.get("action_repeat", 1),
    )
    jit_reset = jax.jit(rollout_env.reset)
    jit_step = jax.jit(rollout_env.step)
    render_every = 5  # 50 Hz physics -> 10 fps video
    rng = jax.random.PRNGKey(_SEED.value)
    state = jit_reset(jax.random.PRNGKey(_SEED.value + num_steps))
    frames = []
    max_steps = min(ppo_params.episode_length, 500)  # ~20 s of sim
    for i in range(max_steps):
      rng, act_rng = jax.random.split(rng)
      if bool(state.done):
        state = jax.jit(rollout_env.reset)(
            jax.random.PRNGKey(_SEED.value + num_steps + i)
        )
      obs = state.obs["state"] if isinstance(state.obs, dict) else state.obs
      action, _ = jit_inference_fn(obs, rng)
      state = jit_step(state, action)
      if i % render_every == 0:
        scene_option = mujoco.MjvOption()
        scene_option.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = False
        frames.append(rollout_env.render([state], height=360, width=640, scene_option=scene_option)[0])
    wandb.log(
        {"eval/video": wandb.Video(np.stack(frames), fps=10, format="mp4")},
        step=num_steps,
    )

  if _WANDB_EVAL_VIDEOS.value > 0 and _USE_WANDB.value and not _PLAY_ONLY.value:
    # A dedicated rollout env for the eval videos: rendered with EGL into
    # frames that go straight to W&B at every eval point.
    eval_video_env = registry.load(
        _ENV_NAME.value,
        config=registry.get_default_config(_ENV_NAME.value),
        config_overrides=dict(env_cfg_overrides),
    )

  def progress(num_steps, metrics):
    times.append(time.monotonic())

    # Log to Weights & Biases
    if _USE_WANDB.value and not _PLAY_ONLY.value:
      wandb.log(metrics, step=num_steps)
      if _WANDB_EVAL_VIDEOS.value > 0 and eval_video_env is not None:
        try:
          log_eval_video(num_steps, metrics, eval_video_env)
        except Exception as exc:  # never kill training over a video
          print(f"eval video logging failed: {exc}", flush=True)

    # Log to TensorBoard
    if _USE_TB.value and not _PLAY_ONLY.value and writer is not None:
      for key, value in metrics.items():
        writer.add_scalar(key, value, num_steps)
      writer.flush()
    if _RUN_EVALS.value:
      print(f"{num_steps}: reward={metrics['eval/episode_reward']:.3f}")
    if _LOG_TRAINING_METRICS.value:
      if "episode/sum_reward" in metrics:
        print(
            f"{num_steps}: mean episode"
            f" reward={metrics['episode/sum_reward']:.3f}"
        )

  eval_env_overrides = dict(env_cfg_overrides)
  if _VISION.value:
    eval_env_overrides["vision_config.nworld"] = num_eval_envs
  eval_env = registry.load(
      _ENV_NAME.value,
      config=registry.get_default_config(_ENV_NAME.value),
      config_overrides=eval_env_overrides,
  )

  # Latest policy snapshot, refreshed by policy_params_fn at every eval
  # point. The eval-video logger below uses it to render rollouts.
  _latest_policy = {"make_policy": None, "params": None}

  def policy_params_fn(current_step, make_policy, params):  # pylint: disable=unused-argument
    _latest_policy["make_policy"] = make_policy
    _latest_policy["params"] = params
  if _RSCOPE_ENVS.value:
    # Interactive visualisation of policy checkpoints
    from rscope import brax as rscope_utils

    if not _VISION.value:
      rscope_env = registry.load(
          _ENV_NAME.value, config=env_cfg, config_overrides=env_cfg_overrides
      )
      rscope_env = wrapper.wrap_for_brax_training(
          rscope_env,
          episode_length=ppo_params.episode_length,
          action_repeat=ppo_params.action_repeat,
          randomization_fn=training_params.get("randomization_fn"),
      )
    else:
      rscope_env = env

    rscope_handle = rscope_utils.BraxRolloutSaver(
        rscope_env,
        ppo_params,
        _VISION.value,
        _RSCOPE_ENVS.value,
        _DETERMINISTIC_RSCOPE.value,
        jax.random.PRNGKey(_SEED.value),
        rscope_fn,
    )

    def policy_params_fn(current_step, make_policy, params):  # pylint: disable=unused-argument
      rscope_handle.set_make_policy(make_policy)
      _latest_policy["make_policy"] = make_policy
      _latest_policy["params"] = params

  # Train or load the model
  make_inference_fn, params, _ = train_fn(  # pylint: disable=no-value-for-parameter
      environment=env,
      progress_fn=progress,
      policy_params_fn=policy_params_fn,
      eval_env=eval_env,
  )

  print("Done training.")
  if len(times) > 1:
    print(f"Time to JIT compile: {times[1] - times[0]}")
    print(f"Time to train: {times[-1] - times[1]}")

  print("Starting inference...")

  # Create inference function.
  inference_fn = make_inference_fn(params, deterministic=True)
  jit_inference_fn = jax.jit(inference_fn)

  infer_env_overrides = dict(env_cfg_overrides)
  if _VISION.value:
    infer_env_overrides["vision_config.nworld"] = _NUM_VIDEOS.value
  infer_env = registry.load(
      _ENV_NAME.value,
      config=registry.get_default_config(_ENV_NAME.value),
      config_overrides=infer_env_overrides,
  )

  # Run evaluation rollouts matching how training handles batched environments.
  wrapped_infer_env = wrapper.wrap_for_brax_training(
      infer_env,
      episode_length=ppo_params.episode_length,
      action_repeat=ppo_params.get("action_repeat", 1),
  )

  rng = jax.random.split(jax.random.PRNGKey(_SEED.value), _NUM_VIDEOS.value)
  reset_states = jax.jit(wrapped_infer_env.reset)(rng)

  empty_data = reset_states.data.__class__(
      **{k: None for k in reset_states.data.__annotations__}
  )  # pytype: disable=attribute-error
  empty_traj = reset_states.__class__(
      **{k: None for k in reset_states.__annotations__}
  )  # pytype: disable=attribute-error
  empty_traj = empty_traj.replace(data=empty_data)

  def step(carry, _):
    state, rng = carry
    rng, act_key = jax.random.split(rng)
    act_keys = jax.random.split(act_key, _NUM_VIDEOS.value)
    act = jax.vmap(jit_inference_fn)(state.obs, act_keys)[0]
    state = wrapped_infer_env.step(state, act)
    traj_data = empty_traj.tree_replace({
        "data.qpos": state.data.qpos,
        "data.qvel": state.data.qvel,
        "data.time": state.data.time,
        "data.ctrl": state.data.ctrl,
        "data.mocap_pos": state.data.mocap_pos,
        "data.mocap_quat": state.data.mocap_quat,
        "data.xfrc_applied": state.data.xfrc_applied,
    })
    return (state, rng), traj_data

  @jax.jit
  def do_rollout(state, rng):
    _, traj = jax.lax.scan(
        step, (state, rng), None, length=ppo_params.episode_length
    )
    return traj

  if _NUM_VIDEOS.value > 0:
    traj_stacked = do_rollout(
        reset_states, jax.random.PRNGKey(_SEED.value + 1)
    )
    # traj_stacked has shape (time, nworld, ...), swap to (nworld, time, ...).
    traj_stacked = jax.tree.map(lambda x: jp.moveaxis(x, 0, 1), traj_stacked)
    trajectories = [None] * _NUM_VIDEOS.value
    for i in range(_NUM_VIDEOS.value):
      t = jax.tree.map(lambda x, i=i: x[i], traj_stacked)
      trajectories[i] = [
          jax.tree.map(lambda x, j=j: x[j], t)
          for j in range(ppo_params.episode_length)
      ]

    # Render and save the rollout.
    render_every = 2
    fps = 1.0 / infer_env.dt / render_every
    print(f"FPS for rendering: {fps}")
    scene_option = mujoco.MjvOption()
    scene_option.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = False
    scene_option.flags[mujoco.mjtVisFlag.mjVIS_PERTFORCE] = False
    scene_option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = False
    for i, rollout in enumerate(trajectories):
      traj = rollout[::render_every]
      frames = infer_env.render(
          traj, height=480, width=640, scene_option=scene_option
      )
      media.write_video(logdir / f"rollout{i}.mp4", frames, fps=fps)
      print(f"Rollout video saved as '{logdir}/rollout{i}.mp4'.")


def run():
  """Entry point for uv/pip script."""
  app.run(main)


if __name__ == "__main__":
  run()
