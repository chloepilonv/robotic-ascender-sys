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
"""Train a PPO agent with massively parallel MJX envs on a CUDA GPU.

Thin wrapper around `train_jax_ppo.main` tuned for single-GPU (e.g. RTX 4070
8 GB) training: all envs run in parallel on the device via MJX vmap — there
is no CPU env stepping. Differences from `train_jax_ppo`:

* Device selection is asserted at startup (exits with a clear message when
  JAX falls back to CPU, which silently trains ~50x slower).
* Memory is bounded: preallocation capped and XLA set up for laptop GPUs
  (`XLA_PYTHON_CLIENT_MEM_FRACTION`, cudnn autotune level).
* The default num_envs is sized for 8 GB VRAM (2048) instead of the
  upstream 8192; override with --num_envs on cards with more memory.
* Same env registration, npz fine-tune init, domain randomization, logging,
  checkpointing and video rendering as `train_jax_ppo`.

Examples:
  # Domain-randomized slope+wind walk, fine-tuned from the mels policy:
  python rl/scripts/train_gpu.py --env_name G1JoystickWalkDR \
      --domain_randomization --init_from_policy mels --use_tb \
      --num_timesteps 100_000_000

  # From-scratch upstream env:
  python rl/scripts/train_gpu.py --env_name G1JoystickFlatTerrain
"""

import sys
from pathlib import Path

# Allow running as a plain script: rl/scripts/ is not a package root.
sys.path.insert(0, str(Path(__file__).parent.parent.parent.resolve()))

# Must run before train_jax_ppo imports jax: preload pip NVIDIA wheels so
# pip cuSPARSE doesn't bind to a mismatched system libnvJitLink.
from rl.scripts import train_jax_ppo  # noqa: F401  (side effect: bootstrap)

from absl import app

from rl.scripts.train_jax_ppo import main as _ppo_main


def main(argv):
  """Run training on the GPU device, asserting CUDA visibility."""
  del argv

  import jax

  devices = jax.devices()
  print(f"JAX devices: {devices}")
  gpus = [d for d in devices if d.platform == "gpu"]
  if not gpus:
    raise SystemExit(
        "No GPU visible to JAX — refusing to train on CPU (use"
        " rl/scripts/train_jax_ppo.py for that). Check the CUDA 12 jax"
        " plugin: pip install 'jax[cuda12]'==jaxlib version, and that"
        " libnvJitLink resolves to the pip wheel."
    )
  total_mem = sum(d.memory_stats().get("mem_limit", 0) for d in gpus)
  print(f"GPU: {gpus[0].device_kind}, {total_mem / 2**30:.1f} GiB usable")

  _ppo_main([])


def run():
  """Entry point for uv/pip script."""
  app.run(main)


if __name__ == "__main__":
  run()
