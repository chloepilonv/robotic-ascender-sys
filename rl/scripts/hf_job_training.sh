#!/bin/bash
# Training entry point inside an HF Jobs container.
# Expects (env): ARTIFACTS_REPO, EXP_NAME, TRAIN_FLAGS; HF_TOKEN secret set.
# Mounted: the artifacts dataset repo read-only at /artifacts.
set -euo pipefail

mkdir -p /work/logs
LOG=/work/logs/job.log
exec > >(tee "$LOG") 2>&1 || true

echo "[job] host: $(hostname)"
nvidia-smi || echo "[job] WARNING: no GPU visible"

# --- deps: jax + CUDA 12, mujoco playground, wandb --------------------------
pip install --no-cache-dir \
    "jax[cuda12]==0.11.1" jaxlib==0.11.1 \
    mujoco==3.12.0 mujoco-mjx==3.12.0 playground==0.2.0 \
    brax==0.14.2 mediapy tensorboardX wandb

# W&B: authenticate from the HF secret if provided
if [[ -n "${WANDB_API_KEY:-}" ]]; then
  wandb login --relogin "$WANDB_API_KEY" > /dev/null 2>&1 || echo "[job] wandb login failed (continuing without W&B)"
fi

export MUJOCO_GL=egl
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.85
export XLA_FLAGS="--xla_gpu_triton_gemm_any=True"

# --- unpack the code snapshot ------------------------------------------------
mkdir -p /work/repo && cd /work/repo
tar xzf /artifacts/code/code.tar.gz

# --- artifacts uploader: pushes new checkpoints/logs every 5 min -------------
(
  while true; do
    sleep 300
    python - <<'PY' || echo "[artifacts] sync failed this round; retrying in 5 min"
import os
from huggingface_hub import HfApi
api = HfApi()
api.upload_folder(
    repo_id=os.environ["ARTIFACTS_REPO"], repo_type="dataset",
    folder_path="/work/logs", path_in_repo=f"runs/{os.environ['EXP_NAME']}",
)
print("[job] artifacts synced to", os.environ["ARTIFACTS_REPO"], flush=True)
PY
  done
) &
ARTIFACT_UPLOADER_PID=$!
trap 'kill $ARTIFACT_UPLOADER_PID 2>/dev/null || true' EXIT

echo "[job] training starting: $TRAIN_FLAGS"
python rl/scripts/train_jax_ppo.py $TRAIN_FLAGS --logdir /work/logs || \
  echo "[job] training exited nonzero — uploading logs anyway"

# --- final full upload --------------------------------------------------------
python - <<'PY'
import os, traceback
from huggingface_hub import HfApi
try:
    api = HfApi()
    repo = os.environ["ARTIFACTS_REPO"]
    exp = os.environ["EXP_NAME"]
    api.upload_folder(
        repo_id=repo, repo_type="dataset",
        folder_path="/work/logs",
        path_in_repo=f"runs/{exp}",
    )
    print("[job] artifacts uploaded to", repo, "runs/", exp)
except Exception:
    import traceback; traceback.print_exc()
PY
echo "[job] done."
