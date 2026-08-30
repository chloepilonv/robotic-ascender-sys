#!/usr/bin/env bash
# One HF Jobs run: fetch code from the Hub -> install mjlab -> train one slope -> export ONNX -> upload.
# The code tarball is uploaded first (from your Mac) so the job only needs HF_TOKEN:
#   git archive -o code.tar.gz <branch> && hf upload iteratehack/g1-ascender code.tar.gz code.tar.gz
# CODE_TAR names the tarball in the repo (default code.tar.gz). The repo root tarball is
# SHARED with other teams in the org -- use a distinct name so a concurrent job that pulls
# code.tar.gz mid-run does not get your tree.
# Then:
#   hf jobs run --flavor a10g-large --timeout 4h --secrets HF_TOKEN \
#     -e TASK=Himalayas-Ascender-Slope10-G1 -e ITERS=3000 -e HF_REPO=iteratehack/g1-ascender \
#     nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04 bash -c "$(cat rl/scripts/hf_job.sh)"
# Optional: RESUME=<hub path of a model_*.pt> to continue from a previous slope.
set -euo pipefail
TASK=${TASK:-Himalayas-Ascender-Slope10-G1}
CODE_TAR=${CODE_TAR:-code.tar.gz}
# mjlab resolves a resume as <log_root>/<experiment_name>/<load_run>/<checkpoint>, and
# experiment_name comes from make_ppo_cfg: g1_ascender_slope<N>.
SLOPE=$(echo "$TASK" | sed -E 's/.*Slope([0-9]+).*/\1/')
EXPERIMENT=${EXPERIMENT:-g1_ascender_slope${SLOPE}}
ITERS=${ITERS:-3000}
HF_REPO=${HF_REPO:?set HF_REPO=<org>/<model-repo>}
export PATH="$HOME/.local/bin:$PATH"

apt-get update -qq && apt-get install -y -qq curl ca-certificates git libegl1 libgl1 libglvnd0 > /dev/null
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
curl -LsSf https://astral.sh/uv/install.sh | sh > /dev/null
uv tool install -q huggingface_hub
hf download "$HF_REPO" "$CODE_TAR" --local-dir /dl
mkdir -p /work && tar -xzf "/dl/$CODE_TAR" -C /work && cd /work
uv venv -q -p 3.11 .venv && uv pip install -q -p .venv/bin/python mjlab onnx onnxscript
# Stock Unitree STLs (build.py --fetch needs USD; do the sparse clone directly).
git clone -q --depth 1 --filter=blob:none --sparse https://github.com/google-deepmind/mujoco_menagerie /men \
  && git -C /men sparse-checkout set unitree_g1 -q \
  && mkdir -p assets/robots/g1/_menagerie && cp -r /men/unitree_g1 assets/robots/g1/_menagerie/
nvidia-smi --query-gpu=name,memory.total --format=csv

RESUME_ARGS=()
if [ -n "${RESUME:-}" ]; then
  hf download "$HF_REPO" "$RESUME" --local-dir /resume
  # Seed dir must live under the experiment name, and must sort BEFORE the date-named run
  # dir that training creates, or the upload step below would pick the seed instead.
  SEED_RUN=0000_resume_seed
  mkdir -p "logs/rsl_rl/$EXPERIMENT/$SEED_RUN"
  cp "/resume/$RESUME" "logs/rsl_rl/$EXPERIMENT/$SEED_RUN/model_0.pt"
  RESUME_ARGS=(--agent.resume True --agent.load-run "$SEED_RUN" --log-root logs/rsl_rl)
fi

.venv/bin/python -m rl.scripts.train_mjlab_ppo "$TASK" --agent.max-iterations "$ITERS" ${EXTRA_ARGS:-} "${RESUME_ARGS[@]}"

RUN=$(ls -dt logs/rsl_rl/*/*/ | head -1)  # newest by mtime = the run just trained
CKPT=$(ls "$RUN"/model_*.pt | sort -V | tail -1)
.venv/bin/python -m rl.scripts.export_onnx "$TASK" "$CKPT" "$RUN/policy.onnx" || echo "export failed (non-fatal)"
hf upload "$HF_REPO" "$RUN" "$TASK/$(basename "$RUN")" --include "model_*.pt" --include "policy.onnx" --include "*.tfevents*"
echo "DONE -> https://huggingface.co/$HF_REPO/tree/main/$TASK"
