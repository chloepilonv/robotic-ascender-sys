#!/usr/bin/env bash
# One HF Jobs run: clone -> install mjlab -> train one slope -> export ONNX -> upload to the Hub.
#
#   hf jobs run --flavor a10g-large --timeout 4h \
#     --secrets HF_TOKEN --secrets GH_TOKEN \
#     -e TASK=Himalayas-Ascender-Slope10-G1 -e ITERS=3000 -e HF_REPO=iteratehack/g1-ascender \
#     nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04 \
#     bash -c "$(curl -fsSL https://raw.githubusercontent.com/chloepilonv/g1-himalayas/feat/rl-ascender/rl/chloe/scripts/hf_job.sh)"
#
# Secrets: HF_TOKEN (write, for the upload), GH_TOKEN (read, the repo is private).
# Optional: RESUME=<hub path of a model_*.pt> to continue from a previous slope.
set -euo pipefail
TASK=${TASK:-Himalayas-Ascender-Slope10-G1}
ITERS=${ITERS:-3000}
HF_REPO=${HF_REPO:?set HF_REPO=<org>/<model-repo>}

apt-get update -qq && apt-get install -y -qq git curl > /dev/null
curl -LsSf https://astral.sh/uv/install.sh | sh; export PATH="$HOME/.local/bin:$PATH"
git clone -q -b feat/rl-ascender "https://${GH_TOKEN}@github.com/chloepilonv/g1-himalayas.git" /work && cd /work
uv venv -q -p 3.11 .venv && uv pip install -q -p .venv/bin/python mjlab onnx onnxscript huggingface_hub
.venv/bin/python assets/robots/mujoco/build.py --fetch

RESUME_ARGS=()
if [ -n "${RESUME:-}" ]; then
  .venv/bin/hf download "$HF_REPO" "$RESUME" --local-dir /resume
  mkdir -p logs/rsl_rl/resume/run && cp "/resume/$RESUME" logs/rsl_rl/resume/run/model_0.pt
  RESUME_ARGS=(--agent.resume --agent.load-run run --log-root logs/rsl_rl)
fi

.venv/bin/python -m rl.chloe.scripts.train_mjlab_ppo "$TASK" --agent.max-iterations "$ITERS" "${RESUME_ARGS[@]}"

RUN=$(ls -d logs/rsl_rl/g1_ascender_slope*/*/ | sort | tail -1)
CKPT=$(ls "$RUN"/model_*.pt | sort -V | tail -1)
.venv/bin/python -m rl.chloe.scripts.export_onnx "$TASK" "$CKPT" "$RUN/policy.onnx"
.venv/bin/hf upload "$HF_REPO" "$RUN" "$TASK/$(basename "$RUN")" --include "model_*.pt" --include "policy.onnx" --include "*.tfevents*"
echo "uploaded to https://huggingface.co/$HF_REPO/tree/main/$TASK"
