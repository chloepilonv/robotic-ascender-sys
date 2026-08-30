#!/usr/bin/env bash
# One HF Jobs run: fetch code from the Hub -> install mjlab -> train one slope -> export ONNX -> upload.
# The code tarball is uploaded first (from your Mac) so the job only needs HF_TOKEN:
#   git archive -o code.tar.gz feat/rl-ascender && hf upload iteratehack/g1-ascender code.tar.gz code.tar.gz
# Then:
#   hf jobs run --flavor a10g-large --timeout 4h --secrets HF_TOKEN \
#     -e TASK=Himalayas-Ascender-Slope10-G1 -e ITERS=3000 -e HF_REPO=iteratehack/g1-ascender \
#     nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04 bash -c "$(cat rl/chloe/scripts/hf_job.sh)"
# Optional: RESUME=<hub path of a model_*.pt> to continue from a previous slope.
set -euo pipefail
TASK=${TASK:-Himalayas-Ascender-Slope10-G1}
ITERS=${ITERS:-3000}
HF_REPO=${HF_REPO:?set HF_REPO=<org>/<model-repo>}
export PATH="$HOME/.local/bin:$PATH"

apt-get update -qq && apt-get install -y -qq curl ca-certificates git > /dev/null
curl -LsSf https://astral.sh/uv/install.sh | sh > /dev/null
uv tool install -q huggingface_hub
hf download "$HF_REPO" code.tar.gz --local-dir /dl
mkdir -p /work && tar -xzf /dl/code.tar.gz -C /work && cd /work
uv venv -q -p 3.11 .venv && uv pip install -q -p .venv/bin/python mjlab onnx onnxscript
.venv/bin/python assets/robots/mujoco/build.py --fetch
nvidia-smi --query-gpu=name,memory.total --format=csv

RESUME_ARGS=()
if [ -n "${RESUME:-}" ]; then
  hf download "$HF_REPO" "$RESUME" --local-dir /resume
  mkdir -p logs/rsl_rl/resume/run && cp "/resume/$RESUME" logs/rsl_rl/resume/run/model_0.pt
  RESUME_ARGS=(--agent.resume --agent.load-run run --log-root logs/rsl_rl)
fi

.venv/bin/python -m rl.chloe.scripts.train_mjlab_ppo "$TASK" --agent.max-iterations "$ITERS" ${EXTRA_ARGS:-} "${RESUME_ARGS[@]}"

RUN=$(ls -d logs/rsl_rl/*/*/ | sort | tail -1)
CKPT=$(ls "$RUN"/model_*.pt | sort -V | tail -1)
.venv/bin/python -m rl.chloe.scripts.export_onnx "$TASK" "$CKPT" "$RUN/policy.onnx" || echo "export failed (non-fatal)"
hf upload "$HF_REPO" "$RUN" "$TASK/$(basename "$RUN")" --include "model_*.pt" --include "policy.onnx" --include "*.tfevents*"
echo "DONE -> https://huggingface.co/$HF_REPO/tree/main/$TASK"
