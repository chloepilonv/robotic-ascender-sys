#!/usr/bin/env bash
# One HF Jobs run: fetch code -> install mjlab -> render a policy to mp4 -> upload.
# Same setup as hf_job.sh, but it renders instead of training, so it is minutes not hours.
#
#   hf jobs run --flavor a10g-large --timeout 30m --secrets HF_TOKEN \
#     -e TASK=Himalayas-Ascender-Slope20-G1 -e HF_REPO=iteratehack/g1-ascender \
#     -e CODE_TAR=dr-guardrails/code.tar.gz \
#     -e CKPT=Himalayas-Ascender-Slope20-G1/<run>/model_4399.pt \
#     nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04 bash -c "$(cat rl/chloe/scripts/hf_render_job.sh)"
set -euo pipefail
TASK=${TASK:-Himalayas-Ascender-Slope20-G1}
HF_REPO=${HF_REPO:?set HF_REPO=<org>/<model-repo>}
CODE_TAR=${CODE_TAR:-code.tar.gz}
CKPT=${CKPT:?set CKPT=<hub path of a model_*.pt>}
SECONDS_=${SECONDS_:-12}
OUT=${OUT:-climb.mp4}
export PATH="$HOME/.local/bin:$PATH"

apt-get update -qq && apt-get install -y -qq curl ca-certificates git libegl1 libgl1 libglvnd0 > /dev/null
# Headless rendering: MuJoCo needs EGL, there is no display in a job.
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
curl -LsSf https://astral.sh/uv/install.sh | sh > /dev/null
uv tool install -q huggingface_hub
hf download "$HF_REPO" "$CODE_TAR" --local-dir /dl
mkdir -p /work && tar -xzf "/dl/$CODE_TAR" -C /work && cd /work
uv venv -q -p 3.11 .venv && uv pip install -q -p .venv/bin/python mjlab
git clone -q --depth 1 --filter=blob:none --sparse https://github.com/google-deepmind/mujoco_menagerie /men \
  && git -C /men sparse-checkout set unitree_g1 -q \
  && mkdir -p assets/robots/g1/_menagerie && cp -r /men/unitree_g1 assets/robots/g1/_menagerie/
nvidia-smi --query-gpu=name --format=csv,noheader

hf download "$HF_REPO" "$CKPT" --local-dir /ckpt
.venv/bin/python -m rl.chloe.scripts.render_video "$TASK" "/ckpt/$CKPT" "$OUT" \
  --seconds "$SECONDS_" ${RENDER_ARGS:-}

hf upload "$HF_REPO" "$OUT" "renders/$(basename "$OUT")"
echo "DONE -> https://huggingface.co/$HF_REPO/blob/main/renders/$(basename "$OUT")"
