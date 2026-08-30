#!/bin/bash
# Launch an RL training job on Hugging Face Jobs (billed to the iteratehack org).
#
# Prerequisites (one-time):
#   pip install "huggingface_hub[jobs]"   # provides the `hf` CLI
#   hf auth login                         # org member with Jobs access
#
# Usage:
#   rl/scripts/launch_hf_training.sh [extra trainer flags...]
#
# Env-var knobs (all optional):
#   ENV_NAME=G1JoystickWalkDR   env to train
#   NUM_TIMESTEPS=100000000     total steps
#   NUM_ENVS=4096               parallel envs on the GPU
#   DR=1                        1 = --domain_randomization --init_from_policy mels
#   FLAVOR=a10g-large           HF Jobs hardware (see `hf jobs hardware`)
#   TIMEOUT=24h                 max job duration
#   ARTIFACTS_REPO=iteratehack/ascender-rl-artifacts
#
# Example — the standard fine-tune:
#   rl/scripts/launch_hf_training.sh
#
# With extra trainer flags:
#   rl/scripts/launch_hf_training.sh -- --playground_config_overrides \
#       '{"dr_config.slope_max_deg": 15.0, "dr_config.wind_max_speed_kmph": 36.0, "dr_config.friction_range": [1.0, 1.0]}'
#
# Checkpoints stream to the artifacts dataset repo every 5 minutes while the
# job runs; download mid-training with:
#   hf download iteratehack/ascender-rl-artifacts --repo-type dataset \
#       --include "runs/<exp>/checkpoints/*" --local-dir ./hf-checkpoints

set -euo pipefail
cd "$(dirname "$0")/../.."

ARTIFACTS_REPO="${ARTIFACTS_REPO:-iteratehack/ascender-rl-artifacts}"
ORG="${HF_JOBS_NAMESPACE:-iteratehack}"
FLAVOR="${FLAVOR:-a10g-large}"
TIMEOUT="${TIMEOUT:-24h}"
ENV_NAME="${ENV_NAME:-G1JoystickWalkDR}"
NUM_TIMESTEPS="${NUM_TIMESTEPS:-100000000}"
NUM_ENVS="${NUM_ENVS:-4096}"
DR="${DR:-1}"

TRAIN_FLAGS="--env_name $ENV_NAME --num_envs $NUM_ENVS --num_timesteps $NUM_TIMESTEPS --use_wandb --wandb_entity project-yeti --wandb_project ascender-rl"
if [[ "${DR:-1}" == "1" ]]; then
  TRAIN_FLAGS="$TRAIN_FLAGS --domain_randomization --init_from_policy mels"
fi
for extra in "$@"; do
  TRAIN_FLAGS="$TRAIN_FLAGS $extra"
done

STAMP=$(date +%Y%m%d-%H%M%S)
EXP_NAME="${ENV_NAME}-hf-${STAMP}"

echo "[launch] packaging code snapshot (incl. gitignored asset trees)..."
SNAP_DIR=$(mktemp -d)
tar czf "$SNAP_DIR/code.tar.gz" \
    --exclude=.git --exclude=__pycache__ --exclude=logs --exclude=build \
    --exclude=app/harness/episodes --exclude=wandb --exclude="*.egg-info" \
    . 2>/dev/null
hf upload "$ARTIFACTS_REPO" "$SNAP_DIR/code.tar.gz" code/code.tar.gz \
    --repo-type dataset --quiet > /dev/null
echo "[launch] code snapshot uploaded"

echo "[launch] experiment: $EXP_NAME  flavor: $FLAVOR (billed to $ORG)"

hf jobs run \
  --namespace iteratehack \
  --flavor "$FLAVOR" \
  --timeout "$TIMEOUT" \
  --detach \
  --secrets HF_TOKEN \
  -e ARTIFACTS_REPO="$ARTIFACTS_REPO" \
  -e EXP_NAME="$STAMP" \
  -e ENV_NAME="$ENV_NAME" \
  -e TRAIN_FLAGS="$TRAIN_FLAGS" \
  -v "hf://datasets/$ARTIFACTS_REPO:/artifacts:ro" \
  "python:3.12" \
  bash /artifacts/code/run_training.sh

echo "[launch] job submitted. Follow:"
echo "  hf jobs ps --namespace iteratehack"
echo "  hf jobs logs --namespace iteratehack <job-id>"
echo "Checkpoints stream to https://huggingface.co/datasets/$ARTIFACTS_REPO"
echo "  runs/<exp>/checkpoints/<step>/ — download mid-run with:"
echo "  hf download $ARTIFACTS_REPO --repo-type dataset --include 'runs/$STAMP/checkpoints/*' --local-dir ./hf-checkpoints"
