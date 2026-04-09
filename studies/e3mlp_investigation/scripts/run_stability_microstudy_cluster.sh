#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../../.."
source mandala-venv/bin/activate

RUN_NAME="${RUN_NAME:-cluster_stability_$(date +%Y%m%d_%H%M%S)}"
VARIANTS="${VARIANTS:-normact,gatemagnitudes,resnormact,resgatemagnitudes,film,bilinear}"
DEPTHS="${DEPTHS:-4,6,10}"
OUTPUT_SCALES="${OUTPUT_SCALES:-0.5,1.0,1.5}"
WEIGHT_INIT_SCALES="${WEIGHT_INIT_SCALES:-0.5,1.0,1.5}"
RESIDUAL_SCALES="${RESIDUAL_SCALES:-0.1,0.25}"
NUM_STEPS="${NUM_STEPS:-16}"
NUM_SEEDS="${NUM_SEEDS:-4}"
BATCH_SIZE="${BATCH_SIZE:-32}"
LR="${LR:-1e-3}"
HIDDEN_IRREPS_PRESET="${HIDDEN_IRREPS_PRESET:-preliminary}"
DEVICE="${DEVICE:-cuda}"

export MPLCONFIGDIR="${MPLCONFIGDIR:-studies/e3mlp_investigation/cache/mplconfig}"

python studies/e3mlp_investigation/scripts/run_stability_microstudy.py \
  --variants "${VARIANTS}" \
  --depths "${DEPTHS}" \
  --output-scales "${OUTPUT_SCALES}" \
  --weight-init-scales "${WEIGHT_INIT_SCALES}" \
  --residual-scales "${RESIDUAL_SCALES}" \
  --num-steps "${NUM_STEPS}" \
  --num-seeds "${NUM_SEEDS}" \
  --batch-size "${BATCH_SIZE}" \
  --lr "${LR}" \
  --hidden-irreps-preset "${HIDDEN_IRREPS_PRESET}" \
  --device "${DEVICE}" \
  --run-name "${RUN_NAME}" \
  --append-log
