#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../../.."
source mandala-venv/bin/activate

RUN_NAME="${RUN_NAME:-batch1_stability_advanced_$(date +%Y%m%d_%H%M%S)}"
DEVICE="${DEVICE:-cuda}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-studies/e3mlp_investigation/cache/mplconfig}"

python studies/e3mlp_investigation/scripts/run_stability_microstudy.py \
  --variants "film,bilinear" \
  --depths "4,6,10" \
  --output-scales "0.35,0.5,1.0,1.5" \
  --weight-init-scales "0.25,0.5,1.0" \
  --residual-scales "0.05,0.1,0.25" \
  --num-steps 24 \
  --num-seeds 3 \
  --batch-size 64 \
  --lr 7e-4 \
  --hidden-irreps-preset preliminary \
  --device "${DEVICE}" \
  --run-name "${RUN_NAME}" \
  --append-log
