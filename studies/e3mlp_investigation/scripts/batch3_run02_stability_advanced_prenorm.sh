#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../../.."
source mandala-venv/bin/activate

RUN_NAME="${RUN_NAME:-batch3_stability_advanced_prenorm_$(date +%Y%m%d_%H%M%S)}"
DEVICE="${DEVICE:-cuda}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-studies/e3mlp_investigation/cache/mplconfig}"

echo "[batch3_run02] starting stability advanced prenorm sweep: ${RUN_NAME}"
python -u studies/e3mlp_investigation/scripts/run_stability_microstudy.py \
  --variants "film,bilinear" \
  --depths "4,6,10" \
  --output-scales "0.2,0.35,0.5,1.0" \
  --weight-init-scales "0.2,0.35,0.5" \
  --residual-scales "0.02,0.05,0.1" \
  --num-steps 16 \
  --num-seeds 2 \
  --batch-size 48 \
  --lr 5e-4 \
  --hidden-irreps-preset preliminary \
  --device "${DEVICE}" \
  --pre-norm \
  --run-name "${RUN_NAME}"
echo "[batch3_run02] finished: ${RUN_NAME}"
