#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../../.."
source mandala-venv/bin/activate

RUN_NAME="${RUN_NAME:-batch1_stability_core_$(date +%Y%m%d_%H%M%S)}"
DEVICE="${DEVICE:-cuda}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-studies/e3mlp_investigation/cache/mplconfig}"

echo "[batch1_run01] starting stability core sweep: ${RUN_NAME}"
python -u studies/e3mlp_investigation/scripts/run_stability_microstudy.py \
  --variants "normact,gate,gatemagnitudes,resnormact,resgatemagnitudes" \
  --depths "6,10,12" \
  --output-scales "0.5,1.0,1.5" \
  --weight-init-scales "0.5,1.0" \
  --residual-scales "0.05,0.25" \
  --num-steps 20 \
  --num-seeds 3 \
  --batch-size 64 \
  --lr 1e-3 \
  --hidden-irreps-preset preliminary \
  --device "${DEVICE}" \
  --run-name "${RUN_NAME}" \
  --append-log
echo "[batch1_run01] finished: ${RUN_NAME}"
