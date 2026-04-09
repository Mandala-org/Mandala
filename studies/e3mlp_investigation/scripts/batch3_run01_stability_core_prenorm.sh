#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../../.."
source mandala-venv/bin/activate

RUN_NAME="${RUN_NAME:-batch3_stability_core_prenorm_$(date +%Y%m%d_%H%M%S)}"
DEVICE="${DEVICE:-cuda}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-studies/e3mlp_investigation/cache/mplconfig}"

echo "[batch3_run01] starting stability core prenorm sweep: ${RUN_NAME}"
python -u studies/e3mlp_investigation/scripts/run_stability_microstudy.py \
  --variants "normact,gate,gatemagnitudes,resnormact,resgatemagnitudes" \
  --depths "4,6,10,12" \
  --output-scales "0.35,0.5,1.0" \
  --weight-init-scales "0.25,0.5,1.0" \
  --residual-scales "0.05,0.1" \
  --num-steps 16 \
  --num-seeds 2 \
  --batch-size 48 \
  --lr 7e-4 \
  --hidden-irreps-preset preliminary \
  --device "${DEVICE}" \
  --pre-norm \
  --run-name "${RUN_NAME}"
echo "[batch3_run01] finished: ${RUN_NAME}"
