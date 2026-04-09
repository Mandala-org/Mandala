#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../../.."
source mandala-venv/bin/activate

RUN_NAME="${RUN_NAME:-batch3_synth_mixed_prenorm_huber_$(date +%Y%m%d_%H%M%S)}"
DEVICE="${DEVICE:-cuda}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-studies/e3mlp_investigation/cache/mplconfig}"

echo "[batch3_run03] starting synthetic mixed prenorm huber sweep: ${RUN_NAME}"
python -u studies/e3mlp_investigation/scripts/run_synthetic_teacher_study.py \
  --teacher-kind mixed \
  --variants "normact,gate,gatemagnitudes,resnormact,resgatemagnitudes,film,bilinear" \
  --depths "2,4,6" \
  --loss-kinds "huber" \
  --num-train 2048 \
  --num-val 512 \
  --batch-size 128 \
  --num-steps 120 \
  --lr 3e-4 \
  --num-seeds 3 \
  --hidden-irreps-preset preliminary \
  --device "${DEVICE}" \
  --output-scale 0.5 \
  --weight-init-scale 0.5 \
  --residual-scale 0.05 \
  --pre-norm \
  --run-name "${RUN_NAME}"
echo "[batch3_run03] finished: ${RUN_NAME}"
