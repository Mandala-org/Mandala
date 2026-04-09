#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../../.."
source mandala-venv/bin/activate

RUN_NAME="${RUN_NAME:-batch1_synth_quadratic_$(date +%Y%m%d_%H%M%S)}"
DEVICE="${DEVICE:-cuda}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-studies/e3mlp_investigation/cache/mplconfig}"

echo "[batch1_run04] starting synthetic quadratic sweep: ${RUN_NAME}"
python -u studies/e3mlp_investigation/scripts/run_synthetic_teacher_study.py \
  --teacher-kind quadratic \
  --variants "normact,gatemagnitudes,resgatemagnitudes,film,bilinear" \
  --depths "2,4,6" \
  --loss-kinds "mse,huber" \
  --num-train 2048 \
  --num-val 512 \
  --batch-size 128 \
  --num-steps 140 \
  --lr 3e-4 \
  --num-seeds 3 \
  --hidden-irreps-preset preliminary \
  --device "${DEVICE}" \
  --run-name "${RUN_NAME}" \
  --append-log
echo "[batch1_run04] finished: ${RUN_NAME}"
