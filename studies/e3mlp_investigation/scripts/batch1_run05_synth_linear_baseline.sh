#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../../.."
source mandala-venv/bin/activate

RUN_NAME="${RUN_NAME:-batch1_synth_linear_$(date +%Y%m%d_%H%M%S)}"
DEVICE="${DEVICE:-cuda}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-studies/e3mlp_investigation/cache/mplconfig}"

echo "[batch1_run05] starting synthetic linear baseline sweep: ${RUN_NAME}"
python -u studies/e3mlp_investigation/scripts/run_synthetic_teacher_study.py \
  --teacher-kind linear \
  --variants "normact,gate,gatemagnitudes,resnormact,resgatemagnitudes" \
  --depths "2,4" \
  --loss-kinds "mse,huber" \
  --num-train 2048 \
  --num-val 512 \
  --batch-size 128 \
  --num-steps 80 \
  --lr 3e-4 \
  --num-seeds 3 \
  --hidden-irreps-preset preliminary \
  --device "${DEVICE}" \
  --run-name "${RUN_NAME}" \
  --append-log
echo "[batch1_run05] finished: ${RUN_NAME}"
