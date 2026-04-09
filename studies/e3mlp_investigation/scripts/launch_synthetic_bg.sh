#!/usr/bin/env bash
set -euo pipefail

cd /home/bartek/casus/mandala
source mandala-venv/bin/activate
export MPLCONFIGDIR="${MPLCONFIGDIR:-studies/e3mlp_investigation/cache/mplconfig}"

RUN_NAME="${RUN_NAME:?RUN_NAME is required}"
OUT_DIR="studies/e3mlp_investigation/artifacts/${RUN_NAME}"
mkdir -p "${OUT_DIR}"

python studies/e3mlp_investigation/scripts/run_synthetic_teacher_study.py \
  --variants "${VARIANTS:-normact,gatemagnitudes,resnormact,film,bilinear}" \
  --depths "${DEPTHS:-2,4}" \
  --loss-kinds "${LOSS_KINDS:-mse,mae,huber}" \
  --teacher-kind "${TEACHER_KIND:-mixed}" \
  --num-train "${NUM_TRAIN:-256}" \
  --num-val "${NUM_VAL:-64}" \
  --batch-size "${BATCH_SIZE:-32}" \
  --num-steps "${NUM_STEPS:-24}" \
  --num-seeds "${NUM_SEEDS:-2}" \
  --run-name "${RUN_NAME}" \
  --append-log \
  > "${OUT_DIR}/stdout.log" 2>&1

echo 0 > "${OUT_DIR}/exit_code.txt"
