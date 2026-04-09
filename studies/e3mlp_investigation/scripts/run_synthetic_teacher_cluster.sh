#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../../.."
source mandala-venv/bin/activate

RUN_NAME="${RUN_NAME:-cluster_synthetic_teacher_$(date +%Y%m%d_%H%M%S)}"
TEACHER_KIND="${TEACHER_KIND:-mixed}"
VARIANTS="${VARIANTS:-normact,gatemagnitudes,resnormact,film,bilinear}"
DEPTHS="${DEPTHS:-2,4,6}"
LOSS_KINDS="${LOSS_KINDS:-mse,mae,huber}"
NUM_TRAIN="${NUM_TRAIN:-512}"
NUM_VAL="${NUM_VAL:-128}"
BATCH_SIZE="${BATCH_SIZE:-64}"
NUM_STEPS="${NUM_STEPS:-64}"
LR="${LR:-1e-3}"
NUM_SEEDS="${NUM_SEEDS:-4}"
HIDDEN_IRREPS_PRESET="${HIDDEN_IRREPS_PRESET:-preliminary}"
DEVICE="${DEVICE:-cuda}"

export MPLCONFIGDIR="${MPLCONFIGDIR:-studies/e3mlp_investigation/cache/mplconfig}"

python -u studies/e3mlp_investigation/scripts/run_synthetic_teacher_study.py \
  --teacher-kind "${TEACHER_KIND}" \
  --variants "${VARIANTS}" \
  --depths "${DEPTHS}" \
  --loss-kinds "${LOSS_KINDS}" \
  --num-train "${NUM_TRAIN}" \
  --num-val "${NUM_VAL}" \
  --batch-size "${BATCH_SIZE}" \
  --num-steps "${NUM_STEPS}" \
  --lr "${LR}" \
  --num-seeds "${NUM_SEEDS}" \
  --hidden-irreps-preset "${HIDDEN_IRREPS_PRESET}" \
  --device "${DEVICE}" \
  --run-name "${RUN_NAME}"
