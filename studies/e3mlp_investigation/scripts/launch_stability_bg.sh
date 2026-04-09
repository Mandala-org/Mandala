#!/usr/bin/env bash
set -euo pipefail

cd /home/bartek/casus/mandala
source mandala-venv/bin/activate
export MPLCONFIGDIR="${MPLCONFIGDIR:-studies/e3mlp_investigation/cache/mplconfig}"

RUN_NAME="${RUN_NAME:?RUN_NAME is required}"
OUT_DIR="studies/e3mlp_investigation/artifacts/${RUN_NAME}"
mkdir -p "${OUT_DIR}"

python studies/e3mlp_investigation/scripts/run_stability_microstudy.py \
  --variants "${VARIANTS:-normact,gatemagnitudes,resnormact,resgatemagnitudes,film,bilinear}" \
  --depths "${DEPTHS:-4,10}" \
  --output-scales "${OUTPUT_SCALES:-0.5,1.0}" \
  --weight-init-scales "${WEIGHT_INIT_SCALES:-0.5,1.0}" \
  --residual-scales "${RESIDUAL_SCALES:-0.1,0.25}" \
  --num-steps "${NUM_STEPS:-8}" \
  --num-seeds "${NUM_SEEDS:-2}" \
  --run-name "${RUN_NAME}" \
  > "${OUT_DIR}/stdout.log" 2>&1

echo 0 > "${OUT_DIR}/exit_code.txt"
