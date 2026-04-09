#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../../.."
source mandala-venv/bin/activate

RUN_NAME="${RUN_NAME:-batch2_silicon_density_mean_$(date +%Y%m%d_%H%M%S)}"
DEVICE="${DEVICE:-cuda}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-studies/e3mlp_investigation/cache/mplconfig}"

echo "[batch2_run03] starting silicon density mean sweep: ${RUN_NAME}"
python -u studies/e3mlp_investigation/scripts/run_silicon_nognn_study.py \
  --train-snapshots "2700K" \
  --eval-snapshots "900K" \
  --target density \
  --aggregation mean \
  --architecture single \
  --variant normact \
  --depth 4 \
  --hidden-irreps-preset preliminary \
  --num-steps 80 \
  --lr 3e-4 \
  --loss-kind huber \
  --topk 8 \
  --device "${DEVICE}" \
  --run-name "${RUN_NAME}" \
  --append-log
echo "[batch2_run03] finished: ${RUN_NAME}"
