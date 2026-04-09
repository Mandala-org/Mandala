#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../../.."
source mandala-venv/bin/activate

RUN_NAME="${RUN_NAME:-batch3_silicon_hamiltonian_mean_lr3x_$(date +%Y%m%d_%H%M%S)}"
DEVICE="${DEVICE:-cuda}"
VARIANT="${VARIANT:-gatemagnitudes}"
EDGE_BATCH_SIZE="${EDGE_BATCH_SIZE:-1024}"
if [[ "${VARIANT}" == "bilinear" ]]; then
  EDGE_BATCH_SIZE="${EDGE_BATCH_SIZE:-128}"
fi
export MPLCONFIGDIR="${MPLCONFIGDIR:-studies/e3mlp_investigation/cache/mplconfig}"

ARGS=(
  --train-snapshots 2700K
  --eval-snapshots 900K
  --target hamiltonian
  --aggregation mean
  --architecture single
  --variant "${VARIANT}"
  --depth 4
  --hidden-irreps-preset silicon
  --num-steps 80
  --lr 9e-4
  --loss-kind huber
  --topk 8
  --device "${DEVICE}"
  --output-scale 0.5
  --weight-init-scale 0.5
  --residual-scale 0.05
  --pre-norm
  --edge-batch-size "${EDGE_BATCH_SIZE}"
  --run-name "${RUN_NAME}"
)

if [[ -n "${RESUME_FROM:-}" ]]; then
  ARGS+=(--resume-from "${RESUME_FROM}")
fi

echo "[batch3_run06] starting silicon hamiltonian mean lr3x run: ${RUN_NAME}"
python -u studies/e3mlp_investigation/scripts/run_silicon_nognn_study.py "${ARGS[@]}"
echo "[batch3_run06] finished: ${RUN_NAME}"
