#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

if [[ -f "${REPO_ROOT}/mandala-venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "${REPO_ROOT}/mandala-venv/bin/activate"
fi

RUN_NAME="${RUN_NAME:-overfit-ablation-smoke}"
DEVICE="${DEVICE:-cuda}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-studies/overfit_ablation_study/checkpoints}"
NUM_EPOCHS="${NUM_EPOCHS:-60}"
LOG_INTERVAL="${LOG_INTERVAL:-20}"
LR_PATIENCE="${LR_PATIENCE:-200}"
LBFGS_STEPS="${LBFGS_STEPS:-20}"

python -u studies/overfit_ablation_study/overfit_water_ablation.py \
  --run-name "${RUN_NAME}" \
  --data-path "data/small/H2O/original/H2O.matrix" \
  --info-path "data/small/H2O/original/H2O.info.out" \
  --hidden-dim 32 \
  --l-max 4 \
  --hidden-irreps "32x0e+32x0o+16x1e+16x1o+16x2e+16x2o+8x3e+8x3o+8x4e" \
  --num-layers 2 \
  --n-radial 64 \
  --lr 1e-3 \
  --num-epochs "${NUM_EPOCHS}" \
  --log-interval "${LOG_INTERVAL}" \
  --adaptive-log-interval \
  --log-data \
  --log-model \
  --log-per-irrep-metrics \
  --log-per-irrep-images \
  --benchmark \
  --device "${DEVICE}" \
  --dtype float64 \
  --checkpoint-dir "${CHECKPOINT_DIR}" \
  --grad-clip 1.0 \
  --lr-factor 0.5 \
  --lr-patience "${LR_PATIENCE}" \
  --loss-aggregation global \
  --sh-mode aligned \
  --train-on-irrep-parts \
  --generate-video \
  --e3layernorm true \
  --norm-kind component \
  --skip-connections true \
  --delta-learning true \
  --lbfgs-steps "${LBFGS_STEPS}" \
  --seed 0
