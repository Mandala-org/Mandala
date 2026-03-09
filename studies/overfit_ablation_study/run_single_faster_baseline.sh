#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

if [[ -f "${REPO_ROOT}/mandala-venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "${REPO_ROOT}/mandala-venv/bin/activate"
fi

RUN_NAME="${RUN_NAME:-overfit-ablation-faster-single}"
DEVICE="${DEVICE:-cuda}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-studies/overfit_ablation_study/checkpoints}"
WANDB_PROJECT="${WANDB_PROJECT:-ablation-study}"

export WANDB_PROJECT

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
  --num-epochs 10000 \
  --log-interval 200 \
  --grad-clip 0.0 \
  --lr-factor 0.5 \
  --lr-patience 600 \
  --loss-aggregation global \
  --sh-mode aligned \
  --e3layernorm false \
  --delta-learning false \
  --norm-kind component \
  --skip-connections true \
  --dtype float64 \
  --device "${DEVICE}" \
  --checkpoint-dir "${CHECKPOINT_DIR}" \
  --lbfgs-steps 20 \
  --seed 0 \
  --adaptive-log-interval \
  --log-data \
  --log-model \
  --log-per-irrep-metrics \
  --log-per-irrep-images \
  --benchmark \
  --generate-video
