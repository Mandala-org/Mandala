#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

# Prefer project venv when present.
if [[ -f "${REPO_ROOT}/mandala-venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "${REPO_ROOT}/mandala-venv/bin/activate"
fi

# Defaults avoid overwriting existing downloaded weights directory.
# WANDB_PROJECT="${WANDB_PROJECT:-100meV-train}"
RUN_NAME="${RUN_NAME:-100meV_train}"
DEVICE="${DEVICE:-cuda}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-studies/minimal_overfit_study/checkpoints}"

python studies/minimal_overfit_study/overfit_water_minimal.py \
  --run-name "${RUN_NAME}" \
  --data-path "data/small/H2O/original/H2O.matrix" \
  --info-path "data/small/H2O/original/H2O.info.out" \
  --convention "e3nn" \
  --training-unit "100meV" \
  --hidden-dim 32 \
  --l-max 4 \
  --hidden-irreps "32x0e+32x0o+16x1e+16x1o+16x2e+16x2o+8x3e+8x3o+8x4e" \
  --num-layers 2 \
  --cutoff-radius 7.5 \
  --n-radial 64 \
  --lr 0.01 \
  --num-epochs 40000 \
  --log-interval 100 \
  --adaptive-log-interval \
  --log-data \
  --log-model \
  --log-per-irrep-metrics \
  --log-per-irrep-images \
  --benchmark \
  --grad-clip 1 \
  --lr-factor 0.5 \
  --lr-patience 1000 \
  --generate-video \
  --device "${DEVICE}" \
  --checkpoint-dir "${CHECKPOINT_DIR}"
