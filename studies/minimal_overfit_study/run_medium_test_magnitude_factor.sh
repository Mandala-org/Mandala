#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

RUN_NAME="run_medium_test_magnitude_factor"

python studies/minimal_overfit_study/overfit_water_minimal.py \
  --run-name "${RUN_NAME}" \
  --num-epochs 10000 \
  --log-interval 200 \
  --lr 1e-2 \
  --cutoff-radius 7.0 \
  --apply-cutoff-to-targets \
  --require-exact-edge-match \
  --generate-video \
  --log-per-irrep-images \
  --adaptive-log-interval \
  --log-data \
  --log-model \
  --train-on-irrep-parts \
  --grad-clip 1.0 \
  --hidden-dim 64 \
  --l-max 4 \
  --num-layers 2 \
  --n-radial 128 \
  --lr-patience 200 \
  --magnitude-factorization \
  --magnitude-lambda 1.0 \
  --benchmark
