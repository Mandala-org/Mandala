#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

RUN_NAME="smoke-test-$(date +%Y%m%d-%H%M%S)"

python studies/minimal_overfit_study/overfit_water_minimal.py \
  --run-name "${RUN_NAME}" \
  --num-epochs 100 \
  --log-interval 1 \
  --lr 1e-2 \
  --normalize-blocks \
  --require-exact-edge-match \
  --generate-video \
  --adaptive-log-interval \
  --log-data \
  --log-model \
  --log-per-irrep-metrics \
  --grad-clip 1.0 \
  --hidden-dim 64 \
  --l-max 4 \
  --num-layers 2 \
  --n-radial 128
