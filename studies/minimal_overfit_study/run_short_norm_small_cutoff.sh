#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

RUN_NAME="short-norm-small-cutoff-$(date +%Y%m%d-%H%M%S)"

python studies/minimal_overfit_study/overfit_water_minimal.py \
  --run-name "${RUN_NAME}" \
  --num-epochs 500 \
  --log-interval 100 \
  --lr 1e-2 \
  --cutoff-radius 4 \
  --normalize-blocks \
  --apply-cutoff-to-targets \
  --generate-video \
  --adaptive-log-interval \
  --grad-clip 1.0 \
  --hidden-dim 64 \
  --l-max 4 \
  --num-layers 2 \
  --n-radial 128
