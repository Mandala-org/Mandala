#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

RUN_NAME="long-norm-small-cutoff-$(date +%Y%m%d-%H%M%S)"

python studies/minimal_overfit_study/overfit_water_minimal.py \
  --run-name "${RUN_NAME}" \
  --num-epochs 30000 \
  --log-interval 200 \
  --lr 3e-3 \
  --cutoff-radius 4.5 \
  --normalize-blocks \
  --apply-cutoff-to-targets \
  --grad-clip 1.0 \
  --hidden-dim 64 \
  --l-max 2 \
  --num-layers 2 \
  --n-radial 32
