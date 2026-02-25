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

# Retrieved from WandB API:
#   run: b-brzoza/mandala-test-variants/raoblm2i
#   name: avid-sweep-12
#   commit: 8c947fbd1d64e9facbb9ef03524f713c3fd76a15
#
# Defaults below avoid clobbering existing downloaded weights in
# studies/minimal_overfit_study/checkpoints/avid-sweep-12.
RUN_NAME="${RUN_NAME:-avid-sweep-12-train-on-parts-mlp-for-scalars}"
DEVICE="${DEVICE:-cuda}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-studies/minimal_overfit_study/checkpoints}"

python studies/minimal_overfit_study/overfit_water_minimal.py \
  --run-name "${RUN_NAME}" \
  --data-path "data/small/H2O/original/H2O.matrix" \
  --info-path "data/small/H2O/original/H2O.info.out" \
  --convention "e3nn" \
  --xyz-permutation "012" \
  --change-box "left" \
  --box-convention "rows" \
  --hidden-dim 32 \
  --l-max 4 \
  --hidden-irreps "32x0e+32x0o+16x1e+16x1o+16x2e+16x2o+8x3e+8x3o+8x4e+8x4o" \
  --num-layers 2 \
  --cutoff-radius 7 \
  --n-radial 64 \
  --lr 0.01 \
  --num-epochs 40000 \
  --log-interval 200 \
  --adaptive-log-interval \
  --grad-clip 1 \
  --lr-factor 0.5 \
  --lr-patience 1000 \
  --normalize-blocks \
  --train-on-irrep-parts \
  --head-mlp-for-scalars \
  --generate-video \
  --device "${DEVICE}" \
  --checkpoint-dir "${CHECKPOINT_DIR}"
