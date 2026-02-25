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

# Based on WandB run:
#   b-brzoza/mandala-test-variants/raoblm2i  (name: avid-sweep-12)
# with additional orbital reduction and expanded debug logging.
#
# Defaults avoid overwriting existing downloaded weights directory.
RUN_NAME="${RUN_NAME:-avid-sweep-12-1s1p-debug}"
DEVICE="${DEVICE:-cuda}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-studies/minimal_overfit_study/checkpoints}"

python studies/minimal_overfit_study/overfit_water_minimal.py \
  --run-name "${RUN_NAME}" \
  --data-path "data/small/H2O/original/H2O.matrix" \
  --info-path "data/small/H2O/original/H2O.info.out" \
  --convention "e3nn" \
  --orbital-selection "1s1p" \
  --xyz-permutation "012" \
  --change-box "left" \
  --box-convention "rows" \
  --hidden-dim 32 \
  --l-max 4 \
  --hidden-irreps "32x0e+16x1e+16x1o+16x2e" \
  --num-layers 2 \
  --cutoff-radius 7 \
  --n-radial 64 \
  --lr 0.01 \
  --num-epochs 40000 \
  --log-interval 100 \
  --adaptive-log-interval \
  --log-data \
  --log-model \
  --log-per-irrep-metrics \
  --log-per-irrep-images \
  --log-activations-wandb \
  --benchmark \
  --grad-clip 1 \
  --lr-factor 0.5 \
  --lr-patience 1000 \
  --normalize-blocks \
  --generate-video \
  --device "${DEVICE}" \
  --checkpoint-dir "${CHECKPOINT_DIR}"
