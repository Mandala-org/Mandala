#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

source mandala-venv/bin/activate

OUTPUT_DIR="${OUTPUT_DIR:-data/pyscf_baseline/results}"
METHOD="${METHOD:-rks}"
XC="${XC:-lda,vwn}"
GRID_LEVEL="${GRID_LEVEL:-0}"
KX="${KX:-1}"
KY="${KY:-1}"
KZ="${KZ:-1}"
MAX_CYCLE="${MAX_CYCLE:-100}"
CONV_TOL="${CONV_TOL:-1e-8}"
OVERWRITE="${OVERWRITE:-1}"

COMMON_ARGS=(
  --output-dir "$OUTPUT_DIR"
  --method "$METHOD"
  --basis-mode openmx_like
  --xc "$XC"
  --grid-level "$GRID_LEVEL"
  --kmesh "$KX" "$KY" "$KZ"
  --max-cycle "$MAX_CYCLE"
  --conv-tol "$CONV_TOL"
)

if [[ "$OVERWRITE" == "1" ]]; then
  COMMON_ARGS+=(--overwrite)
fi

python data/pyscf_baseline/calc_pyscf_baseline.py \
  --info-path data/small/H2O/original/H2O.info.out \
  --run-name h2o_original_rks_openmx_like \
  "${COMMON_ARGS[@]}"

python data/pyscf_baseline/calc_pyscf_baseline.py \
  --info-path data/small/H2O/rotated/H2O.info.out \
  --run-name h2o_rotated_rks_openmx_like \
  "${COMMON_ARGS[@]}"

python data/pyscf_baseline/verify_pyscf_equivariance.py \
  --original-npz "$OUTPUT_DIR/h2o_original_rks_openmx_like.npz" \
  --rotated-npz "$OUTPUT_DIR/h2o_rotated_rks_openmx_like.npz" \
  --output-dir "$OUTPUT_DIR/equivariance"
