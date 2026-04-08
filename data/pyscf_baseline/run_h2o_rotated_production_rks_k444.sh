#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

source mandala-venv/bin/activate

OUTPUT_DIR="${OUTPUT_DIR:-data/pyscf_baseline/results}"
RUN_NAME="${RUN_NAME:-h2o_rotated_rks_openmx_like_prod_k444}"
XC="${XC:-pbe,pbe}"
GRID_LEVEL="${GRID_LEVEL:-3}"
KX="${KX:-4}"
KY="${KY:-4}"
KZ="${KZ:-4}"
DF_BACKEND="${DF_BACKEND:-gdf}"
DF_AUXBASIS="${DF_AUXBASIS:-}"
PBC_PRECISION="${PBC_PRECISION:-1e-8}"
KE_CUTOFF="${KE_CUTOFF:-}"
MAX_MEMORY_MB="${MAX_MEMORY_MB:-1200000}"
MAX_CYCLE="${MAX_CYCLE:-200}"
CONV_TOL="${CONV_TOL:-1e-9}"
OVERWRITE="${OVERWRITE:-1}"
REQUIRE_FORCES_STRESS="${REQUIRE_FORCES_STRESS:-0}"

mkdir -p "$OUTPUT_DIR"

ARGS=(
  --info-path data/small/H2O/rotated/H2O.info.out
  --output-dir "$OUTPUT_DIR"
  --run-name "$RUN_NAME"
  --xc "$XC"
  --grid-level "$GRID_LEVEL"
  --kmesh "$KX" "$KY" "$KZ"
  --df-backend "$DF_BACKEND"
  --pbc-precision "$PBC_PRECISION"
  --max-memory-mb "$MAX_MEMORY_MB"
  --max-cycle "$MAX_CYCLE"
  --conv-tol "$CONV_TOL"
)

if [[ -n "$DF_AUXBASIS" ]]; then
  ARGS+=(--df-auxbasis "$DF_AUXBASIS")
fi
if [[ -n "$KE_CUTOFF" ]]; then
  ARGS+=(--ke-cutoff "$KE_CUTOFF")
fi

if [[ "$OVERWRITE" == "1" ]]; then
  ARGS+=(--overwrite)
fi
if [[ "$REQUIRE_FORCES_STRESS" == "1" ]]; then
  ARGS+=(--require-forces-stress)
fi

echo "Running production KRKS (rotated) with kmesh=($KX $KY $KZ), xc=$XC, df_backend=$DF_BACKEND, run_name=$RUN_NAME"
python data/pyscf_baseline/calc_pyscf_baseline.py "${ARGS[@]}"

echo
echo "Done. Artifacts:"
echo "  $OUTPUT_DIR/$RUN_NAME.npz"
echo "  $OUTPUT_DIR/$RUN_NAME.json"
