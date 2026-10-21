#!/usr/bin/env bash
set -euo pipefail
# Run from the frozen repository checkout root in the Mandala environment.
if [[ $# -ne 0 ]]; then
  echo "This frozen launcher takes no arguments; use the Python CLI for variants." >&2
  exit 2
fi
export PYTHONPATH="$(pwd)/src:$(pwd)"
export PYTHONHASHSEED=42
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export MPLCONFIGDIR=/tmp/mandala-siox-models-mpl
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=1
exec python -u -m scripts.train_siox_models.run prepare --workers 8
