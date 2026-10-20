#!/usr/bin/env bash
set -euo pipefail
# Run from the repository root in the project environment. No submission here.
export PYTHONHASHSEED=42
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export MPLCONFIGDIR=/tmp/mandala-publication-mpl
exec python -u scripts/reproduce_examples/train.py --recipe scripts/reproduce_examples/sio2.json "$@"
