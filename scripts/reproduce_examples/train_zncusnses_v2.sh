#!/usr/bin/env bash
set -euo pipefail
# Run from the repository root in the project environment. No submission here.
# The chain restarted on 10 Nov 2026 without the shifted-self head, so it
# resumes from the current source; the legacy frozen copy is SiO2-only now.
export PYTHONHASHSEED=42
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export MPLCONFIGDIR=/tmp/mandala-publication-mpl
exec python -u scripts/reproduce_examples/train.py --recipe scripts/reproduce_examples/zncusnses_v2.json "$@"
