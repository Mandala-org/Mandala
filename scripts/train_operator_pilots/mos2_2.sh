#!/usr/bin/env bash
set -euo pipefail
if [[ $# != 0 ]]; then
  echo "This frozen launcher takes no arguments." >&2
  exit 2
fi
export PYTHONPATH="$(pwd)/src:$(pwd)"
export PYTHONHASHSEED=42 CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1
export MPLCONFIGDIR=/tmp/mandala-operator-pilots-mpl
exec python -u -m scripts.run_training_segment --config configs/pilots/mos2_mandala.yaml --output /bigdata/casus/wdm/hamiltonian_learning/models/checkpoints/operator_benchmark_pilots_v1/mos2 --records researchops/runs/operator_benchmark_pilots_v1/mos2 --segment 2 --segments 2 --budget-seconds 162000 --accelerator gpu --evaluate-final-test
