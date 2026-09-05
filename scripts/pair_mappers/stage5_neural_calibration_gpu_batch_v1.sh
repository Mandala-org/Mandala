#!/usr/bin/env bash
set -euo pipefail

# One restartable H100 batch, using the allocation's 16 host CPUs.
export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16

bash scripts/pair_mappers/run_stage5_neural_calibration_gpu_v1.sh
bash scripts/pair_mappers/aggregate_stage5_neural_calibration_v1.sh

echo "Stage 5 neural calibration complete; return the grid manifest and aggregate_v1 plus each run's config, summary, history, and validation metrics."
