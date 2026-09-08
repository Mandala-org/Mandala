#!/usr/bin/env bash
set -euo pipefail
export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16
python -u scripts/pair_mappers/run_stage7_offsite_gpu.py --task-index 3 --stage7-root artifacts/pair_stage7/sio2_batch_a_v1 --stage6-root artifacts/pair_stage6/sio2_independent_v1
