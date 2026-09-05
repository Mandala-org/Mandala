#!/usr/bin/env bash
set -euo pipefail

# H100 allocations provide 16 host CPUs. CPU-only launchers retain the
# repository-wide 64-worker default.
export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16

mkdir -p artifacts/pair_stage4/sio2_mapper_correctness_v1
python -u scripts/pair_mappers/run_stage4_correctness_suite.py \
  --output-dir artifacts/pair_stage4/sio2_mapper_correctness_v1 \
  --architectures m0 m1 m2 m3 m5 m7 \
  --device cuda \
  --seed 20260905 \
  --fit-samples 384 \
  --fit-validation-samples 128 \
  --fit-steps 800 \
  --fit-learning-rate 0.003 \
  --benchmark-batch-size 256 \
  --benchmark-warmup 5 \
  --benchmark-steps 20 \
  --equivariance-tolerance 1e-8 \
  --reversal-tolerance 1e-10 \
  --synthetic-fit-relative-rms 0.05 \
  2>&1 | tee artifacts/pair_stage4/sio2_mapper_correctness_v1/launcher.log
