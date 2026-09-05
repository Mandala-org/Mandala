#!/usr/bin/env bash
set -euo pipefail

mkdir -p artifacts/pair_stage3/sio2_confirmation_cases_v1
python -u scripts/pair_descriptors/generate_stage3_confirmation_cases.py \
  --d1-cache-dir /bigdata/casus/wdm/hamiltonian_learning/data/pair_descriptor_hamiltonian/d1_sio2_v1 \
  --d1-registry artifacts/pair_stage2/d1_sio2_precompute_v1/shards.csv \
  --output-dir artifacts/pair_stage3/sio2_confirmation_cases_v1 \
  --cutoffs-angstrom 6.5 7.5 8.5 10.5 \
  --neighbor-counts 4 6 8 \
  --synthetic-cases-per-neighbor-count 34 \
  --real-cases-per-cutoff 8 \
  --real-neighbor-count 8 \
  --seed 20260905 \
  --num-workers 64 \
  2>&1 | tee artifacts/pair_stage3/sio2_confirmation_cases_v1/launcher.log
