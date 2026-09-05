#!/usr/bin/env bash
set -euo pipefail

export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16

mkdir -p artifacts/pair_stage5/sio2_m0_d3_validation_v1
python -u scripts/pair_mappers/run_stage5_m0_screen.py \
  --family d3 \
  --descriptor-cache-dir /bigdata/casus/wdm/hamiltonian_learning/data/pair_descriptor_hamiltonian/d3_sio2_v1 \
  --descriptor-summary artifacts/pair_stage2/d3_sio2_precompute_v1/summary.json \
  --descriptor-schemas artifacts/pair_stage2/d3_sio2_precompute_v1/descriptor_schemas.json \
  --promotions artifacts/pair_stage3/sio2_descriptor_promotions_v1/promotions.json \
  --baseline-cache-dir /bigdata/casus/wdm/hamiltonian_learning/data/pair_descriptor_hamiltonian/sio2_baseline_cache_v1 \
  --baseline-summary artifacts/pair_stage1/sio2_baseline_cache_v1/summary.json \
  --shard-registry artifacts/pair_stage1/sio2_baseline_cache_v1/shards.csv \
  --range-envelope artifacts/pair_stage1/sio2_baseline_cache_v1/range_envelope.json \
  --output-dir artifacts/pair_stage5/sio2_m0_d3_validation_v1 \
  --device cuda --seed 20260905 \
  --bond-radial-count 2 --bond-l-max 4 --bond-cutoff-angstrom 6.5 \
  --ridge 1e-8 --batch-size 2048 \
  --envelope-floor-hartree 1e-8 --distance-bin-width-angstrom 0.5 \
  2>&1 | tee artifacts/pair_stage5/sio2_m0_d3_validation_v1/launcher.log
