#!/usr/bin/env bash
set -euo pipefail

export MPLCONFIGDIR="${TMPDIR:-/tmp}/mandala-matplotlib-stage3"
mkdir -p "$MPLCONFIGDIR"
mkdir -p artifacts/pair_stage3/sio2_information_suite_v1
python -u scripts/pair_descriptors/run_stage3_information_suite.py \
  --family d1 artifacts/pair_stage2/d1_sio2_precompute_v1/descriptor_schemas.json \
  --family d2 artifacts/pair_stage2/d2_sio2_precompute_v1/descriptor_schemas.json \
  --family d3 artifacts/pair_stage2/d3_sio2_precompute_v1/descriptor_schemas.json \
  --family d4 artifacts/pair_stage2/d4_sio2_precompute_v1/descriptor_schemas.json \
  --normalization-json artifacts/pair_stage2/sio2_descriptor_normalization_v1/normalization.json \
  --d1-cache-dir /bigdata/casus/wdm/hamiltonian_learning/data/pair_descriptor_hamiltonian/d1_sio2_v1 \
  --d1-registry artifacts/pair_stage2/d1_sio2_precompute_v1/shards.csv \
  --output-dir artifacts/pair_stage3/sio2_information_suite_v1 \
  --num-workers 64 \
  --seed 20260905 \
  --neighbor-counts 4 6 8 \
  --synthetic-repeats 2 \
  --real-case-count 2 \
  --real-neighbor-count 8 \
  --inverse-starts 8 \
  --inverse-steps 500 \
  --inverse-polish-steps 50 \
  --inverse-learning-rate 0.03 \
  --collision-starts 4 \
  --collision-steps 350 \
  --collision-learning-rate 0.02 \
  --collision-minimum-geometry-rms-angstrom 0.05 \
  --continuation-steps 100 \
  --continuation-learning-rate 0.01 \
  --descriptor-match-tolerance 1e-7 \
  --reconstruction-rmsd-tolerance-angstrom 1e-4 \
  --rank-relative-tolerance 1e-8 \
  --resume \
  2>&1 | tee artifacts/pair_stage3/sio2_information_suite_v1/launcher.log
