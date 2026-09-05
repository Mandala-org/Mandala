#!/usr/bin/env bash
set -euo pipefail

mkdir -p artifacts/pair_stage3/sio2_reconstruction_rescue_v1
python -u scripts/pair_descriptors/run_stage3_reconstruction_rescue.py \
  --family d1 artifacts/pair_stage2/d1_sio2_precompute_v1/descriptor_schemas.json \
  --family d2 artifacts/pair_stage2/d2_sio2_precompute_v1/descriptor_schemas.json \
  --family d3 artifacts/pair_stage2/d3_sio2_precompute_v1/descriptor_schemas.json \
  --normalization-json artifacts/pair_stage2/sio2_descriptor_normalization_v1/normalization.json \
  --cases-json artifacts/pair_stage3/sio2_information_suite_v1/cases.json \
  --output-dir artifacts/pair_stage3/sio2_reconstruction_rescue_v1 \
  --num-workers 64 \
  --seed 20260905 \
  --radial-starts 12 \
  --inverse-starts 16 \
  --inverse-steps 400 \
  --inverse-polish-steps 50 \
  --inverse-polish-candidates 6 \
  --inverse-learning-rate 0.02 \
  --descriptor-match-tolerance 1e-7 \
  --reconstruction-rmsd-tolerance-angstrom 1e-4 \
  --resume \
  2>&1 | tee artifacts/pair_stage3/sio2_reconstruction_rescue_v1/launcher.log
