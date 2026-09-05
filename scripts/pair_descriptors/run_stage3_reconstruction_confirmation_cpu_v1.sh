#!/usr/bin/env bash
set -euo pipefail

mkdir -p artifacts/pair_stage3/sio2_reconstruction_confirmation_v1
python -u scripts/pair_descriptors/run_stage3_reconstruction_rescue.py \
  --family d1 artifacts/pair_stage2/d1_sio2_precompute_v1/descriptor_schemas.json \
  --family d2 artifacts/pair_stage2/d2_sio2_precompute_v1/descriptor_schemas.json \
  --family d3 artifacts/pair_stage2/d3_sio2_precompute_v1/descriptor_schemas.json \
  --normalization-json artifacts/pair_stage2/sio2_descriptor_normalization_v1/normalization.json \
  --cases-json artifacts/pair_stage3/sio2_confirmation_cases_v1/cases.json \
  --output-dir artifacts/pair_stage3/sio2_reconstruction_confirmation_v1 \
  --config-key r6p5_spherical_bessel_compact_n4_l3 \
  --config-key r6p5_spherical_bessel_high_n8_l6 \
  --config-key r6p5_zernike_compact_n4_l3 \
  --config-key r6p5_zernike_high_n8_l6 \
  --config-key r6p5_d2_degree8_o8_l8 \
  --config-key r6p5_d2_degree10_o10_l10 \
  --config-key r6p5_d3_compact_o4_l3 \
  --config-key r6p5_d3_high_o8_l6 \
  --config-key r7p5_spherical_bessel_compact_n4_l3 \
  --config-key r7p5_spherical_bessel_high_n8_l6 \
  --config-key r7p5_zernike_compact_n4_l3 \
  --config-key r7p5_zernike_high_n8_l6 \
  --config-key r7p5_d2_degree8_o8_l8 \
  --config-key r7p5_d2_degree10_o10_l10 \
  --config-key r7p5_d3_compact_o4_l3 \
  --config-key r7p5_d3_high_o8_l6 \
  --config-key r8p5_spherical_bessel_compact_n4_l3 \
  --config-key r8p5_spherical_bessel_high_n8_l6 \
  --config-key r8p5_zernike_compact_n4_l3 \
  --config-key r8p5_zernike_high_n8_l6 \
  --config-key r8p5_d2_degree8_o8_l8 \
  --config-key r8p5_d2_degree10_o10_l10 \
  --config-key r8p5_d3_compact_o4_l3 \
  --config-key r8p5_d3_high_o8_l6 \
  --config-key r10p5_spherical_bessel_compact_n4_l3 \
  --config-key r10p5_spherical_bessel_high_n8_l6 \
  --config-key r10p5_zernike_compact_n4_l3 \
  --config-key r10p5_zernike_high_n8_l6 \
  --config-key r10p5_d2_degree8_o8_l8 \
  --config-key r10p5_d2_degree10_o10_l10 \
  --config-key r10p5_d3_compact_o4_l3 \
  --config-key r10p5_d3_high_o8_l6 \
  --num-workers 64 \
  --seed 20260905 \
  --radial-starts 24 \
  --inverse-starts 64 \
  --inverse-steps 1200 \
  --inverse-polish-steps 100 \
  --inverse-polish-candidates 8 \
  --inverse-learning-rate 0.02 \
  --descriptor-match-tolerance 1e-7 \
  --reconstruction-rmsd-tolerance-angstrom 1e-4 \
  --resume \
  2>&1 | tee artifacts/pair_stage3/sio2_reconstruction_confirmation_v1/launcher.log
