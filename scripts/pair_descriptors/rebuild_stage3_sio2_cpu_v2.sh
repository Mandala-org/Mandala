#!/usr/bin/env bash
set -euo pipefail

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export MPLCONFIGDIR="${TMPDIR:-/tmp}/mandala-matplotlib-stage3-v2"
mkdir -p "${MPLCONFIGDIR}"

mkdir -p artifacts/pair_stage3/sio2_information_suite_v2
if [[ ! -f artifacts/pair_stage3/sio2_information_suite_v2/summary.json ]]; then
  python -u scripts/pair_descriptors/run_stage3_information_suite.py \
  --family d1 artifacts/pair_stage2/d1_sio2_precompute_v2/descriptor_schemas.json \
  --family d2 artifacts/pair_stage2/d2_sio2_precompute_v2/descriptor_schemas.json \
  --family d3 artifacts/pair_stage2/d3_sio2_precompute_v2/descriptor_schemas.json \
  --family d4 artifacts/pair_stage2/d4_sio2_precompute_v2/descriptor_schemas.json \
  --normalization-json artifacts/pair_stage2/sio2_descriptor_normalization_v2/normalization.json \
  --d1-cache-dir /bigdata/casus/wdm/hamiltonian_learning/data/pair_descriptor_hamiltonian/d1_sio2_v2 \
  --d1-registry artifacts/pair_stage2/d1_sio2_precompute_v2/shards.csv \
  --output-dir artifacts/pair_stage3/sio2_information_suite_v2 \
  --num-workers 64 --seed 20260905 --neighbor-counts 4 6 8 \
  --synthetic-repeats 2 --real-case-count 2 --real-neighbor-count 8 \
  --inverse-starts 8 --inverse-steps 500 --inverse-polish-steps 50 \
  --inverse-learning-rate 0.03 --collision-starts 4 --collision-steps 350 \
  --collision-learning-rate 0.02 --collision-minimum-geometry-rms-angstrom 0.05 \
  --continuation-steps 100 --continuation-learning-rate 0.01 \
  --descriptor-match-tolerance 1e-7 --reconstruction-rmsd-tolerance-angstrom 1e-4 \
  --rank-relative-tolerance 1e-8 --resume \
    2>&1 | tee -a artifacts/pair_stage3/sio2_information_suite_v2/launcher.log
fi

mkdir -p artifacts/pair_stage3/sio2_confirmation_cases_v2
if [[ ! -f artifacts/pair_stage3/sio2_confirmation_cases_v2/summary.json ]]; then
  python -u scripts/pair_descriptors/generate_stage3_confirmation_cases.py \
  --d1-cache-dir /bigdata/casus/wdm/hamiltonian_learning/data/pair_descriptor_hamiltonian/d1_sio2_v2 \
  --d1-registry artifacts/pair_stage2/d1_sio2_precompute_v2/shards.csv \
  --output-dir artifacts/pair_stage3/sio2_confirmation_cases_v2 \
  --cutoffs-angstrom 6.5 7.5 8.5 10.5 --neighbor-counts 4 6 8 \
  --synthetic-cases-per-neighbor-count 34 --real-cases-per-cutoff 8 \
  --real-neighbor-count 8 --seed 20260905 --num-workers 64 \
    2>&1 | tee artifacts/pair_stage3/sio2_confirmation_cases_v2/launcher.log
fi

mkdir -p artifacts/pair_stage3/sio2_reconstruction_confirmation_v2
configuration_args=()
for radius in r6p5 r7p5 r8p5 r10p5; do
  configuration_args+=(--config-key "${radius}_spherical_bessel_compact_n4_l3")
  configuration_args+=(--config-key "${radius}_spherical_bessel_high_n8_l6")
  configuration_args+=(--config-key "${radius}_zernike_compact_n4_l3")
  configuration_args+=(--config-key "${radius}_zernike_high_n8_l6")
  configuration_args+=(--config-key "${radius}_d2_degree8_o8_l8")
  configuration_args+=(--config-key "${radius}_d2_degree10_o10_l10")
  configuration_args+=(--config-key "${radius}_d3_compact_o4_l3")
  configuration_args+=(--config-key "${radius}_d3_high_o8_l6")
done
if [[ ! -f artifacts/pair_stage3/sio2_reconstruction_confirmation_v2/summary.json ]]; then
  python -u scripts/pair_descriptors/run_stage3_reconstruction_rescue.py \
  --family d1 artifacts/pair_stage2/d1_sio2_precompute_v2/descriptor_schemas.json \
  --family d2 artifacts/pair_stage2/d2_sio2_precompute_v2/descriptor_schemas.json \
  --family d3 artifacts/pair_stage2/d3_sio2_precompute_v2/descriptor_schemas.json \
  --normalization-json artifacts/pair_stage2/sio2_descriptor_normalization_v2/normalization.json \
  --cases-json artifacts/pair_stage3/sio2_confirmation_cases_v2/cases.json \
  --output-dir artifacts/pair_stage3/sio2_reconstruction_confirmation_v2 \
  "${configuration_args[@]}" --num-workers 64 --seed 20260905 \
  --radial-starts 24 --inverse-starts 64 --inverse-steps 1200 \
  --inverse-polish-steps 100 --inverse-polish-candidates 8 \
  --inverse-learning-rate 0.02 --descriptor-match-tolerance 1e-7 \
  --reconstruction-rmsd-tolerance-angstrom 1e-4 --resume \
    2>&1 | tee -a artifacts/pair_stage3/sio2_reconstruction_confirmation_v2/launcher.log
fi

mkdir -p artifacts/pair_stage3/sio2_descriptor_promotions_v2
if [[ ! -f artifacts/pair_stage3/sio2_descriptor_promotions_v2/summary.json ]]; then
  python -u scripts/pair_descriptors/freeze_stage3_promotions.py \
  --certification-dir artifacts/pair_stage2/sio2_descriptor_certification_v2 \
  --information-dir artifacts/pair_stage3/sio2_information_suite_v2 \
  --confirmation-dir artifacts/pair_stage3/sio2_reconstruction_confirmation_v2 \
  --cases-dir artifacts/pair_stage3/sio2_confirmation_cases_v2 \
  --family-artifact d1 artifacts/pair_stage2/d1_sio2_precompute_v2 \
  --family-artifact d2 artifacts/pair_stage2/d2_sio2_precompute_v2 \
  --family-artifact d3 artifacts/pair_stage2/d3_sio2_precompute_v2 \
  --family-artifact d4 artifacts/pair_stage2/d4_sio2_precompute_v2 \
  --output-dir artifacts/pair_stage3/sio2_descriptor_promotions_v2 \
    --synthetic-success-threshold 0.99
fi

python -c 'import json; p=json.load(open("artifacts/pair_stage3/sio2_descriptor_promotions_v2/summary.json")); assert p["passed"] and p["preferred_high_gate_pass_count"] == 16 and not p["selection_uses_hamiltonian"]'
