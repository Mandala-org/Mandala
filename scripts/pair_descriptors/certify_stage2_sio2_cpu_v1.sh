#!/usr/bin/env bash
set -euo pipefail

mkdir -p artifacts/pair_stage2/sio2_descriptor_certification_v1
python -u scripts/pair_descriptors/certify_stage2_sio2.py \
  --family d1 /bigdata/casus/wdm/hamiltonian_learning/data/pair_descriptor_hamiltonian/d1_sio2_v1 artifacts/pair_stage2/d1_sio2_precompute_v1 \
  --family d2 /bigdata/casus/wdm/hamiltonian_learning/data/pair_descriptor_hamiltonian/d2_sio2_v1 artifacts/pair_stage2/d2_sio2_precompute_v1 \
  --family d3 /bigdata/casus/wdm/hamiltonian_learning/data/pair_descriptor_hamiltonian/d3_sio2_v1 artifacts/pair_stage2/d3_sio2_precompute_v1 \
  --family d4 /bigdata/casus/wdm/hamiltonian_learning/data/pair_descriptor_hamiltonian/d4_sio2_v1 artifacts/pair_stage2/d4_sio2_precompute_v1 \
  --normalization-summary artifacts/pair_stage2/sio2_descriptor_normalization_v1/summary.json \
  --output-dir artifacts/pair_stage2/sio2_descriptor_certification_v1 \
  --num-workers 64 \
  --seed 20260904 \
  --symmetry-tolerance 1e-9 \
  --permutation-tolerance 1e-12 \
  --puncture-tolerance 1e-10 \
  --float32-tolerance 5e-7 \
  --pbc-displacement-tolerance-angstrom 2e-5 \
  --storage-budget-bytes 1979120929996 \
  2>&1 | tee artifacts/pair_stage2/sio2_descriptor_certification_v1/launcher.log
