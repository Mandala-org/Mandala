#!/usr/bin/env bash
set -euo pipefail

mkdir -p artifacts/pair_stage2/sio2_descriptor_normalization_v1
python -u scripts/pair_descriptors/fit_descriptor_normalization_sio2.py \
  --family d1 /bigdata/casus/wdm/hamiltonian_learning/data/pair_descriptor_hamiltonian/d1_sio2_v1 artifacts/pair_stage2/d1_sio2_precompute_v1 artifacts/pair_stage2/d1_sio2_precompute_v1/descriptor_schemas.json \
  --family d2 /bigdata/casus/wdm/hamiltonian_learning/data/pair_descriptor_hamiltonian/d2_sio2_v1 artifacts/pair_stage2/d2_sio2_precompute_v1 artifacts/pair_stage2/d2_sio2_precompute_v1/descriptor_schemas.json \
  --family d3 /bigdata/casus/wdm/hamiltonian_learning/data/pair_descriptor_hamiltonian/d3_sio2_v1 artifacts/pair_stage2/d3_sio2_precompute_v1 artifacts/pair_stage2/d3_sio2_precompute_v1/descriptor_schemas.json \
  --family d4 /bigdata/casus/wdm/hamiltonian_learning/data/pair_descriptor_hamiltonian/d4_sio2_v1 artifacts/pair_stage2/d4_sio2_precompute_v1 artifacts/pair_stage2/d4_sio2_precompute_v1/descriptor_schemas.json \
  --output-dir artifacts/pair_stage2/sio2_descriptor_normalization_v1 \
  --num-workers 64 \
  --scale-floor 1e-12 \
  2>&1 | tee artifacts/pair_stage2/sio2_descriptor_normalization_v1/launcher.log
