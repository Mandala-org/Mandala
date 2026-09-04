#!/usr/bin/env bash
set -euo pipefail

mkdir -p artifacts/pair_stage2/d4_sio2_precompute_v1
python -u scripts/pair_descriptors/precompute_d4_sio2.py \
  --input-d1-cache-dir /bigdata/casus/wdm/hamiltonian_learning/data/pair_descriptor_hamiltonian/d1_sio2_v1 \
  --d1-summary artifacts/pair_stage2/d1_sio2_precompute_v1/summary.json \
  --d1-registry artifacts/pair_stage2/d1_sio2_precompute_v1/shards.csv \
  --output-cache-dir /bigdata/casus/wdm/hamiltonian_learning/data/pair_descriptor_hamiltonian/d4_sio2_v1 \
  --output-dir artifacts/pair_stage2/d4_sio2_precompute_v1 \
  --cutoffs-angstrom 6.5 7.5 8.5 10.5 \
  --radial-bases spherical_bessel zernike \
  --resolution compact 4 3 \
  --resolution high 8 6 \
  --body-degrees 2 3 \
  --lift-max-input-l 3 \
  --lift-max-radial-index 1 \
  --lift-radial-degree-budget 2 \
  --lift-max-intermediate-l 4 \
  --max-paths-per-degree-irrep 48 \
  --num-workers 64 \
  --resume \
  2>&1 | tee artifacts/pair_stage2/d4_sio2_precompute_v1/launcher.log
