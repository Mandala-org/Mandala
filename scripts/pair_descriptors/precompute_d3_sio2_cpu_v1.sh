#!/usr/bin/env bash
set -euo pipefail

mkdir -p artifacts/pair_stage2/d3_sio2_precompute_v1
python -u scripts/pair_descriptors/precompute_d2_d3_sio2.py \
  --family d3 \
  --input-oracle-dir /bigdata/casus/wdm/hamiltonian_learning/data/pair_descriptor_hamiltonian/d1_sio2_v1 \
  --oracle-summary artifacts/pair_stage2/d1_sio2_precompute_v1/summary.json \
  --oracle-registry artifacts/pair_stage2/d1_sio2_precompute_v1/shards.csv \
  --output-cache-dir /bigdata/casus/wdm/hamiltonian_learning/data/pair_descriptor_hamiltonian/d3_sio2_v1 \
  --output-dir artifacts/pair_stage2/d3_sio2_precompute_v1 \
  --cutoffs-angstrom 6.5 7.5 8.5 10.5 \
  --resolution compact 4 3 \
  --resolution high 8 6 \
  --num-workers 64 \
  --resume \
  2>&1 | tee artifacts/pair_stage2/d3_sio2_precompute_v1/launcher.log
