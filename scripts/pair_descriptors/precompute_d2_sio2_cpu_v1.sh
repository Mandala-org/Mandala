#!/usr/bin/env bash
set -euo pipefail

mkdir -p artifacts/pair_stage2/d2_sio2_precompute_v1
python -u scripts/pair_descriptors/precompute_d2_d3_sio2.py \
  --family d2 \
  --input-oracle-dir /bigdata/casus/wdm/hamiltonian_learning/data/pair_descriptor_hamiltonian/d1_sio2_v1 \
  --oracle-summary artifacts/pair_stage2/d1_sio2_precompute_v1/summary.json \
  --oracle-registry artifacts/pair_stage2/d1_sio2_precompute_v1/shards.csv \
  --output-cache-dir /bigdata/casus/wdm/hamiltonian_learning/data/pair_descriptor_hamiltonian/d2_sio2_v1 \
  --output-dir artifacts/pair_stage2/d2_sio2_precompute_v1 \
  --cutoffs-angstrom 6.5 7.5 8.5 10.5 \
  --resolution degree4 4 4 \
  --resolution degree6 6 6 \
  --resolution degree8 8 8 \
  --resolution degree10 10 10 \
  --num-workers 64 \
  --resume \
  2>&1 | tee artifacts/pair_stage2/d2_sio2_precompute_v1/launcher.log
