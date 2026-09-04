#!/usr/bin/env bash
set -euo pipefail

cd /home/brzoza73/casus/mandala
mkdir -p artifacts/pair_stage2/d1_sio2_precompute_v1

python -u scripts/pair_descriptors/precompute_d1_sio2.py \
  --input-cache-dir /bigdata/casus/wdm/hamiltonian_learning/data/pair_descriptor_hamiltonian/sio2_baseline_cache_v1 \
  --cache-summary artifacts/pair_stage1/sio2_baseline_cache_v1/summary.json \
  --shard-registry artifacts/pair_stage1/sio2_baseline_cache_v1/shards.csv \
  --output-cache-dir /bigdata/casus/wdm/hamiltonian_learning/data/pair_descriptor_hamiltonian/d1_sio2_v1 \
  --output-dir artifacts/pair_stage2/d1_sio2_precompute_v1 \
  --cutoffs-angstrom 6.5 7.5 8.5 10.5 \
  --radial-bases spherical_bessel zernike \
  --resolution compact 4 3 \
  --resolution high 8 6 \
  --num-workers 32 \
  --resume \
  2>&1 | tee artifacts/pair_stage2/d1_sio2_precompute_v1/launcher.log
