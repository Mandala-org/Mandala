#!/usr/bin/env bash
set -euo pipefail

cd /home/brzoza73/casus/mandala
mkdir -p artifacts/pair_stage1/native_ace_sio2_h100_v1

python -u scripts/pair_baselines/train_native_ace_sio2.py \
  --cache-dir /bigdata/casus/wdm/hamiltonian_learning/data/pair_descriptor_hamiltonian/sio2_baseline_cache_v1 \
  --cache-summary /home/brzoza73/casus/mandala/artifacts/pair_stage1/sio2_baseline_cache_v1/summary.json \
  --shard-registry /home/brzoza73/casus/mandala/artifacts/pair_stage1/sio2_baseline_cache_v1/shards.csv \
  --range-envelope /home/brzoza73/casus/mandala/artifacts/pair_stage1/sio2_baseline_cache_v1/range_envelope.json \
  --core-validation-summary /home/brzoza73/casus/mandala/artifacts/pair_stage1/native_ace_core_h100_v1/summary.json \
  --output-dir /home/brzoza73/casus/mandala/artifacts/pair_stage1/native_ace_sio2_h100_v1 \
  --device cuda \
  --seed 20260904 \
  --onsite-correlation-order 2 \
  --onsite-max-degree 6 \
  --bond-radial-count 2 \
  --bond-l-max 4 \
  --bond-cutoff-angstrom 6.5 \
  --offsite-max-degree 6 \
  --ridge 1e-8 \
  --batch-size 2048 \
  --envelope-floor-hartree 1e-8 \
  --distance-bin-width-angstrom 0.5 \
  2>&1 | tee artifacts/pair_stage1/native_ace_sio2_h100_v1/launcher.log
