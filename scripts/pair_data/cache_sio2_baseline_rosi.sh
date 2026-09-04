#!/usr/bin/env bash
set -euo pipefail

cd /home/brzoza73/casus/mandala
mkdir -p /bigdata/casus/wdm/hamiltonian_learning/data/pair_descriptor_hamiltonian/sio2_baseline_cache_v1
mkdir -p artifacts/pair_stage1/sio2_baseline_cache_v1

python -u scripts/pair_data/cache_sio2_baseline.py \
  --input-npz /bigdata/casus/wdm/hamiltonian_learning/data/pair_descriptor_hamiltonian/sio2_source_v1/graph_data.npz \
  --audit-summary /home/brzoza73/casus/mandala/artifacts/pair_stage1/sio2_data_audit_v1/summary.json \
  --split-indices /home/brzoza73/casus/mandala/artifacts/pair_stage1/sio2_data_audit_v1/split_indices.npz \
  --cache-dir /bigdata/casus/wdm/hamiltonian_learning/data/pair_descriptor_hamiltonian/sio2_baseline_cache_v1 \
  --output-dir /home/brzoza73/casus/mandala/artifacts/pair_stage1/sio2_baseline_cache_v1 \
  --hamiltonian-cutoff-angstrom 6.5 \
  --descriptor-cutoff-angstrom 8.5 \
  --density-radial-count 2 \
  --density-l-max 2 \
  --envelope-bin-width-angstrom 0.05 \
  --nao-max 14 \
  --num-workers 8 \
  --resume \
  2>&1 | tee artifacts/pair_stage1/sio2_baseline_cache_v1/launcher.log
