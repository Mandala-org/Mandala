#!/usr/bin/env bash
set -euo pipefail

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

mkdir -p /bigdata/casus/wdm/hamiltonian_learning/data/pair_descriptor_hamiltonian/sio2_baseline_cache_v2
mkdir -p artifacts/pair_stage1/sio2_baseline_cache_v2
if [[ ! -f artifacts/pair_stage1/sio2_baseline_cache_v2/summary.json ]]; then
  python -u scripts/pair_data/cache_sio2_baseline.py \
  --input-npz /bigdata/casus/wdm/hamiltonian_learning/data/pair_descriptor_hamiltonian/sio2_source_v1/graph_data.npz \
  --audit-summary artifacts/pair_stage1/sio2_data_audit_v1/summary.json \
  --split-indices artifacts/pair_stage1/sio2_data_audit_v1/split_indices.npz \
  --cache-dir /bigdata/casus/wdm/hamiltonian_learning/data/pair_descriptor_hamiltonian/sio2_baseline_cache_v2 \
  --output-dir artifacts/pair_stage1/sio2_baseline_cache_v2 \
  --hamiltonian-cutoff-angstrom 6.5 \
  --descriptor-cutoff-angstrom 8.5 \
  --density-radial-count 2 \
  --density-l-max 2 \
  --envelope-bin-width-angstrom 0.05 \
  --plot-max-points-per-pair 50000 \
  --nao-max 14 \
  --num-workers 64 \
  --resume \
    2>&1 | tee -a artifacts/pair_stage1/sio2_baseline_cache_v2/launcher.log
fi

python -c 'import json; p=json.load(open("artifacts/pair_stage1/sio2_baseline_cache_v2/summary.json")); assert p["passed"] and p["cache_metadata"]["offsite_supervision"] == "all directed reverse-pair members"'
