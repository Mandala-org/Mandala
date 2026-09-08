#!/usr/bin/env bash
set -euo pipefail
export OMP_NUM_THREADS=64 MKL_NUM_THREADS=64 OPENBLAS_NUM_THREADS=64
root=artifacts/pair_stage7/sio2_batch_a_v1/onsite_ridge_extension
if [[ ! -f "${root}/summary.json" ]]; then
  attempt=$(mktemp -d artifacts/pair_stage7/sio2_batch_a_v1/onsite_ridge_extension.work.XXXXXX)
  python -u scripts/pair_mappers/run_stage6_onsite_ridge_screen.py \
    --family d4 \
    --descriptor-cache-dir /bigdata/casus/wdm/hamiltonian_learning/data/pair_descriptor_hamiltonian/d4_sio2_v2 \
    --descriptor-summary artifacts/pair_stage2/d4_sio2_precompute_v2/summary.json \
    --descriptor-schemas artifacts/pair_stage2/d4_sio2_precompute_v2/descriptor_schemas.json \
    --normalization artifacts/pair_stage2/sio2_descriptor_normalization_v2/normalization.json \
    --normalization-summary artifacts/pair_stage2/sio2_descriptor_normalization_v2/summary.json \
    --promotions artifacts/pair_stage3/sio2_descriptor_promotions_v2/promotions.json \
    --baseline-cache-dir /bigdata/casus/wdm/hamiltonian_learning/data/pair_descriptor_hamiltonian/sio2_baseline_cache_v2 \
    --baseline-summary artifacts/pair_stage1/sio2_baseline_cache_v2/summary.json \
    --shard-registry artifacts/pair_stage1/sio2_baseline_cache_v2/shards.csv \
    --output-dir "${attempt}" --device cpu --seed 20260907 \
    --ridge-values 1e-5 1e-4 1e-3 1e-2 --distance-bin-width-angstrom 0.5 \
    2>&1 | tee "${attempt}/launcher.log"
  mv "${attempt}" "${root}"
fi
python -c 'import json; p=json.load(open("artifacts/pair_stage7/sio2_batch_a_v1/onsite_ridge_extension/summary.json")); assert p["completed"] and p["passed"] and not p["test_shards_read"]'
