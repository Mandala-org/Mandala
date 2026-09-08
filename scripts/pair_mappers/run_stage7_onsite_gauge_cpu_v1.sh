#!/usr/bin/env bash
set -euo pipefail
export OMP_NUM_THREADS=64 MKL_NUM_THREADS=64 OPENBLAS_NUM_THREADS=64
root=artifacts/pair_stage7/sio2_batch_a_v1/onsite_gauge_audit
if [[ ! -f "${root}/summary.json" ]]; then
  attempt=$(mktemp -d artifacts/pair_stage7/sio2_batch_a_v1/onsite_gauge_audit.work.XXXXXX)
  python -u scripts/pair_mappers/audit_stage7_onsite_identity_gauge.py \
    --descriptor-cache-dir /bigdata/casus/wdm/hamiltonian_learning/data/pair_descriptor_hamiltonian/d4_sio2_v2 \
    --descriptor-schemas artifacts/pair_stage2/d4_sio2_precompute_v2/descriptor_schemas.json \
    --normalization artifacts/pair_stage2/sio2_descriptor_normalization_v2/normalization.json \
    --baseline-cache-dir /bigdata/casus/wdm/hamiltonian_learning/data/pair_descriptor_hamiltonian/sio2_baseline_cache_v2 \
    --shard-registry artifacts/pair_stage1/sio2_baseline_cache_v2/shards.csv \
    --model-checkpoint artifacts/pair_stage6/sio2_independent_v1/onsite_ridge/d4/models/r6p5_d4_spherical_bessel_high_n8_l6_b3__affine_ridge_1em06.pt \
    --descriptor-key r6p5_d4_spherical_bessel_high_n8_l6_b3 --ridge 1e-6 \
    --cpu-manifest artifacts/pair_stage7/sio2_batch_a_v1/cpu_manifest.json \
    --output-dir "${attempt}" 2>&1 | tee "${attempt}/launcher.log"
  mv "${attempt}" "${root}"
fi
python -c 'import json; p=json.load(open("artifacts/pair_stage7/sio2_batch_a_v1/onsite_gauge_audit/summary.json")); assert p["completed"] and p["passed"] and p["diagnostic_only"] and not p["test_shards_read"]'
