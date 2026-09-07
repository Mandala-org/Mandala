#!/usr/bin/env bash
set -euo pipefail

# CPU allocation: use the project default of 64 workers/BLAS threads.
export OMP_NUM_THREADS=64
export MKL_NUM_THREADS=64
export OPENBLAS_NUM_THREADS=64

python -c 'import json; p=json.load(open("artifacts/pair_stage6/sio2_independent_v1/summary.json")); assert p["passed"] and p["offsite_task_count"] == 2 and p["onsite_neural_task_count"] == 12'

root=artifacts/pair_stage6/sio2_independent_v1/onsite_ridge
for family in d1 d2 d3 d4; do
  output="${root}/${family}"
  extra=()
  if [[ "${family}" == d1 ]]; then
    extra+=(--include-mean-baseline)
  fi
  if [[ ! -f "${output}/summary.json" ]]; then
    mkdir -p "${output}"
    python -u scripts/pair_mappers/run_stage6_onsite_ridge_screen.py \
      --family "${family}" \
      --descriptor-cache-dir "/bigdata/casus/wdm/hamiltonian_learning/data/pair_descriptor_hamiltonian/${family}_sio2_v2" \
      --descriptor-summary "artifacts/pair_stage2/${family}_sio2_precompute_v2/summary.json" \
      --descriptor-schemas "artifacts/pair_stage2/${family}_sio2_precompute_v2/descriptor_schemas.json" \
      --normalization artifacts/pair_stage2/sio2_descriptor_normalization_v2/normalization.json \
      --normalization-summary artifacts/pair_stage2/sio2_descriptor_normalization_v2/summary.json \
      --promotions artifacts/pair_stage3/sio2_descriptor_promotions_v2/promotions.json \
      --baseline-cache-dir /bigdata/casus/wdm/hamiltonian_learning/data/pair_descriptor_hamiltonian/sio2_baseline_cache_v2 \
      --baseline-summary artifacts/pair_stage1/sio2_baseline_cache_v2/summary.json \
      --shard-registry artifacts/pair_stage1/sio2_baseline_cache_v2/shards.csv \
      --output-dir "${output}" --device cpu --seed 20260907 \
      --ridge-values 1e-10 1e-8 1e-6 --distance-bin-width-angstrom 0.5 \
      "${extra[@]}" \
      2>&1 | tee "${output}/launcher.log"
  fi
  python -c 'import json,sys; p=json.load(open(sys.argv[1])); assert p["completed"] and p["passed"] and not p["test_shards_read"]' "${output}/summary.json"
done
