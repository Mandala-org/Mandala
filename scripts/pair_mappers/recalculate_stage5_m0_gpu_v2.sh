#!/usr/bin/env bash
set -euo pipefail

export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16

for family in d1 d2 d3 d4; do
  output="artifacts/pair_stage5/sio2_m0_${family}_validation_v2"
  mkdir -p "${output}"
  if [[ ! -f "${output}/summary.json" ]]; then
    python -u scripts/pair_mappers/run_stage5_m0_screen.py \
    --family "${family}" \
    --descriptor-cache-dir "/bigdata/casus/wdm/hamiltonian_learning/data/pair_descriptor_hamiltonian/${family}_sio2_v2" \
    --descriptor-summary "artifacts/pair_stage2/${family}_sio2_precompute_v2/summary.json" \
    --descriptor-schemas "artifacts/pair_stage2/${family}_sio2_precompute_v2/descriptor_schemas.json" \
    --promotions artifacts/pair_stage3/sio2_descriptor_promotions_v2/promotions.json \
    --baseline-cache-dir /bigdata/casus/wdm/hamiltonian_learning/data/pair_descriptor_hamiltonian/sio2_baseline_cache_v2 \
    --baseline-summary artifacts/pair_stage1/sio2_baseline_cache_v2/summary.json \
    --shard-registry artifacts/pair_stage1/sio2_baseline_cache_v2/shards.csv \
    --range-envelope artifacts/pair_stage1/sio2_baseline_cache_v2/range_envelope.json \
    --output-dir "${output}" --device cuda --seed 20260905 \
    --bond-radial-count 2 --bond-l-max 4 --bond-cutoff-angstrom 6.5 \
    --ridge 1e-8 --range-fit-mode physical_design --batch-size 2048 \
    --envelope-floor-hartree 1e-8 --distance-bin-width-angstrom 0.5 \
      2>&1 | tee "${output}/launcher.log"
  fi
  python -c 'import json,sys; p=json.load(open(sys.argv[1])); assert p["completed"] and p["passed"] and not p["test_shards_read"]' "${output}/summary.json"
done

aggregate=artifacts/pair_stage5/sio2_m0_validation_aggregate_v2
mkdir -p "${aggregate}"
if [[ ! -f "${aggregate}/summary.json" ]]; then
  python -u scripts/pair_mappers/aggregate_stage5_m0_screen.py \
  --family-result d1 artifacts/pair_stage5/sio2_m0_d1_validation_v2 \
  --family-result d2 artifacts/pair_stage5/sio2_m0_d2_validation_v2 \
  --family-result d3 artifacts/pair_stage5/sio2_m0_d3_validation_v2 \
  --family-result d4 artifacts/pair_stage5/sio2_m0_d4_validation_v2 \
  --promotions artifacts/pair_stage3/sio2_descriptor_promotions_v2/promotions.json \
    --output-dir "${aggregate}"
fi

python -c 'import json; p=json.load(open("artifacts/pair_stage5/sio2_m0_validation_aggregate_v2/summary.json")); assert p["completed"] and p["passed"] and not p["test_shards_read"]'
