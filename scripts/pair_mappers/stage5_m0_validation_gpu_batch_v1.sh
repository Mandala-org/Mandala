#!/usr/bin/env bash
set -euo pipefail

# One ordered H100 batch. Every GPU operation is limited to the allocation's
# 16 host CPUs; this batch deliberately stops before any test-set evaluation.
export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16

python -c 'import json; value=json.load(open("artifacts/pair_stage3/sio2_descriptor_promotions_v1/summary.json")); assert value["passed"] and value["promotion_count"] == 32 and not value["selection_uses_hamiltonian"]'

bash scripts/pair_mappers/finalize_stage5_m0_validation_recovery_v1.sh

if [[ ! -f artifacts/pair_stage4/sio2_mapper_correctness_v2/summary.json ]]; then
  bash scripts/pair_mappers/run_stage4_correctness_suite_gpu_v2.sh
fi
python -c 'import json; value=json.load(open("artifacts/pair_stage4/sio2_mapper_correctness_v2/summary.json")); assert value["completed"] and value["passed"]'

for family in d1 d2 d3 d4; do
  if [[ ! -f "artifacts/pair_stage5/sio2_m0_${family}_validation_v1/summary.json" ]]; then
    bash "scripts/pair_mappers/run_stage5_m0_${family}_gpu_v1.sh"
  fi
  python -c 'import json, sys; value=json.load(open(sys.argv[1])); assert value["completed"] and value["passed"] and not value["test_shards_read"]' "artifacts/pair_stage5/sio2_m0_${family}_validation_v1/summary.json"
done

if [[ ! -f artifacts/pair_stage5/sio2_m0_validation_aggregate_v1/summary.json ]]; then
  bash scripts/pair_mappers/aggregate_stage5_m0_validation_v1.sh
fi
python -c 'import json; value=json.load(open("artifacts/pair_stage5/sio2_m0_validation_aggregate_v1/summary.json")); assert value["completed"] and value["passed"] and value["configuration_count"] == 32 and not value["test_shards_read"]'

echo "Stage 5 M0 validation batch complete; return sio2_mapper_correctness_v2 and sio2_m0_validation_aggregate_v1."
