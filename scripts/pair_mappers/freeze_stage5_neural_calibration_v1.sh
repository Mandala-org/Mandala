#!/usr/bin/env bash
set -euo pipefail

if [[ ! -f artifacts/pair_stage5/sio2_neural_calibration_grid_v1/summary.json ]]; then
  python -u scripts/pair_mappers/freeze_stage5_neural_calibration.py \
    --m0-aggregate artifacts/pair_stage5/sio2_m0_validation_aggregate_v1/summary.json \
    --m0-results artifacts/pair_stage5/sio2_m0_validation_aggregate_v1/results.csv \
    --promotions artifacts/pair_stage3/sio2_descriptor_promotions_v1/promotions.json \
    --output-dir artifacts/pair_stage5/sio2_neural_calibration_grid_v1
fi
python -c 'import json; value=json.load(open("artifacts/pair_stage5/sio2_neural_calibration_grid_v1/summary.json")); assert value["completed"] and value["passed"] and value["task_count"] == 12 and not value["test_shards_read"]'
