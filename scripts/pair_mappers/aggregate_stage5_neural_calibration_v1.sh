#!/usr/bin/env bash
set -euo pipefail

output=artifacts/pair_stage5/sio2_neural_calibration_grid_v1/aggregate_v1
if [[ ! -f "${output}/summary.json" ]]; then
  python -u scripts/pair_mappers/aggregate_stage5_neural_calibration.py \
    --grid-dir artifacts/pair_stage5/sio2_neural_calibration_grid_v1 \
    --output-dir "${output}"
fi
python -c 'import json; value=json.load(open("artifacts/pair_stage5/sio2_neural_calibration_grid_v1/aggregate_v1/summary.json")); assert value["completed"] and value["passed"] and value["configuration_count"] == 12 and not value["test_shards_read"]'
