#!/usr/bin/env bash
set -euo pipefail

root=artifacts/pair_stage6/sio2_independent_v1
if [[ ! -f "${root}/summary.json" ]]; then
  mkdir -p "${root}"
  python -u scripts/pair_mappers/freeze_stage6_split_training.py \
    --calibration-aggregate artifacts/pair_stage5/sio2_neural_calibration_grid_v2/aggregate_v2/summary.json \
    --promotions artifacts/pair_stage3/sio2_descriptor_promotions_v2/promotions.json \
    --output-dir "${root}"
fi
python -c 'import json; p=json.load(open("artifacts/pair_stage6/sio2_independent_v1/summary.json")); assert p["passed"] and p["offsite_task_count"] == 2 and p["onsite_neural_task_count"] == 12 and not p["test_shards_read"]'
