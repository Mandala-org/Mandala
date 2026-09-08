#!/usr/bin/env bash
set -euo pipefail

root=artifacts/pair_stage7/sio2_batch_a_v1
if [[ ! -f "${root}/summary.json" ]]; then
  mkdir -p artifacts/pair_stage7
  attempt=$(mktemp -d artifacts/pair_stage7/sio2_batch_a_v1.work.XXXXXX)
  python -u scripts/pair_mappers/freeze_stage7_batch_a.py \
    --stage6-root artifacts/pair_stage6/sio2_independent_v1 \
    --promotions artifacts/pair_stage3/sio2_descriptor_promotions_v2/promotions.json \
    --output-dir "${attempt}"
  mv "${attempt}" "${root}"
fi
python -c 'import json; p=json.load(open("artifacts/pair_stage7/sio2_batch_a_v1/summary.json")); assert p["passed"] and p["gpu_task_count"] == 4 and p["cpu_task_count"] == 2 and not p["test_shards_read"]'
