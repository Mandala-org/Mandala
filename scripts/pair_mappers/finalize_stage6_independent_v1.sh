#!/usr/bin/env bash
set -euo pipefail

root=artifacts/pair_stage6/sio2_independent_v1
output="${root}/aggregate_v1"
if [[ ! -f "${output}/summary.json" ]]; then
  mkdir -p "${output}"
  python -u scripts/pair_mappers/aggregate_stage6_independent.py \
    --stage6-root "${root}" --output-dir "${output}"
fi
python -c 'import json; p=json.load(open("artifacts/pair_stage6/sio2_independent_v1/aggregate_v1/summary.json")); assert p["completed"] and p["passed"] and not p["test_shards_read"]'
