#!/usr/bin/env bash
set -euo pipefail
root=artifacts/pair_stage7/sio2_batch_a_v1
aggregate="${root}/offsite/aggregate_v1"
if [[ ! -f "${aggregate}/summary.json" ]]; then
  aggregate_attempt=$(mktemp -d "${root}"/offsite/aggregate_v1.work.XXXXXX)
  python -u scripts/pair_mappers/aggregate_stage5_neural_calibration.py \
    --grid-dir "${root}"/offsite --output-dir "${aggregate_attempt}"
  mv "${aggregate_attempt}" "${aggregate}"
fi
gate="${root}/gate_v1"
if [[ ! -f "${gate}/summary.json" ]]; then
  gate_attempt=$(mktemp -d "${root}"/gate_v1.work.XXXXXX)
  python -u scripts/pair_mappers/finalize_stage7_batch_a.py \
    --stage7-root "${root}" \
    --stage6-root artifacts/pair_stage6/sio2_independent_v1 \
    --output-dir "${gate_attempt}"
  mv "${gate_attempt}" "${gate}"
fi
python -c 'import json; p=json.load(open("artifacts/pair_stage7/sio2_batch_a_v1/gate_v1/summary.json")); assert p["completed"] and p["passed"] and not p["test_shards_read"]'
