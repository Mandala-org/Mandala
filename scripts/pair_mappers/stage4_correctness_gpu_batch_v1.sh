#!/usr/bin/env bash
set -euo pipefail

bash scripts/pair_descriptors/freeze_stage3_promotions_v1.sh
# Result-dependent Stage-3 gate. This prevents accidental Stage-4 execution
# on an unfrozen or incomplete descriptor selection.
python -c 'import json; value=json.load(open("artifacts/pair_stage3/sio2_descriptor_promotions_v1/summary.json")); assert value["passed"] and value["preferred_high_gate_pass_count"] == 16'

bash scripts/pair_mappers/run_stage4_correctness_suite_gpu_v1.sh
python -c 'import json; value=json.load(open("artifacts/pair_stage4/sio2_mapper_correctness_v1/summary.json")); assert value["completed"] and value["passed"]'
