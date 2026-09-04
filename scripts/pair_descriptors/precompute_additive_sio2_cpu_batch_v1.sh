#!/usr/bin/env bash
set -euo pipefail

# Stage-2 additive descriptor batch. D2 and D3 are independent once the frozen
# D1 neighbor oracle exists, so both are intentionally executed before review.
if [[ ! -f artifacts/pair_stage2/d2_sio2_precompute_v1/summary.json ]]; then
  bash scripts/pair_descriptors/precompute_d2_sio2_cpu_v1.sh
fi
python -c 'import json; assert json.load(open("artifacts/pair_stage2/d2_sio2_precompute_v1/summary.json"))["passed"]'

if [[ ! -f artifacts/pair_stage2/d3_sio2_precompute_v1/summary.json ]]; then
  bash scripts/pair_descriptors/precompute_d3_sio2_cpu_v1.sh
fi
python -c 'import json; assert json.load(open("artifacts/pair_stage2/d3_sio2_precompute_v1/summary.json"))["passed"]'
