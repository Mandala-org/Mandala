#!/usr/bin/env bash
set -euo pipefail

# The original blind inverse optimizer failed despite full Jacobian rank. This
# result-informed batch uses a constructive l=0/l=1 initializer and confirms
# the analytically eligible resolution grid on 102 synthetic + 8 real cases.
if [[ ! -f artifacts/pair_stage3/sio2_confirmation_cases_v1/summary.json ]]; then
  bash scripts/pair_descriptors/generate_stage3_confirmation_cases_cpu_v1.sh
fi
python -c 'import json; assert json.load(open("artifacts/pair_stage3/sio2_confirmation_cases_v1/summary.json"))["passed"]'

if [[ ! -f artifacts/pair_stage3/sio2_reconstruction_confirmation_v1/summary.json ]]; then
  bash scripts/pair_descriptors/run_stage3_reconstruction_confirmation_cpu_v1.sh
fi
python -c 'import json; assert json.load(open("artifacts/pair_stage3/sio2_reconstruction_confirmation_v1/summary.json"))["completed"]'
