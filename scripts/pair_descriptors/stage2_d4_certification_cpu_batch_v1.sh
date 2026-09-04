#!/usr/bin/env bash
set -euo pipefail

# D4 must exist before common train-only normalization and the joint D1--D4 gate.
if [[ ! -f artifacts/pair_stage2/d4_sio2_precompute_v1/summary.json ]]; then
  bash scripts/pair_descriptors/precompute_d4_sio2_cpu_v1.sh
fi
python -c 'import json; assert json.load(open("artifacts/pair_stage2/d4_sio2_precompute_v1/summary.json"))["passed"]'

if [[ ! -f artifacts/pair_stage2/sio2_descriptor_normalization_v1/summary.json ]]; then
  bash scripts/pair_descriptors/fit_descriptor_normalization_sio2_cpu_v1.sh
fi
python -c 'import json; assert json.load(open("artifacts/pair_stage2/sio2_descriptor_normalization_v1/summary.json"))["passed"]'

if [[ ! -f artifacts/pair_stage2/sio2_descriptor_certification_v1/summary.json ]]; then
  bash scripts/pair_descriptors/certify_stage2_sio2_cpu_v1.sh
fi
python -c 'import json; assert json.load(open("artifacts/pair_stage2/sio2_descriptor_certification_v1/summary.json"))["passed"]'
