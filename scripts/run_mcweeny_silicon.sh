#!/usr/bin/env bash
set -euo pipefail
# Frozen CPU evaluation batch. Run from the repository root in the active env.
python scripts/evaluate_density_purification.py \
  --manifest studies/mcweeny/silicon_cases.json \
  --output-dir analysis_outputs/mcweeny_silicon \
  --threads 64 \
  --chunk-size 4096
