#!/usr/bin/env bash
set -euo pipefail

python -u scripts/pair_mappers/aggregate_stage5_m0_screen.py \
  --family-result d1 artifacts/pair_stage5/sio2_m0_d1_validation_v1 \
  --family-result d2 artifacts/pair_stage5/sio2_m0_d2_validation_v1 \
  --family-result d3 artifacts/pair_stage5/sio2_m0_d3_validation_v1 \
  --family-result d4 artifacts/pair_stage5/sio2_m0_d4_validation_v1 \
  --promotions artifacts/pair_stage3/sio2_descriptor_promotions_v1/promotions.json \
  --output-dir artifacts/pair_stage5/sio2_m0_validation_aggregate_v1
