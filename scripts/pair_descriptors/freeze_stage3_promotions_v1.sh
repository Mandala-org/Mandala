#!/usr/bin/env bash
set -euo pipefail

python -u scripts/pair_descriptors/freeze_stage3_promotions.py \
  --artifacts-root artifacts \
  --output-dir artifacts/pair_stage3/sio2_descriptor_promotions_v1 \
  --synthetic-success-threshold 0.99
