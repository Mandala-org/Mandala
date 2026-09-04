#!/usr/bin/env bash
set -euo pipefail

cd /home/brzoza73/casus/mandala
mkdir -p artifacts/pair_stage1/native_ace_core_h100_v1

python -u scripts/pair_baselines/validate_native_ace_core.py \
  --output-dir /home/brzoza73/casus/mandala/artifacts/pair_stage1/native_ace_core_h100_v1 \
  --device cuda \
  --seed 20260904 \
  --descriptor-cutoff-angstrom 8.5 \
  --density-radial-count 2 \
  --density-l-max 2 \
  --onsite-correlation-order 2 \
  --onsite-max-degree 6 \
  --bond-radial-count 2 \
  --bond-l-max 4 \
  --bond-cutoff-angstrom 6.5 \
  --offsite-max-degree 6 \
  --ridge 1e-8 \
  --pair-batch-size 512 \
  --warmup-iterations 5 \
  --benchmark-iterations 20 \
  2>&1 | tee artifacts/pair_stage1/native_ace_core_h100_v1/run.log
