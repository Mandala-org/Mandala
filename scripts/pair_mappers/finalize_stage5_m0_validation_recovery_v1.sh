#!/usr/bin/env bash
set -euo pipefail

export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16

for family in d1 d2 d3 d4; do
  output="artifacts/pair_stage5/sio2_m0_${family}_validation_v1"
  if [[ ! -f "${output}/summary.json" ]] \
    && [[ -f "${output}/config.json" ]] \
    && [[ -f "${output}/fit_diagnostics.json" ]] \
    && [[ -f "${output}/validation_metrics.json" ]]; then
    python -u scripts/pair_mappers/finalize_stage5_m0_screen.py \
      --output-dir "${output}" \
      --descriptor-summary "artifacts/pair_stage2/${family}_sio2_precompute_v1/summary.json" \
      --promotions artifacts/pair_stage3/sio2_descriptor_promotions_v1/promotions.json
  fi
done
