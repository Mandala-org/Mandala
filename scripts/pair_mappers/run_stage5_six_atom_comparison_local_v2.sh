#!/usr/bin/env bash
set -euo pipefail

output=artifacts/pair_stage5/sio2_six_atom_comparison_v2
if [[ ! -f "${output}/summary.json" ]]; then
  mkdir -p "${output}"
  mandala-venv/bin/python -u scripts/pair_mappers/plot_stage5_six_atom_comparison.py \
    --baseline-cache-dir local_cache/sio2_baseline_cache_v2 \
    --baseline-summary artifacts/pair_stage1/sio2_baseline_cache_v2/summary.json \
    --shard-registry artifacts/pair_stage1/sio2_baseline_cache_v2/shards.csv \
    --range-envelope artifacts/pair_stage1/sio2_baseline_cache_v2/range_envelope.json \
    --native-config artifacts/pair_stage1/native_ace_sio2_h100_physical_v2/config.json \
    --native-summary artifacts/pair_stage1/native_ace_sio2_h100_physical_v2/summary.json \
    --native-checkpoint artifacts/pair_stage1/native_ace_sio2_h100_physical_v2/model.pt \
    --d1-cache-dir local_cache/d1_sio2_v2 \
    --d1-schemas artifacts/pair_stage2/d1_sio2_precompute_v2/descriptor_schemas.json \
    --normalization artifacts/pair_stage2/sio2_descriptor_normalization_v2/normalization.json \
    --m5-config artifacts/pair_stage5/sio2_neural_calibration_grid_v2/runs/m5_large_lr1em3_physical/config.json \
    --m5-summary artifacts/pair_stage5/sio2_neural_calibration_grid_v2/runs/m5_large_lr1em3_physical/summary.json \
    --m5-checkpoint artifacts/pair_stage5/sio2_neural_calibration_grid_v2/runs/m5_large_lr1em3_physical/best_model.pt \
    --m3-config artifacts/pair_stage5/sio2_neural_calibration_grid_v2/runs/m3_large_lr1em3_physical/config.json \
    --m3-summary artifacts/pair_stage5/sio2_neural_calibration_grid_v2/runs/m3_large_lr1em3_physical/summary.json \
    --m3-checkpoint artifacts/pair_stage5/sio2_neural_calibration_grid_v2/runs/m3_large_lr1em3_physical/best_model.pt \
    --output-dir "${output}" --atom-count 6 \
    --color-limit-hartree 0.05 --envelope-floor-hartree 1e-8 \
    2>&1 | tee "${output}/launcher.log"
fi
