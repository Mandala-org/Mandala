#!/usr/bin/env bash
set -euo pipefail

# Submit this launcher on one H100 allocation with 16 host CPUs. All selection
# remains validation-only; no test shard is read.
bash scripts/pair_mappers/recalculate_stage4_gpu_v3.sh
bash scripts/pair_baselines/recalculate_native_ace_gpu_v2.sh
bash scripts/pair_mappers/recalculate_stage5_m0_gpu_v2.sh
bash scripts/pair_mappers/recalculate_stage5_neural_gpu_v2.sh

echo "GPU recalculation v2 complete. Return the compact artifact directories for joint evaluation."
