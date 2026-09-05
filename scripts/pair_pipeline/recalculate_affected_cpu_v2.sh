#!/usr/bin/env bash
set -euo pipefail

# Submit this launcher on a 64-CPU node. It is restartable at cache-shard and
# reconstruction-task boundaries and stops at the validation-only model gate.
bash scripts/pair_data/cache_sio2_directed_cpu_v2.sh
bash scripts/pair_descriptors/rebuild_stage2_sio2_cpu_v2.sh
bash scripts/pair_descriptors/rebuild_stage3_sio2_cpu_v2.sh

echo "CPU recalculation v2 complete. The directed cache, all descriptor families, certification, and geometry-only promotions passed."
