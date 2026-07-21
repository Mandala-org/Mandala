#!/usr/bin/env bash
set -euo pipefail

SHARD_INDEX="${SLURM_ARRAY_TASK_ID:-${1:-}}"
NUM_SHARDS="${NUM_SHARDS:-64}"

if [[ -z "${SHARD_INDEX}" ]]; then
  echo "SLURM_ARRAY_TASK_ID is unset; provide a shard index as argument 1." >&2
  exit 1
fi
if ! [[ "${SHARD_INDEX}" =~ ^[0-9]+$ ]]; then
  echo "Shard index must be an integer, got: ${SHARD_INDEX}" >&2
  exit 1
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

python -u scripts/warmup_zncusnses_big_cache.py \
  --data-path /bigdata/casus/wdm/hamiltonian_learning/data/ZnCuSnSeS_big \
  --snapshot-cache-dir /bigdata/casus/wdm/hamiltonian_learning/data/ZnCuSnSeS_big/snapshot_cache \
  --scales 1,2 \
  --expected-per-scale 400 \
  --num-shards "${NUM_SHARDS}" \
  --shard-index "${SHARD_INDEX}" \
  --envelope-path eval_outputs/zncusnses_radial_fit_study/slater_soft_cutoff_envelope.json
