#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 <siox|zncusnses>" >&2
  exit 2
fi

DATASET_KIND=$1
SHARD_INDEX=${SLURM_ARRAY_TASK_ID:?Run this script as a Slurm array job}
NUM_SHARDS=${DEEPE3_CONVERT_SHARDS:-2}
NUM_WORKERS=${DEEPE3_CONVERT_WORKERS:-96}
DATA_ROOT=/bigdata/casus/wdm/hamiltonian_learning/data

case "${DATASET_KIND}" in
  siox)
    INPUT_DIR=${DATA_ROOT}/SiOx_new
    OUTPUT_DIR=${DATA_ROOT}/DeepH-E3/SiOx_new
    EXTRA_ARGS=()
    ;;
  zncusnses)
    INPUT_DIR=${DATA_ROOT}/ZnCuSnSeS
    OUTPUT_DIR=${DATA_ROOT}/DeepH-E3/ZnCuSnSeS
    EXTRA_ARGS=(--scales 1 2)
    ;;
  *)
    echo "Unsupported dataset kind: ${DATASET_KIND}" >&2
    exit 2
    ;;
esac

echo "=== DeepH-E3 conversion shard ==="
echo "dataset=${DATASET_KIND} shard=${SHARD_INDEX}/${NUM_SHARDS} workers=${NUM_WORKERS}"
echo "input=${INPUT_DIR}"
echo "output=${OUTPUT_DIR}"

python -u scripts/convert_openmx_to_deeph_e3.py \
  --input-dir "${INPUT_DIR}" \
  --output-dir "${OUTPUT_DIR}" \
  --dataset-kind "${DATASET_KIND}" \
  --num-shards "${NUM_SHARDS}" \
  --shard-index "${SHARD_INDEX}" \
  --num-workers "${NUM_WORKERS}" \
  --skip-existing \
  "${EXTRA_ARGS[@]}"
