#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ -f "${HOME}/casus/mandala-venv/bin/activate" ]]; then
  source "${HOME}/casus/mandala-venv/bin/activate"
elif [[ -f "${ROOT_DIR}/mandala-venv/bin/activate" ]]; then
  source "${ROOT_DIR}/mandala-venv/bin/activate"
else
  echo "Could not find mandala virtualenv." >&2
  exit 1
fi

cd "${ROOT_DIR}"

CHECKPOINT="${CHECKPOINT:-/home/brzoza73/casus/mandala/checkpoints/silicon/zesty-sweep-135/best_model.pt}"
DATA_ROOT="${DATA_ROOT:-/bigdata/casus/wdm/hamiltonian_learning/data/silicon_very_big/dataset_A}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT_DIR}/eval_outputs/zesty-sweep-135_snapshot0_gt_overlap}"
SNAPSHOT_INDEX="${SNAPSHOT_INDEX:-0}"
TEMPERATURE="${1:-${TEMPERATURE:-}}"
if [[ -z "${TEMPERATURE}" ]]; then
  echo "Usage: $0 <temperature>" >&2
  echo "Example: $0 300K" >&2
  exit 1
fi

NUM_WORKERS="${NUM_WORKERS:-24}"
CHUNK_SIZE="${CHUNK_SIZE:-12}"
NUM_POINTS="${NUM_POINTS:-240}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mplconfig}"

mkdir -p "${OUTPUT_ROOT}"

snapshot_path="${DATA_ROOT}/${TEMPERATURE}/${SNAPSHOT_INDEX}"
output_dir="${OUTPUT_ROOT}/${TEMPERATURE}"
log_path="${output_dir}/run.log"

mkdir -p "${output_dir}"

if [[ ! -d "${snapshot_path}" ]]; then
  echo "Snapshot path does not exist: ${snapshot_path}" >&2
  exit 1
fi

echo "=== Starting ${TEMPERATURE} (snapshot ${SNAPSHOT_INDEX}) ==="
echo "snapshot_path=${snapshot_path}"
echo "output_dir=${output_dir}"
echo "log_path=${log_path}"

python -u scripts/evaluate_checkpoint_materials.py \
  --checkpoint "${CHECKPOINT}" \
  --mode snapshot \
  --snapshot-path "${snapshot_path}" \
  --output-dir "${output_dir}" \
  --device cpu \
  --plot-title "Silicon ${TEMPERATURE} snapshot ${SNAPSHOT_INDEX}" \
  --use-gt-overlap-for-eigs \
  --dos-energy-min -20 \
  --dos-energy-max 25 \
  --num-points "${NUM_POINTS}" \
  --chunk-size "${CHUNK_SIZE}" \
  --num-workers "${NUM_WORKERS}" \
  > "${log_path}" 2>&1

echo "=== Finished ${TEMPERATURE}; log saved to ${log_path} ==="
