#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

SCRATCH_DIR="${SCRATCH:?SCRATCH must be set}"
PGROOT="${PGROOT:-${SCRATCH_DIR}/mandala_optuna_short/postgres}"
URL_FILE="${URL_FILE:-${PGROOT}/storage_url.txt}"
STUDY_YAML="${STUDY_YAML:-${ROOT_DIR}/sweeps/train_silicon_hamiltonian_energy_halfgt_short_optuna.yaml}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${SCRATCH_DIR}/mandala_optuna_short_hamiltonian_energy/checkpoints}"
WANDB_MODE_VALUE="${WANDB_MODE_VALUE:-offline}"
AGENT_LABEL="${AGENT_LABEL:-$(hostname -s)-$$}"

if [[ ! -f "${URL_FILE}" ]]; then
  echo "Storage URL file not found: ${URL_FILE}" >&2
  echo "Start the PostgreSQL job first." >&2
  exit 1
fi

STORAGE_URL="$(cat "${URL_FILE}")"

echo "Optuna hamiltonian short agent starting"
echo "HOSTNAME=$(hostname -f 2>/dev/null || hostname)"
echo "STUDY_YAML=${STUDY_YAML}"
echo "URL_FILE=${URL_FILE}"
echo "STORAGE_URL=${STORAGE_URL}"
echo "CHECKPOINT_DIR=${CHECKPOINT_DIR}"
echo "WANDB_MODE_VALUE=${WANDB_MODE_VALUE}"
echo "AGENT_LABEL=${AGENT_LABEL}"

source ~/casus/mandala-venv/bin/activate

mkdir -p "${CHECKPOINT_DIR}" "${SCRATCH_DIR}/cache"

python -u scripts/optuna_agent.py \
  --study-yaml "${STUDY_YAML}" \
  --storage "${STORAGE_URL}" \
  --wandb-mode "${WANDB_MODE_VALUE}" \
  --checkpoint-dir "${CHECKPOINT_DIR}" \
  --agent-label "${AGENT_LABEL}"
