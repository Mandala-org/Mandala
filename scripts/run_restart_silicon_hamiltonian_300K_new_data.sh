#!/usr/bin/env bash

set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo "Usage: $0 <learning-rate> <seed> [run-tag]" >&2
  echo "Examples:" >&2
  echo "  $0 0.001 42" >&2
  echo "  $0 0.0005 123 lr-scan-a" >&2
  exit 1
fi

LR="$1"
SEED="$2"
RUN_TAG="${3:-}"

if ! [[ "${SEED}" =~ ^[0-9]+$ ]]; then
  echo "Seed must be an integer, got: ${SEED}" >&2
  exit 1
fi

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

CHECKPOINT_DIR="checkpoints/silicon_hamiltonian_300K_new_data_restart"
RESUME_CHECKPOINT="/data/home2/brzoza73/casus/mandala/checkpoints/silicon_hamiltonian_300K_new_data/sage-sweep-73/latest_checkpoint.pt"
SWEEP_YAML="sweeps/train_silicon_hamiltonian_300K_new_data.yaml"
WANDB_PROJECT="mandala-silicon-hamiltonian-rosi"

LR_TAG="${LR//./p}"
DEFAULT_RUN_TAG="sage-sweep-73-lr${LR_TAG}-seed${SEED}"
if [[ -n "${RUN_TAG}" ]]; then
  RUN_TAG="${RUN_TAG}"
else
  RUN_TAG="${DEFAULT_RUN_TAG}"
fi

python -u scripts/wandb_run.py \
  --sweep-yaml "${SWEEP_YAML}" \
  --wandb-project "${WANDB_PROJECT}" \
  --wandb-mode online \
  --checkpoint-dir "${CHECKPOINT_DIR}" \
  --resume-from-checkpoint "${RESUME_CHECKPOINT}" \
  --run-name "${RUN_TAG}" \
  --num-workers 3 \
  --max-wall-clock-hours 47 \
  --lr "${LR}" \
  --seed "${SEED}"
