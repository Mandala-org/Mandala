#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 3 ]]; then
  echo "Usage: $0 <wandb_run_id> [additional_epochs=1000] [wandb_project=mandala-minimal-silicon-study]" >&2
  exit 1
fi

RUN_ID="$1"
ADDITIONAL_EPOCHS="${2:-1000}"
WANDB_PROJECT="${3:-mandala-minimal-silicon-study}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

if [[ -f "${REPO_ROOT}/mandala-venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "${REPO_ROOT}/mandala-venv/bin/activate"
fi

srun --ntasks=1 --unbuffered python -u studies/minimal_silicon_study/train_silicon_minimal.py \
  --resume-from-run-id "${RUN_ID}" \
  --wandb-project "${WANDB_PROJECT}" \
  --num-epochs "${ADDITIONAL_EPOCHS}"
