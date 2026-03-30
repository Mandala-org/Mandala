#!/usr/bin/env bash
set -euo pipefail

ADDITIONAL_EPOCHS="${1:-1000}"
WANDB_PROJECT="${2:-mandala-minimal-silicon-study}"
RUN_NAME_PREFIX="${3:-q3on0gqb_restart}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

if [[ -f "${REPO_ROOT}/mandala-venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "${REPO_ROOT}/mandala-venv/bin/activate"
fi

srun --ntasks=1 --unbuffered python -u studies/minimal_silicon_study/train_silicon_minimal.py \
  --resume-from-run-id q3on0gqb \
  --fresh-run true \
  --randomize-seed true \
  --wandb-project "${WANDB_PROJECT}" \
  --run-name "${RUN_NAME_PREFIX}" \
  --num-epochs "${ADDITIONAL_EPOCHS}"
