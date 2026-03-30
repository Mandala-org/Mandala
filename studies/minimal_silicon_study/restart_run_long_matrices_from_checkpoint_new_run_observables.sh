#!/usr/bin/env bash
set -euo pipefail

DEFAULT_CHECKPOINT_PATH="studies/minimal_silicon_study/checkpoints/long_matrices_1000snap_1000ep/"

if [[ $# -gt 4 ]]; then
  echo "Usage: $0 [checkpoint_path=${DEFAULT_CHECKPOINT_PATH}] [additional_epochs=1000] [run_name_prefix=long_matrices_restart_observables] [loss_coef_observables=1e-9]" >&2
  exit 1
fi

CHECKPOINT_PATH="${1:-${DEFAULT_CHECKPOINT_PATH}}"
ADDITIONAL_EPOCHS="${2:-1000}"
RUN_NAME_PREFIX="${3:-long_matrices_restart_observables}"
LOSS_COEF_OBSERVABLES="${4:-1e-9}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

if [[ -f "${REPO_ROOT}/mandala-venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "${REPO_ROOT}/mandala-venv/bin/activate"
fi

srun --ntasks=1 --unbuffered python -u studies/minimal_silicon_study/train_silicon_minimal.py \
  --resume-from-checkpoint "${CHECKPOINT_PATH}" \
  --run-name "${RUN_NAME_PREFIX}" \
  --enable-energy true \
  --enable-num-electrons true \
  --train-on-energy true \
  --train-on-num-electrons true \
  --loss-coef-observables "${LOSS_COEF_OBSERVABLES}" \
  --num-epochs "${ADDITIONAL_EPOCHS}"
