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

python -u scripts/wandb_run.py \
  --resume-from-wandb https://wandb.ai/b-brzoza/mandala-matrices/sweeps/xqqtet2h/runs/e9yo3gkj \
  --wandb-mode online \
  "$@"
