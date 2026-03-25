#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

if [[ -f "${REPO_ROOT}/mandala-venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "${REPO_ROOT}/mandala-venv/bin/activate"
fi

python studies/minimal_silicon_study/train_silicon_minimal.py \
  --resume-from-checkpoint studies/minimal_silicon_study/checkpoints/autumn-sweep-23/final_model.pt \
  --run-name test_restart \
  --data-path /bigdata/casus/wdm/hamiltonian_learning/data/silicon_very_big/dataset_A \
  --snapshot-cache-dir /bigdata/casus/wdm/hamiltonian_learning/data/silicon_very_big/snapshot_cache \
  --train-temps 2700 \
  --val-temp 2700 \
  --n-snapshots-per-temp 100 \
  --val-n-snapshots 10 \
  --num-epochs 500 \
  --lr 0.0174 \
  --loss-coef-density-matrix 10 \
  --lr-patience 80 \
  --log-interval 10 \
  --adaptive-log-interval true
