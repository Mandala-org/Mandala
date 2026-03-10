#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

if [[ -f "${REPO_ROOT}/mandala-venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "${REPO_ROOT}/mandala-venv/bin/activate"
fi

RUN_NAME="${RUN_NAME:-eV_train_edge_encoder_tensor_square_silicon}"
DEVICE="${DEVICE:-cuda}"
DATA_PATH="${DATA_PATH:-/bigdata/casus/wdm/hamiltonian_learning/data/silicon_very_big/dataset_A}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-studies/minimal_silicon_study/checkpoints}"

python studies/minimal_silicon_study/train_silicon_minimal.py \
  --run-name "${RUN_NAME}" \
  --data-path "${DATA_PATH}" \
  --val-temp 2700 \
  --train-temps 2700 \
  --n-snapshots-per-temp 100 \
  --val-n-snapshots 10 \
  --convention "e3nn" \
  --hidden-dim 32 \
  --l-max 4 \
  --hidden-irreps "32x0e+32x0o+16x1e+16x1o+16x2e+16x2o+8x3e+8x3o+8x4e" \
  --num-layers 2 \
  --cutoff-radius 7.5 \
  --n-radial 64 \
  --edge-encoder-use-sh-tensor-square \
  --lr 0.01 \
  --num-epochs 4000 \
  --log-interval 100 \
  --adaptive-log-interval \
  --log-data \
  --log-model \
  --log-per-irrep-metrics \
  --log-per-irrep-images \
  --benchmark \
  --grad-clip 1 \
  --lr-factor 0.5 \
  --lr-patience 400 \
  --generate-video \
  --device "${DEVICE}" \
  --checkpoint-dir "${CHECKPOINT_DIR}"
