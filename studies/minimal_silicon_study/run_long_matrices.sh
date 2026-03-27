#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

if [[ -f "${REPO_ROOT}/mandala-venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "${REPO_ROOT}/mandala-venv/bin/activate"
fi

srun --ntasks=1 --unbuffered python -u studies/minimal_silicon_study/train_silicon_minimal.py \
  --run-name long_matrices_1000snap_1000ep \
  --checkpoint-dir studies/minimal_silicon_study/checkpoints \
  --data-path /bigdata/casus/wdm/hamiltonian_learning/data/silicon_very_big/dataset_A \
  --snapshot-cache-dir /bigdata/casus/wdm/hamiltonian_learning/data/silicon_very_big/snapshot_cache \
  --train-temps 2700 \
  --val-temp 2700 \
  --n-snapshots-per-temp 1000 \
  --val-n-snapshots 20 \
  --device cuda \
  --dtype float32 \
  --training-unit ev \
  --matrix-targets hamiltonian,overlap,density \
  --enable-energy true \
  --enable-forces false \
  --enable-num-electrons true \
  --train-on-energy false \
  --train-on-forces false \
  --train-on-num-electrons false \
  --rescale-density-to-num-electrons true \
  --lr 0.015 \
  --lr-factor 0.2 \
  --lr-patience 8 \
  --loss-coef-density-matrix 10 \
  --num-epochs 1000 \
  --randomize-seed true \
  --log-interval 1 \
  --adaptive-log-interval false \
  --benchmark true \
  --log-data true \
  --log-model true \
  --log-per-irrep-metrics true \
  --print-per-irrep-metrics false \
  --log-per-irrep-images true \
  --generate-video true \
  --grad-clip 1 \
  --hidden-dim 32 \
  --hidden-irreps 32x0e+24x1e+24x1o+16x2e+16x2o+12x3e+12x3o+8x4e \
  --l-max 4 \
  --num-layers 2 \
  --n-radial 64 \
  --head-e3mlp-layers 2 \
  --e3layernorm true \
  --edge-encoder-use-sh-tensor-square true \
  --head-use-tensor-square true \
  --head-use-node-embeddings-for-self-edges true \
  --radial-embedding-scale none \
  --separate-shifted-self true \
  --cutoff-radius 7 \
  --symmetrize-preds true \
  --apply-cutoff-to-targets true \
  --require-exact-edge-match true
