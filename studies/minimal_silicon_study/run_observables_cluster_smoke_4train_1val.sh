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
  --run-name silicon_observables_cluster_smoke_8train_2val_10ep \
  --data-path /bigdata/casus/wdm/hamiltonian_learning/data/silicon_very_big/dataset_A \
  --train-temps 2700 \
  --val-temp 2700 \
  --n-snapshots-per-temp 4 \
  --val-n-snapshots 1 \
  --device cuda \
  --dtype float32 \
  --training-unit ev \
  --matrix-targets hamiltonian,overlap,density \
  --enable-energy true \
  --enable-num-electrons true \
  --train-on-energy true \
  --train-on-num-electrons true \
  --loss-coef-observables 1e-6 \
  --enable-forces true \
  --train-on-forces true \
  --loss-coef-forces 1e-8 \
  --symmetrize-preds true \
  --hidden-irreps 8x0e+8x1e+8x1o+4x2e+4x2o+4x3e+4x3o+2x4e \
  --hidden-dim 32 \
  --l-max 4 \
  --num-layers 2 \
  --n-radial 16 \
  --head-e3mlp-layers 2 \
  --e3layernorm true \
  --edge-encoder-use-sh-tensor-square true \
  --head-use-tensor-square true \
  --head-use-node-embeddings-for-self-edges true \
  --radial-embedding-scale none \
  --separate-shifted-self true \
  --cutoff-radius 7.0 \
  --lr 1e-3 \
  --lr-factor 0.2 \
  --lr-patience 20 \
  --grad-clip 1.0 \
  --num-epochs 10 \
  --log-interval 1 \
  --adaptive-log-interval true \
  --benchmark true \
  --apply-cutoff-to-targets true \
  --require-exact-edge-match true \
  --train-on-irrep-parts true \
  --log-data true \
  --log-model true \
  --log-forward false \
  --verbose-forward false \
  --log-per-irrep-metrics true \
  --log-per-irrep-images true \
  --generate-video true
