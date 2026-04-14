#!/bin/bash

set -euo pipefail

source mandala-venv/bin/activate

export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mpl-${USER}}"
mkdir -p "${MPLCONFIGDIR}"

python -u scripts/train_silicon.py \
  --precision 32-true \
  --gpus 1 \
  --num-workers 4 \
  --wandb-project mandala-minimal-silicon-e3mlp-sweep \
  --run-name silicon_integrity_complement \
  --data-path /bigdata/casus/wdm/hamiltonian_learning/data/silicon_very_big/dataset_A \
  --snapshot-cache-dir /bigdata/casus/wdm/hamiltonian_learning/data/silicon_very_big/snapshot_cache \
  --checkpoint-dir checkpoints/silicon_integrity_complement \
  --matrix-targets density \
  --enable-energy true \
  --enable-num-electrons true \
  --train-on-energy false \
  --train-on-num-electrons false \
  --loss-coef-observables 0.0 \
  --enable-forces false \
  --train-on-forces false \
  --loss-coef-forces 0.0 \
  --symmetrize-output false \
  --hidden-irreps "32x0e+32x0o+16x1e+16x1o+8x2e+8x2o+8x3e+8x3o+8x4e" \
  --hidden-base-dim 32 \
  --l-max 4 \
  --num-layers-gnn 2 \
  --n-radial 64 \
  --cutoff-radius 7.0 \
  --edge-encoder-style mandala \
  --e3layernorm true \
  --separate-shifted-self false \
  --edge-encoder-use-sh-tensor-square false \
  --radial-embedding-scale none \
  --apply-cutoff-to-targets true \
  --require-exact-edge-match true \
  --dtype float32 \
  --device cuda \
  --grad-clip-val 1.0 \
  --accumulate-grad-batches 1 \
  --lr-scheduler-factor 0.2 \
  --lr-scheduler-patience 100 \
  --max-epochs 1000 \
  --log-interval 10 \
  --adaptive-log-interval true \
  --benchmark true \
  --log-data true \
  --log-model true \
  --log-forward false \
  --verbose-forward false \
  --log-per-irrep-metrics true \
  --print-per-irrep-metrics true \
  --log-per-irrep-images true \
  --generate-video true \
  --num-train 20 \
  --num-val 5 \
  --lr 0.0003 \
  --head-use-tensor-square false \
  --head-use-node-embeddings-for-self-edges false \
  --head-e3mlp-layers 1 \
  --internal-e3mlp-layers 0 \
  --head-e3mlp-variant gate \
  --internal-e3mlp-variant resgatemagnitudes \
  --e3mlp-pre-norm true \
  --e3mlp-output-scale 0.25 \
  --e3mlp-weight-init-scale 0.1 \
  --e3mlp-residual-scale 0.03 \
  --e3mlp-film-hidden-dim 64 \
  --init-weights-factor 0.01 \
  --head-diag-output-scale 0.75 \
  --head-offdiag-output-scale 0.03
