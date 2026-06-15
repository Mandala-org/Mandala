#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

source "${HOME}/casus/mandala/mandala-venv/bin/activate"

cd "${ROOT_DIR}"

export WANDB_MODE=online

python -u scripts/wandb_run.py \
  --dataset-kind ZnCuSnSeS \
  --data-path /bigdata/casus/wdm/hamiltonian_learning/data/ZnCuSnSeS \
  --snapshot-cache-dir /bigdata/casus/wdm/hamiltonian_learning/data/ZnCuSnSeS/snapshot_cache \
  --checkpoint-dir /bigdata/casus/wdm/hamiltonian_learning/models/checkpoints/ZnCuSnSeS_small_sampled \
  --wandb-project mandala-ZnCuSnSeS-hdo-rosi \
  --wandb-mode online \
  --resume-mode latest \
  --seed 42 \
  --precision bf16-mixed \
  --convention e3nn \
  --dataset-device cuda \
  --num-workers 8 \
  --scales '[1]' \
  --num-train-per-scale 10 \
  --num-val-per-scale 2 \
  --lr 0.01 \
  --use-lr-scheduler true \
  --cutoff-radius 11.0 \
  --l-max 6 \
  --hidden-base-dim 64 \
  --hidden-irreps 64x0e+64x0o+32x1e+32x1o+16x2e+16x2o+16x3e+16x3o+16x4e+16x4o+8x5e+8x5o+8x6e+8x6o \
  --edge-type-emb-dim 32 \
  --emb-use-odd-features true \
  --edge-encoder-style distance \
  --edge-encoder-use-sh-tensor-square false \
  --e3layernorm false \
  --num-layers-gnn 3 \
  --tp-type separate_weight \
  --use-self-connection true \
  --edge-update-node-combine concat \
  --edge-update-residual true \
  --node-update-message-agg sum \
  --node-update-residual true \
  --head-use-mlp-log-scale false \
  --neck-depth 2 \
  --internal-e3mlp-layers 2 \
  --head-e3mlp-layers 2 \
  --head-use-node-embeddings-for-self-edges true \
  --separate-shifted-self true \
  --head-use-tensor-square false \
  --head-log-scale-mlp-n-layers 1 \
  --e3mlp-variant normact \
  --internal-e3mlp-variant normact \
  --head-e3mlp-variant normact \
  --e3mlp-pre-norm false \
  --e3mlp-norm-eps 1e-8 \
  --activation-odd-scalar tanh \
  --activation-odd-gate tanh \
  --nonlin-kind normact \
  --activation-scalar leakyrelu \
  --activation-gate softplus \
  --s2act-res 128 \
  --norm-kind component \
  --dropout 0.0 \
  --l1-reg-coef 0.0 \
  --l2-reg-coef 0.0 \
  --grad-clip-val 1 \
  --accumulate-grad-batches 1 \
  --n-radial 128 \
  --radial-layers '[128,128,128]' \
  --benchmark true \
  --adaptive-log-interval true \
  --log-interval 10 \
  --log-data true \
  --log-model true \
  --log-forward false \
  --log-per-irrep-metrics true \
  --print-per-irrep-metrics true \
  --log-per-irrep-images true \
  --log-per-pair-loss-metrics true \
  --log-hamiltonian-irrep-contrib-metrics true \
  --log-hamiltonian-pair-contrib-metrics true \
  --matrix-targets hamiltonian,density,overlap \
  --enable-energy true \
  --enable-num-electrons true \
  --enable-forces false \
  --train-on-energy false \
  --train-on-num-electrons false \
  --train-on-forces false \
  --train-observables-on-gt false \
  --log-partial-gt-observables true \
  --loss-coef-observables 0.0 \
  --loss-coef-forces 0.0 \
  --symmetrize-output false \
  --apply-cutoff-to-targets true \
  --require-exact-edge-match true \
  --safety-checks true \
  --precompute-edge-features true \
  --radial-embedding-scale none \
  --revert-on-spike true \
  --revert-monitor val/loss_total \
  --revert-decay-patience 4 \
  --revert-decay-rate 0.8 \
  --revert-spike-factor 2.0 \
  --log-train-metrics false \
  --batch-size 1
