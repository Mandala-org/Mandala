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
  --dataset-kind siox \
  --data-path /bigdata/casus/wdm/hamiltonian_learning/data/SiOx \
  --checkpoint-dir checkpoints/siox \
  --snapshot-cache-dir /bigdata/casus/wdm/hamiltonian_learning/data/SiOx/snapshot_cache \
  --min-temp 3000 \
  --max-temp 3000 \
  --temp-step 300 \
  --val-temp 3000 \
  --num-train 140 \
  --num-val 25 \
  --seed 42 \
  --precision 32-true \
  --resume-mode latest \
  --generate-video true \
  --log-artifacts true \
  --wandb-mode online \
  --convention e3nn \
  --use-lr-scheduler true \
  --lr 0.01 \
  --max-epochs 200 \
  --batch-size 1 \
  --cutoff-radius 7.5 \
  --l-max 4 \
  --hidden-base-dim 32 \
  --hidden-irreps 32x0e+32x0o+16x1e+16x1o+8x2e+8x2o+8x3e+8x3o+8x4e \
  --edge-type-emb-dim 32 \
  --emb-use-odd-features true \
  --edge-encoder-style distance \
  --edge-encoder-use-sh-tensor-square false \
  --e3layernorm false \
  --num-layers-gnn 2 \
  --tp-type separate_weight \
  --use-self-connection true \
  --edge-update-node-combine concat \
  --edge-update-residual true \
  --node-update-message-agg sum \
  --node-update-residual true \
  --head-use-mlp-log-scale false \
  --neck-depth 1 \
  --internal-e3mlp-layers 1 \
  --head-e3mlp-layers 2 \
  --head-use-node-embeddings-for-self-edges true \
  --separate-shifted-self true \
  --head-use-tensor-square false \
  --head-diag-output-scale 2 \
  --head-offdiag-output-scale 1 \
  --head-log-scale-mlp-n-layers 1 \
  --e3mlp-variant basic \
  --internal-e3mlp-variant normact \
  --head-e3mlp-variant film \
  --e3mlp-output-scale 1 \
  --e3mlp-weight-init-scale 0.5 \
  --e3mlp-residual-scale 1 \
  --e3mlp-pre-norm false \
  --e3mlp-norm-eps 1e-8 \
  --e3mlp-film-hidden-dim 64 \
  --activation-odd-scalar tanh \
  --activation-odd-gate tanh \
  --nonlin-kind normact \
  --activation-scalar leakyrelu \
  --activation-gate softplus \
  --s2act-res 128 \
  --norm-kind component \
  --lr-scheduler-factor 0.2 \
  --lr-scheduler-patience 20 \
  --lr-scheduler-min-lr 1e-8 \
  --lr-scheduler-target val/loss_total \
  --dropout 0.0 \
  --l1-reg-coef 0.0 \
  --l2-reg-coef 0.0 \
  --init-weights-factor 1 \
  --grad-clip-val 1 \
  --accumulate-grad-batches 1 \
  --n-radial 64 \
  --radial-layers '[64,64]' \
  --device cuda \
  --gpus 1 \
  --num-workers 8 \
  --dataset-device cuda \
  --benchmark true \
  --adaptive-log-interval true \
  --log-interval 10 \
  --log-data true \
  --log-model true \
  --log-forward false \
  --log-per-irrep-metrics true \
  --print-per-irrep-metrics true \
  --log-per-irrep-images true \
  --matrix-targets hamiltonian,density,overlap \
  --enable-energy true \
  --enable-num-electrons true \
  --enable-forces false \
  --train-on-energy false \
  --train-on-num-electrons false \
  --train-on-forces false \
  --train-observables-on-gt true \
  --log-partial-gt-observables true \
  --loss-coef-observables 0.0 \
  --loss-coef-forces 0.0 \
  --symmetrize-output false \
  --apply-cutoff-to-targets true \
  --require-exact-edge-match true \
  --precompute-edge-features true \
  --radial-embedding-scale none \
  --temp-step 300 \
  --wandb-project mandala-minimal-siox-hamiltonian-energy-halfgt-rosi
