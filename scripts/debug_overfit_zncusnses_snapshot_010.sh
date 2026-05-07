#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

source "${HOME}/casus/mandala/mandala-venv/bin/activate"

cd "${ROOT_DIR}"

export WANDB_MODE=online

python -u -Xfrozen_modules=off -m debugpy --listen 5678 --wait-for-client scripts/wandb_run.py \
  --dataset-kind ZnCuSnSeS_small \
  --data-path data/small/ZnCu2Sn_SeS_2_scale_1_010 \
  --checkpoint-dir checkpoints/zinc_copper_tin_selenide_sulfide_overfit_snapshot_010 \
  --snapshot-cache-dir checkpoints/zinc_copper_tin_selenide_sulfide_overfit_snapshot_010/snapshot_cache \
  --num-train 1 \
  --num-val 0 \
  --seed 42 \
  --precision 32-true \
  --generate-video true \
  --log-artifacts true \
  --wandb-mode online \
  --convention e3nn \
  --use-lr-scheduler false \
  --lr 0.01 \
  --max-epochs 1000 \
  --batch-size 1 \
  --cutoff-radius 11.0 \
  --l-max 5 \
  --hidden-base-dim 32 \
  --hidden-irreps 32x0e+32x0o+16x1e+16x1o+8x2e+8x2o+8x3e+8x3o+8x4e \
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
  --neck-depth 1 \
  --internal-e3mlp-layers 1 \
  --head-e3mlp-layers 3 \
  --head-use-node-embeddings-for-self-edges true \
  --separate-shifted-self true \
  --head-use-tensor-square false \
  --head-diag-output-scale 2 \
  --head-offdiag-output-scale 1 \
  --head-log-scale-mlp-n-layers 1 \
  --e3mlp-variant film \
  --internal-e3mlp-variant film \
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
  --lr-scheduler-patience 100 \
  --lr-scheduler-min-lr 1e-8 \
  --lr-scheduler-target train/loss_total \
  --dropout 0.0 \
  --l1-reg-coef 0.0 \
  --l2-reg-coef 0.0 \
  --init-weights-factor 1 \
  --grad-clip-val 1 \
  --accumulate-grad-batches 1 \
  --n-radial 128 \
  --radial-layers '[128,128,128]' \
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
  --train-observables-on-gt false \
  --log-partial-gt-observables true \
  --loss-coef-observables 0.0 \
  --loss-coef-forces 0.0 \
  --symmetrize-output true \
  --apply-cutoff-to-targets false \
  --require-exact-edge-match true \
  --safety-checks true \
  --revert-on-spike false \
  --precompute-edge-features true \
  --radial-embedding-scale none \
  --wandb-project mandala-zinc-copper-tin-selenide-sulfide-overfit-test \
  --run-name overfit-zinc-copper-tin-selenide-sulfide-snapshot-010-cutoff11
