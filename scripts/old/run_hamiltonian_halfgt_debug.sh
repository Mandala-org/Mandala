#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

source "${HOME}/casus/mandala-venv/bin/activate"

export WANDB_MODE="${WANDB_MODE:-offline}"

cd "${ROOT_DIR}"

python scripts/wandb_run.py \
  --dataset-kind silicon \
  --data-path /bigdata/casus/wdm/hamiltonian_learning/data/silicon_very_big/dataset_A \
  --checkpoint-dir /bigdata/casus/wdm/hamiltonian_learning/models/checkpoints/silicon \
  --min-temp 3000 \
  --max-temp 3000 \
  --temp-step 300 \
  --n-snapshots-per-temp 12 \
  --val-temp 3000 \
  --val-n-snapshots 2 \
  --num-train 10 \
  --num-val 2 \
  --dataset-device cpu \
  --num-workers 0 \
  --matrix-targets hamiltonian \
  --enable-energy true \
  --enable-num-electrons false \
  --train-on-energy true \
  --train-on-num-electrons false \
  --train-observables-on-gt true \
  --log-partial-gt-observables true \
  --loss-coef-observables 0.0001 \
  --enable-forces false \
  --train-on-forces false \
  --loss-coef-forces 0.0 \
  --symmetrize-output false \
  --hidden-irreps 32x0e+32x0o+16x1e+16x1o+8x2e+8x2o+8x3e+8x3o+8x4e \
  --hidden-base-dim 32 \
  --l-max 4 \
  --num-layers-gnn 3 \
  --n-radial 64 \
  --cutoff-radius 7.5 \
  --edge-encoder-style distance \
  --e3layernorm false \
  --separate-shifted-self true \
  --edge-encoder-use-sh-tensor-square false \
  --radial-embedding-scale none \
  --apply-cutoff-to-targets true \
  --require-exact-edge-match true \
  --precompute-edge-features true \
  --dtype float32 \
  --device cuda \
  --gpus 1 \
  --grad-clip-val 1 \
  --accumulate-grad-batches 1 \
  --lr-scheduler-factor 0.2 \
  --lr-scheduler-patience 100 \
  --max-epochs 500 \
  --log-interval 10 \
  --adaptive-log-interval true \
  --benchmark true \
  --log-data true \
  --log-model true \
  --log-forward true \
  --log-per-irrep-metrics true \
  --print-per-irrep-metrics true \
  --log-per-irrep-images true \
  --generate-video true \
  --wandb-project mandala-minimal-silicon-hamiltonian-energy-halfgt-rosi \
  --run-name debug_tp_mismatch_hamiltonian_halfgt
