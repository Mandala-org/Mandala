#!/usr/bin/env bash

set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo "Usage: $0 <learning-rate> <seed> [run-tag]" >&2
  echo "Examples:" >&2
  echo "  $0 0.001 42" >&2
  echo "  $0 0.0005 123 lr-scan-a" >&2
  exit 1
fi

LR="$1"
SEED="$2"
RUN_TAG="${3:-}"

if ! [[ "${SEED}" =~ ^[0-9]+$ ]]; then
  echo "Seed must be an integer, got: ${SEED}" >&2
  exit 1
fi

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

CHECKPOINT_DIR="checkpoints/silicon_hamiltonian_300K_new_data_restart"
RESUME_CHECKPOINT="/data/home2/brzoza73/casus/mandala/checkpoints/silicon_hamiltonian_300K_new_data/sage-sweep-73/latest_checkpoint.pt"
SWEEP_YAML="sweeps/train_silicon_hamiltonian_300K_new_data.yaml"
WANDB_PROJECT="mandala-silicon-hamiltonian-rosi"
DATA_PATH="/bigdata/casus/wdm/hamiltonian_learning/data/silicon_very_big_new"
SNAPSHOT_CACHE_DIR="/bigdata/casus/wdm/hamiltonian_learning/data/silicon_very_big_new/snapshot_cache"

LR_TAG="${LR//./p}"
DEFAULT_RUN_TAG="sage-sweep-73-lr${LR_TAG}-seed${SEED}"
if [[ -n "${RUN_TAG}" ]]; then
  RUN_TAG="${RUN_TAG}"
else
  RUN_TAG="${DEFAULT_RUN_TAG}"
fi

python -u scripts/wandb_run.py \
  --sweep-yaml "${SWEEP_YAML}" \
  --wandb-project "${WANDB_PROJECT}" \
  --wandb-mode online \
  --checkpoint-dir "${CHECKPOINT_DIR}" \
  --resume-from-checkpoint "${RESUME_CHECKPOINT}" \
  --run-name "${RUN_TAG}" \
  --dataset-kind silicon \
  --data-path "${DATA_PATH}" \
  --snapshot-cache-dir "${SNAPSHOT_CACHE_DIR}" \
  --dataset-device cpu \
  --gpus 1 \
  --num-workers 3 \
  --max-wall-clock-hours 47 \
  --min-temp 300 \
  --max-temp 300 \
  --temp-step 300 \
  --val-temp 300 \
  --num-train 80 \
  --num-val 20 \
  --generate-video true \
  --log-artifacts true \
  --max-epochs 5000 \
  --precision 32-true \
  --convention e3nn \
  --use-lr-scheduler true \
  --lr-scheduler-factor 0.5 \
  --lr-scheduler-patience 60 \
  --lr-scheduler-min-lr 1e-8 \
  --lr-scheduler-target val/hamiltonian_mae \
  --revert-on-spike true \
  --revert-monitor val/hamiltonian_mae \
  --revert-decay-patience 4 \
  --revert-decay-rate 0.8 \
  --revert-spike-factor 2.0 \
  --cutoff-radius 8.0 \
  --l-max 4 \
  --hidden-base-dim 32 \
  --edge-type-emb-dim 32 \
  --emb-use-odd-features true \
  --edge-encoder-style rich \
  --edge-encoder-use-sh-tensor-square false \
  --e3layernorm false \
  --num-layers-gnn 2 \
  --tp-type separate_weight \
  --edge-update-node-combine concat \
  --head-use-mlp-log-scale false \
  --head-use-node-embeddings-for-self-edges true \
  --separate-shifted-self true \
  --head-use-tensor-square false \
  --head-offdiag-output-scale 1.0 \
  --head-log-scale-mlp-n-layers 1 \
  --e3mlp-output-scale 1.0 \
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
  --init-weights-factor 1.0 \
  --n-radial 64 \
  --radial-layers "[64, 64]" \
  --device cuda \
  --benchmark false \
  --adaptive-log-interval true \
  --log-interval 10 \
  --log-data true \
  --log-model true \
  --log-forward false \
  --log-per-irrep-metrics true \
  --print-per-irrep-metrics false \
  --log-per-irrep-images false \
  --log-partial-gt-observables true \
  --log-per-pair-loss-metrics true \
  --log-hamiltonian-irrep-contrib-metrics true \
  --log-hamiltonian-pair-contrib-metrics true \
  --log-activation-mag false \
  --train-on-irrep-parts false \
  --matrix-targets "[\"hamiltonian\"]" \
  --enable-energy true \
  --enable-num-electrons true \
  --enable-forces false \
  --enable-stress false \
  --train-on-energy false \
  --train-on-num-electrons false \
  --train-on-forces false \
  --train-on-stress false \
  --train-observables-on-gt false \
  --loss-coef-observables 0.0 \
  --loss-coef-forces 0.0 \
  --loss-coef-stress 0.0 \
  --symmetrize-output true \
  --symmetrize-hamiltonian-targets true \
  --rescale-density-to-num-electrons false \
  --precompute-edge-features true \
  --radial-embedding-scale none \
  --apply-cutoff-to-targets true \
  --require-exact-edge-match true \
  --lr "${LR}" \
  --seed "${SEED}"
