#!/usr/bin/env bash

set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 <learning-rate>" >&2
  exit 1
fi

START_LR="$1"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

source mandala-venv/bin/activate

export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mpl-${USER}}"
mkdir -p "${MPLCONFIGDIR}"

SOURCE_RUN_NAME="dutiful-sweep-12-restart47h9-pat60"
RESUME_CHECKPOINT="/bigdata/casus/wdm/hamiltonian_learning/models/checkpoints/SiOx_hamiltonian_envelope_selected_47h9/${SOURCE_RUN_NAME}/latest_checkpoint.pt"
CHECKPOINT_DIR="/bigdata/casus/wdm/hamiltonian_learning/models/checkpoints/SiOx_hamiltonian_envelope_selected_95h8_mae"
WANDB_PROJECT="mandala-SiOx-hamiltonian-envelope-selected-95h8-mae"
DATA_PATH="/bigdata/casus/wdm/hamiltonian_learning/data/SiOx_new"
SNAPSHOT_CACHE_DIR="/bigdata/casus/wdm/hamiltonian_learning/data/SiOx_new/snapshot_cache"
ENVELOPE_PATH="eval_outputs/SiOx_0_50372000_radial_fit_study/slater_exp_quad_soft_wall_envelope.json"

if [[ ! -f "${RESUME_CHECKPOINT}" ]]; then
  echo "Resume checkpoint not found: ${RESUME_CHECKPOINT}" >&2
  exit 1
fi

if [[ ! -f "${ENVELOPE_PATH}" ]]; then
  echo "Envelope artifact not found: ${ENVELOPE_PATH}" >&2
  exit 1
fi

RUN_NAME="${SOURCE_RUN_NAME}-restart95h8-mae-lr$(printf '%s' "${START_LR}" | sed 's/\./p/g' | sed 's/-/m/g')"

CMD=(
  python -u scripts/wandb_run.py
  --wandb-project "${WANDB_PROJECT}"
  --checkpoint-dir "${CHECKPOINT_DIR}"
  --resume-from-checkpoint "${RESUME_CHECKPOINT}"
  --compatibility true
  --resume-mode latest
  --wandb-mode online
  --generate-video false
  --log-artifacts true
  --dataset-kind siox
  --data-path "${DATA_PATH}"
  --snapshot-cache-dir "${SNAPSHOT_CACHE_DIR}"
  --dataset-device cuda
  --num-workers 3
  --num-train 90
  --num-val 10
  --max-wall-clock-hours 47.9
  --lr "${START_LR}"
  --max_epochs 3000
  --grad-clip-val 2.0
  --accumulate-grad-batches 1
  --use-lr-scheduler true
  --lr-scheduler-factor 0.5
  --lr-scheduler-patience 60
  --lr-scheduler-min-lr 1e-8
  --lr-scheduler-target val/loss_total
  --precision 32-true
  --convention e3nn
  --cutoff-radius 10.0
  --l-max 4
  --hidden-base-dim 64
  --hidden-irreps "128x0e+64x0o+32x1e+32x1o+24x2e+24x2o+16x3e+16x3o+8x4e"
  --edge-type-emb-dim 32
  --emb-use-odd-features true
  --edge-encoder-style rich
  --edge-encoder-use-sh-tensor-square true
  --e3layernorm false
  --num-layers-gnn 2
  --tp-type separate_weight
  --use-self-connection false
  --edge-update-node-combine concat
  --edge-update-residual false
  --node-update-message-agg attention
  --node-update-attention-scalar-dim 64
  --node-update-attention-heads 4
  --node-update-residual false
  --e3mlp-variant film
  --internal-e3mlp-variant resnormact
  --head-e3mlp-variant normact
  --neck-depth 1
  --internal-e3mlp-layers 1
  --head-e3mlp-layers 1
  --head-use-node-embeddings-for-self-edges true
  --separate-shifted-self true
  --head-use-tensor-square false
  --head-use-mlp-log-scale false
  --head-log-scale-mlp-n-layers 1
  --head-diag-output-scale 1.0
  --head-offdiag-output-scale 1.0
  --e3mlp-pre-norm false
  --e3mlp-norm-eps 1e-8
  --e3mlp-film-hidden-dim 64
  --e3mlp-output-scale 1.0
  --e3mlp-weight-init-scale 1.0
  --e3mlp-residual-scale 0.25
  --activation-odd-scalar tanh
  --activation-odd-gate tanh
  --nonlin-kind normact
  --activation-scalar leakyrelu
  --activation-gate softplus
  --s2act-res 128
  --norm-kind component
  --dropout 0.0
  --l1-reg-coef 0.0
  --l2-reg-coef 0.0
  --n-radial 128
  --radial-layers "[128, 128, 128]"
  --pair-conditioned-radial-mlp false
  --pair-distance-normalization off
  --device cuda
  --gpus 1
  --benchmark false
  --adaptive-log-interval true
  --log-interval 10
  --log-data false
  --log-model false
  --log-forward false
  --log-per-irrep-metrics true
  --print-per-irrep-metrics false
  --log-per-irrep-images false
  --log-hamiltonian-irrep-contrib-metrics true
  --log-hamiltonian-pair-contrib-metrics true
  --log-partial-gt-observables true
  --matrix-targets "[\"hamiltonian\"]"
  --enable-energy true
  --enable-num-electrons true
  --enable-forces false
  --train-on-energy false
  --train-on-num-electrons false
  --train-on-forces false
  --train-on-irrep-parts false
  --train-observables-on-gt false
  --loss-coef-observables 0.0
  --loss-coef-forces 0.0
  --loss-l1-fraction 1.0
  --symmetrize-output false
  --symmetrize-hamiltonian-targets true
  --rescale-density-to-num-electrons false
  --precompute-edge-features true
  --radial-embedding-scale none
  --apply-cutoff-to-targets true
  --require-exact-edge-match true
  --allow-incomplete-dataset false
  --hamiltonian-envelope-path "${ENVELOPE_PATH}"
  --hamiltonian-envelope-mode multiply_prediction
  --loss-weighting-mode off
  --revert-on-spike true
  --revert-monitor val/loss_total
  --revert-decay-patience 4
  --revert-decay-rate 0.5
  --revert-spike-factor 2.0
  --run-name "${RUN_NAME}"
  --seed 42
)

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  printf '%q ' "${CMD[@]}"
  printf '\n'
else
  printf '\n=== Launching %s ===\n' "${RUN_NAME}"
  printf '=== Resume checkpoint: %s ===\n' "${RESUME_CHECKPOINT}"
  printf '=== Mode: MAE fine-tune from weights only ===\n'
  printf '=== Starting LR: %s ===\n' "${START_LR}"
  "${CMD[@]}"
fi
