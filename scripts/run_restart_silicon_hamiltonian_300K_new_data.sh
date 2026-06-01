#!/bin/bash

set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "Usage: $0 <seed> <train_on_irrep_parts:true|false>" >&2
  echo "Examples:" >&2
  echo "  $0 42 true" >&2
  echo "  $0 45 false" >&2
  exit 1
fi

SEED="$1"
TRAIN_ON_IRREP_PARTS="$2"

if ! [[ "${SEED}" =~ ^[0-9]+$ ]]; then
  echo "Seed must be an integer, got: ${SEED}" >&2
  exit 1
fi

case "${TRAIN_ON_IRREP_PARTS}" in
  true|false) ;;
  *)
    echo "train_on_irrep_parts must be 'true' or 'false', got: ${TRAIN_ON_IRREP_PARTS}" >&2
    exit 1
    ;;
esac

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

source mandala-venv/bin/activate

export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mpl-${USER}}"
mkdir -p "${MPLCONFIGDIR}"

CHECKPOINT_DIR="/bigdata/casus/wdm/hamiltonian_learning/models/checkpoints/silicon_hamiltonian_300K_new_data_restart_sage_sweep_73"
RESUME_CHECKPOINT="${RESUME_CHECKPOINT:-/bigdata/casus/wdm/hamiltonian_learning/models/checkpoints/silicon_hamiltonian_300K_new_data_restart_sage_sweep_73/silicon_hamiltonian_300K_new_data_restart_sage_sweep_73_lr0p001_seed42/best_model.pt}"
WANDB_PROJECT="mandala-silicon-hamiltonian-rosi"
LR="0.001"
IRREP_SUFFIX="plain"
if [[ "${TRAIN_ON_IRREP_PARTS}" == "true" ]]; then
  IRREP_SUFFIX="irrep"
fi
RUN_NAME="silicon_hamiltonian_300K_new_data_restart_sage_sweep_73_lr${LR//./p}_seed${SEED}_${IRREP_SUFFIX}"
DATA_PATH="/bigdata/casus/wdm/hamiltonian_learning/data/silicon_very_big_new"
SNAPSHOT_CACHE_DIR="/bigdata/casus/wdm/hamiltonian_learning/data/silicon_very_big_new/snapshot_cache"

if [[ ! -f "${RESUME_CHECKPOINT}" ]]; then
  echo "Resume checkpoint not found: ${RESUME_CHECKPOINT}" >&2
  echo "Set RESUME_CHECKPOINT to the correct latest_checkpoint.pt path before launching." >&2
  exit 1
fi

CMD=(
  python -u scripts/wandb_run.py
  --checkpoint-dir "${CHECKPOINT_DIR}"
  --resume-from-checkpoint "${RESUME_CHECKPOINT}"
  --fork-run true
  --resume-mode latest
  --wandb-mode online
  --wandb-project "${WANDB_PROJECT}"
  --run-name "${RUN_NAME}"
  --accumulate-grad-batches 1
  --activation-gate softplus
  --activation-odd-gate tanh
  --activation-odd-scalar tanh
  --activation-scalar leakyrelu
  --adaptive-log-interval True
  --apply-cutoff-to-targets True
  --benchmark False
  --convention e3nn
  --cutoff-radius 8
  --data-path "${DATA_PATH}"
  --dataset-device cpu
  --dataset-kind silicon
  --device cuda
  --dropout 0
  --e3layernorm False
  --e3mlp-film-hidden-dim 128
  --e3mlp-norm-eps 1e-08
  --e3mlp-output-scale 1
  --e3mlp-pre-norm False
  --e3mlp-residual-scale 1
  --e3mlp-variant film
  --e3mlp-weight-init-scale 1
  --edge-encoder-style rich
  --edge-encoder-use-sh-tensor-square True
  --edge-type-emb-dim 32
  --edge-update-node-combine concat
  --edge-update-residual False
  --emb-use-odd-features True
  --enable-energy True
  --enable-forces False
  --enable-num-electrons True
  --enable-stress False
  --generate-video True
  --gpus 1
  --grad-clip-val 2
  --head-diag-output-scale 2
  --head-e3mlp-layers 2
  --head-log-scale-mlp-n-layers 1
  --head-offdiag-output-scale 1
  --head-use-mlp-log-scale False
  --head-use-node-embeddings-for-self-edges True
  --head-use-tensor-square False
  --hidden-base-dim 32
  --hidden-irreps "128x0e+128x0o+64x1e+64x1o+32x2e+32x2o+16x3e+16x3o+16x4e"
  --init-weights-factor 1
  --internal-e3mlp-layers 1
  --l-max 4
  --l1-reg-coef 0
  --l2-reg-coef 0
  --log-activation-mag False
  --log-artifacts True
  --log-data True
  --log-forward False
  --log-hamiltonian-irrep-contrib-metrics True
  --log-hamiltonian-pair-contrib-metrics True
  --log-interval 10
  --log-model True
  --log-partial-gt-observables True
  --log-per-irrep-images False
  --log-per-irrep-metrics True
  --log-per-pair-loss-metrics True
  --loss-coef-forces 0
  --loss-coef-observables 0
  --loss-coef-stress 0
  --lr-scheduler-factor 0.5
  --lr-scheduler-min-lr 1e-08
  --lr-scheduler-patience 60
  --lr-scheduler-target val/hamiltonian_mae
  --matrix-targets "[\"hamiltonian\"]"
  --max-epochs 5000
  --max-temp 300
  --max-wall-clock-hours 47
  --min-temp 300
  --n-radial 128
  --neck-depth 2
  --node-update-attention-heads 4
  --node-update-attention-scalar-dim 128
  --node-update-message-agg attention
  --node-update-residual False
  --nonlin-kind normact
  --norm-kind component
  --num-layers-gnn 2
  --num-train 80
  --num-val 20
  --num-workers 3
  --precision 32-true
  --precompute-edge-features True
  --print-per-irrep-metrics False
  --radial-embedding-scale none
  --radial-layers "[128, 128]"
  --require-exact-edge-match True
  --rescale-density-to-num-electrons False
  --revert-decay-patience 4
  --revert-decay-rate 0.8
  --revert-monitor val/hamiltonian_mae
  --revert-on-spike True
  --revert-spike-factor 2
  --s2act-res 128
  --separate-shifted-self True
  --snapshot-cache-dir "${SNAPSHOT_CACHE_DIR}"
  --symmetrize-hamiltonian-targets True
  --symmetrize-output False
  --temp-step 300
  --tp-type separate_weight
  --train-observables-on-gt False
  --train-on-energy False
  --train-on-forces False
  --train-on-irrep-parts "${TRAIN_ON_IRREP_PARTS}"
  --train-on-num-electrons False
  --train-on-stress False
  --use-lr-scheduler True
  --use-self-connection True
  --val-temp 300
  --lr "${LR}"
  --seed "${SEED}"
)

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  printf '%q ' "${CMD[@]}"
  printf '\n'
else
  printf '\n=== Launching %s ===\n' "${RUN_NAME}"
  "${CMD[@]}"
fi
