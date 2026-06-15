#!/bin/bash

set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 <lr>" >&2
  echo "Examples:" >&2
  echo "  $0 8e-4" >&2
  echo "  $0 0.0005" >&2
  exit 1
fi

LR="$1"

case "${LR}" in
  8e-4|0.0008) LR_TAG="8em4" ;;
  5e-4|0.0005) LR_TAG="5em4" ;;
  3e-4|0.0003) LR_TAG="3em4" ;;
  2e-4|0.0002) LR_TAG="2em4" ;;
  *)
    echo "Unsupported lr: ${LR}. Expected one of: 8e-4, 5e-4, 3e-4, 2e-4" >&2
    exit 1
    ;;
esac

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

source mandala-venv/bin/activate

export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mpl-${USER}}"
mkdir -p "${MPLCONFIGDIR}"

CHECKPOINT_DIR="/bigdata/casus/wdm/hamiltonian_learning/models/checkpoints/silicon_perturbed_scales_hdo_compat_from_likely_sweep_22_47h5"
SOURCE_RUN_NAME="sleek-sweep-8-restart-lr2em3"
RESUME_CHECKPOINT="${RESUME_CHECKPOINT:-${CHECKPOINT_DIR}/${SOURCE_RUN_NAME}/best_model.pt}"
WANDB_PROJECT="mandala-silicon-perturbed-hdo-compat"
RUN_NAME="sleek-sweep-8-restart-lr${LR_TAG}-p120"
DATA_PATH="/bigdata/casus/wdm/hamiltonian_learning/data/perturbed_snapshots_Si"
SNAPSHOT_CACHE_DIR="/bigdata/casus/wdm/hamiltonian_learning/data/perturbed_snapshots_Si/snapshot_cache"

if [[ ! -f "${RESUME_CHECKPOINT}" ]]; then
  echo "Resume checkpoint not found: ${RESUME_CHECKPOINT}" >&2
  exit 1
fi

CMD=(
  python -u scripts/wandb_run.py
  --checkpoint-dir "${CHECKPOINT_DIR}"
  --resume-from-checkpoint "${RESUME_CHECKPOINT}"
  --compatibility true
  --resume-mode best
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
  --allow-incomplete-dataset True
  --convention e3nn
  --cutoff-radius 8
  --data-path "${DATA_PATH}"
  --dataset-device cpu
  --dataset-kind silicon_scales
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
  --generate-video False
  --gpus 1
  --grad-clip-val 1
  --head-diag-output-scale 2
  --head-e3mlp-layers 2
  --head-e3mlp-variant film
  --head-log-scale-mlp-n-layers 1
  --head-offdiag-output-scale 1
  --head-use-mlp-log-scale False
  --head-use-node-embeddings-for-self-edges True
  --head-use-tensor-square False
  --hidden-base-dim 32
  --hidden-irreps "128x0e+128x0o+64x1e+64x1o+32x2e+32x2o+16x3e+16x3o+16x4e"
  --init-weights-factor 1
  --internal-e3mlp-layers 1
  --internal-e3mlp-variant film
  --l-max 4
  --l1-reg-coef 0
  --l2-reg-coef 0
  --log-activation-mag False
  --log-artifacts True
  --log-data False
  --log-forward False
  --log-hamiltonian-irrep-contrib-metrics True
  --log-hamiltonian-pair-contrib-metrics True
  --log-interval 10
  --log-model False
  --log-partial-gt-observables True
  --log-per-irrep-images False
  --log-per-irrep-metrics True
  --log-per-pair-loss-metrics True
  --loss-coef-forces 0
  --loss-coef-observables 1e-09
  --loss-coef-stress 0
  --lr-scheduler-factor 0.5
  --lr-scheduler-min-lr 1e-08
  --lr-scheduler-patience 120
  --lr-scheduler-target val/loss_total
  --matrix-targets "[\"hamiltonian\", \"density\", \"overlap\"]"
  --max-epochs 5000
  --max-wall-clock-hours 47.5
  --n-radial 128
  --neck-depth 2
  --node-update-attention-heads 4
  --node-update-attention-scalar-dim 128
  --node-update-message-agg attention
  --node-update-residual False
  --nonlin-kind normact
  --norm-kind component
  --num-layers-gnn 2
  --num-train-per-scale 80
  --num-val-per-scale 20
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
  --revert-monitor val/loss_total
  --revert-on-spike True
  --revert-spike-factor 2
  --s2act-res 128
  --scales "[1]"
  --seed 43
  --separate-shifted-self True
  --snapshot-cache-dir "${SNAPSHOT_CACHE_DIR}"
  --symmetrize-hamiltonian-targets True
  --symmetrize-output False
  --tp-type separate_weight
  --train-observables-on-gt False
  --train-on-energy False
  --train-on-forces False
  --train-on-irrep-parts False
  --train-on-num-electrons False
  --train-on-stress False
  --use-lr-scheduler True
  --use-self-connection True
  --lr "${LR}"
)

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  printf '%q ' "${CMD[@]}"
  printf '\n'
else
  printf '\n=== Launching %s ===\n' "${RUN_NAME}"
  "${CMD[@]}"
fi
