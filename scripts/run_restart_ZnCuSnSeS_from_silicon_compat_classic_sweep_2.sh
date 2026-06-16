#!/bin/bash

set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 <lr>" >&2
  echo "Examples:" >&2
  echo "  $0 1e-3" >&2
  echo "  $0 5e-4" >&2
  exit 1
fi

LR="$1"

case "${LR}" in
  1e-3|0.001) LR_TAG="1em3" ;;
  5e-4|0.0005) LR_TAG="5em4" ;;
  2e-4|0.0002) LR_TAG="2em4" ;;
  1e-4|0.0001) LR_TAG="1em4" ;;
  *)
    echo "Unsupported lr: ${LR}. Expected one of: 1e-3, 5e-4, 2e-4, 1e-4" >&2
    exit 1
    ;;
esac

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

source mandala-venv/bin/activate

export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mpl-${USER}}"
mkdir -p "${MPLCONFIGDIR}"

CHECKPOINT_DIR="/bigdata/casus/wdm/hamiltonian_learning/models/checkpoints/ZnCuSnSeS_from_silicon_compat_11h5"
RESUME_CHECKPOINT="${RESUME_CHECKPOINT:-/bigdata/casus/wdm/hamiltonian_learning/models/checkpoints/ZnCuSnSeS_from_silicon_compat_11h5/classic-sweep-2/latest_checkpoint.pt}"
WANDB_PROJECT="mandala-ZnCuSnSeS-hamiltonian-compat-from-silicon"
RUN_NAME="classic-sweep-2-restart-lr${LR_TAG}"
DATA_PATH="/bigdata/casus/wdm/hamiltonian_learning/data/ZnCuSnSeS"
SNAPSHOT_CACHE_DIR="/bigdata/casus/wdm/hamiltonian_learning/data/ZnCuSnSeS/snapshot_cache"

if [[ ! -f "${RESUME_CHECKPOINT}" ]]; then
  echo "Resume checkpoint not found: ${RESUME_CHECKPOINT}" >&2
  exit 1
fi

CMD=(
  python -u scripts/wandb_run.py
  --accumulate-grad-batches 1
  --activation-gate softplus
  --activation-odd-gate tanh
  --activation-odd-scalar tanh
  --activation-scalar leakyrelu
  --adaptive-log-interval True
  --apply-cutoff-to-targets True
  --benchmark False
  --checkpoint-dir "${CHECKPOINT_DIR}"
  --compatibility True
  --convention e3nn
  --cutoff-radius 12
  --data-path "${DATA_PATH}"
  --dataset-device cuda
  --dataset-kind ZnCuSnSeS
  --device cuda
  --dropout 0
  --e3layernorm False
  --e3mlp-film-hidden-dim 64
  --e3mlp-norm-eps 1e-08
  --e3mlp-output-scale 1
  --e3mlp-pre-norm False
  --e3mlp-residual-scale 0.25
  --e3mlp-variant film
  --e3mlp-weight-init-scale 1
  --edge-encoder-style rich
  --edge-encoder-use-sh-tensor-square False
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
  --grad-clip-val 0.5
  --head-diag-output-scale 1
  --head-e3mlp-layers 1
  --head-e3mlp-variant film
  --head-log-scale-mlp-n-layers 1
  --head-offdiag-output-scale 1
  --head-pair-mode shared_conditioned
  --head-use-mlp-log-scale False
  --head-use-node-embeddings-for-self-edges True
  --head-use-tensor-square False
  --hidden-base-dim 64
  --hidden-irreps "128x0e+16x0o+8x1e+64x1o+24x2e+8x2o+8x3e+24x3o+16x4e+4x4o+4x5e+12x5o+8x6e"
  --internal-e3mlp-layers 1
  --internal-e3mlp-variant resnormact
  --l-max 6
  --l1-reg-coef 0
  --l2-reg-coef 0
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
  --loss-coef-forces 0
  --loss-coef-observables 0
  --loss-coef-stress 0
  --lr "${LR}"
  --lr-scheduler-factor 0.5
  --lr-scheduler-min-lr 1e-08
  --lr-scheduler-patience 60
  --lr-scheduler-target val/hamiltonian_mae
  --matrix-targets "[\"hamiltonian\"]"
  --max-epochs 3000
  --max-wall-clock-hours 47.5
  --n-radial 128
  --neck-depth 1
  --node-update-attention-heads 4
  --node-update-attention-scalar-dim 64
  --node-update-message-agg attention
  --node-update-residual False
  --nonlin-kind normact
  --norm-kind component
  --num-layers-gnn 2
  --num-train-per-scale 80
  --num-val-per-scale 20
  --num-workers 0
  --precision 32-true
  --precompute-edge-features True
  --print-per-irrep-metrics False
  --radial-embedding-scale none
  --radial-layers "[128, 128, 128]"
  --require-exact-edge-match True
  --resume-from-checkpoint "${RESUME_CHECKPOINT}"
  --resume-mode latest
  --revert-decay-patience 200
  --revert-decay-rate 0.5
  --revert-monitor val/hamiltonian_mae
  --revert-on-spike True
  --revert-spike-factor 2
  --run-name "${RUN_NAME}"
  --s2act-res 128
  --scales "[1]"
  --seed 43
  --separate-shifted-self True
  --snapshot-cache-dir "${SNAPSHOT_CACHE_DIR}"
  --symmetrize-hamiltonian-targets True
  --symmetrize-output True
  --tp-type separate_weight
  --train-observables-on-gt False
  --train-on-energy False
  --train-on-forces False
  --train-on-irrep-parts False
  --train-on-num-electrons False
  --train-on-stress False
  --use-lr-scheduler True
  --use-self-connection False
  --wandb-mode online
  --wandb-project "${WANDB_PROJECT}"
)

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  printf '%q ' "${CMD[@]}"
  printf '\n'
else
  printf '\n=== Launching %s ===\n' "${RUN_NAME}"
  "${CMD[@]}"
fi
