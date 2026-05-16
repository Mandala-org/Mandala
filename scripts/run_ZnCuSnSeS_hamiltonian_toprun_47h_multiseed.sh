#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "Usage: $0 <seed> [scale1|scales1234]" >&2
  echo "   or: DRY_RUN=1 $0 <seed> [scale1|scales1234]" >&2
  echo "Examples:" >&2
  echo "  $0 43" >&2
  echo "  $0 101 scales1234" >&2
  echo "  DRY_RUN=1 $0 43" >&2
  exit 1
fi

SEED="$1"
SCALE_MODE="${2:-scale1}"

if ! [[ "${SEED}" =~ ^[0-9]+$ ]]; then
  echo "Seed must be an integer, got: ${SEED}" >&2
  exit 1
fi

if [[ "${SCALE_MODE}" != "scale1" && "${SCALE_MODE}" != "scales1234" ]]; then
  echo "Second argument must be 'scale1' or 'scales1234', got: ${SCALE_MODE}" >&2
  exit 1
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-python}"
CHECKPOINT_DIR="checkpoints/ZnCuSnSeS_hamiltonian_toprun_47h"
RESUME_CHECKPOINT="/data/home2/brzoza73/casus/mandala/checkpoints/ZnCuSnSeS_hamiltonian_narrow/polar-sweep-35/latest_checkpoint.pt"
WANDB_PROJECT="mandala-ZnCuSnSeS-hamiltonian-rosi"
DATA_PATH="/bigdata/casus/wdm/hamiltonian_learning/data/ZnCuSnSeS"
SNAPSHOT_CACHE_DIR="/bigdata/casus/wdm/hamiltonian_learning/data/ZnCuSnSeS/snapshot_cache"

if [[ "${SCALE_MODE}" == "scale1" ]]; then
  RUN_NAME="zncusnses-toprun47h-seed${SEED}-scale1"
  SCALES="[1]"
else
  RUN_NAME="zncusnses-toprun47h-seed${SEED}-scales1234"
  SCALES="[1, 2, 3, 4]"
fi

CMD=(
  "${PYTHON_BIN}"
  scripts/wandb_run.py
  --wandb-project "${WANDB_PROJECT}"
  --checkpoint-dir "${CHECKPOINT_DIR}"
  --resume-from-checkpoint "${RESUME_CHECKPOINT}"
  --fork-run true
  --resume-mode latest
  --wandb-mode online
  --generate-video false
  --log-artifacts true
  --dataset-kind ZnCuSnSeS
  --data-path "${DATA_PATH}"
  --snapshot-cache-dir "${SNAPSHOT_CACHE_DIR}"
  --dataset-device cpu
  --num-workers 3
  --num-train-per-scale 80
  --num-val-per-scale 20
  --max-wall-clock-hours 47
  --max-epochs 10000
  --lr 0.010803383053097332
  --precision 32-true
  --convention e3nn
  --use-lr-scheduler true
  --cutoff-radius 11.0
  --l-max 6
  --hidden-base-dim 64
  --hidden-irreps "128x0e+16x0o+8x1e+64x1o+24x2e+8x2o+8x3e+24x3o+16x4e+4x4o+4x5e+12x5o+8x6e"
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
  --head-use-mlp-log-scale false
  --neck-depth 2
  --internal-e3mlp-layers 1
  --head-e3mlp-layers 1
  --head-use-node-embeddings-for-self-edges true
  --separate-shifted-self true
  --head-use-tensor-square false
  --head-diag-output-scale 1.0
  --head-offdiag-output-scale 1.0
  --head-log-scale-mlp-n-layers 1
  --e3mlp-variant film
  --internal-e3mlp-variant resnormact
  --head-e3mlp-variant normact
  --e3mlp-output-scale 1.0
  --e3mlp-weight-init-scale 1.0
  --e3mlp-residual-scale 0.25
  --e3mlp-pre-norm false
  --e3mlp-norm-eps 1e-8
  --e3mlp-film-hidden-dim 64
  --activation-odd-scalar tanh
  --activation-odd-gate tanh
  --nonlin-kind normact
  --activation-scalar leakyrelu
  --activation-gate softplus
  --s2act-res 128
  --norm-kind component
  --grad-clip-val 2.0
  --accumulate-grad-batches 1
  --dropout 0.0
  --l1-reg-coef 0.0
  --l2-reg-coef 0.0
  --n-radial 128
  --radial-layers "[128, 128, 128]"
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
  --matrix-targets "[\"hamiltonian\"]"
  --enable-energy true
  --enable-num-electrons true
  --enable-forces false
  --train-on-energy false
  --train-on-num-electrons false
  --train-on-forces false
  --train-observables-on-gt false
  --log-partial-gt-observables true
  --loss-coef-observables 0.0
  --loss-coef-forces 0.0
  --symmetrize-output false
  --symmetrize-hamiltonian-targets true
  --rescale-density-to-num-electrons false
  --precompute-edge-features true
  --radial-embedding-scale none
  --apply-cutoff-to-targets true
  --require-exact-edge-match true
  --revert-on-spike true
  --revert-monitor val/loss_total
  --revert-decay-patience 4
  --revert-decay-rate 0.5
  --revert-spike-factor 2.0
  --run-name "${RUN_NAME}"
  --seed "${SEED}"
  --scales "${SCALES}"
)

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  printf '%q ' "${CMD[@]}"
  printf '\n'
else
  printf '\n=== Launching %s ===\n' "${RUN_NAME}"
  "${CMD[@]}"
fi
