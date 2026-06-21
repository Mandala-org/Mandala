#!/bin/bash

set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 <source-run-name>" >&2
  echo "Selected source runs:" >&2
  echo "  eternal-sweep-41" >&2
  echo "  hopeful-sweep-44" >&2
  echo "  dandy-sweep-34" >&2
  echo "  lively-sweep-33" >&2
  echo "  balmy-sweep-38" >&2
  echo "  astral-sweep-39" >&2
  echo "  icy-sweep-47" >&2
  echo "  dauntless-sweep-4" >&2
  exit 1
fi

SOURCE_RUN_NAME="$1"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

source mandala-venv/bin/activate

export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mpl-${USER}}"
mkdir -p "${MPLCONFIGDIR}"

SOURCE_CHECKPOINT_DIR="/bigdata/casus/wdm/hamiltonian_learning/models/checkpoints/ZnCuSnSeS_hamiltonian_envelope_stageA_2x_11h5"
CHECKPOINT_DIR="/bigdata/casus/wdm/hamiltonian_learning/models/checkpoints/ZnCuSnSeS_hamiltonian_envelope_stageA_selected_47h5"
WANDB_PROJECT="mandala-ZnCuSnSeS-hamiltonian-envelope-stageA-selected-47h5"
DATA_PATH="/bigdata/casus/wdm/hamiltonian_learning/data/ZnCuSnSeS"
SNAPSHOT_CACHE_DIR="/bigdata/casus/wdm/hamiltonian_learning/data/ZnCuSnSeS/snapshot_cache"
ENVELOPE_PATH="eval_outputs/zncusnses_radial_fit_study/slater_soft_cutoff_envelope.json"
BASE_LR="0.010803383053097332"

case "${SOURCE_RUN_NAME}" in
  eternal-sweep-41)
    SEED=43
    HAMILTONIAN_ENVELOPE_MODE="multiply_prediction"
    PAIR_CONDITIONED_RADIAL_MLP="True"
    PAIR_DISTANCE_NORMALIZATION="off"
    ;;
  hopeful-sweep-44)
    SEED=46
    HAMILTONIAN_ENVELOPE_MODE="multiply_prediction"
    PAIR_CONDITIONED_RADIAL_MLP="True"
    PAIR_DISTANCE_NORMALIZATION="off"
    ;;
  dandy-sweep-34)
    SEED=44
    HAMILTONIAN_ENVELOPE_MODE="multiply_prediction"
    PAIR_CONDITIONED_RADIAL_MLP="False"
    PAIR_DISTANCE_NORMALIZATION="off"
    ;;
  lively-sweep-33)
    SEED=43
    HAMILTONIAN_ENVELOPE_MODE="multiply_prediction"
    PAIR_CONDITIONED_RADIAL_MLP="False"
    PAIR_DISTANCE_NORMALIZATION="off"
    ;;
  balmy-sweep-38)
    SEED=44
    HAMILTONIAN_ENVELOPE_MODE="multiply_prediction"
    PAIR_CONDITIONED_RADIAL_MLP="False"
    PAIR_DISTANCE_NORMALIZATION="pair_r0"
    ;;
  astral-sweep-39)
    SEED=45
    HAMILTONIAN_ENVELOPE_MODE="multiply_prediction"
    PAIR_CONDITIONED_RADIAL_MLP="False"
    PAIR_DISTANCE_NORMALIZATION="pair_r0"
    ;;
  icy-sweep-47)
    SEED=45
    HAMILTONIAN_ENVELOPE_MODE="multiply_prediction"
    PAIR_CONDITIONED_RADIAL_MLP="True"
    PAIR_DISTANCE_NORMALIZATION="pair_r0"
    ;;
  dauntless-sweep-4)
    SEED=46
    HAMILTONIAN_ENVELOPE_MODE="off"
    PAIR_CONDITIONED_RADIAL_MLP="False"
    PAIR_DISTANCE_NORMALIZATION="off"
    ;;
  *)
    echo "Unsupported source run: ${SOURCE_RUN_NAME}" >&2
    exit 1
    ;;
esac

RESUME_CHECKPOINT="${RESUME_CHECKPOINT:-${SOURCE_CHECKPOINT_DIR}/${SOURCE_RUN_NAME}/latest_checkpoint.pt}"
RUN_NAME="${SOURCE_RUN_NAME}-restart47h5"

if [[ ! -f "${RESUME_CHECKPOINT}" ]]; then
  echo "Resume checkpoint not found: ${RESUME_CHECKPOINT}" >&2
  exit 1
fi

if [[ ! -f "${ENVELOPE_PATH}" ]]; then
  echo "Envelope artifact not found: ${ENVELOPE_PATH}" >&2
  exit 1
fi

CMD=(
  python -u scripts/wandb_run.py
  --wandb-project "${WANDB_PROJECT}"
  --checkpoint-dir "${CHECKPOINT_DIR}"
  --resume-from-checkpoint "${RESUME_CHECKPOINT}"
  --resume-mode latest
  --wandb-mode online
  --generate-video false
  --log-artifacts true
  --dataset-kind ZnCuSnSeS
  --data-path "${DATA_PATH}"
  --snapshot-cache-dir "${SNAPSHOT_CACHE_DIR}"
  --dataset-device cpu
  --num-workers 3
  --scales "[1]"
  --num-train-per-scale 80
  --num-val-per-scale 20
  --max-wall-clock-hours 47.5
  --lr "${BASE_LR}"
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
  --e3mlp-variant film
  --internal-e3mlp-variant resnormact
  --head-e3mlp-variant normact
  --neck-depth 2
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
  --pair-conditioned-radial-mlp "${PAIR_CONDITIONED_RADIAL_MLP}"
  --pair-distance-normalization "${PAIR_DISTANCE_NORMALIZATION}"
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
  --symmetrize-output false
  --symmetrize-hamiltonian-targets true
  --rescale-density-to-num-electrons false
  --precompute-edge-features true
  --radial-embedding-scale none
  --apply-cutoff-to-targets true
  --require-exact-edge-match true
  --allow-incomplete-dataset false
  --hamiltonian-envelope-path "${ENVELOPE_PATH}"
  --hamiltonian-envelope-mode "${HAMILTONIAN_ENVELOPE_MODE}"
  --loss-weighting-mode off
  --revert-on-spike true
  --revert-monitor val/loss_total
  --revert-decay-patience 4
  --revert-decay-rate 0.5
  --revert-spike-factor 2.0
  --run-name "${RUN_NAME}"
  --seed "${SEED}"
)

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  printf '%q ' "${CMD[@]}"
  printf '\n'
else
  printf '\n=== Launching %s ===\n' "${RUN_NAME}"
  printf '=== Source checkpoint: %s ===\n' "${RESUME_CHECKPOINT}"
  printf '=== Envelope mode: %s | pair-mlp: %s | distance norm: %s ===\n' \
    "${HAMILTONIAN_ENVELOPE_MODE}" "${PAIR_CONDITIONED_RADIAL_MLP}" "${PAIR_DISTANCE_NORMALIZATION}"
  "${CMD[@]}"
fi
