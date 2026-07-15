#!/usr/bin/env bash

set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "Usage: $0 <full|half-gt> <3e-3|1e-2>" >&2
  exit 1
fi

ENERGY_MODE="$1"
OBS_COEF="$2"

case "${ENERGY_MODE}" in
  full) TRAIN_ON_GT="false" ;;
  half-gt) TRAIN_ON_GT="true" ;;
  *) echo "Unsupported energy mode: ${ENERGY_MODE}" >&2; exit 1 ;;
esac
case "${OBS_COEF}" in
  3e-3|1e-2) ;;
  *) echo "Unsupported observable coefficient: ${OBS_COEF}" >&2; exit 1 ;;
esac

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"
source mandala-venv/bin/activate

export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mpl-${USER}}"
mkdir -p "${MPLCONFIGDIR}"

SOURCE_RUN="leafy-sweep-15"
SOURCE_ROOT="/bigdata/casus/wdm/hamiltonian_learning/models/checkpoints/silicon_perturbed_hdo_observables_envelope_finetune_11h9"
RESUME_CHECKPOINT="${SOURCE_ROOT}/${SOURCE_RUN}/best_model.pt"
CHECKPOINT_DIR="/bigdata/casus/wdm/hamiltonian_learning/models/checkpoints/silicon_hdo_energy_stage2_47h25"
WANDB_PROJECT="mandala-silicon-hdo-energy-stage2-47h25"
DATA_PATH="/bigdata/casus/wdm/hamiltonian_learning/data/perturbed_snapshots_Si"
SNAPSHOT_CACHE_DIR="${DATA_PATH}/snapshot_cache"
COEF_TAG="${OBS_COEF//-/m}"
MODE_TAG="${ENERGY_MODE//-/_}"
RUN_NAME="${SOURCE_RUN}-${MODE_TAG}-coef${COEF_TAG}-lr3em4-47h25"

if [[ "${DRY_RUN:-0}" != "1" && ! -f "${RESUME_CHECKPOINT}" ]]; then
  echo "Resume checkpoint not found: ${RESUME_CHECKPOINT}" >&2
  exit 1
fi

CMD=(
  python -u scripts/wandb_run.py
  --wandb-project "${WANDB_PROJECT}"
  --checkpoint-dir "${CHECKPOINT_DIR}"
  --resume-from-checkpoint "${RESUME_CHECKPOINT}"
  --compatibility true
  --resume-mode best
  --wandb-mode online
  --run-name "${RUN_NAME}"
  --generate-video false
  --log-artifacts true
  --dataset-kind silicon_scales
  --data-path "${DATA_PATH}"
  --snapshot-cache-dir "${SNAPSHOT_CACHE_DIR}"
  --dataset-device cpu
  --num-workers 3
  --scales '[1]'
  --num-train-per-scale 80
  --num-val-per-scale 20
  --data-split-seed 42
  --max-wall-clock-hours 47.25
  --lr 3e-4
  --max-epochs 5000
  --grad-clip-val 1
  --accumulate-grad-batches 1
  --use-lr-scheduler true
  --lr-scheduler-factor 0.5
  --lr-scheduler-patience 120
  --lr-scheduler-min-lr 1e-8
  --lr-scheduler-target val/loss_total
  --checkpoint-monitor val/energy_mae
  --precision 32-true
  --convention e3nn
  --cutoff-radius 8
  --hidden-base-dim 32
  --hidden-irreps "128x0e+128x0o+64x1e+64x1o+32x2e+32x2o+16x3e+16x3o+16x4e"
  --l-max 4
  --edge-type-emb-dim 32
  --emb-use-odd-features true
  --edge-encoder-style rich
  --edge-encoder-use-sh-tensor-square true
  --e3layernorm false
  --num-layers-gnn 2
  --tp-type separate_weight
  --use-self-connection true
  --edge-update-node-combine concat
  --edge-update-residual false
  --node-update-message-agg attention
  --node-update-attention-scalar-dim 128
  --node-update-attention-heads 4
  --node-update-residual false
  --e3mlp-variant film
  --internal-e3mlp-variant film
  --head-e3mlp-variant film
  --neck-depth 2
  --internal-e3mlp-layers 1
  --head-e3mlp-layers 2
  --head-use-node-embeddings-for-self-edges true
  --separate-shifted-self true
  --head-use-tensor-square false
  --head-use-mlp-log-scale false
  --head-log-scale-mlp-n-layers 1
  --head-diag-output-scale 2
  --head-offdiag-output-scale 1
  --e3mlp-pre-norm false
  --e3mlp-norm-eps 1e-8
  --e3mlp-film-hidden-dim 128
  --e3mlp-output-scale 1
  --e3mlp-weight-init-scale 1
  --e3mlp-residual-scale 1
  --activation-odd-scalar tanh
  --activation-odd-gate tanh
  --nonlin-kind normact
  --activation-scalar leakyrelu
  --activation-gate softplus
  --s2act-res 128
  --norm-kind component
  --dropout 0
  --l1-reg-coef 0
  --l2-reg-coef 0
  --init-weights-factor 1
  --n-radial 128
  --radial-layers '[128, 128]'
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
  --log-per-pair-loss-metrics true
  --log-hamiltonian-irrep-contrib-metrics true
  --log-hamiltonian-pair-contrib-metrics true
  --log-partial-gt-observables true
  --matrix-targets '["hamiltonian", "density", "overlap"]'
  --enable-energy true
  --enable-num-electrons true
  --enable-forces false
  --enable-stress false
  --train-on-energy true
  --train-on-num-electrons false
  --train-on-forces false
  --train-on-stress false
  --train-on-irrep-parts false
  --train-observables-on-gt "${TRAIN_ON_GT}"
  --loss-coef-observables "${OBS_COEF}"
  --loss-coef-forces 0
  --loss-coef-stress 0
  --loss-l1-fraction 1
  --symmetrize-output false
  --symmetrize-hamiltonian-targets true
  --rescale-density-to-num-electrons true
  --precompute-edge-features true
  --radial-embedding-scale none
  --apply-cutoff-to-targets true
  --require-exact-edge-match true
  --allow-incomplete-dataset true
  --hamiltonian-envelope-mode off
  --loss-weighting-mode off
  --revert-on-spike true
  --revert-monitor val/loss_total
  --revert-decay-patience 4
  --revert-decay-rate 0.8
  --revert-spike-factor 2
  --seed 43
)

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  printf '%q ' "${CMD[@]}"
  printf '\n'
else
  printf '\n=== Launching %s ===\n' "${RUN_NAME}"
  printf '=== Energy mode: %s; observable coefficient: %s; electron-count training: off ===\n' \
    "${ENERGY_MODE}" "${OBS_COEF}"
  "${CMD[@]}"
fi
