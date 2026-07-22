#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 <learning-rate>" >&2
  echo "Allowed values: 0.005, 0.002, 0.001, 0.0005" >&2
  exit 1
fi

LR="$1"
case "${LR}" in
  0.005|5e-3) LR_TAG="5em3" ;;
  0.002|2e-3) LR_TAG="2em3" ;;
  0.001|1e-3) LR_TAG="1em3" ;;
  0.0005|5e-4) LR_TAG="5em4" ;;
  *)
    echo "Unsupported learning rate: ${LR}" >&2
    echo "Allowed values: 0.005, 0.002, 0.001, 0.0005" >&2
    exit 1
    ;;
esac

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-python}"
DATA_PATH="/bigdata/casus/wdm/hamiltonian_learning/data/ZnCuSnSeS_big"
SNAPSHOT_CACHE_DIR="${DATA_PATH}/snapshot_cache"
SOURCE_CHECKPOINT="/bigdata/casus/wdm/hamiltonian_learning/models/checkpoints/ZnCuSnSeS_big_3x_b200/zncusnses-big-3x-scale1-seed42/best_model.pt"
CHECKPOINT_DIR="/bigdata/casus/wdm/hamiltonian_learning/models/checkpoints/ZnCuSnSeS_big_3x_b200_restarts"
WANDB_PROJECT="mandala-ZnCuSnSeS-big-3x-b200-restarts"
RUN_NAME="zncusnses-big-3x-scale1-seed42-restart-lr${LR_TAG}-clip0p5-pat12"
ENVELOPE_PATH="eval_outputs/zncusnses_radial_fit_study/slater_soft_cutoff_envelope.json"

HIDDEN_IRREPS="384x0e+48x0o+24x1e+192x1o+72x2e+24x2o+24x3e+72x3o+48x4e+12x4o+12x5e+36x5o+24x6e"

if [[ ! -f "${SOURCE_CHECKPOINT}" ]]; then
  echo "Source checkpoint not found: ${SOURCE_CHECKPOINT}" >&2
  exit 1
fi
if [[ ! -f "${ENVELOPE_PATH}" ]]; then
  echo "Envelope artifact not found: ${ENVELOPE_PATH}" >&2
  exit 1
fi

CMD=(
  "${PYTHON_BIN}" -u scripts/wandb_run.py
  --wandb-project "${WANDB_PROJECT}"
  --wandb-group "ZnCuSnSeS_big_3x_scale1_seed42_restarts"
  --wandb-tags "[big_dataset, b200, checkpoint_restart, 3x_hidden_irreps, stability_tuning]"
  --checkpoint-dir "${CHECKPOINT_DIR}"
  --run-name "${RUN_NAME}"
  --resume-from-checkpoint "${SOURCE_CHECKPOINT}"
  --resume-mode best
  --compatibility true
  --wandb-mode online
  --generate-video false
  --log-artifacts true
  --dataset-kind ZnCuSnSeS
  --data-path "${DATA_PATH}"
  --snapshot-cache-dir "${SNAPSHOT_CACHE_DIR}"
  --dataset-device cuda
  --num-workers 3
  --scales "[1]"
  --num-train-per-scale 350
  --num-val-per-scale 50
  --num-test-per-scale 0
  --data-split-seed 42
  --allow-incomplete-dataset false
  --evaluate-test-after-fit false
  --max-wall-clock-hours 47
  --max-epochs 10000
  --lr "${LR}"
  --precision 32-true
  --device cuda
  --gpus 1
  --benchmark false
  --convention e3nn
  --cutoff-radius 11.0
  --l-max 6
  --hidden-base-dim 64
  --hidden-irreps "${HIDDEN_IRREPS}"
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
  --n-radial 128
  --radial-layers "[128, 128, 128]"
  --pair-conditioned-radial-mlp false
  --pair-distance-normalization off
  --dropout 0.0
  --l1-reg-coef 0.0
  --l2-reg-coef 0.0
  --grad-clip-val 0.5
  --accumulate-grad-batches 1
  --use-lr-scheduler true
  --lr-scheduler-factor 0.5
  --lr-scheduler-patience 12
  --lr-scheduler-min-lr 1e-8
  --lr-scheduler-target val/hamiltonian_mae
  --revert-on-spike true
  --revert-monitor val/hamiltonian_mae
  --revert-decay-patience 4
  --revert-decay-rate 0.5
  --revert-spike-factor 2.0
  --checkpoint-monitor val/hamiltonian_mae
  --matrix-targets "[\"hamiltonian\"]"
  --loss-l1-fraction 1.0
  --symmetrize-output false
  --symmetrize-hamiltonian-targets true
  --apply-cutoff-to-targets true
  --require-exact-edge-match true
  --precompute-edge-features true
  --radial-embedding-scale none
  --hamiltonian-envelope-path "${ENVELOPE_PATH}"
  --hamiltonian-envelope-mode multiply_prediction
  --loss-weighting-mode off
  --enable-energy true
  --enable-num-electrons true
  --enable-forces false
  --enable-stress false
  --train-on-energy false
  --train-on-num-electrons false
  --train-on-forces false
  --train-on-stress false
  --train-observables-on-gt true
  --log-partial-gt-observables true
  --loss-coef-observables 0.0
  --loss-coef-forces 0.0
  --loss-coef-stress 0.0
  --adaptive-log-interval true
  --log-interval 10
  --log-data false
  --log-model false
  --log-forward false
  --log-per-irrep-metrics true
  --log-hamiltonian-irrep-contrib-metrics true
  --log-hamiltonian-pair-contrib-metrics true
  --print-per-irrep-metrics false
  --log-per-irrep-images false
  --seed 42
)

printf '\n=== Launching %s ===\n' "${RUN_NAME}"
printf 'Source checkpoint: %s\n' "${SOURCE_CHECKPOINT}"
printf 'lr=%s, grad_clip=0.5, scheduler_patience=12\n' "${LR}"
printf 'Restart mode: exact weights via compatibility mode; fresh optimizer/scheduler\n'

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  printf '%q ' "${CMD[@]}"
  printf '\n'
else
  "${CMD[@]}"
fi
