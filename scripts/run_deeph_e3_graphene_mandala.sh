#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 <seed>" >&2
  echo "Example: $0 42" >&2
  exit 1
fi

SEED="$1"
if ! [[ "${SEED}" =~ ^[0-9]+$ ]]; then
  echo "Seed must be an integer, got: ${SEED}" >&2
  exit 1
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-python}"
DATA_PATH="${DATA_PATH:-/bigdata/casus/wdm/hamiltonian_learning/data/DeepH-E3/Monolayer_graphene_dataset}"
SNAPSHOT_CACHE_DIR="${SNAPSHOT_CACHE_DIR:-${DATA_PATH}/mandala_snapshot_cache_cutoff6p35_e3nn}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-/bigdata/casus/wdm/hamiltonian_learning/models/checkpoints/DeepH_E3_graphene_mandala}"
ENVELOPE_PATH="configs/envelopes/deeph_e3_graphene_slater_exp_quad_soft_wall.json"
WANDB_PROJECT="mandala-DeepH-E3-graphene"
RUN_NAME="deeph-e3-graphene-mandala-seed${SEED}"

HIDDEN_IRREPS="64x0e+16x0o+8x1e+32x1o+16x2e+4x2o+8x3o+2x3e+8x4e"

for path in "${DATA_PATH}" "${ENVELOPE_PATH}"; do
  if [[ ! -e "${path}" ]]; then
    echo "Required path does not exist: ${path}" >&2
    exit 1
  fi
done

CMD=(
  "${PYTHON_BIN}"
  -u
  scripts/wandb_run.py
  --wandb-project "${WANDB_PROJECT}"
  --wandb-group "deeph_e3_graphene_350_50_50"
  --wandb-tags "[deeph_e3, graphene, comparable_size, envelope, fresh_initialization]"
  --checkpoint-dir "${CHECKPOINT_DIR}"
  --run-name "${RUN_NAME}"
  --wandb-mode online
  --generate-video false
  --log-artifacts true
  --dataset-kind deeph_e3
  --data-path "${DATA_PATH}"
  --snapshot-cache-dir "${SNAPSHOT_CACHE_DIR}"
  --dataset-device cuda
  --num-workers 3
  --num-train 350
  --num-val 50
  --num-test 50
  --data-split-seed 42
  --allow-incomplete-dataset false
  --evaluate-test-after-fit true
  --max-wall-clock-hours 47.5
  --max-epochs 3000
  --lr 0.008
  --precision 32-true
  --device cuda
  --gpus 1
  --benchmark false
  --convention e3nn
  --graph-source target_edges
  --cutoff-radius 6.35
  --l-max 4
  --hidden-base-dim 64
  --hidden-irreps "${HIDDEN_IRREPS}"
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
  --node-update-attention-scalar-dim 64
  --node-update-attention-heads 4
  --node-update-residual false
  --neck-depth 1
  --internal-e3mlp-layers 1
  --head-e3mlp-layers 1
  --head-pair-mode split
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
  --grad-clip-val 2.0
  --accumulate-grad-batches 1
  --use-lr-scheduler true
  --lr-scheduler-factor 0.5
  --lr-scheduler-patience 60
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
  --enable-energy false
  --enable-num-electrons false
  --enable-forces false
  --enable-stress false
  --train-on-energy false
  --train-on-num-electrons false
  --train-on-forces false
  --train-on-stress false
  --train-observables-on-gt false
  --log-partial-gt-observables false
  --loss-coef-observables 0.0
  --loss-coef-forces 0.0
  --loss-coef-stress 0.0
  --spectral-loss-enabled false
  --train-on-spectral false
  --spectral-loss-coef 0.0
  --adaptive-log-interval true
  --log-interval 10
  --log-data false
  --log-model false
  --log-forward false
  --log-per-irrep-metrics true
  --log-per-pair-loss-metrics true
  --log-hamiltonian-irrep-contrib-metrics true
  --log-hamiltonian-pair-contrib-metrics true
  --print-per-irrep-metrics false
  --log-per-irrep-images false
  --seed "${SEED}"
)

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  printf '%q ' "${CMD[@]}"
  printf '\n'
else
  printf '\n=== Launching %s ===\n' "${RUN_NAME}"
  printf 'Dataset split: train=350, val=50, test=50 (split seed=42)\n'
  printf 'Architecture: parameter count is reported by Mandala at startup; hidden irreps=%s\n' "${HIDDEN_IRREPS}"
  printf 'Envelope: %s\n' "${ENVELOPE_PATH}"
  "${CMD[@]}"
fi
