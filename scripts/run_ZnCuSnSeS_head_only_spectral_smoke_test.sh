#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

source mandala-venv/bin/activate

export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mpl-${USER}}"
mkdir -p "${MPLCONFIGDIR}"

CHECKPOINT_DIR="/bigdata/casus/wdm/hamiltonian_learning/models/checkpoints/ZnCuSnSeS_head_only_spectral_smoke_test"
RESUME_CHECKPOINT="${RESUME_CHECKPOINT:-/bigdata/casus/wdm/hamiltonian_learning/models/checkpoints/ZnCuSnSeS_hamiltonian_envelope_stageA_selected_47h5/dandy-sweep-34-restart47h5/best_model.pt}"
WANDB_PROJECT="mandala-ZnCuSnSeS-head-only-spectral-smoke-test"
RUN_NAME="zncusnses-head-only-spectral-smoke-test"
DATA_PATH="/bigdata/casus/wdm/hamiltonian_learning/data/ZnCuSnSeS"
SNAPSHOT_CACHE_DIR="/bigdata/casus/wdm/hamiltonian_learning/data/ZnCuSnSeS/snapshot_cache"
ENVELOPE_PATH="eval_outputs/zncusnses_radial_fit_study/slater_soft_cutoff_envelope.json"
SPECTRAL_FERMI_CACHE_PATH="/bigdata/casus/wdm/hamiltonian_learning/data/ZnCuSnSeS/spectral_fermi_cache_cutoff11_k4x4x4_merged.pt"

if [[ ! -f "${RESUME_CHECKPOINT}" ]]; then
  echo "Resume checkpoint not found: ${RESUME_CHECKPOINT}" >&2
  exit 1
fi

if [[ ! -f "${ENVELOPE_PATH}" ]]; then
  echo "Envelope artifact not found: ${ENVELOPE_PATH}" >&2
  exit 1
fi

if [[ ! -e "${SPECTRAL_FERMI_CACHE_PATH}" ]]; then
  echo "Spectral Fermi cache not found: ${SPECTRAL_FERMI_CACHE_PATH}" >&2
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
  --generate-video false
  --log-artifacts true
  --dataset-kind ZnCuSnSeS
  --data-path "${DATA_PATH}"
  --snapshot-cache-dir "${SNAPSHOT_CACHE_DIR}"
  --dataset-device cpu
  --num-workers 1
  --scales "[1]"
  --num-train-per-scale 2
  --num-val-per-scale 1
  --max-wall-clock-hours 0.33
  --max_epochs 40
  --lr 5e-4
  --precision 32-true
  --convention e3nn
  --use-lr-scheduler true
  --lr-scheduler-factor 0.5
  --lr-scheduler-patience 20
  --lr-scheduler-min-lr 1e-8
  --lr-scheduler-target val/loss_total
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
  --grad-clip-val 1.0
  --accumulate-grad-batches 1
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
  --matrix-targets "[\"hamiltonian\"]"
  --enable-energy true
  --enable-num-electrons true
  --enable-forces false
  --train-on-energy false
  --train-on-num-electrons false
  --train-on-forces false
  --train-on-irrep-parts false
  --train-observables-on-gt false
  --log-partial-gt-observables true
  --loss-coef-observables 0.0
  --loss-coef-forces 0.0
  --symmetrize-output false
  --symmetrize-hamiltonian-targets true
  --precompute-edge-features true
  --radial-embedding-scale none
  --apply-cutoff-to-targets true
  --require-exact-edge-match true
  --allow-incomplete-dataset false
  --hamiltonian-envelope-path "${ENVELOPE_PATH}"
  --hamiltonian-envelope-mode multiply_prediction
  --loss-weighting-mode off
  --spectral-loss-enabled true
  --spectral-loss-coef 0.05
  --spectral-loss-kmesh 2x2x2
  --spectral-loss-window-ev 10.0
  --spectral-loss-taper-ev 2.0
  --spectral-loss-huber-delta-ev 0.1
  --spectral-loss-overlap-psd-cleanup false
  --spectral-loss-overlap-jitter true
  --freeze-backbone-train-heads-only true
  --revert-on-spike true
  --revert-monitor val/loss_total
  --revert-decay-patience 4
  --revert-decay-rate 0.5
  --revert-spike-factor 2.0
  --run-name "${RUN_NAME}"
  --seed 43
)
CMD+=(--spectral-fermi-cache-path "${SPECTRAL_FERMI_CACHE_PATH}")

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  printf '%q ' "${CMD[@]}"
  printf '\n'
else
  printf '\n=== Launching %s ===\n' "${RUN_NAME}"
  "${CMD[@]}"
fi
