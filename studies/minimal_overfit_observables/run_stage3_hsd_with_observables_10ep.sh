#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

if [[ -f "${REPO_ROOT}/mandala-venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "${REPO_ROOT}/mandala-venv/bin/activate"
fi

export WANDB_PROJECT="${WANDB_PROJECT:-mandala-minimal-overfit-observables-stages}"

RUN_NAME="${RUN_NAME:-obs_stage3_hsd_with_observables_10ep}"
DEVICE="${DEVICE:-cuda}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-studies/minimal_overfit_observables/checkpoints}"
DATA_PATH="${DATA_PATH:-data/small/H2O/original/H2O.matrix}"
INFO_PATH="${INFO_PATH:-data/small/H2O/original/H2O.info.out}"
CUTOFF_RADIUS="${CUTOFF_RADIUS:-7.0}"

python studies/minimal_overfit_observables/overfit_observables_minimal.py \
  --run-name "${RUN_NAME}" \
  --data-path "${DATA_PATH}" \
  --info-path "${INFO_PATH}" \
  --device "${DEVICE}" \
  --checkpoint-dir "${CHECKPOINT_DIR}" \
  --matrix-targets "hamiltonian,overlap,density" \
  --enable-energy true \
  --enable-num-electrons true \
  --train-on-energy true \
  --train-on-num-electrons true \
  --loss-coef-observables 1e-5 \
  --hidden-dim 32 \
  --l-max 2 \
  --num-layers 2 \
  --n-radial 16 \
  --cutoff-radius "${CUTOFF_RADIUS}" \
  --lr 0.01 \
  --num-epochs 10 \
  --log-interval 1 \
  --adaptive-log-interval \
  --separate-shifted-self \
  --log-data \
  --log-model \
  --log-per-irrep-metrics \
  --log-per-irrep-images \
  --log-activations-wandb \
  --generate-video \
  --benchmark
