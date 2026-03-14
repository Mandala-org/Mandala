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

RUN_NAME="${RUN_NAME:-obs_stage2_hsd_no_observables_10ep}"
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
  --enable-energy false \
  --enable-num-electrons false \
  --train-on-energy false \
  --train-on-num-electrons false \
  --loss-coef-observables 0.0 \
  --hidden-irreps "32x0e+32x1e+32x1o+16x2e+16x2o+16x3e+16x3o+8x4e" \
  --num-layers 2 \
  --n-radial 16 \
  --cutoff-radius "${CUTOFF_RADIUS}" \
  --lr 0.01 \
  --num-epochs 10 \
  --log-interval 1 \
  --adaptive-log-interval \
  --log-data \
  --log-model \
  --log-per-irrep-metrics \
  --log-per-irrep-images \
  --log-activations-wandb \
  --generate-video \
  --benchmark
