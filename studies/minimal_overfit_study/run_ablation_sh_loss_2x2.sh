#!/usr/bin/env zsh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

STAMP="$(date +%Y%m%d-%H%M%S)"
RUN_PREFIX="${RUN_PREFIX:-ablation-sh-loss-2x2-${STAMP}}"
WANDB_PROJECT_NAME="${WANDB_PROJECT_NAME:-mandala-minimal-overfit-study-ablation-sh-loss-2x2}"

# Tunables (override with env vars on HPC launcher).
DEVICE="${DEVICE:-cuda}"
DTYPE="${DTYPE:-float64}"
EPOCHS="${EPOCHS:-3000}"
LR="${LR:-1e-2}"
LOG_INTERVAL="${LOG_INTERVAL:-200}"
CUTOFF="${CUTOFF:-4.0}"
HIDDEN_DIM="${HIDDEN_DIM:-64}"
L_MAX="${L_MAX:-7}"
NUM_LAYERS="${NUM_LAYERS:-2}"
N_RADIAL="${N_RADIAL:-128}"

COMMON_ARGS=(
  --device "${DEVICE}"
  --dtype "${DTYPE}"
  --num-epochs "${EPOCHS}"
  --lr "${LR}"
  --log-interval "${LOG_INTERVAL}"
  --adaptive-log-interval
  --cutoff-radius "${CUTOFF}"
  --hidden-dim "${HIDDEN_DIM}"
  --l-max "${L_MAX}"
  --num-layers "${NUM_LAYERS}"
  --n-radial "${N_RADIAL}"
  --apply-cutoff-to-targets
  --require-exact-edge-match
  --grad-clip 1.0
)

run_case() {
  local sh_mode="$1"
  local loss_agg="$2"
  local radial_scale="$3"
  local run_name="${RUN_PREFIX}-${sh_mode}-${loss_agg}"
  local run_cmd submit_cmd

  echo "================================================================================"
  echo "Submitting: ${run_name}"
  echo "  sh_mode=${sh_mode}, radial_scale=${radial_scale}, loss_aggregation=${loss_agg}"
  echo "  wandb_project=${WANDB_PROJECT_NAME}"
  echo "================================================================================"

  run_cmd=(
    python studies/minimal_overfit_study/overfit_water_minimal.py
    --run-name "${run_name}"
    --sh-mode "${sh_mode}"
    --radial-embedding-scale "${radial_scale}"
    --loss-aggregation "${loss_agg}"
    "${COMMON_ARGS[@]}"
  )

  printf -v submit_cmd '%q ' "${run_cmd[@]}"
  submit_cmd="cd ${REPO_ROOT} && export WANDB_PROJECT=${WANDB_PROJECT_NAME} && ${submit_cmd}"

  echo "~/scripts/hpc/hpc.py run --gpu h100 --gpus 1 ${submit_cmd}"
  ~/scripts/hpc/hpc.py run --gpu h100 --gpus 1 "${submit_cmd}"
}

# 2x2: {legacy features, aligned features} x {per_key, global loss}
run_case "legacy" "per_key" "sqrt_n_radial"
run_case "legacy" "global" "sqrt_n_radial"
run_case "aligned" "per_key" "none"
run_case "aligned" "global" "none"

echo "Done. Submitted 4 independent jobs with prefix: ${RUN_PREFIX}"
echo "WandB project: ${WANDB_PROJECT_NAME}"
