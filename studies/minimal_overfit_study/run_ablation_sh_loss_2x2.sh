#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd -P)"
cd "${REPO_ROOT}"

STAMP="$(date +%Y%m%d-%H%M%S)"
RUN_PREFIX="${RUN_PREFIX:-ablation-sh-loss-2x2-${STAMP}}"
WANDB_PROJECT_NAME="${WANDB_PROJECT_NAME:-mandala-minimal-overfit-study-ablation-sh-loss-2x2}"
VENV_ACTIVATE="${VENV_ACTIVATE:-${REPO_ROOT}/mandala-venv/bin/activate}"

# Tunables
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
  local train_cmd_str repo_root_q venv_q project_q

  echo "================================================================================"
  echo "Submitting: ${run_name}"
  echo "  sh_mode=${sh_mode}, radial_scale=${radial_scale}, loss_aggregation=${loss_agg}"
  echo "  wandb_project=${WANDB_PROJECT_NAME}"
  echo "  venv_activate=${VENV_ACTIVATE}"
  echo "  repo_root=${REPO_ROOT}"
  echo "================================================================================"

  local -a train_cmd=(
    python studies/minimal_overfit_study/overfit_water_minimal.py
    --run-name "${run_name}"
    --sh-mode "${sh_mode}"
    --radial-embedding-scale "${radial_scale}"
    --loss-aggregation "${loss_agg}"
    "${COMMON_ARGS[@]}"
  )
  printf -v train_cmd_str '%q ' "${train_cmd[@]}"
  printf -v repo_root_q '%q' "${REPO_ROOT}"
  printf -v venv_q '%q' "${VENV_ACTIVATE}"
  printf -v project_q '%q' "${WANDB_PROJECT_NAME}"

  ~/scripts/hpc/hpc.py run --gpu h100 --gpus 1 -- \
    bash -lc "cd ${repo_root_q} && source ${venv_q} && export WANDB_PROJECT=${project_q} && ${train_cmd_str}"
}

run_case "legacy"  "per_key" "sqrt_n_radial"
run_case "legacy"  "global"  "sqrt_n_radial"
run_case "aligned" "per_key" "none"
run_case "aligned" "global"  "none"

echo "Done. Submitted 4 independent jobs with prefix: ${RUN_PREFIX}"
echo "WandB project: ${WANDB_PROJECT_NAME}"
