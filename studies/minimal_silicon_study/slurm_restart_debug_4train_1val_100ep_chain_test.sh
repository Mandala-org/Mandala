#!/usr/bin/env bash
#SBATCH --chdir /data/home2/brzoza73/casus/mandala
#SBATCH --output slurm_%j.out
#SBATCH --error slurm_%j.out
#SBATCH --job-name silicon-restart-test
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --ntasks-per-node 1
#SBATCH --gres gpu:1
#SBATCH --time 00:10:00
#SBATCH --signal=B:USR1@300
#SBATCH --mem 264G
#SBATCH --cpus-per-task 16
#SBATCH -A casus
#SBATCH -p gpu-h100

set -euo pipefail

REPO_ROOT="/data/home2/brzoza73/casus/mandala"
cd "${REPO_ROOT}"

source "$HOME/scripts/python.profile"
if [[ -f "${REPO_ROOT}/mandala-venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "${REPO_ROOT}/mandala-venv/bin/activate"
fi

RUN_NAME="silicon_restart_debug_4train_1val_100ep_chain_test"
GROUP_NAME="silicon_restart_debug_4train_1val_100ep_chain_test"
WANDB_PROJECT_NAME="resume-tests"
CHECKPOINT_DIR="studies/minimal_silicon_study/checkpoints/${RUN_NAME}"
LATEST_CHECKPOINT="${CHECKPOINT_DIR}/latest_checkpoint.pt"
TOTAL_EPOCHS=100
MAX_CHAIN_JOBS="${MAX_CHAIN_JOBS:-10}"
CHAIN_INDEX="${CHAIN_INDEX:-1}"

export WANDB_RUN_GROUP="${GROUP_NAME}"
export WANDB_PROJECT="${WANDB_PROJECT_NAME}"

TRAIN_PID=""
INTERRUPTED_BY_SIGNAL=0
TRAIN_EXIT=0

forward_sigint() {
  echo "[SLURM] Received pre-timeout signal. Forwarding SIGINT to training process."
  INTERRUPTED_BY_SIGNAL=1
  if [[ -n "${TRAIN_PID}" ]] && kill -0 "${TRAIN_PID}" 2>/dev/null; then
    kill -INT -- -"${TRAIN_PID}" 2>/dev/null || kill -INT "${TRAIN_PID}" || true
  fi
}
trap forward_sigint USR1

COMMON_ARGS=(
  --wandb-project "${WANDB_PROJECT_NAME}"
  --data-path /bigdata/casus/wdm/hamiltonian_learning/data/silicon_very_big/dataset_A
  --snapshot-cache-dir /bigdata/casus/wdm/hamiltonian_learning/data/silicon_very_big/snapshot_cache
  --train-temps 2700
  --val-temp 2700
  --n-snapshots-per-temp 4
  --val-n-snapshots 1
  --device cuda
  --dtype float32
  --training-unit ev
  --matrix-targets hamiltonian,overlap,density
  --enable-energy true
  --enable-num-electrons true
  --enable-forces true
  --train-on-energy true
  --train-on-num-electrons true
  --train-on-forces true
  --loss-coef-observables 1e-6
  --loss-coef-density-matrix 1
  --loss-coef-forces 1e-8
  --symmetrize-preds true
  --hidden-irreps 24x0e+16x1e+16x1o+8x2e+8x2o+8x3e+8x3o+4x4e
  --hidden-dim 32
  --l-max 4
  --num-layers 2
  --n-radial 16
  --head-e3mlp-layers 2
  --e3layernorm true
  --edge-encoder-use-sh-tensor-square true
  --head-use-tensor-square true
  --head-use-node-embeddings-for-self-edges true
  --radial-embedding-scale none
  --separate-shifted-self true
  --cutoff-radius 7.0
  --lr 1e-3
  --lr-factor 0.2
  --lr-patience 20
  --grad-clip 1.0
  --log-interval 1
  --adaptive-log-interval false
  --benchmark true
  --apply-cutoff-to-targets true
  --require-exact-edge-match true
  --log-data false
  --log-model false
  --log-forward false
  --verbose-forward false
  --log-per-irrep-metrics true
  --log-per-irrep-images true
  --generate-video true
)

get_completed_epochs() {
  python - <<'PY'
from pathlib import Path
import torch
path = Path("studies/minimal_silicon_study/checkpoints/silicon_restart_debug_4train_1val_100ep_chain_test/latest_checkpoint.pt")
if not path.exists():
    print(0)
else:
    ckpt = torch.load(path, map_location="cpu")
    print(int(ckpt.get("epoch", -1)) + 1)
PY
}

get_run_id() {
  python - <<'PY'
from pathlib import Path
import torch
path = Path("studies/minimal_silicon_study/checkpoints/silicon_restart_debug_4train_1val_100ep_chain_test/latest_checkpoint.pt")
ckpt = torch.load(path, map_location="cpu")
run_id = ckpt.get("wandb_run_id")
if not run_id:
    raise SystemExit("Could not find wandb_run_id in latest_checkpoint.pt")
print(run_id)
PY
}

COMPLETED_EPOCHS="$(get_completed_epochs)"
REMAINING_EPOCHS=$(( TOTAL_EPOCHS - COMPLETED_EPOCHS ))

echo "[SLURM] chain index: ${CHAIN_INDEX}/${MAX_CHAIN_JOBS}"
echo "[SLURM] completed epochs before launch: ${COMPLETED_EPOCHS}/${TOTAL_EPOCHS}"

if (( REMAINING_EPOCHS <= 0 )); then
  echo "[SLURM] Training already complete. Nothing to do."
  exit 0
fi

if [[ -f "${LATEST_CHECKPOINT}" ]]; then
  RUN_ID="$(get_run_id)"
  echo "[SLURM] Resuming run id ${RUN_ID} from ${LATEST_CHECKPOINT}"
  setsid python -u studies/minimal_silicon_study/train_silicon_minimal.py \
    --resume-from-run-id "${RUN_ID}" \
    --resume-from-checkpoint "${LATEST_CHECKPOINT}" \
    --run-name "${RUN_NAME}" \
    --num-epochs "${REMAINING_EPOCHS}" \
    "${COMMON_ARGS[@]}" &
else
  echo "[SLURM] Starting fresh run ${RUN_NAME}"
  setsid python -u studies/minimal_silicon_study/train_silicon_minimal.py \
    --run-name "${RUN_NAME}" \
    --num-epochs "${REMAINING_EPOCHS}" \
    "${COMMON_ARGS[@]}" &
fi

TRAIN_PID=$!
while true; do
  set +e
  wait "${TRAIN_PID}"
  TRAIN_EXIT=$?
  set -e
  if [[ "${TRAIN_EXIT}" -eq 0 ]]; then
    break
  fi
  if kill -0 "${TRAIN_PID}" 2>/dev/null; then
    echo "[SLURM] wait was interrupted (exit=${TRAIN_EXIT}) but training is still running; continuing to wait."
    continue
  fi
  break
done

if [[ "${TRAIN_EXIT}" -ne 0 ]]; then
  if [[ "${INTERRUPTED_BY_SIGNAL}" -eq 1 ]]; then
    echo "[SLURM] Training exited with code ${TRAIN_EXIT} after timeout signal; inspecting checkpoint state."
  else
    echo "[SLURM] Training process exited with code ${TRAIN_EXIT}" >&2
    exit "${TRAIN_EXIT}"
  fi
fi

COMPLETED_EPOCHS="$(get_completed_epochs)"
REMAINING_EPOCHS=$(( TOTAL_EPOCHS - COMPLETED_EPOCHS ))
echo "[SLURM] completed epochs after launch: ${COMPLETED_EPOCHS}/${TOTAL_EPOCHS}"

if (( REMAINING_EPOCHS <= 0 )); then
  echo "[SLURM] Training finished all ${TOTAL_EPOCHS} epochs."
  exit 0
fi

if (( CHAIN_INDEX >= MAX_CHAIN_JOBS )); then
  echo "[SLURM] Reached MAX_CHAIN_JOBS=${MAX_CHAIN_JOBS} with ${REMAINING_EPOCHS} epochs still remaining." >&2
  exit 1
fi

NEXT_CHAIN_INDEX=$(( CHAIN_INDEX + 1 ))
echo "[SLURM] Resubmitting next job with CHAIN_INDEX=${NEXT_CHAIN_INDEX}"
sbatch --export=ALL,CHAIN_INDEX="${NEXT_CHAIN_INDEX}",MAX_CHAIN_JOBS="${MAX_CHAIN_JOBS}" "$0"
