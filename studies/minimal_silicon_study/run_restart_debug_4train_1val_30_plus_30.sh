#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

if [[ -f "${REPO_ROOT}/mandala-venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "${REPO_ROOT}/mandala-venv/bin/activate"
fi

PHASE1_RUN_NAME="silicon_restart_debug_4train_1val_30ep_phase1"
PHASE2_RUN_NAME="silicon_restart_debug_4train_1val_30ep_phase2"
GROUP_NAME="silicon_restart_debug_4train_1val_30_plus_30"
PHASE1_CHECKPOINT_DIR="studies/minimal_silicon_study/checkpoints/${PHASE1_RUN_NAME}"

export WANDB_RUN_GROUP="${GROUP_NAME}"
export WANDB_PROJECT="resume-tests"

COMMON_ARGS=(
  --data-path /bigdata/casus/wdm/hamiltonian_learning/data/silicon_very_big/dataset_A
  --snapshot-cache-dir /bigdata/casus/wdm/hamiltonian_learning/data/silicon_very_big/snapshot_cache
  --wandb-project resume-tests
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
  --benchmark false
  --apply-cutoff-to-targets true
  --require-exact-edge-match true
  --log-data false
  --log-model false
  --log-forward false
  --verbose-forward false
  --log-per-irrep-metrics false
  --log-per-irrep-images false
  --generate-video false
)

python studies/minimal_silicon_study/train_silicon_minimal.py \
  --run-name "${PHASE1_RUN_NAME}" \
  --num-epochs 30 \
  "${COMMON_ARGS[@]}"

if [[ ! -f "${PHASE1_CHECKPOINT_DIR}/latest_checkpoint.pt" ]]; then
  echo "Expected checkpoint not found: ${PHASE1_CHECKPOINT_DIR}/latest_checkpoint.pt" >&2
  exit 1
fi

PHASE1_RUN_ID="$(python - <<'PY'
import torch
ckpt = torch.load(
    'studies/minimal_silicon_study/checkpoints/silicon_restart_debug_10train_1val_100ep_phase1/latest_checkpoint.pt',
    map_location='cpu',
)
run_id = ckpt.get('wandb_run_id')
if not run_id:
    raise SystemExit('Could not find wandb_run_id in latest_checkpoint.pt')
print(run_id)
PY
)"

python studies/minimal_silicon_study/train_silicon_minimal.py \
  --resume-from-run-id "${PHASE1_RUN_ID}" \
  --resume-from-checkpoint "${PHASE1_CHECKPOINT_DIR}/latest_checkpoint.pt" \
  --run-name "${PHASE2_RUN_NAME}" \
  --num-epochs 30 \
  "${COMMON_ARGS[@]}"
