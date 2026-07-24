#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

export ZNCUSNSES_BIG_SOURCE_CHECKPOINT="/bigdata/casus/wdm/hamiltonian_learning/models/checkpoints/ZnCuSnSeS_big_3x_b200_restarts/zncusnses-big-3x-scale1-seed42-restart-lr5em3-clip0p5-pat12/best_model.pt"
export ZNCUSNSES_BIG_CHECKPOINT_DIR="/bigdata/casus/wdm/hamiltonian_learning/models/checkpoints/ZnCuSnSeS_big_3x_b200_scales12_finetune"
export ZNCUSNSES_BIG_WANDB_PROJECT="mandala-ZnCuSnSeS-big-3x-b200-scales12-finetune"
export ZNCUSNSES_BIG_WANDB_GROUP="ZnCuSnSeS_big_3x_scales12_from_best_scale1"
export ZNCUSNSES_BIG_RUN_NAME="zncusnses-big-3x-scales12-from-scale1-best-lr5em3-clip0p5-pat12-seed42"
export ZNCUSNSES_BIG_SCALES="[1, 2]"

exec bash "${ROOT_DIR}/scripts/run_restart_ZnCuSnSeS_big_scale1_seed42_b200.sh" 0.005
