#!/bin/bash
#SBATCH --job-name=e3mlp-synth
#SBATCH --output=studies/e3mlp_investigation/artifacts/slurm_%x_%j.out
#SBATCH --error=studies/e3mlp_investigation/artifacts/slurm_%x_%j.err
#SBATCH --time=01:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

set -euo pipefail
cd /home/bartek/casus/mandala

export RUN_NAME="${RUN_NAME:-slurm_synth_${SLURM_JOB_ID}}"
export TEACHER_KIND="${TEACHER_KIND:-mixed}"
export HIDDEN_IRREPS_PRESET="${HIDDEN_IRREPS_PRESET:-preliminary}"
export VARIANTS="${VARIANTS:-normact,gatemagnitudes,resnormact,film,bilinear}"
export DEPTHS="${DEPTHS:-2,4,6}"
export LOSS_KINDS="${LOSS_KINDS:-mse,mae,huber}"
export NUM_TRAIN="${NUM_TRAIN:-512}"
export NUM_VAL="${NUM_VAL:-128}"
export BATCH_SIZE="${BATCH_SIZE:-64}"
export NUM_STEPS="${NUM_STEPS:-64}"
export NUM_SEEDS="${NUM_SEEDS:-4}"
export LR="${LR:-1e-3}"
export DEVICE="${DEVICE:-cuda}"

bash studies/e3mlp_investigation/scripts/run_synthetic_teacher_cluster.sh
