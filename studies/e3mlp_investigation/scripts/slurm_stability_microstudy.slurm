#!/bin/bash
#SBATCH --job-name=e3mlp-stab
#SBATCH --output=studies/e3mlp_investigation/artifacts/slurm_%x_%j.out
#SBATCH --error=studies/e3mlp_investigation/artifacts/slurm_%x_%j.err
#SBATCH --time=01:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

set -euo pipefail
cd /home/bartek/casus/mandala

export RUN_NAME="${RUN_NAME:-slurm_stability_${SLURM_JOB_ID}}"
export HIDDEN_IRREPS_PRESET="${HIDDEN_IRREPS_PRESET:-preliminary}"
export VARIANTS="${VARIANTS:-normact,gatemagnitudes,resnormact,resgatemagnitudes,film,bilinear}"
export DEPTHS="${DEPTHS:-4,6,10}"
export OUTPUT_SCALES="${OUTPUT_SCALES:-0.5,1.0,1.5}"
export WEIGHT_INIT_SCALES="${WEIGHT_INIT_SCALES:-0.5,1.0,1.5}"
export RESIDUAL_SCALES="${RESIDUAL_SCALES:-0.1,0.25}"
export NUM_STEPS="${NUM_STEPS:-16}"
export NUM_SEEDS="${NUM_SEEDS:-4}"
export BATCH_SIZE="${BATCH_SIZE:-32}"
export LR="${LR:-1e-3}"
export DEVICE="${DEVICE:-cuda}"

bash studies/e3mlp_investigation/scripts/run_stability_microstudy_cluster.sh
