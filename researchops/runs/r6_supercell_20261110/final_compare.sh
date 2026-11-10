#!/usr/bin/env bash
#SBATCH --job-name=r6-final
#SBATCH --partition=gpu-h100-casus
#SBATCH --account=casus
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=researchops/runs/r6_supercell_20261110/final-compare-%j.out
set -euo pipefail
export HPC_PROJECT=mandala
source ~/scripts/profiles/mandala.sh
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export PYTHONHASHSEED=42 CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONPATH="$PWD/src:$PWD"
python -u researchops/runs/r6_supercell_20261110/analyze_tiling.py
python -u researchops/runs/r6_supercell_20261110/compare.py
python -u researchops/runs/r6_supercell_20261110/summarize.py
python -u researchops/runs/r6_supercell_20261110/compare_calibration.py
python -u researchops/runs/r6_supercell_20261110/collect_status.py
