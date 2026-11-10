#!/usr/bin/env bash
#SBATCH --job-name=r6-eval-si-5x
#SBATCH --partition=gpu-h100-casus
#SBATCH --account=casus
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --output=researchops/runs/r6_supercell_20261110/eval-si-5x-%j.out
set -euo pipefail
export HPC_PROJECT=mandala
source ~/scripts/profiles/mandala.sh
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export PYTHONHASHSEED=42 CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONPATH="$PWD/src:$PWD"
python -u researchops/runs/r6_supercell_20261110/evaluate.py --material silicon --names si-5x5x5-1 si-5x5x5-tiled
python -u researchops/runs/r6_supercell_20261110/route_parent.py --names si-5x5x5-1 si-5x5x5-tiled
python -u researchops/runs/r6_supercell_20261110/compare.py --reuse-diagnostics
python -u researchops/runs/r6_supercell_20261110/analyze_tiling.py
python -u researchops/runs/r6_supercell_20261110/summarize.py
