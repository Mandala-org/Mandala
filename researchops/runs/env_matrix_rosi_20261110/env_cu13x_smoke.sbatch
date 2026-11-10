#!/bin/bash
#SBATCH --job-name=env-cu13x
#SBATCH --partition=gpu-h100-casus
#SBATCH --account=casus
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:45:00
#SBATCH --output=researchops/runs/env_matrix_rosi_20261110/slurm-%j.log
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 MAX_JOBS=4
cd /data/home2/brzoza73/casus/mandala_worktrees/envs2
if ! command -v uv >/dev/null; then curl -LsSf https://astral.sh/uv/install.sh | sh; fi
nvidia-smi
nvidia-smi --query-gpu=driver_version --format=csv
# All installation and compilation happens on the allocated GPU node.
uv python install 3.13
runner="$PWD/researchops/runs/env_matrix_rosi_20261110/verify_environment.py"
python_runner=$(uv python find --managed-python 3.13)
compat_dir=$("$python_runner" "$PWD/researchops/runs/env_matrix_rosi_20261110/install_cuda_compat.py") || exit 1
export LD_LIBRARY_PATH="$compat_dir${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export MANDALA_CUDA_COMPAT="cuda-compat-13-4 / user-space driver 615.71.10; kernel driver unchanged"
result=0
for version in 3.13 3.14; do
    "$python_runner" "$runner" --env "py${version/./}-latest-cu13x" --python "$version" --gpu-only --workers 4 || result=1
done
exit "$result"
