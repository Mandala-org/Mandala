#!/bin/bash
#SBATCH -J pyscf_h2o_rks_k444
#SBATCH -A casus
#SBATCH -p cpu-genoa
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --mem=1200000
#SBATCH -t 3-00:00:00
#SBATCH -o /home/bartek/casus/mandala/data/pyscf_baseline/results/slurm-%x-%j.out
#SBATCH -e /home/bartek/casus/mandala/data/pyscf_baseline/results/slurm-%x-%j.err

set -euo pipefail

cd /home/bartek/casus/mandala
source mandala-venv/bin/activate

mkdir -p data/pyscf_baseline/results

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export VECLIB_MAXIMUM_THREADS="${SLURM_CPUS_PER_TASK}"

# Optional run-time tuning overrides (can be set before sbatch):
#   export XC=pbe,pbe
#   export DF_BACKEND=gdf
#   export DF_AUXBASIS=weigend
#   export MAX_MEMORY_MB=1200000
#   export KX=4; export KY=4; export KZ=4
#   export REQUIRE_FORCES_STRESS=1

bash data/pyscf_baseline/run_h2o_original_production_rks_k444.sh

# Produce a post-run size report for quick sanity checks.
python data/pyscf_baseline/analyze_shift_resolved_size.py \
  --pyscf-npz data/pyscf_baseline/results/h2o_original_rks_openmx_like_prod_k444.npz \
  --report-json data/pyscf_baseline/results/h2o_original_rks_openmx_like_prod_k444.size_report.json
