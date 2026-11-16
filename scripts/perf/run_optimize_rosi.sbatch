#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
# H100 verification of the inference optimisation harness.
# Usage (from the run directory): sbatch run_optimize_rosi.sbatch <material> [variants...]
# The run directory holds code/ (this branch), configs/ (trimmed, /bigdata
# paths) and uses the inference-speed benchmark's pinned upstream sources.
set -uo pipefail
material=$1; shift
variants=${*:-"deeph deeph_in_mandala deeph_e3 deeph_e3_in_mandala mace_h mace_h_in_mandala mandala"}
here=$(pwd -P)
bench=/data/home2/brzoza73/casus/mandala/researchops/runs/mandala_inference_speed_v1
source "$HOME/scripts/profiles/mandala.sh"
# The optimiser reads configs/ and upstream/ under --root.
ln -sfn "$bench/upstream" "$here/upstream"
export PYTHONPATH="$here/code/src:$here/code" PYTHONDONTWRITEBYTECODE=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 PYTHONHASHSEED=42
export TORCHINDUCTOR_CACHE_DIR="$here/inductor_cache/$SLURM_JOB_ID" TORCHINDUCTOR_COMPILE_THREADS=4
cd "$here/code"
nvidia-smi -L
for v in $variants; do
  echo "=== $v $(date -Is) load $(cut -d' ' -f1-3 /proc/loadavg)"
  timeout 5400 python -u scripts/perf/optimize_reference_inference.py \
    --root "$here" --material "$material" --variant "$v" --jit-mode eager \
    --repeats "${REPEATS:-10}" --output "$here/results/${material}_${v}_eager.json" \
    || echo "FAILED $v"
done
echo "ALL_DONE $(date -Is)"
