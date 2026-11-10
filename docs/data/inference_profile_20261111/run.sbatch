#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
# Stage-level latency profile of native vs in-MANDALA inference.
# Usage: sbatch run.sbatch <graphene|mos2> [extra profiler args]
# VARIANTS env selects variants (default: all seven, one process each).
set -uo pipefail
material=$1; shift
variants=${VARIANTS:-"mandala deeph deeph_in_mandala deeph_e3 deeph_e3_in_mandala mace_h mace_h_in_mandala"}
here=/data/home2/brzoza73/casus/mandala/researchops/runs/mandala_inference_profile_v1
bench=/data/home2/brzoza73/casus/mandala/researchops/runs/mandala_inference_speed_v1
source "$HOME/scripts/profiles/mandala.sh"
# Benchmark's frozen framework (commit 3ec49ae + attempt-4 patch), used read-only.
export PYTHONPATH="$bench/framework/src:$bench/framework" PYTHONDONTWRITEBYTECODE=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 PYTHONHASHSEED=42
cd "$bench/framework"
nvidia-smi -L
status=0
for v in $variants; do
  echo "=== $v $(date -Is)"
  timeout 1500 python -u "$here/profile_reference_inference.py" --root "$bench" \
    --material "$material" --variant "$v" --output-dir "$here/results/$SLURM_JOB_ID" "$@" \
    || { echo "FAILED $v"; status=1; }
done
exit $status
