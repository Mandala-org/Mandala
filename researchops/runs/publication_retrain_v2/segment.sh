#!/usr/bin/env bash
set -eo pipefail
source "$HOME/scripts/profiles/mandala.sh"
set -u

material=${1:?Usage: segment.sh silicon|sio2|zncusnses}
case "$material" in silicon|sio2|zncusnses) ;; *) exit 2 ;; esac
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    # sbatch spools this file elsewhere; --chdir pins the repository root.
    root=$(pwd -P)
else
    root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
fi
cd "$root"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export MKL_NUM_THREADS=$OMP_NUM_THREADS
export OPENBLAS_NUM_THREADS=$OMP_NUM_THREADS
export NUMEXPR_NUM_THREADS=$OMP_NUM_THREADS
export PYTHONPATH="$root/src:$root${PYTHONPATH:+:$PYTHONPATH}"

# The recipe wrapper locks the run, resumes its safe latest checkpoint, and
# returns 0 without training when checkpoint epoch >= the total epoch cap.
# Run in a Slurm step so --signal=INT@600 reaches Python/Lightning.
exec srun --ntasks=1 --cpus-per-task="$OMP_NUM_THREADS" \
    bash "scripts/reproduce_examples/train_${material}_v2.sh"
