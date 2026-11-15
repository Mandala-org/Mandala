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
# Lightning answers that SIGINT with a graceful shutdown followed by
# sys.exit(1), which would break the afterok chain even though the epoch
# boundary checkpoint is intact. Treat that case as a clean segment end.
checkpoint=/bigdata/casus/wdm/hamiltonian_learning/models/checkpoints/publication_retrain_v2/$material/latest_checkpoint.pt
segment_start=$(date +%s)
rc=0
if [[ "$material" == zncusnses ]]; then
    # train_zncusnses_v2.sh falls back to the frozen legacy source (recipe with
    # separate_shifted_self=true) whenever a checkpoint exists, but the Zn chain
    # restarted on 10 Nov without that head must resume from the current source.
    # The wrapper is hashed into the reproducibility contract, so it stays
    # byte-identical and the recipe trainer is called here with the exact
    # environment of the wrapper's fresh-start branch (seed 42, as recorded).
    train_cmd=(env PYTHONHASHSEED=42 CUBLAS_WORKSPACE_CONFIG=:4096:8
        MPLCONFIGDIR=/tmp/mandala-publication-mpl
        python -u scripts/reproduce_examples/train.py
        --recipe scripts/reproduce_examples/zncusnses_v2.json)
else
    train_cmd=(bash "scripts/reproduce_examples/train_${material}_v2.sh")
fi
srun --ntasks=1 --cpus-per-task="$OMP_NUM_THREADS" "${train_cmd[@]}" || rc=$?
if (( rc != 0 )) && [[ -n "${SLURM_JOB_ID:-}" ]]; then
    end_time=$(scontrol show job "$SLURM_JOB_ID" | grep -oP "EndTime=\K\S+" | head -n 1)
    seconds_left=$(( $(date -d "$end_time" +%s) - $(date +%s) ))
    if (( seconds_left <= 900 )) && [[ -f "$checkpoint" ]] \
        && (( $(stat -c %Y "$checkpoint") > segment_start )); then
        echo "--- segment.sh: training stopped by the Slurm time-limit signal (exit $rc, ${seconds_left}s left) after saving a checkpoint in this job; reporting success so the next segment resumes ---"
        rc=0
    fi
fi
exit "$rc"
