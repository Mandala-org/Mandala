#!/usr/bin/env bash
# Terok (no Slurm). Usage: run_optimize_terok.sh <tag> <gpu> <material> <jit-mode> [variants...]
# CPUS selects the pinned cores (default 236-239; check they are idle first).
set -uo pipefail
tag=$1 gpu=$2 material=$3 jit=$4; shift 4
variants=${*:-"mandala deeph deeph_in_mandala deeph_e3 deeph_e3_in_mandala mace_h mace_h_in_mandala"}
base=~/casus/mandala_perf
. $base/venv/bin/activate
export PYTHONPATH=$base/code/src:$base/code PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=$gpu
export CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 PYTHONHASHSEED=42
export TORCHINDUCTOR_CACHE_DIR=$base/runs/$tag/inductor_cache
cd $base/code
nvidia-smi -i $gpu --query-gpu=name,memory.used,utilization.gpu --format=csv,noheader
for v in $variants; do
  echo "=== $v $(date -Is)"
  timeout 3600 taskset -c ${CPUS:-236-239} python -u scripts/perf/optimize_reference_inference.py \
    --root $base --material $material --variant $v --jit-mode $jit \
    --output $base/runs/$tag/${material}_${v}_${jit}.json \
    || echo "FAILED $v"
done
echo "ALL_DONE $(date -Is)"
