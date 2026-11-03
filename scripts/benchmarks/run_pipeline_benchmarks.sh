#!/usr/bin/env bash
# Frozen benchmark batch for the modular pipeline (one GPU, 4 CPUs).
#
# 1. A/B: forward and training-step time of identical models on the
#    pre-decomposition source (BASE_COMMIT) and on this checkout.
# 2. Per-stage profiles of the reference recipes at three structure sizes.
# 3. Cost-model constants fitted from the profiles.
# Results go to benchmark_data/pipeline_modularity_v1/<job id>; the batch
# stops at the first failure.
set -euo pipefail
if [[ $# != 0 ]]; then
  echo "This frozen launcher takes no arguments." >&2
  exit 2
fi
BASE_COMMIT=3ec49ae277d76c922c35849c14089ba2b393d310
ROOT="$(pwd)"
OUT="benchmark_data/pipeline_modularity_v1/${SLURM_JOB_ID:-local-$(date +%Y%m%d-%H%M%S)}"
export PYTHONHASHSEED=42 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
export MPLCONFIGDIR=/tmp/mandala-benchmarks-mpl
# Large graphs fragment the caching allocator (job 820553 ran out of memory
# with 30 GiB reserved but unallocated).
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$OUT/base_source" "$OUT/ab" "$OUT/profiles"
git archive "$BASE_COMMIT" src configs | tar -x -C "$OUT/base_source"
git rev-parse HEAD > "$OUT/modular_commit.txt"
git status --porcelain > "$OUT/modular_worktree_status.txt"
echo "$BASE_COMMIT" > "$OUT/base_commit.txt"

SIO2='{"O": [0, 0, 1, 1, 2], "Si": [0, 0, 1, 1, 2]}'
SI='{"Si": [0, 0, 1, 1, 2]}'
CASES=(
  "paper_sio2_96|configs/reference/mandala_paper.yaml|$SIO2|96"
  "paper_sio2_192|configs/reference/mandala_paper.yaml|$SIO2|192"
  "first_model_si_64|examples/first_model/silicon.yaml|$SI|64"
  # The reference models also run at 32 atoms: at 64, the pre-decomposition
  # DeepH training step needs more than 55 GiB, and an out-of-memory run is
  # recorded instead of timed.
  "deeph_si_32|configs/reference/deeph.yaml|$SI|32"
  "deeph_e3_si_32|configs/reference/deeph_e3.yaml|$SI|32"
  "mace_h_si_32|configs/reference/mace_h.yaml|$SI|32"
  "deeph_si_64|configs/reference/deeph.yaml|$SI|64"
  "deeph_e3_si_64|configs/reference/deeph_e3.yaml|$SI|64"
  "mace_h_si_64|configs/reference/mace_h.yaml|$SI|64"
)
for case in "${CASES[@]}"; do
  IFS='|' read -r name config shells atoms <<< "$case"
  # modular_nocache recomputes structure-only stages like the base tree, which
  # separates per-stage dispatch overhead from the static-feature cache.
  for tree in base modular modular_nocache; do
    # Each tree runs its own recipes: the reference models were monolithic
    # transforms before the decomposition and library blocks after it.
    if [[ $tree == base ]]; then
      source_root="$OUT/base_source/src"
      recipe="$config"
      [[ -f "$OUT/base_source/$config" ]] && recipe="$OUT/base_source/$config"
    else
      source_root="$ROOT/src"
      recipe="$config"
    fi
    PYTHONPATH="$source_root:$ROOT" python -u scripts/benchmarks/forward_timing.py \
      --config "$recipe" --shells "$shells" --atoms "$atoms" --device cuda \
      --warmup 5 --repeats 20 --label "$tree" --output "$OUT/ab/${name}_${tree}.json" \
      $([[ $tree == modular_nocache ]] && echo --no-static-cache)
  done
done
PYTHONPATH="$ROOT/src:$ROOT" python scripts/benchmarks/compare_timings.py "$OUT/ab" --output "$OUT/ab_summary.md"

RECIPES=(
  "configs/reference/mandala_paper.yaml|$SIO2"
  "configs/reference/deeph.yaml|$SI"
  "configs/reference/deeph_e3.yaml|$SI"
  "configs/reference/mace_h.yaml|$SI"
  "configs/hybrids/tensor_products_degree_expansion_shell_readout.yaml|$SI"
  "configs/hybrids/equivariant_nodes_bond_frame_readout.yaml|$SI"
)
for recipe in "${RECIPES[@]}"; do
  IFS='|' read -r config shells <<< "$recipe"
  for atoms in 32 64 128; do
    PYTHONPATH="$ROOT/src:$ROOT" python -u scripts/benchmarks/stage_profile.py \
      --config "$config" --shells "$shells" --atoms "$atoms" --device cuda \
      --output "$OUT/profiles/$(basename "$config" .yaml)_${atoms}.json"
  done
done
PYTHONPATH="$ROOT/src:$ROOT" python scripts/benchmarks/fit_cost_constants.py "$OUT"/profiles/*.json \
  --output "$OUT/cost_constants.json"
echo "Done: $OUT (copy cost_constants.json to src/net/cost_constants.json to calibrate the cost model)"
