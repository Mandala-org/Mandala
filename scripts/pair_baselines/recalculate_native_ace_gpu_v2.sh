#!/usr/bin/env bash
set -euo pipefail

# H100 allocations provide 16 host CPUs.
export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16

mkdir -p artifacts/pair_stage1/native_ace_core_h100_v2
if [[ ! -f artifacts/pair_stage1/native_ace_core_h100_v2/summary.json ]]; then
  python -u scripts/pair_baselines/validate_native_ace_core.py \
  --output-dir artifacts/pair_stage1/native_ace_core_h100_v2 \
  --device cuda --seed 20260904 --descriptor-cutoff-angstrom 8.5 \
  --density-radial-count 2 --density-l-max 2 --onsite-correlation-order 2 \
  --onsite-max-degree 6 --bond-radial-count 2 --bond-l-max 4 \
  --bond-cutoff-angstrom 6.5 --offsite-max-degree 6 --ridge 1e-8 \
  --pair-batch-size 512 --warmup-iterations 5 --benchmark-iterations 20 \
    2>&1 | tee artifacts/pair_stage1/native_ace_core_h100_v2/run.log
fi

for fit_mode in physical_design weighted_normalized_target; do
  if [[ "${fit_mode}" == physical_design ]]; then label=physical; else label=weighted_h_over_g; fi
  output="artifacts/pair_stage1/native_ace_sio2_h100_${label}_v2"
  mkdir -p "${output}"
  if [[ ! -f "${output}/summary.json" ]]; then
    python -u scripts/pair_baselines/train_native_ace_sio2.py \
    --cache-dir /bigdata/casus/wdm/hamiltonian_learning/data/pair_descriptor_hamiltonian/sio2_baseline_cache_v2 \
    --cache-summary artifacts/pair_stage1/sio2_baseline_cache_v2/summary.json \
    --shard-registry artifacts/pair_stage1/sio2_baseline_cache_v2/shards.csv \
    --range-envelope artifacts/pair_stage1/sio2_baseline_cache_v2/range_envelope.json \
    --core-validation-summary artifacts/pair_stage1/native_ace_core_h100_v2/summary.json \
    --output-dir "${output}" --device cuda --seed 20260904 \
    --onsite-correlation-order 2 --onsite-max-degree 6 \
    --bond-radial-count 2 --bond-l-max 4 --bond-cutoff-angstrom 6.5 \
    --offsite-max-degree 6 --ridge 1e-8 --range-fit-mode "${fit_mode}" \
    --batch-size 2048 --envelope-floor-hartree 1e-8 \
    --distance-bin-width-angstrom 0.5 \
      2>&1 | tee "${output}/launcher.log"
  fi
done

python -c 'import json; paths=["artifacts/pair_stage1/native_ace_sio2_h100_physical_v2/summary.json", "artifacts/pair_stage1/native_ace_sio2_h100_weighted_h_over_g_v2/summary.json"]; values=[json.load(open(p)) for p in paths]; assert all(v["passed"] for v in values); assert abs(values[0]["headline_validation_matrix_mae_mev"]-values[1]["headline_validation_matrix_mae_mev"]) < 1e-4'
