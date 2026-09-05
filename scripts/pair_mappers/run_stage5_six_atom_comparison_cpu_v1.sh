#!/usr/bin/env bash
set -euo pipefail

export OMP_NUM_THREADS=64
export MKL_NUM_THREADS=64
export OPENBLAS_NUM_THREADS=64
export NUMEXPR_NUM_THREADS=64

output_dir="artifacts/pair_stage5/six_atom_validation_comparison_v1"
mkdir -p "${output_dir}"

python -u scripts/pair_mappers/plot_stage5_six_atom_comparison.py \
  --baseline-cache-dir /bigdata/casus/wdm/hamiltonian_learning/data/pair_descriptor_hamiltonian/sio2_baseline_cache_v1 \
  --baseline-summary artifacts/pair_stage1/sio2_baseline_cache_v1/summary.json \
  --shard-registry artifacts/pair_stage1/sio2_baseline_cache_v1/shards.csv \
  --range-envelope artifacts/pair_stage1/sio2_baseline_cache_v1/range_envelope.json \
  --native-config artifacts/pair_stage1/native_ace_sio2_h100_v1/config.json \
  --native-checkpoint artifacts/pair_stage1/native_ace_sio2_h100_v1/model.pt \
  --d4-cache-dir /bigdata/casus/wdm/hamiltonian_learning/data/pair_descriptor_hamiltonian/d4_sio2_v1 \
  --d4-schemas artifacts/pair_stage2/d4_sio2_precompute_v1/descriptor_schemas.json \
  --m0-high-checkpoint artifacts/pair_stage5/sio2_m0_d4_validation_v1/models/r6p5_d4_spherical_bessel_high_n8_l6_b3.pt \
  --m0-compact-checkpoint artifacts/pair_stage5/sio2_m0_d4_validation_v1/models/r6p5_d4_spherical_bessel_compact_n4_l3_b2.pt \
  --output-dir "${output_dir}" \
  --atom-count 6 \
  --color-limit-hartree 0.05 \
  --envelope-floor-hartree 1e-8 \
  2>&1 | tee "${output_dir}/launcher.log"
