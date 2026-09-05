#!/usr/bin/env bash
set -euo pipefail

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

baseline=/bigdata/casus/wdm/hamiltonian_learning/data/pair_descriptor_hamiltonian/sio2_baseline_cache_v2

mkdir -p artifacts/pair_stage2/d1_sio2_precompute_v2
if [[ ! -f artifacts/pair_stage2/d1_sio2_precompute_v2/summary.json ]]; then
  python -u scripts/pair_descriptors/precompute_d1_sio2.py \
  --input-cache-dir "${baseline}" \
  --cache-summary artifacts/pair_stage1/sio2_baseline_cache_v2/summary.json \
  --shard-registry artifacts/pair_stage1/sio2_baseline_cache_v2/shards.csv \
  --output-cache-dir /bigdata/casus/wdm/hamiltonian_learning/data/pair_descriptor_hamiltonian/d1_sio2_v2 \
  --output-dir artifacts/pair_stage2/d1_sio2_precompute_v2 \
  --cutoffs-angstrom 6.5 7.5 8.5 10.5 \
  --radial-bases spherical_bessel zernike \
  --resolution compact 4 3 --resolution high 8 6 \
  --num-workers 64 --resume \
    2>&1 | tee -a artifacts/pair_stage2/d1_sio2_precompute_v2/launcher.log
fi

for family in d2 d3; do
  mkdir -p "artifacts/pair_stage2/${family}_sio2_precompute_v2"
  if [[ "${family}" == d2 ]]; then
    resolution=(--resolution degree4 4 4 --resolution degree6 6 6 --resolution degree8 8 8 --resolution degree10 10 10)
  else
    resolution=(--resolution compact 4 3 --resolution high 8 6)
  fi
  if [[ ! -f "artifacts/pair_stage2/${family}_sio2_precompute_v2/summary.json" ]]; then
    python -u scripts/pair_descriptors/precompute_d2_d3_sio2.py \
    --family "${family}" \
    --input-oracle-dir /bigdata/casus/wdm/hamiltonian_learning/data/pair_descriptor_hamiltonian/d1_sio2_v2 \
    --oracle-summary artifacts/pair_stage2/d1_sio2_precompute_v2/summary.json \
    --oracle-registry artifacts/pair_stage2/d1_sio2_precompute_v2/shards.csv \
    --output-cache-dir "/bigdata/casus/wdm/hamiltonian_learning/data/pair_descriptor_hamiltonian/${family}_sio2_v2" \
    --output-dir "artifacts/pair_stage2/${family}_sio2_precompute_v2" \
    --cutoffs-angstrom 6.5 7.5 8.5 10.5 \
    "${resolution[@]}" --num-workers 64 --resume \
      2>&1 | tee -a "artifacts/pair_stage2/${family}_sio2_precompute_v2/launcher.log"
  fi
done

mkdir -p artifacts/pair_stage2/d4_sio2_precompute_v2
if [[ ! -f artifacts/pair_stage2/d4_sio2_precompute_v2/summary.json ]]; then
  python -u scripts/pair_descriptors/precompute_d4_sio2.py \
  --input-d1-cache-dir /bigdata/casus/wdm/hamiltonian_learning/data/pair_descriptor_hamiltonian/d1_sio2_v2 \
  --d1-summary artifacts/pair_stage2/d1_sio2_precompute_v2/summary.json \
  --d1-registry artifacts/pair_stage2/d1_sio2_precompute_v2/shards.csv \
  --output-cache-dir /bigdata/casus/wdm/hamiltonian_learning/data/pair_descriptor_hamiltonian/d4_sio2_v2 \
  --output-dir artifacts/pair_stage2/d4_sio2_precompute_v2 \
  --cutoffs-angstrom 6.5 7.5 8.5 10.5 --radial-bases spherical_bessel zernike \
  --resolution compact 4 3 --resolution high 8 6 --body-degrees 2 3 \
  --lift-max-input-l 3 --lift-max-radial-index 1 --lift-radial-degree-budget 2 \
  --lift-max-intermediate-l 4 --max-paths-per-degree-irrep 48 \
  --num-workers 64 --resume \
    2>&1 | tee -a artifacts/pair_stage2/d4_sio2_precompute_v2/launcher.log
fi

mkdir -p artifacts/pair_stage2/sio2_descriptor_normalization_v2
if [[ ! -f artifacts/pair_stage2/sio2_descriptor_normalization_v2/summary.json ]]; then
  python -u scripts/pair_descriptors/fit_descriptor_normalization_sio2.py \
  --family d1 /bigdata/casus/wdm/hamiltonian_learning/data/pair_descriptor_hamiltonian/d1_sio2_v2 artifacts/pair_stage2/d1_sio2_precompute_v2 artifacts/pair_stage2/d1_sio2_precompute_v2/descriptor_schemas.json \
  --family d2 /bigdata/casus/wdm/hamiltonian_learning/data/pair_descriptor_hamiltonian/d2_sio2_v2 artifacts/pair_stage2/d2_sio2_precompute_v2 artifacts/pair_stage2/d2_sio2_precompute_v2/descriptor_schemas.json \
  --family d3 /bigdata/casus/wdm/hamiltonian_learning/data/pair_descriptor_hamiltonian/d3_sio2_v2 artifacts/pair_stage2/d3_sio2_precompute_v2 artifacts/pair_stage2/d3_sio2_precompute_v2/descriptor_schemas.json \
  --family d4 /bigdata/casus/wdm/hamiltonian_learning/data/pair_descriptor_hamiltonian/d4_sio2_v2 artifacts/pair_stage2/d4_sio2_precompute_v2 artifacts/pair_stage2/d4_sio2_precompute_v2/descriptor_schemas.json \
  --output-dir artifacts/pair_stage2/sio2_descriptor_normalization_v2 \
  --num-workers 64 --scale-floor 1e-12 \
    2>&1 | tee artifacts/pair_stage2/sio2_descriptor_normalization_v2/launcher.log
fi

mkdir -p artifacts/pair_stage2/sio2_descriptor_certification_v2
if [[ ! -f artifacts/pair_stage2/sio2_descriptor_certification_v2/summary.json ]]; then
  python -u scripts/pair_descriptors/certify_stage2_sio2.py \
  --family d1 /bigdata/casus/wdm/hamiltonian_learning/data/pair_descriptor_hamiltonian/d1_sio2_v2 artifacts/pair_stage2/d1_sio2_precompute_v2 \
  --family d2 /bigdata/casus/wdm/hamiltonian_learning/data/pair_descriptor_hamiltonian/d2_sio2_v2 artifacts/pair_stage2/d2_sio2_precompute_v2 \
  --family d3 /bigdata/casus/wdm/hamiltonian_learning/data/pair_descriptor_hamiltonian/d3_sio2_v2 artifacts/pair_stage2/d3_sio2_precompute_v2 \
  --family d4 /bigdata/casus/wdm/hamiltonian_learning/data/pair_descriptor_hamiltonian/d4_sio2_v2 artifacts/pair_stage2/d4_sio2_precompute_v2 \
  --normalization-summary artifacts/pair_stage2/sio2_descriptor_normalization_v2/summary.json \
  --output-dir artifacts/pair_stage2/sio2_descriptor_certification_v2 \
  --num-workers 64 --seed 20260904 --symmetry-tolerance 1e-9 \
  --permutation-tolerance 1e-12 --puncture-tolerance 1e-10 --float32-tolerance 5e-7 \
  --pbc-displacement-tolerance-angstrom 2e-5 --storage-budget-bytes 1979120929996 \
    2>&1 | tee artifacts/pair_stage2/sio2_descriptor_certification_v2/launcher.log
fi

python -c 'import json; assert all(json.load(open(f"artifacts/pair_stage2/{p}/summary.json"))["passed"] for p in ("d1_sio2_precompute_v2", "d2_sio2_precompute_v2", "d3_sio2_precompute_v2", "d4_sio2_precompute_v2", "sio2_descriptor_normalization_v2", "sio2_descriptor_certification_v2"))'
