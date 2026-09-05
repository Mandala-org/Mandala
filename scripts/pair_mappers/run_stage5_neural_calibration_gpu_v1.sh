#!/usr/bin/env bash
set -euo pipefail

# The H100 allocation provides 16 host CPUs.
export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16

bash scripts/pair_mappers/freeze_stage5_neural_calibration_v1.sh

run_task() {
  local architecture="$1"
  local band="$2"
  local learning_rate="$3"
  local rate_label="$4"
  local cap hidden rank
  case "${band}" in
    small) cap=2; hidden=64; rank=8 ;;
    medium) cap=4; hidden=128; rank=32 ;;
    large) cap=8; hidden=256; rank=128 ;;
    *) echo "Unknown resource band: ${band}" >&2; return 2 ;;
  esac
  local task_id="${architecture}_${band}_lr${rate_label}"
  local output="artifacts/pair_stage5/sio2_neural_calibration_grid_v1/runs/${task_id}"
  if [[ ! -f "${output}/summary.json" ]]; then
    mkdir -p "${output}"
    python -u scripts/pair_mappers/train_stage5_neural_calibration.py \
      --family d1 \
      --descriptor-key r6p5_spherical_bessel_high_n8_l6 \
      --descriptor-cache-dir /bigdata/casus/wdm/hamiltonian_learning/data/pair_descriptor_hamiltonian/d1_sio2_v1 \
      --descriptor-summary artifacts/pair_stage2/d1_sio2_precompute_v1/summary.json \
      --descriptor-schemas artifacts/pair_stage2/d1_sio2_precompute_v1/descriptor_schemas.json \
      --normalization artifacts/pair_stage2/sio2_descriptor_normalization_v1/normalization.json \
      --normalization-summary artifacts/pair_stage2/sio2_descriptor_normalization_v1/summary.json \
      --promotions artifacts/pair_stage3/sio2_descriptor_promotions_v1/promotions.json \
      --calibration-manifest artifacts/pair_stage5/sio2_neural_calibration_grid_v1/calibration_manifest.json \
      --task-id "${task_id}" \
      --baseline-cache-dir /bigdata/casus/wdm/hamiltonian_learning/data/pair_descriptor_hamiltonian/sio2_baseline_cache_v1 \
      --baseline-summary artifacts/pair_stage1/sio2_baseline_cache_v1/summary.json \
      --shard-registry artifacts/pair_stage1/sio2_baseline_cache_v1/shards.csv \
      --range-envelope artifacts/pair_stage1/sio2_baseline_cache_v1/range_envelope.json \
      --output-dir "${output}" \
      --architecture "${architecture}" \
      --resource-band "${band}" \
      --descriptor-multiplicity-cap "${cap}" \
      --generator-multiplicity "${cap}" \
      --hidden-multiplicity "${cap}" \
      --hidden-l-max 4 \
      --invariant-hidden "${hidden}" \
      --factorization-rank "${rank}" \
      --bond-radial-count 2 \
      --bond-l-max 4 \
      --bond-cutoff-angstrom 6.5 \
      --learning-rate "${learning_rate}" \
      --weight-decay 1e-6 \
      --steps 1500 \
      --eval-interval 250 \
      --early-stopping-evaluations 3 \
      --onsite-batch-size 256 \
      --offsite-batch-size 512 \
      --evaluation-batch-size 2048 \
      --train-fraction 0.25 \
      --seed 20260905 \
      --device cuda \
      --envelope-floor-hartree 1e-8 \
      --distance-bin-width-angstrom 0.5 \
      --float32-symmetry-tolerance 2e-5 \
      --resume \
      2>&1 | tee -a "${output}/launcher.log"
  fi
  python -c 'import json, sys; value=json.load(open(sys.argv[1])); assert value["completed"] and value["passed"] and not value["test_shards_read"]' "${output}/summary.json"
}

for architecture in m3 m5; do
  for band in small medium large; do
    run_task "${architecture}" "${band}" 3e-4 3em4
    run_task "${architecture}" "${band}" 1e-3 1em3
  done
done
