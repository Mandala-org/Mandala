#!/usr/bin/env bash
set -euo pipefail

export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16

grid=artifacts/pair_stage5/sio2_neural_calibration_grid_v2
if [[ ! -f "${grid}/summary.json" ]]; then
  mkdir -p "${grid}"
  python -u scripts/pair_mappers/freeze_stage5_neural_calibration.py \
    --promotions artifacts/pair_stage3/sio2_descriptor_promotions_v2/promotions.json \
    --output-dir "${grid}"
fi
python -c 'import json; p=json.load(open("artifacts/pair_stage5/sio2_neural_calibration_grid_v2/summary.json")); assert p["passed"] and p["task_count"] == 24 and not p["test_shards_read"]'

run_task() {
  local architecture="$1" band="$2" learning_rate="$3" rate_label="$4"
  local range_loss="$5" loss_label="$6"
  local cap hidden rank
  case "${band}" in
    small) cap=2; hidden=64; rank=8 ;;
    medium) cap=4; hidden=128; rank=32 ;;
    large) cap=8; hidden=256; rank=128 ;;
    *) echo "Unknown resource band: ${band}" >&2; return 2 ;;
  esac
  local task_id="${architecture}_${band}_lr${rate_label}_${loss_label}"
  local output="${grid}/runs/${task_id}"
  if [[ ! -f "${output}/summary.json" ]]; then
    mkdir -p "${output}"
    python -u scripts/pair_mappers/train_stage5_neural_calibration.py \
      --family d1 --descriptor-key r6p5_spherical_bessel_high_n8_l6 \
      --descriptor-cache-dir /bigdata/casus/wdm/hamiltonian_learning/data/pair_descriptor_hamiltonian/d1_sio2_v2 \
      --descriptor-summary artifacts/pair_stage2/d1_sio2_precompute_v2/summary.json \
      --descriptor-schemas artifacts/pair_stage2/d1_sio2_precompute_v2/descriptor_schemas.json \
      --normalization artifacts/pair_stage2/sio2_descriptor_normalization_v2/normalization.json \
      --normalization-summary artifacts/pair_stage2/sio2_descriptor_normalization_v2/summary.json \
      --promotions artifacts/pair_stage3/sio2_descriptor_promotions_v2/promotions.json \
      --calibration-manifest "${grid}/calibration_manifest.json" \
      --task-id "${task_id}" \
      --baseline-cache-dir /bigdata/casus/wdm/hamiltonian_learning/data/pair_descriptor_hamiltonian/sio2_baseline_cache_v2 \
      --baseline-summary artifacts/pair_stage1/sio2_baseline_cache_v2/summary.json \
      --shard-registry artifacts/pair_stage1/sio2_baseline_cache_v2/shards.csv \
      --range-envelope artifacts/pair_stage1/sio2_baseline_cache_v2/range_envelope.json \
      --output-dir "${output}" --architecture "${architecture}" \
      --resource-band "${band}" --descriptor-multiplicity-cap "${cap}" \
      --generator-multiplicity "${cap}" --hidden-multiplicity "${cap}" \
      --hidden-l-max 4 --invariant-hidden "${hidden}" --factorization-rank "${rank}" \
      --bond-radial-count 2 --bond-l-max 4 --bond-cutoff-angstrom 6.5 \
      --learning-rate "${learning_rate}" --weight-decay 1e-6 \
      --range-loss-mode "${range_loss}" --steps 1500 --eval-interval 250 \
      --early-stopping-evaluations 3 --onsite-batch-size 256 \
      --offsite-batch-size 512 --evaluation-batch-size 2048 \
      --train-fraction 0.25 --seed 20260905 --device cuda \
      --envelope-floor-hartree 1e-8 --distance-bin-width-angstrom 0.5 \
      --float32-symmetry-tolerance 2e-5 --resume \
      2>&1 | tee -a "${output}/launcher.log"
  fi
  python -c 'import json,sys; p=json.load(open(sys.argv[1])); assert p["completed"] and p["passed"] and not p["test_shards_read"]' "${output}/summary.json"
}

for architecture in m3 m5; do
  for band in small medium large; do
    for rate_spec in "3e-4 3em4" "1e-3 1em3"; do
      read -r rate rate_label <<< "${rate_spec}"
      run_task "${architecture}" "${band}" "${rate}" "${rate_label}" physical_mse physical
      run_task "${architecture}" "${band}" "${rate}" "${rate_label}" weighted_normalized_mse weighted_h_over_g
    done
  done
done

aggregate="${grid}/aggregate_v2"
if [[ ! -f "${aggregate}/summary.json" ]]; then
  mkdir -p "${aggregate}"
  python -u scripts/pair_mappers/aggregate_stage5_neural_calibration.py \
    --grid-dir "${grid}" --output-dir "${aggregate}"
fi
python -c 'import json; p=json.load(open("artifacts/pair_stage5/sio2_neural_calibration_grid_v2/aggregate_v2/summary.json")); assert p["completed"] and p["passed"] and p["configuration_count"] == 24'
