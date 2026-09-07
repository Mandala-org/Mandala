#!/usr/bin/env bash
set -euo pipefail

# H100 allocation: 16 host CPUs.
export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16

grid=artifacts/pair_stage6/sio2_full_training_v1
if [[ ! -f "${grid}/summary.json" ]]; then
  mkdir -p "${grid}"
  python -u scripts/pair_mappers/freeze_stage6_full_training.py \
    --calibration-aggregate artifacts/pair_stage5/sio2_neural_calibration_grid_v2/aggregate_v2/summary.json \
    --promotions artifacts/pair_stage3/sio2_descriptor_promotions_v2/promotions.json \
    --output-dir "${grid}"
fi
python -c 'import json; p=json.load(open("artifacts/pair_stage6/sio2_full_training_v1/summary.json")); assert p["passed"] and p["task_count"] == 6'

run_task() {
  local architecture="$1" weight="$2" weight_label="$3"
  local task_id="${architecture}_large_lr1em3_onsite${weight_label}"
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
      --resource-band large --descriptor-multiplicity-cap 8 \
      --generator-multiplicity 8 --hidden-multiplicity 8 --hidden-l-max 4 \
      --invariant-hidden 256 --factorization-rank 128 \
      --bond-radial-count 2 --bond-l-max 4 --bond-cutoff-angstrom 6.5 \
      --learning-rate 1e-3 --weight-decay 1e-6 --range-loss-mode physical_mse \
      --steps 10000 --eval-interval 500 --early-stopping-evaluations 6 \
      --onsite-batch-size 256 --offsite-batch-size 512 \
      --evaluation-batch-size 2048 --train-fraction 1.0 \
      --onsite-loss-weight "${weight}" --seed 20260907 --device cuda \
      --envelope-floor-hartree 1e-8 --distance-bin-width-angstrom 0.5 \
      --float32-symmetry-tolerance 2e-5 --resume \
      2>&1 | tee -a "${output}/launcher.log"
  fi
  python -c 'import json,sys; p=json.load(open(sys.argv[1])); assert p["completed"] and p["passed"] and not p["test_shards_read"]' "${output}/summary.json"
}

for architecture in m3 m5; do
  run_task "${architecture}" 0.05 0p05
  run_task "${architecture}" 0.2 0p2
  run_task "${architecture}" 0.5 0p5
done

aggregate="${grid}/aggregate_v1"
if [[ ! -f "${aggregate}/summary.json" ]]; then
  mkdir -p "${aggregate}"
  python -u scripts/pair_mappers/aggregate_stage5_neural_calibration.py \
    --grid-dir "${grid}" --output-dir "${aggregate}"
fi
python -c 'import json; p=json.load(open("artifacts/pair_stage6/sio2_full_training_v1/aggregate_v1/summary.json")); assert p["completed"] and p["passed"] and p["configuration_count"] == 6 and not p["test_shards_read"]'
