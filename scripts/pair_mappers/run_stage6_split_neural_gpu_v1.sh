#!/usr/bin/env bash
set -euo pipefail

# H100 allocation: 16 host CPUs.
export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16

root=artifacts/pair_stage6/sio2_independent_v1
python -c 'import json; p=json.load(open("artifacts/pair_stage6/sio2_independent_v1/summary.json")); assert p["passed"] and p["offsite_task_count"] == 2 and p["onsite_neural_task_count"] == 12'

run_neural() {
  local grid="$1" task_id="$2" family="$3" descriptor_key="$4"
  local architecture="$5" band="$6" cap="$7" generator_count="$8"
  local hidden="$9" invariant_hidden="${10}" rank="${11}" scope="${12}"
  local baseline="${13}" steps="${14}" eval_interval="${15}" patience="${16}"
  local bond_radial="${17}" bond_lmax="${18}" onsite_batch="${19}" offsite_batch="${20}"
  local output="${grid}/runs/${task_id}"
  if [[ ! -f "${output}/summary.json" ]]; then
    mkdir -p "${output}"
    python -u scripts/pair_mappers/train_stage5_neural_calibration.py \
      --family "${family}" --descriptor-key "${descriptor_key}" \
      --descriptor-cache-dir "/bigdata/casus/wdm/hamiltonian_learning/data/pair_descriptor_hamiltonian/${family}_sio2_v2" \
      --descriptor-summary "artifacts/pair_stage2/${family}_sio2_precompute_v2/summary.json" \
      --descriptor-schemas "artifacts/pair_stage2/${family}_sio2_precompute_v2/descriptor_schemas.json" \
      --normalization artifacts/pair_stage2/sio2_descriptor_normalization_v2/normalization.json \
      --normalization-summary artifacts/pair_stage2/sio2_descriptor_normalization_v2/summary.json \
      --promotions artifacts/pair_stage3/sio2_descriptor_promotions_v2/promotions.json \
      --calibration-manifest "${grid}/calibration_manifest.json" --task-id "${task_id}" \
      --baseline-cache-dir /bigdata/casus/wdm/hamiltonian_learning/data/pair_descriptor_hamiltonian/sio2_baseline_cache_v2 \
      --baseline-summary artifacts/pair_stage1/sio2_baseline_cache_v2/summary.json \
      --shard-registry artifacts/pair_stage1/sio2_baseline_cache_v2/shards.csv \
      --range-envelope artifacts/pair_stage1/sio2_baseline_cache_v2/range_envelope.json \
      --output-dir "${output}" --architecture "${architecture}" \
      --resource-band "${band}" --descriptor-multiplicity-cap "${cap}" \
      --generator-multiplicity "${generator_count}" --hidden-multiplicity "${hidden}" \
      --hidden-l-max 4 --invariant-hidden "${invariant_hidden}" --factorization-rank "${rank}" \
      --bond-radial-count "${bond_radial}" --bond-l-max "${bond_lmax}" \
      --bond-cutoff-angstrom 6.5 --learning-rate 1e-3 --weight-decay 1e-6 \
      --range-loss-mode physical_mse --steps "${steps}" --eval-interval "${eval_interval}" \
      --early-stopping-evaluations "${patience}" --onsite-batch-size "${onsite_batch}" \
      --offsite-batch-size "${offsite_batch}" --evaluation-batch-size 2048 \
      --train-fraction 1.0 --target-scope "${scope}" --onsite-baseline "${baseline}" \
      --separate-descriptor-projections --seed 20260907 --device cuda \
      --envelope-floor-hartree 1e-8 --distance-bin-width-angstrom 0.5 \
      --float32-symmetry-tolerance 2e-5 --resume \
      2>&1 | tee -a "${output}/launcher.log"
  fi
  python scripts/pair_mappers/validate_stage6_scoped_run.py --run-dir "${output}"
}

offsite_grid="${root}/offsite"
for architecture in m3 m5; do
  run_neural "${offsite_grid}" "offsite_${architecture}_large_lr1em3" \
    d1 r6p5_spherical_bessel_high_n8_l6 "${architecture}" large \
    8 8 8 256 128 offsite none 10000 500 8 2 4 256 512
done

onsite_grid="${root}/onsite_neural"
families=(d1 d2 d3 d4)
keys=(
  r6p5_spherical_bessel_high_n8_l6
  r6p5_d2_degree10_o10_l10
  r6p5_d3_high_o8_l6
  r6p5_d4_spherical_bessel_high_n8_l6_b3
)
for index in 0 1 2 3; do
  family="${families[$index]}"
  key="${keys[$index]}"
  run_neural "${onsite_grid}" "onsite_${family}_medium_none" \
    "${family}" "${key}" m3 medium 4 4 4 128 64 onsite none 6000 250 10 1 0 512 1
  run_neural "${onsite_grid}" "onsite_${family}_medium_invariant_mean" \
    "${family}" "${key}" m3 medium 4 4 4 128 64 onsite invariant_mean 6000 250 10 1 0 512 1
  run_neural "${onsite_grid}" "onsite_${family}_large_invariant_mean" \
    "${family}" "${key}" m3 large 8 8 8 256 128 onsite invariant_mean 6000 250 10 1 0 512 1
done

for grid in "${offsite_grid}" "${onsite_grid}"; do
  aggregate="${grid}/aggregate_v1"
  if [[ ! -f "${aggregate}/summary.json" ]]; then
    mkdir -p "${aggregate}"
    python -u scripts/pair_mappers/aggregate_stage5_neural_calibration.py \
      --grid-dir "${grid}" --output-dir "${aggregate}"
  fi
  python -c 'import json,sys; p=json.load(open(sys.argv[1])); assert p["completed"] and p["passed"] and not p["test_shards_read"]' "${aggregate}/summary.json"
done
