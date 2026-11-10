#!/usr/bin/env bash
#SBATCH --job-name=r6-legacy-si
#SBATCH --partition=gpu-h100-casus
#SBATCH --account=casus
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=01:00:00
#SBATCH --chdir=/data/home2/brzoza73/casus/mandala_worktrees/retrain_v2
#SBATCH --output=/data/home2/brzoza73/casus/mandala_worktrees/retrain_v2/researchops/runs/r6_no_shifted_self_20261110/legacy_validation_%j.log
#SBATCH --error=/data/home2/brzoza73/casus/mandala_worktrees/retrain_v2/researchops/runs/r6_no_shifted_self_20261110/legacy_validation_%j.log
set -eo pipefail
source "$HOME/scripts/profiles/mandala.sh"
set -u
export PYTHONPATH="$PWD/src:$PWD"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4
export PYTHONHASHSEED=42 CUBLAS_WORKSPACE_CONFIG=:4096:8
run=researchops/runs/r6_no_shifted_self_20261110
printf 'Source commit: '
git rev-parse HEAD
# Read only: verify a running SiO2 recovery checkpoint under the new loader.
CUDA_VISIBLE_DEVICES='' python -u "$run/check_sio2_checkpoint.py"
python -u scripts/report/evaluate_validation_error_distribution.py \
  --checkpoint /bigdata/casus/wdm/hamiltonian_learning/models/checkpoints/silicon_hdo_energy_stage2_47h25/leafy-sweep-15-half_gt-coef3em3-lr3em4-47h25/best_model.pt \
  --dataset-kind silicon_scales \
  --data-path /bigdata/casus/wdm/hamiltonian_learning/data/perturbed_snapshots_Si \
  --scales '[1]' --num-train-per-scale 80 --num-val-per-scale 20 --data-split-seed 43 \
  --snapshot-cache-dir /bigdata/casus/wdm/hamiltonian_learning/data/perturbed_snapshots_Si/snapshot_cache \
  --dataset-device cpu --device cuda --convention e3nn --film-semantics legacy_2026_07 \
  --metadata-policy paper_2026_07 --density-convention paper_restricted_total \
  --output-json "$run/silicon_legacy_validation.json" \
  --scratch-dir "/bigdata/casus/wdm/hamiltonian_learning/runs/r6_no_shifted_self_20261110/$SLURM_JOB_ID"
python -u - <<'PY'
import json
from pathlib import Path
result = json.loads(Path('researchops/runs/r6_no_shifted_self_20261110/silicon_legacy_validation.json').read_text())
mae_mev = result['metrics']['hamiltonian']['mae'] * 1000
print(f'Legacy silicon H MAE: {mae_mev:.6f} meV', flush=True)
assert f'{mae_mev:.6f}' == '0.435164', mae_mev
print(json.dumps(result['metrics'], indent=2))
PY
