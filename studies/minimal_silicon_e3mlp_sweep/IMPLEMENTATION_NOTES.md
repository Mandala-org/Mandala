# Implementation Notes

## Fallbacks Implemented

These are the fallbacks or compatibility paths used while building the new study.

1. Energy evaluation fallback for density-only runs:
   If `hamiltonian` is not predicted, energy is computed as `Tr(H_gt D_pred)` using the ground-truth Hamiltonian.
   File: [train_density_energy_minimal.py](/home/bartek/casus/mandala/studies/minimal_silicon_e3mlp_sweep/train_density_energy_minimal.py)

2. Shifted-self head scale fallback:
   If no separate shifted-self output scale is provided, it falls back to the diagonal head scale.
   File: [common.py](/home/bartek/casus/mandala/studies/minimal_silicon_e3mlp_sweep/common.py)

3. Global-split smoke-test fallback during local validation:
   For local dry-runs, I used `--num-train/--num-val` because the small repo-local silicon folder does not match the HPC temperature-split layout expected by the cluster dataset.
   This was only used for local validation, not in the sweep config.

4. Import precedence fallback:
   The new study script forces its own folder to the front of `sys.path` so `common.py`, `strict_checks.py`, and `detailed_logging.py` resolve to the new study-local copies instead of older study modules with the same names.
   File: [train_density_energy_minimal.py](/home/bartek/casus/mandala/studies/minimal_silicon_e3mlp_sweep/train_density_energy_minimal.py)

The temporary basic-metric fallback to unaligned metrics was removed. The current code uses explicit edge-set alignment and then the strict aligned metric path.

## Issues Encountered And Resolutions

1. The copied silicon trainer imported the wrong `common.py`.
   Cause: the original script prepended older study folders to `sys.path`, so the copied script still imported the old module.
   Resolution: inserted the new study directory at the front of `sys.path`.

2. Density-only training was rejected by the original validation rules.
   Cause: the original script required `hamiltonian` in `--matrix-targets`.
   Resolution: removed that requirement and changed energy handling so density-only runs are valid.

3. Energy metrics originally required both predicted Hamiltonian and predicted density.
   Cause: the original code only computed energy from `H_pred` and `D_pred`.
   Resolution: added density-only energy evaluation with `H_gt` and `D_pred`, while keeping `train_on_energy=false` in this sweep.

4. The aligned basic matrix metric path failed with shape mismatches.
   Cause: the model predicts on the full graph edge set, while target block matrices can live on a smaller aligned subset.
   Resolution: added explicit alignment helpers that project predicted `BlockMatrix` and `IrrepsBlockData` objects onto the exact target edge set before computing strict aligned metrics.

5. The first local dry-run used a temperature split that did not exist in the repo-local silicon folder.
   Cause: local folder only has `2700K` and `900K`, not the large HPC dataset structure.
   Resolution: switched the smoke test to global split mode for local validation only.

6. The first bare-python smoke test failed because `torch` was not installed in the default environment.
   Cause: local shell `python` was outside the project venv.
   Resolution: used `source mandala-venv/bin/activate` for all real validation.

7. The temporary relaxed matrix-metric fallback was not acceptable.
   Cause: it hid the real alignment issue.
   Resolution: removed it entirely and replaced it with explicit prediction-to-target alignment.

8. Git status over the studies tree hit an LFS clean-filter failure unrelated to this study.
   Cause: existing tracked artifact files under `studies/e3mlp_investigation/artifacts/`.
   Resolution: ignored it for implementation work; it did not affect the new study files.

## Functional Differences vs `minimal_silicon_study/train_silicon_minimal.py`

This is the full functional diff at the study-script level.

1. New study location:
   Old: [train_silicon_minimal.py](/home/bartek/casus/mandala/studies/minimal_silicon_study/train_silicon_minimal.py)
   New: [train_density_energy_minimal.py](/home/bartek/casus/mandala/studies/minimal_silicon_e3mlp_sweep/train_density_energy_minimal.py)

2. New network implementation source:
   Old script imported the shared minimal network path used by the older studies.
   New script uses a study-local [common.py](/home/bartek/casus/mandala/studies/minimal_silicon_e3mlp_sweep/common.py) fork.

3. Density-only default target:
   Old default matrix targets: `hamiltonian,overlap,density`
   New default matrix target: `density`

4. No Hamiltonian target requirement:
   Old script required `hamiltonian` in `--matrix-targets`.
   New script allows density-only runs.

5. Energy objective behavior:
   Old script computed energy only from predicted Hamiltonian and predicted density.
   New script can evaluate energy from ground-truth Hamiltonian and predicted density.

6. Energy training in sweep:
   Old study supported `train_on_energy=True`.
   New sweep is configured with `train_on_energy=false`.

7. Number-of-electrons defaults:
   Old defaults enabled number-of-electrons metrics/loss.
   New defaults disable them for this sweep.

8. New E3MLP head variant selection:
   Added `--head-e3mlp-variant`.

9. New internal E3MLP refinement:
   Added `--internal-e3mlp-variant` and `--internal-e3mlp-layers`.
   Each message block can now refine node and edge hidden features with an E3MLP after the existing Gate+norm update.

10. New E3MLP normalization and init controls:
   Added `--e3mlp-pre-norm`
   Added `--e3mlp-output-scale`
   Added `--e3mlp-weight-init-scale`
   Added `--e3mlp-residual-scale`
   Added `--e3mlp-film-hidden-dim`

11. New diagonal/off-diagonal head scaling:
   Added `--head-diag-output-scale`
   Added `--head-offdiag-output-scale`
   This is new relative to the old minimal silicon study and was added to address the observed random-init scale mismatch.

12. Head architecture change:
   Old head used fixed `E3GateMLP` projections.
   New head uses the E3MLP zoo via `build_variant(...)`.

13. Internal architecture change:
   Old message blocks ended with Gate+norm.
   New message blocks optionally apply a second learned E3MLP refinement to node and edge states.

14. Logging change:
   New config logging prints head/internal E3MLP variants, layer counts, init scales, normalization choice, and diagonal/off-diagonal head scales.

15. Sweep support files:
   New study adds:
   [sweep_density_energy_bayes.yaml](/home/bartek/casus/mandala/studies/minimal_silicon_e3mlp_sweep/sweep_density_energy_bayes.yaml)
   [launch_sweep.sh](/home/bartek/casus/mandala/studies/minimal_silicon_e3mlp_sweep/launch_sweep.sh)
   [run_agent.sh](/home/bartek/casus/mandala/studies/minimal_silicon_e3mlp_sweep/run_agent.sh)

16. HPC defaults in sweep:
   New sweep uses the same HPC-style silicon dataset and snapshot-cache paths as the existing cluster silicon scripts.

17. Prediction-to-target metric alignment:
   Old copied path assumed aligned shapes for aligned metrics.
   New study explicitly aligns predicted matrices and irrep tensors to the target edge set before aligned metric computation.

18. Prediction-to-target loss alignment:
   Old copied path used the inherited prefix-based block-loss comparison.
   New study explicitly aligns predicted matrices to the target edge set before block-loss computation.

19. Import resolution:
   New study pins its own module directory first on `sys.path` to avoid accidental imports from older studies with the same filenames.

20. Checkpoint directory default:
   Old default: `studies/minimal_silicon_study/checkpoints`
   New default: `studies/minimal_silicon_e3mlp_sweep/checkpoints`

21. W&B project default:
   Old default project was the minimal silicon study project.
   New default project is `mandala-minimal-silicon-e3mlp-sweep`.

## Remaining Known Limitations

1. The copied final textual summary still prints the Hamiltonian-style summary header even in density-only mode.
   The energy metric itself is correct, but the textual label is inherited from the older reporting path.
