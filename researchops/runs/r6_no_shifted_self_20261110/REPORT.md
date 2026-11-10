# Shifted-self retirement report

Branch: `referee-r6-no-shifted-self`; ROSI integration branch: `referee-retrain-v2`.

New training defaults to one pair head and rejects `separate_shifted_self=true` or `partial_train=shifted_self`, including nested overrides. Saved architectures use `E3GNN.load_from_checkpoint` or `from_checkpoint_config` to retain legacy modules and state-dict keys. `on_fit_start` also rejects training a restored legacy model. Default graphs omit shifted-self selectors; preprocessed cache version is v9.

## File-by-file changes

- `docs/theory/architecture.rst`: Document retired routing and legacy checkpoint compatibility.
- `docs/theory/objectives.rst`: Document retired routing and legacy checkpoint compatibility.
- `docs/user_guide/configuration.rst`: Document retired routing and legacy checkpoint compatibility.
- `researchops/runs/r6_no_shifted_self_20261110/check_sio2_checkpoint.py`: Record reproducible checks, logs or preserve active legacy continuation source.
- `researchops/runs/r6_no_shifted_self_20261110/legacy_validation.sbatch`: Record reproducible checks, logs or preserve active legacy continuation source.
- `researchops/runs/r6_no_shifted_self_20261110/logs/compat_workflow.log`: Record reproducible checks, logs or preserve active legacy continuation source.
- `researchops/runs/r6_no_shifted_self_20261110/logs/cpu.log`: Record reproducible checks, logs or preserve active legacy continuation source.
- `researchops/runs/r6_no_shifted_self_20261110/logs/focused.log`: Record reproducible checks, logs or preserve active legacy continuation source.
- `researchops/runs/r6_no_shifted_self_20261110/logs/gpu.log`: Record reproducible checks, logs or preserve active legacy continuation source.
- `researchops/runs/r6_no_shifted_self_20261110/preserve_running_chains.py`: Record reproducible checks, logs or preserve active legacy continuation source.
- `researchops/runs/r6_no_shifted_self_20261110/verification.json`: Record reproducible checks, logs or preserve active legacy continuation source.
- `scripts/benchmark_silicon_cif_scaling.py`: Construct saved models through the explicit checkpoint compatibility path.
- `scripts/benchmark_silicon_inference.py`: Construct saved models through the explicit checkpoint compatibility path.
- `scripts/check_compat_reload_equivalence.py`: Construct saved models through the explicit checkpoint compatibility path.
- `scripts/debug_hamiltonian_mae_discrepancy.py`: Construct saved models through the explicit checkpoint compatibility path.
- `scripts/debug_prediction_support.py`: Construct saved models through the explicit checkpoint compatibility path.
- `scripts/evaluate_checkpoint_materials.py`: Construct saved models through the explicit checkpoint compatibility path.
- `scripts/evaluate_checkpoint_snapshot.py`: Construct saved models through the explicit checkpoint compatibility path.
- `scripts/export_prediction.py`: Construct saved models through the explicit checkpoint compatibility path.
- `scripts/report/evaluate_physical_constraints.py`: Construct saved models through the explicit checkpoint compatibility path.
- `scripts/report/evaluate_validation_error_distribution.py`: Construct saved models through the explicit checkpoint compatibility path.
- `scripts/report/evaluate_validation_relative_metrics.py`: Construct saved models through the explicit checkpoint compatibility path.
- `scripts/report/export_paper_operators.py`: Construct saved models through the explicit checkpoint compatibility path.
- `scripts/reproduce_examples/README.md`: Explain historical compatibility and fresh v2 policy.
- `scripts/reproduce_examples/silicon_v2.json`: Disable separate shifted-self head for fresh v2 training.
- `scripts/reproduce_examples/sio2_v2.json`: Disable separate shifted-self head for fresh v2 training.
- `scripts/reproduce_examples/train.py`: Set training policy and reject retired configurations early.
- `scripts/reproduce_examples/zncusnses_v2.json`: Disable separate shifted-self head for fresh v2 training.
- `scripts/train.py`: Set training policy and reject retired configurations early.
- `src/data/edge_alignment.py`: Omit selectors for default routing and invalidate preprocessed caches.
- `src/data/gnn_dataset.py`: Omit selectors for default routing and invalidate preprocessed caches.
- `src/data/graph_features.py`: Omit selectors for default routing and invalidate preprocessed caches.
- `src/net/checkpoint_compat.py`: Document existing shared projection migration as legacy compatibility.
- `src/net/common.py`: Set training policy and reject retired configurations early.
- `src/net/e3gnn.py`: Explicit legacy construction/load boundary; guard training and inference routing.
- `src/net/evaluation.py`: Construct saved models through the explicit checkpoint compatibility path.
- `src/net/heads.py`: Keep named legacy modules and isolate legacy routing helper.
- `tests/equivariance/test_equivariance.py`: Adapt defaults or verify routing, legacy loading, cache behavior and training rejection.
- `tests/physics/test_pyscf_rotation_reference_files.py`: Adapt defaults or verify routing, legacy loading, cache behavior and training rejection.
- `tests/unit/data/test_gnn_dataset_cache.py`: Adapt defaults or verify routing, legacy loading, cache behavior and training rejection.
- `tests/unit/data/test_operator_dataset.py`: Adapt defaults or verify routing, legacy loading, cache behavior and training rejection.
- `tests/unit/data/test_structure_inference.py`: Adapt defaults or verify routing, legacy loading, cache behavior and training rejection.
- `tests/unit/net/test_e3gnn.py`: Adapt defaults or verify routing, legacy loading, cache behavior and training rejection.
- `tests/unit/net/test_e3gnn_configs.py`: Adapt defaults or verify routing, legacy loading, cache behavior and training rejection.
- `tests/unit/net/test_hybrid_pipelines.py`: Adapt defaults or verify routing, legacy loading, cache behavior and training rejection.
- `tests/unit/net/test_stress.py`: Adapt defaults or verify routing, legacy loading, cache behavior and training rejection.
- `tests/unit/scripts/test_reproduce_examples.py`: Adapt defaults or verify routing, legacy loading, cache behavior and training rejection.

## Commits

e6e60a9 Retire shifted-self heads for new training and preserve checkpoint inference
1868d73 Cover legacy shifted-self reloads and unit-cell invariant pair predictions
4815363 Adapt training setup tests and classify generated PySCF reference data
a419efe Add allocated paper-checkpoint and live SiO2 compatibility checks
4d72c0b Keep retired selectors in legacy inference and refresh stress fixture seed
4cf5617 Preserve active legacy retrain chains with verified immutable source
3570253 Use the supported full-validation CLI for the paper compatibility gate
171f993 Record passing CPU, A40 and checkpoint-resume verification

## Verification

- `pytest -n 16 -q tests/unit tests/physics tests/equivariance -m '(unit or physics or equivariance) and not (gpu or slow or large_data or network or integration)'`: {"passed": 734, "skipped": 68, "subtests_passed": 52, "warnings": 17, "seconds": 95.12, "log": "logs/cpu.log"}
- `MANDALA_RUN_GPU_TESTS=1 CUDA_VISIBLE_DEVICES=0 pytest -q tests/gpu/test_cuda_smoke.py`: {"passed": 2, "seconds": 7.85, "log": "logs/gpu.log"}
- `pytest -q tests/analysis/test_validation_error_distribution.py tests/workflow/test_epoch_boundary_resume.py`: {"passed": 11, "skipped": 2, "warnings": 16, "seconds": 15.48, "log": "logs/compat_workflow.log"}
- `git diff -z --name-only github/develop -- '*.py' | xargs -0 ruff check --isolated --select E9,F --ignore E741,F541`: {"result": "passed"}
- `git diff --check github/develop`: {"result": "passed"}
- `git fetch github develop && git rebase github/develop`: {"result": "up to date; no conflicts"}

ROSI: `sbatch researchops/runs/r6_no_shifted_self_20261110/legacy_validation.sbatch`, job 837502, COMPLETED 0:0. H MAE = 0.4351643092546589 meV (0.435164 to six decimals), 16 validation structures. See the sbatch file for the exact evaluation CLI. Strict CPU SiO2 load passed at epoch 151 with 2491 keys in job 837500.

Restart: cancelled 837005–837008 and preserved silicon checkpoint directory with `_separate_shifted_self_cancelled_20261110`. `python researchops/runs/publication_retrain_v2/submit.py --materials silicon --append` submitted 837505–837508. Registered Second Brain attempt 5.

## Choices and limits

- Active SiO2/Zn recipes used true, contrary to the brief: their continuing segments use an immutable original source snapshot, with 153 source hashes per chain and recipe hashes verified; their checkpoints and jobs are retained. New v2 recipes use false.
- Silicon first allocation shortened to 42 hours to fit before maintenance; remaining segments retain 45 hours and checkpoint signal handling is retained.
- Stress fixture seed changed to 751 after parameter initialization order changed; assertions and tolerances retained.
- Generated PySCF reference-file test classified large_data because its generated inputs are absent.
- Full retraining and final accuracy are not yet verified. CPU skips and the two workflow skips are recorded in logs.
- paper/, root README.md, frozen benchmark configs and historical recipes are untouched.
