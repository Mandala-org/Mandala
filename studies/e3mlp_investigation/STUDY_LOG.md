# E3MLP Investigation Study Log

This file is append-only.

Rules:

- Every meaningful run gets a dated entry.
- Every entry should link the most important artifacts.
- If a preliminary result is used to justify a next step, mark it as provisional.
- If a larger run overturns a smaller run, explicitly note the override.

## 2026-04-09

Initialized study folder and fixed plan.

Canonical plan:

- [PLAN.md](/home/bartek/casus/mandala/studies/e3mlp_investigation/PLAN.md)

Current status:

- no new experiments executed yet
- no claims accepted yet

Open starting assumptions to test:

- proper scaling and initialization may rescue several E3MLP families
- practical best depth may be `4-6`, with required stability at `10-12`
- no-GNN silicon pretext tasks should be built around multiple information-incorporation schemes, not a single aggregator

Plan update:

- preliminary/synthetic/smoke studies will use a smaller `l_max=4` irreps
- real silicon studies will use the full target hidden irreps
- dry-runs and smoke tests should be followed by background execution while other work continues
- graphical artifacts are mandatory for every experiment family


## 2026-04-09 16:27 - smoke_stability_01

Executed stability micro-study.

- run dir: [smoke_stability_01](/home/bartek/casus/mandala/studies/e3mlp_investigation/artifacts/smoke_stability_01)
- hidden irreps preset: `preliminary`
- hidden irreps: `16x0e+16x0o+8x1e+8x1o+4x2e+4x2o+4x3e+4x3o+4x4e`
- smoke test: `True`

Top provisional rows:
- `gatemagnitudes` depth=4 output_scale=1.0 weight_init_scale=1.0 residual_scale=0.1 loss=1.0167e+00 ratio=0.297 grad=3.469e-02 nan=False
- `normact` depth=4 output_scale=1.0 weight_init_scale=1.0 residual_scale=0.1 loss=1.0542e+00 ratio=0.355 grad=3.521e-02 nan=False
- `film` depth=4 output_scale=1.0 weight_init_scale=1.0 residual_scale=0.1 loss=1.4337e+00 ratio=0.682 grad=1.825e+00 nan=False
- `resnormact` depth=4 output_scale=1.0 weight_init_scale=1.0 residual_scale=0.1 loss=1.7435e+00 ratio=0.918 grad=1.529e-01 nan=False
- `bilinear` depth=4 output_scale=1.0 weight_init_scale=1.0 residual_scale=0.1 loss=1.9622e+00 ratio=0.997 grad=1.675e-01 nan=False

Artifacts:
- [summary.csv](/home/bartek/casus/mandala/studies/e3mlp_investigation/artifacts/smoke_stability_01/summary.csv)
- [loss curves](/home/bartek/casus/mandala/studies/e3mlp_investigation/artifacts/smoke_stability_01/plots/loss_curves.png)
- [gradient curves](/home/bartek/casus/mandala/studies/e3mlp_investigation/artifacts/smoke_stability_01/plots/gradient_curves.png)

Status:
- provisional local result; larger runs may override it


## 2026-04-09 16:31 - smoke_synth_teacher_01

Executed synthetic teacher study.

- run dir: [smoke_synth_teacher_01](/home/bartek/casus/mandala/studies/e3mlp_investigation/artifacts/smoke_synth_teacher_01)
- teacher kind: `mixed`
- hidden irreps preset: `preliminary`
- hidden irreps: `16x0e+16x0o+8x1e+8x1o+4x2e+4x2o+4x3e+4x3o+4x4e`
- smoke test: `True`

Top provisional rows:
- `gatemagnitudes` depth=2 loss=mse val_mae=5.3748e-01 val_mse=4.9529e-01
- `gatemagnitudes` depth=2 loss=mae val_mae=5.3750e-01 val_mse=4.9534e-01
- `normact` depth=2 loss=mse val_mae=5.4237e-01 val_mse=5.0220e-01
- `normact` depth=2 loss=mae val_mae=5.4251e-01 val_mse=5.0230e-01
- `film` depth=2 loss=mae val_mae=7.0455e-01 val_mse=8.8167e-01

Artifacts:
- [summary.csv](/home/bartek/casus/mandala/studies/e3mlp_investigation/artifacts/smoke_synth_teacher_01/summary.csv)
- [validation loss curves](/home/bartek/casus/mandala/studies/e3mlp_investigation/artifacts/smoke_synth_teacher_01/plots/val_loss_curves.png)
- [sample target vs prediction](/home/bartek/casus/mandala/studies/e3mlp_investigation/artifacts/smoke_synth_teacher_01/plots/sample_target_vs_prediction.png)

Status:
- provisional local result; larger runs may override it


## 2026-04-09 16:36 - silicon_pair_cache_2700K

Built the first silicon pair cache using the main dataset pipeline.

- output dir: [2700K](/home/bartek/casus/mandala/studies/e3mlp_investigation/cache/silicon_pairs/2700K)
- cutoff radius: `7.0`
- l_max: `4`
- n_radial: `64`
- snapshot: `2700K`

Artifacts:

- [pair_cache.pt](/home/bartek/casus/mandala/studies/e3mlp_investigation/cache/silicon_pairs/2700K/pair_cache.pt)
- [edge distance histogram](/home/bartek/casus/mandala/studies/e3mlp_investigation/cache/silicon_pairs/2700K/edge_distance_hist.png)
- [Hamiltonian norm vs distance](/home/bartek/casus/mandala/studies/e3mlp_investigation/cache/silicon_pairs/2700K/hamiltonian_Si-Si_norm_vs_distance.png)
- [Density norm vs distance](/home/bartek/casus/mandala/studies/e3mlp_investigation/cache/silicon_pairs/2700K/density_Si-Si_norm_vs_distance.png)

Use:

- reusable pair-level cache for later no-GNN silicon studies
- preliminary visual reference for how target magnitude depends on interatomic distance


## 2026-04-09 16:34 - smoke_synth_teacher_02

Executed synthetic teacher study.

- run dir: [smoke_synth_teacher_02](/home/bartek/casus/mandala/studies/e3mlp_investigation/artifacts/smoke_synth_teacher_02)
- teacher kind: `mixed`
- hidden irreps preset: `preliminary`
- hidden irreps: `16x0e+16x0o+8x1e+8x1o+4x2e+4x2o+4x3e+4x3o+4x4e`
- smoke test: `True`

Top provisional rows:
- `gatemagnitudes` depth=2 loss=mse val_mae=5.3748e-01 val_mse=4.9529e-01
- `gatemagnitudes` depth=2 loss=mae val_mae=5.3750e-01 val_mse=4.9534e-01
- `normact` depth=2 loss=mse val_mae=5.4237e-01 val_mse=5.0220e-01
- `normact` depth=2 loss=mae val_mae=5.4251e-01 val_mse=5.0230e-01
- `film` depth=2 loss=mae val_mae=7.0455e-01 val_mse=8.8167e-01

Artifacts:
- [summary.csv](/home/bartek/casus/mandala/studies/e3mlp_investigation/artifacts/smoke_synth_teacher_02/summary.csv)
- [validation loss curves](/home/bartek/casus/mandala/studies/e3mlp_investigation/artifacts/smoke_synth_teacher_02/plots/val_loss_curves.png)
- [sample target vs prediction](/home/bartek/casus/mandala/studies/e3mlp_investigation/artifacts/smoke_synth_teacher_02/plots/sample_target_vs_prediction.png)
- [per-irrep validation MAE](/home/bartek/casus/mandala/studies/e3mlp_investigation/artifacts/smoke_synth_teacher_02/plots/per_irrep_val_mae.png)

Status:
- provisional local result; larger runs may override it


## 2026-04-09 16:35 - smoke_synth_teacher_03

Executed synthetic teacher study.

- run dir: [smoke_synth_teacher_03](/home/bartek/casus/mandala/studies/e3mlp_investigation/artifacts/smoke_synth_teacher_03)
- teacher kind: `mixed`
- hidden irreps preset: `preliminary`
- hidden irreps: `16x0e+16x0o+8x1e+8x1o+4x2e+4x2o+4x3e+4x3o+4x4e`
- smoke test: `True`

Top provisional rows:
- `gatemagnitudes` depth=2 loss=mse val_mae=5.3748e-01 val_mse=4.9529e-01
- `normact` depth=2 loss=mse val_mae=5.4237e-01 val_mse=5.0220e-01

Artifacts:
- [summary.csv](/home/bartek/casus/mandala/studies/e3mlp_investigation/artifacts/smoke_synth_teacher_03/summary.csv)
- [validation loss curves](/home/bartek/casus/mandala/studies/e3mlp_investigation/artifacts/smoke_synth_teacher_03/plots/val_loss_curves.png)
- [sample target vs prediction](/home/bartek/casus/mandala/studies/e3mlp_investigation/artifacts/smoke_synth_teacher_03/plots/sample_target_vs_prediction.png)
- [per-irrep validation MAE](/home/bartek/casus/mandala/studies/e3mlp_investigation/artifacts/smoke_synth_teacher_03/plots/per_irrep_val_mae.png)

Status:
- provisional local result; larger runs may override it


## 2026-04-09 - silicon_pair_cache_900K

Built silicon pair cache.

- output dir: [900K](/home/bartek/casus/mandala/studies/e3mlp_investigation/cache/silicon_pairs/900K)
- cutoff radius: `7.0`
- l_max: `4`
- n_radial: `64`
- atoms: `216`
- edges: `16468`

Artifacts:
- [pair_cache.pt](/home/bartek/casus/mandala/studies/e3mlp_investigation/cache/silicon_pairs/900K/pair_cache.pt)
- [summary.json](/home/bartek/casus/mandala/studies/e3mlp_investigation/cache/silicon_pairs/900K/summary.json)
- [edge distance histogram](/home/bartek/casus/mandala/studies/e3mlp_investigation/cache/silicon_pairs/900K/edge_distance_hist.png)


## 2026-04-09 16:40 - infrastructure_milestone

Built fresh study infrastructure under [studies/e3mlp_investigation/](/home/bartek/casus/mandala/studies/e3mlp_investigation).

New code added:

- fresh irreps presets
- fresh E3MLP variant module
- stability micro-study runner
- synthetic teacher runner
- silicon pair-cache builder
- local background launchers
- cluster shell wrappers
- Slurm templates

Implemented new E3MLP ideas in the fresh codebase:

- invariant FiLM variant
- layer-scale residual behavior inside residual blocks

Current caution:

- long-running background process handling inside the sandbox is awkward, so run completion should be verified from artifacts rather than assumed from shell session state


## 2026-04-09 17:25 - cluster_batch1_prepared

Prepared the first interactive-GPU batch as plain bash launchers.

Scripts:

- [batch1_run01_stability_core.sh](/home/bartek/casus/mandala/studies/e3mlp_investigation/scripts/batch1_run01_stability_core.sh)
- [batch1_run02_stability_advanced.sh](/home/bartek/casus/mandala/studies/e3mlp_investigation/scripts/batch1_run02_stability_advanced.sh)
- [batch1_run03_synth_mixed.sh](/home/bartek/casus/mandala/studies/e3mlp_investigation/scripts/batch1_run03_synth_mixed.sh)
- [batch1_run04_synth_quadratic.sh](/home/bartek/casus/mandala/studies/e3mlp_investigation/scripts/batch1_run04_synth_quadratic.sh)
- [batch1_run05_synth_linear_baseline.sh](/home/bartek/casus/mandala/studies/e3mlp_investigation/scripts/batch1_run05_synth_linear_baseline.sh)
- [run_cluster_batch1_all.sh](/home/bartek/casus/mandala/studies/e3mlp_investigation/scripts/run_cluster_batch1_all.sh)

Validation:

- bash syntax checked for all batch-1 launchers
- Python study runners compiled successfully after adding explicit `--device` support to the synthetic teacher runner

Use:

- intended for interactive GPU jobs, not Slurm
- results from this batch should override current smoke-study impressions where they disagree


## 2026-04-09 16:58 - batch1_synth_linear_20260409_165309

Executed synthetic teacher study.

- run dir: [batch1_synth_linear_20260409_165309](/home/bartek/casus/mandala/studies/e3mlp_investigation/artifacts/batch1_synth_linear_20260409_165309)
- teacher kind: `linear`
- hidden irreps preset: `preliminary`
- hidden irreps: `16x0e+16x0o+8x1e+8x1o+4x2e+4x2o+4x3e+4x3o+4x4e`
- smoke test: `False`

Top provisional rows:
- `gatemagnitudes` depth=4 loss=mse val_mae=5.0740e-01 val_mse=4.3416e-01
- `gatemagnitudes` depth=4 loss=huber val_mae=5.0753e-01 val_mse=4.3447e-01
- `normact` depth=4 loss=mse val_mae=5.3080e-01 val_mse=4.7630e-01
- `normact` depth=4 loss=huber val_mae=5.3093e-01 val_mse=4.7668e-01
- `gatemagnitudes` depth=2 loss=mse val_mae=5.3321e-01 val_mse=4.9777e-01

Artifacts:
- [summary.csv](/home/bartek/casus/mandala/studies/e3mlp_investigation/artifacts/batch1_synth_linear_20260409_165309/summary.csv)
- [validation loss curves](/home/bartek/casus/mandala/studies/e3mlp_investigation/artifacts/batch1_synth_linear_20260409_165309/plots/val_loss_curves.png)
- [sample target vs prediction](/home/bartek/casus/mandala/studies/e3mlp_investigation/artifacts/batch1_synth_linear_20260409_165309/plots/sample_target_vs_prediction.png)
- [per-irrep validation MAE](/home/bartek/casus/mandala/studies/e3mlp_investigation/artifacts/batch1_synth_linear_20260409_165309/plots/per_irrep_val_mae.png)

Status:
- provisional local result; larger runs may override it
