# Execution Plan for DeepH-E3 vs Minimal vs src Comparison

This plan is designed to isolate the key mismatch causing poor overfit performance, with strict reproducibility and minimal confounders.

## Phase 0: Lock baseline and environment

1. Freeze software and seed settings for each pipeline.
2. Lock dataset/snapshot set and edge cutoff.
3. Disable all optional convention toggles in minimal baseline:
   - `xyz-permutation=012`
   - `change-box=right` only if needed by baseline convention
   - `box-convention=rows`
4. Enable strict edge-set checks in minimal:
   - `--require-exact-edge-match`

Deliverable:

- `baseline_manifest.yaml` with versions, commit SHAs, and run commands.

## Phase 1: Convention and basis parity (no learning)

Objective:

- Prove that target representations are numerically equivalent across pipelines for the same block inputs.

Steps:

1. Compare OpenMX->e3nn converted blocks from:
   - DeepH-E3 `Rotate.openmx2wiki_H` path,
   - main src `OpenMXE3NNConverter` path.
2. Compare irrep-vector maps from:
   - DeepH-E3 `e3TensorDecomp.get_net_out`,
   - `BlockIrrepMapper.blocks_to_vectors` (allowing known reindex/permutation if needed).
3. Round-trip checks in each framework (`block->vector->block`).

Acceptance criteria:

- max abs diff <= 1e-6 for parity tests in float32 (or tighter in float64).

Deliverables:

- `reports/phase1_basis_parity.yaml`
- `reports/phase1_basis_parity_plots.png`

## Phase 2: Graph and SH feature parity

Objective:

- Prove graph edges and SH/radial features match for the same snapshot and cutoff.

Steps:

1. Build edges with DeepH-E3 graph path and src/minimal path.
2. Compare edge sets exactly (not prefix), including multiplicities.
3. Compare edge displacements and lengths.
4. Compare SH tensors under current minimal setting and under DeepH-compatible setting.
5. Compare radial embeddings (shape and value ranges).

Acceptance criteria:

- exact edge-set equality,
- SH and radial diffs near numerical precision when using same API mode.

Deliverables:

- `reports/phase2_graph_sh_parity.yaml`
- `reports/phase2_edge_mismatch_examples.json` (empty on pass)

## Phase 3: Loss-definition parity on fixed predictions

Objective:

- Quantify how objective definitions differ before training changes.

Steps:

1. Use one frozen prediction tensor set.
2. Compute:
   - DeepH-style masked global MSE,
   - minimal per-key-sum MSE,
   - src per-key-sum matrix losses.
3. Compare scalar loss values and per-parameter gradient norms for one backward pass.

Acceptance criteria:

- Explicit quantified delta between objectives and gradient directions.

Deliverables:

- `reports/phase3_loss_parity.yaml`
- `reports/phase3_gradient_cosine_similarity.yaml`

## Phase 4: Minimal script ablations (single snapshot overfit)

Objective:

- Isolate the primary contributor to the two-order gap.

Ablation order (one change at a time):

1. SH mode parity (`normalize=True, normalization='component'` path).
2. Loss parity (global masked MSE emulation).
3. Scheduler step ordering fix.
4. Symmetrization policy alignment.
5. Optional branches off (magnitude factorization, scalar-head variants, extra logging overhead).

For each run:

- Same seed, epochs, lr, cutoff, strict edge checks.
- Report final train MAE/MSE and convergence speed.

Deliverables:

- `reports/phase4_ablation_table.csv`
- `reports/phase4_ablation_summary.md`

## Phase 5: End-to-end training parity against DeepH-E3 protocol

Objective:

- Reproduce a comparable small benchmark under matching train/val setup.

Steps:

1. Build a small, contamination-controlled dataset split.
2. Align training objective, metric definitions, and logging.
3. Train both pipelines with matched core hyperparameters.
4. Compare:
   - train loss trajectory,
   - val metric trajectory,
   - block-type/per-irrep error slices.

Deliverables:

- `reports/phase5_training_parity.json`
- `reports/phase5_curves.png`

## Phase 6: Final diagnosis and fix recommendation

Objective:

- Produce one ranked explanation list with confidence levels and code-level fixes.

Deliverables:

- `final_diagnosis.md` including:
  - confirmed root cause(s),
  - secondary contributors,
  - patch list with expected impact,
  - residual unknowns.

## Priority checks to run first (fastest signal)

1. Phase 2 SH parity check (minimal vs src feature path).
2. Phase 3 loss parity check (global masked MSE vs per-key sum).
3. Scheduler-step-order A/B on minimal overfit run.

These three should quickly indicate whether the gap is mainly feature-convention, objective weighting, or optimization schedule.
