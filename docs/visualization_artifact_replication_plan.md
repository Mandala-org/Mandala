# Visualization And Artifact Replication Plan

This batch implements the scalar groundwork only:

- per-irrep scalar loss/MAE/MSE logging in `E3GNN`
- edge partition counts and ratios (`diag`, `shifted_self`, `offdiag`)
- activation-magnitude summaries in `BenchmarkCallback`, including flattened logger metrics
- config flags for future richer artifact generation:
  - `log_per_irrep_metrics`
  - `log_per_irrep_images`

## What should stay in the training loop

- Cheap scalar metrics that are useful for monitoring and sweep comparison
- Activation-magnitude summaries and benchmark YAML reports
- Per-irrep scalar metrics that can be aggregated by the logger
- Stable run metadata needed to identify representative samples later

## What should move to artifact callbacks or offline analysis

- Per-irrep block visualizations
- Prediction-vs-target matrix image dumps
- Distance-bucketed or block-class-bucketed error plots
- Sweep-level comparison dashboards
- Large image artifacts for activation distributions or irrep decompositions

## Recommended implementation split

### Phase 1: lightweight callback hooks in main code

- Add a dedicated artifact callback instead of mixing image generation into `training_step`
- Select a small stable validation subset by deterministic sample ids
- Persist raw payloads needed for offline plotting:
  - predicted blocks
  - target blocks
  - per-irrep projected blocks
  - edge metadata (`shift`, `src`, `dst`, pair key)
- Store them under a run-local directory such as:
  - `artifacts/<run_name>/val_samples/epoch_0005/...`

### Phase 2: reusable plotting utilities

- Add plotting utilities that operate on persisted payloads, not on live model objects
- First plot families to port from the studies:
  - per-irrep matrix images for `k = (0, 0, 0)`
  - prediction/target/error triptychs
  - per-irrep validation metric trend plots
  - activation-magnitude histograms or summaries
  - block-class metrics for `diag`, `shifted_self`, and `offdiag`

### Phase 3: sweep summaries

- Reuse the `studies/e3mlp_investigation/scripts/` pattern for summary generation
- Generate sweep-level CSV or parquet tables first
- Build summary plots from those tables instead of scraping logs directly

## Minimal APIs worth adding next

- A stable sample identifier on dataset outputs
- A callback-friendly method on `E3GNN` to expose:
  - matrix predictions
  - irrep projections
  - edge partition masks
- A small artifact writer interface that can write:
  - YAML/JSON summaries
  - tensor payloads
  - PNGs produced by plotting utilities

## Suggested first artifact to port

- Per-irrep Hamiltonian validation images for a fixed H2O sample and `k = (0, 0, 0)`

Reason:

- It is the most directly comparable artifact from the studies
- It exercises the new irrep projector path
- It is valuable for checking data treatment and head behavior, not just scalar loss
