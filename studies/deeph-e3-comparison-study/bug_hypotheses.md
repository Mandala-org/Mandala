# Simple Bug and Regression Hypotheses

Ordered by expected impact on your reported underperformance.

## High probability

### 1. SH normalization mismatch in minimal study

Evidence:

- Minimal uses manual vector normalization and `normalize=False`:
  - `studies/minimal_overfit_study/overfit_water_minimal.py:664`
  - `studies/minimal_overfit_study/overfit_water_minimal.py:670`
- DeepH-E3/main src use `normalize=True, normalization='component'`:
  - `external/DeepH-E3/deephe3/model.py:355`
  - `src/data/graph_features.py:193`

Why this can be large:

- Changes the angular feature basis seen by TP layers and can systematically distort learned mapping.

Quick check:

- Log max/mean abs diff between SH tensors from minimal path vs `src/data/graph_features.py` path on same edges.

### 2. Loss weighting mismatch versus DeepH-E3

Evidence:

- DeepH-E3: one masked global mean MSE over all selected elements.
  - `external/DeepH-E3/deephe3/utils.py:40`
- Minimal/main src: sum of per-key MSE means.
  - `studies/minimal_overfit_study/overfit_water_minimal.py:1477`
  - `src/net/e3gnn.py:345`

Why this can be large:

- Equal key-level weighting overweights rare key groups and underweights dominant ones.

Quick check:

- Recompute minimal loss once as DeepH-style masked global mean and compare gradients/loss scale.

### 3. Scheduler step ordering in minimal loop

Evidence:

- `scheduler.step(loss)` before `optimizer.step()`:
  - `studies/minimal_overfit_study/overfit_water_minimal.py:1546`
  - `studies/minimal_overfit_study/overfit_water_minimal.py:1601`

Why this matters:

- Can change LR schedule timing and convergence behavior.

Quick check:

- Move scheduler step to post-optimizer and compare LR trace + convergence on same seed.

### 4. Non-exact edge matching default can hide edge-set mismatches

Evidence:

- Strict alignment default is prefix-only unless `--require-exact-edge-match`.
  - `studies/minimal_overfit_study/strict_checks.py:63`
  - `studies/minimal_overfit_study/overfit_water_minimal.py:343`

Why this matters:

- Missing/extra edges can be silently dropped via `min_n` truncation during loss.

Quick check:

- Always run with `--require-exact-edge-match` during parity experiments.

## Medium probability

### 5. Symmetrization policy mismatch

Evidence:

- Minimal symmetrizes target before training and prediction before metrics.
  - `studies/minimal_overfit_study/overfit_water_minimal.py:559`
  - `studies/minimal_overfit_study/overfit_water_minimal.py:1308`
- DeepH-E3 path does not mirror this exact policy in train kernel.

Why this matters:

- Alters objective and can hide antisymmetric prediction error.

Quick check:

- Compare training runs with and without pre-target symmetrization, using identical metrics computed both ways.

### 6. Convention toggles introduce accidental inconsistency

Evidence:

- `xyz-permutation`, `change-box`, `box-convention` options can alter geometry path.
  - `studies/minimal_overfit_study/overfit_water_minimal.py:285`

Why this matters:

- Easy to align graph path but misalign target path (or vice versa) if one branch misses a transform.

Quick check:

- Disable all convention knobs in parity baseline; re-enable one-by-one.

### 7. Analyze script inconsistency for `mu_H` when filtered metrics are requested

Evidence:

- `compute_metrics` in analyze script filters MAE/MSE but calls `compute_mu_H` on full matrices.
  - `studies/minimal_overfit_study/analyze_model.py:124`
  - `studies/minimal_overfit_study/analyze_model.py:161`

Why this matters:

- Reported modified metric can be inconsistent with filtered subset.

Quick check:

- Compute filtered and unfiltered `mu_H` side-by-side and report both.

## Lower probability but worth checking

### 8. DeepH-E3 split contamination risk for correlated structures

Evidence:

- Train/val split is random over structure index list unless constrained via `extra_validation`.
  - `external/DeepH-E3/deephe3/kernel.py:828`

Why this matters:

- For highly correlated snapshots, random split can overestimate generalization quality.
- This does not explain your overfit underperformance directly, but it affects fairness when comparing reported val metrics.

### 9. Prior local comparison script is incomplete for full equivalence claims

Evidence:

- Equivariance comparison intentionally skipped.
  - `studies/deeph-e3_comparison/run_equivariance_study.py:10`

Why this matters:

- Existing "zero-diff" basis report is useful but not sufficient for end-to-end training parity claims.
