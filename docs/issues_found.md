# Issues Found

This file lists concrete issues identified during repository review.

## 1. `compute_graph_features` return-value mismatch with `E3GNN.forward`
- Severity: High
- Location:
  - `src/data/graph_features.py:205`
  - `src/net/e3gnn.py:186`
- Problem:
  - `compute_graph_features()` returns 6 values, but `E3GNN.forward()` (when `precompute_edge_features=False`) unpacks 7 values (`index_gnn_cutoff` + `num_self_edges`).
- Impact:
  - Runtime unpacking error path when edge features are computed on the fly.

## 2. Potential `UnboundLocalError` for `E_true` / `N_true` in partial-GT observable block
- Severity: High
- Location:
  - `src/net/e3gnn.py:382`
  - `src/net/e3gnn.py:399`
  - `src/net/e3gnn.py:409`
  - `src/net/e3gnn.py:428`
- Problem:
  - `E_true` and `N_true` are defined only in enabled observable branches, but referenced later in the partial ground-truth block without guaranteed definition.
- Impact:
  - Fails for configurations where `log_partial_gt_observables`/`train_observables_on_gt` is enabled while one of the standard observable branches is disabled.

## 3. `E3GNNDataset.to()` uses `self.device` before initialization
- Severity: Medium
- Location:
  - `src/data/gnn_dataset.py:233`
- Problem:
  - `to()` compares with `self.device`, but `self.device` is not initialized in `__init__`.
- Impact:
  - First call to `dataset.to(...)` can raise `AttributeError`.

## 4. Incorrect type annotations for dataset sample structure
- Severity: Low
- Location:
  - `src/data/gnn_dataset.py:78`
  - `src/data/gnn_dataset.py:88`
  - `src/data/gnn_dataset.py:217`
- Problem:
  - Several annotations use `Tuple[Dict, Dict, Dict]`, but actual samples are `(x, y)` two-tuples.
- Impact:
  - Static typing confusion, weaker IDE/linter signal quality.

## 5. Snapshot serialization gates `stress` on `box` presence
- Severity: Medium
- Location:
  - `src/data/snapshot.py:238`
- Problem:
  - `_payload()` writes `stress` only if `box is not None`, not if `stress is not None`.
- Impact:
  - Incorrect serialization behavior for edge cases and unclear intent.

## 6. `Snapshot.from_openmx` gates `stress` assignment on `info.box`
- Severity: Medium
- Location:
  - `src/data/snapshot.py:555`
- Problem:
  - `snap.stress = info.stress if info.box.numel() else None` checks box instead of stress tensor presence.
- Impact:
  - Potentially drops or mis-handles stress data.

## 7. `_edge_displacements` docstring says minimal-image, implementation does not wrap
- Severity: Medium
- Location:
  - `src/data/snapshot.py:441`
  - `src/data/snapshot.py:461`
- Problem:
  - Method claims minimal-image displacement vectors, but wrapping logic is commented out and raw shifted delta is returned.
- Impact:
  - Possible distance/filtering inconsistency under periodic boundaries.

## 8. Hard-coded data paths in model analysis script
- Severity: Medium
- Location:
  - `studies/minimal_overfit_study/analyze_model.py:492`
  - `studies/minimal_overfit_study/analyze_model.py:494`
- Problem:
  - Uses fixed absolute file paths instead of checkpoint/config-provided paths.
- Impact:
  - Poor portability and brittle reproducibility.

## 9. Snapshot docstring claims constructor edge reordering that constructor does not perform
- Severity: Low
- Location:
  - `src/data/snapshot.py:14`
  - `src/data/snapshot.py:53`
- Problem:
  - Top-level description says edges are reordered on construction, but `__init__` does not canonicalize; canonicalization occurs in factory constructors (`from_openmx`, `from_fhiaims`).
- Impact:
  - Misleading documentation and incorrect expectations for direct constructor usage.

## 10. `lookup` type hints in block containers do not match real key shape
- Severity: Low
- Location:
  - `src/data/block_matrix.py:47`
  - `src/data/block_matrix.py:745`
- Problem:
  - Annotated as `Dict[Tuple[int, int], ...]`, but actual keys are 5-tuples `(sx, sy, sz, i, j)`.
- Impact:
  - Type-level inconsistency and avoidable maintenance friction.
