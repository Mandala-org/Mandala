# Three-Way Comparison: DeepH-E3 vs Minimal Study vs Main src

This is the detailed side-by-side comparison for:

- DeepH-E3 reference (`external/DeepH-E3`),
- explicit study (`studies/minimal_overfit_study`),
- main production-like code (`src/`, `scripts/train_silicon.py`).

## A. Data handling

### A1. Input format and preprocessing

DeepH-E3:

- Expects preprocessed folders and graph-cache workflow (`AijData`), with optional direct graph loading.
  - `external/DeepH-E3/deephe3/data.py:17`
  - `external/DeepH-E3/deephe3/data.py:85`
- Uses parser/config-driven train/eval split from one dataset object.
  - `external/DeepH-E3/deephe3/kernel.py:802`

Minimal study:

- Loads exactly one snapshot by default via `Snapshot.from_openmx(...)`.
  - `studies/minimal_overfit_study/overfit_water_minimal.py:442`
- Many manual toggles can change coordinate handling before graph/target canonicalization.
  - `studies/minimal_overfit_study/overfit_water_minimal.py:285`
  - `studies/minimal_overfit_study/overfit_water_minimal.py:503`

Main src:

- Uses `DatasetFactory` and cached per-snapshot processing via `E3GNNDataset`.
  - `src/data/factory.py:36`
  - `src/data/gnn_dataset.py:84`

Impact:

- DeepH-E3 and main src are multi-snapshot by design; minimal script is overfit-oriented and easier to accidentally diverge from dataset protocol.

### A2. Basis and coordinate conversion

DeepH-E3:

- Maintains OpenMX matrix basis and performs OpenMX/wiki/e3nn conversion inside output decomposition modules (`Rotate`, `e3TensorDecomp`).
  - `external/DeepH-E3/deephe3/e3modules.py:44`
  - `external/DeepH-E3/deephe3/e3modules.py:315`

Minimal + src:

- Convert whole snapshot to e3nn basis early (`Snapshot.from_openmx(..., convention='e3nn')`).
  - `studies/minimal_overfit_study/overfit_water_minimal.py:445`
  - `src/data/snapshot.py:735`
- Main src also applies coordinate basis permutation in snapshot conversion.
  - `src/data/snapshot.py:366`

Impact:

- Same final convention can be achieved, but the point in pipeline where conversion occurs differs. This is a major place for subtle mismatch.

## B. Graph construction and edge ordering

### B1. Edge generation

DeepH-E3:

- Graph features from `(distance, displacement)` with periodic shifts and key tuples.
  - `external/DeepH-E3/deephe3/graph.py:68`
  - `external/DeepH-E3/deephe3/graph.py:266`

Minimal:

- Uses ASE `neighbor_list` then explicit self-edge insertion and canonical re-sorting.
  - `studies/minimal_overfit_study/overfit_water_minimal.py:592`
  - `studies/minimal_overfit_study/overfit_water_minimal.py:612`

Main src:

- Same ASE-based approach and canonical sorting.
  - `src/data/graph_features.py:60`
  - `src/data/graph_features.py:149`

### B2. Canonicalization policy

Minimal + src:

- Deterministic policy: self-edges first, off-diagonals sorted by `(distance, sx, sy, sz, src, dst)`.
  - `studies/minimal_overfit_study/common.py:1523`
  - `src/data/snapshot.py:142`
  - `src/data/graph_features.py:118`

DeepH-E3:

- No explicit equivalent canonical sorter in the same style; it relies on generated graph/key order being internally consistent.
  - `external/DeepH-E3/deephe3/graph.py:244`

Impact:

- Minimal/src guarantee deterministic tie-breaking order. DeepH-E3 consistency is internal but not expressed as a separate canonicalization routine.

### B3. Edge-set strictness

Minimal:

- Has strict checks but exact edge-set equality is optional (`--require-exact-edge-match`, default false).
  - `studies/minimal_overfit_study/overfit_water_minimal.py:343`
  - `studies/minimal_overfit_study/strict_checks.py:63`

Main src:

- Safety checks can assert edge equality during loss only if enabled.
  - `src/net/e3gnn.py:340`

DeepH-E3:

- Loss uses masked labels generated from graph and Aij mapping.
  - `external/DeepH-E3/deephe3/data.py:206`
  - `external/DeepH-E3/deephe3/kernel.py:648`

Impact:

- If exact matching is not enforced, truncation/prefix matching can hide dropped or extra edges.

## C. Network architecture

### C1. Message passing blocks

DeepH-E3:

- Edge + node update blocks with EquiConv, gating, norms, optional self-connections.
  - `external/DeepH-E3/deephe3/model.py:119`
  - `external/DeepH-E3/deephe3/model.py:225`

Minimal:

- Similar conceptual two-step message block, explicit prints and diagnostics.
  - `studies/minimal_overfit_study/common.py:309`

Main src:

- `MessageBlock` abstraction with edge update then node update.
  - `src/net/layers.py:567`

### C2. Head/readout

DeepH-E3:

- Uses `e3TensorDecomp` (`get_H`, `get_net_out`) with Wigner-3j based algebra and OpenMX/wiki conversion.
  - `external/DeepH-E3/deephe3/e3modules.py:315`
  - `external/DeepH-E3/deephe3/e3modules.py:348`

Minimal:

- Per-edge-type projections with separate diag/offdiag/shifted-self branches, optional magnitude factorization.
  - `studies/minimal_overfit_study/common.py:477`

Main src:

- `DeepHead`: shared trunk + per-pair heads, optional log-scale branch.
  - `src/net/heads.py:21`

Impact:

- Readout mathematics are not the same style. DeepH-E3 couples output basis conversion to decomposition; minimal/src separate mapper conversions from head output.

## D. Spherical harmonics and radial encoding

DeepH-E3:

- SH called with `normalize=True`, `normalization='component'` and vector reordered to `(y,z,x)`.
  - `external/DeepH-E3/deephe3/model.py:353`
  - `external/DeepH-E3/deephe3/model.py:401`

Minimal:

- Computes `edge_vec_norm = edge_vec / ||edge_vec||` then calls SH with `normalize=False`.
  - `studies/minimal_overfit_study/overfit_water_minimal.py:664`
  - `studies/minimal_overfit_study/overfit_water_minimal.py:670`

Main src:

- SH with `normalize=True`, `normalization='component'`.
  - `src/data/graph_features.py:193`

Impact:

- This is one of the strongest candidates for a major quality gap. Minimal SH input/output normalization path is different from both DeepH-E3 and main src.

## E. Loss function and optimization

### E1. Loss aggregation style

DeepH-E3:

- Global masked MSE over all selected elements (`torch.masked_select(...).mean()`).
  - `external/DeepH-E3/deephe3/utils.py:40`

Minimal:

- Per-key MSE sums (`loss += mse(key)`), not weighted by number of elements per key.
  - `studies/minimal_overfit_study/overfit_water_minimal.py:1477`

Main src:

- Same per-key additive accumulation style in matrix losses.
  - `src/net/e3gnn.py:345`

Impact:

- Per-key equal weighting can distort objective versus DeepH-E3's element-wise global mean, especially on imbalanced edge-type counts.

### E2. Scheduler behavior

DeepH-E3:

- Uses `RevertDecayLR` wrapper and step after epoch-level val loss.
  - `external/DeepH-E3/deephe3/kernel.py:299`

Minimal:

- `ReduceLROnPlateau` stepped inside training loop before `optimizer.step()`.
  - `studies/minimal_overfit_study/overfit_water_minimal.py:1546`
  - `studies/minimal_overfit_study/overfit_water_minimal.py:1601`

Main src:

- Scheduler managed by Lightning and configured against monitored metric.
  - `src/net/e3gnn.py:536`

Impact:

- LR dynamics differ significantly between DeepH-E3 and minimal; this alone can change convergence quality.

## F. Metrics and evaluation

DeepH-E3:

- Core train criterion is masked MSE.
- Analyzer reports global MSE/MAE and block/error plots from `test_result.h5`.
  - `external/DeepH-E3/deephe3/analyzer.py:161`
  - `external/DeepH-E3/deephe3/analyzer.py:245`

Minimal:

- Uses custom `compute_detailed_metrics` including modified metrics with `mu_H` correction.
  - `studies/minimal_overfit_study/common.py:959`
- Also logs DOS/eigen and distance curves.
  - `studies/minimal_overfit_study/overfit_water_minimal.py:2006`
  - `studies/minimal_overfit_study/overfit_water_minimal.py:2078`

Main src:

- Matrix losses plus optional observables (energy/electron count/forces/stress).
  - `src/net/e3gnn.py:383`

Impact:

- You are not comparing a single identical scalar metric pipeline by default. Reported quality can differ even if raw prediction quality were closer.

## G. Symmetrization behavior

Minimal:

- Symmetrizes ground truth Hamiltonian before training objective.
  - `studies/minimal_overfit_study/overfit_water_minimal.py:559`
- Symmetrizes predictions before metrics.
  - `studies/minimal_overfit_study/overfit_water_minimal.py:1308`

Main src:

- Symmetrizes matrix output in loss path when configured.
  - `src/net/e3gnn.py:312`

DeepH-E3:

- No equivalent explicit same-step symmetrization in the same style found in kernel/model path.

Impact:

- Different symmetry constraints can alter both optimization landscape and reported metrics.

## H. Existing in-repo comparison attempts

- Old folder already exists: `studies/deeph-e3_comparison`.
- It includes a basis-equivalence test with zero reported difference for tested path:
  - `studies/deeph-e3_comparison/basis_equivalence_report.yaml`
- Equivariance comparison script was intentionally left incomplete.
  - `studies/deeph-e3_comparison/run_equivariance_study.py:10`

This is useful prior art, but not yet a full three-way training/evaluation parity harness.
