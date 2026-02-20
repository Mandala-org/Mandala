# `overfit_model_minimal` Detailed Summary and Differences

Note: there is no file named `studies/minimal_overfit_study/overfit_model_minimal.py` in this repository.
This summary is based on `studies/minimal_overfit_study/overfit_water_minimal.py`, which appears to be the intended target.

## 1. What this script is for

`studies/minimal_overfit_study/overfit_water_minimal.py` is a single-structure overfitting study driver.
It intentionally bypasses the main Lightning training stack and builds a smaller explicit pipeline around:

- one OpenMX snapshot (`H2O.matrix` + `H2O.info.out`)
- one custom model (`MinimalNetwork` in `studies/minimal_overfit_study/common.py`)
- one target matrix family (Hamiltonian only)
- heavy diagnostics/logging for debugging conventions and irreps behavior.

Entry point: `studies/minimal_overfit_study/overfit_water_minimal.py:81`.

## 2. CLI and configuration surface

The script defines a broad set of controls:

- data and convention controls (`--convention`, `--xyz-permutation`, `--change-box`, `--box-convention`) at `studies/minimal_overfit_study/overfit_water_minimal.py:100`, `studies/minimal_overfit_study/overfit_water_minimal.py:256`, `studies/minimal_overfit_study/overfit_water_minimal.py:262`, `studies/minimal_overfit_study/overfit_water_minimal.py:269`
- architecture controls (`--hidden-dim`, `--l-max`, `--hidden-irreps`, `--num-layers`, `--n-radial`) at `studies/minimal_overfit_study/overfit_water_minimal.py:106`, `studies/minimal_overfit_study/overfit_water_minimal.py:112`, `studies/minimal_overfit_study/overfit_water_minimal.py:118`, `studies/minimal_overfit_study/overfit_water_minimal.py:124`, `studies/minimal_overfit_study/overfit_water_minimal.py:136`
- training controls (`--lr`, `--num-epochs`, scheduler and clipping flags) at `studies/minimal_overfit_study/overfit_water_minimal.py:142`, `studies/minimal_overfit_study/overfit_water_minimal.py:148`, `studies/minimal_overfit_study/overfit_water_minimal.py:214`, `studies/minimal_overfit_study/overfit_water_minimal.py:220`, `studies/minimal_overfit_study/overfit_water_minimal.py:226`
- partial-target controls (`--partial-train`, `--train-on-irrep-parts`, `--normalize-blocks`, `--apply-cutoff-to-targets`) at `studies/minimal_overfit_study/overfit_water_minimal.py:232`, `studies/minimal_overfit_study/overfit_water_minimal.py:239`, `studies/minimal_overfit_study/overfit_water_minimal.py:276`, `studies/minimal_overfit_study/overfit_water_minimal.py:282`
- strict alignment checks (`--require-exact-edge-match`) at `studies/minimal_overfit_study/overfit_water_minimal.py:288`
- logging/media controls (`--log-*`, `--generate-video`) at `studies/minimal_overfit_study/overfit_water_minimal.py:166`, `studies/minimal_overfit_study/overfit_water_minimal.py:245`.

Parsed args are copied into a plain `CONFIG` dict at `studies/minimal_overfit_study/overfit_water_minimal.py:309`.

## 3. Data path and geometric convention handling

### 3.1 Snapshot loading

The script loads a single snapshot using:

- `Snapshot.from_openmx(..., convention=CONFIG["convention"], symmetrize_density=True)`
  at `studies/minimal_overfit_study/overfit_water_minimal.py:380`.

Optional cutoff can be applied directly to target matrices with `--apply-cutoff-to-targets` at `studies/minimal_overfit_study/overfit_water_minimal.py:391`.

### 3.2 Box interpretation and XYZ permutations

After loading:

- box may be transposed for `box_convention="cols"` at `studies/minimal_overfit_study/overfit_water_minimal.py:408`
- user-defined axis permutation is converted to a 3x3 permutation matrix using `permutation_to_matrix(...)` at `studies/minimal_overfit_study/common.py:1998`
- positions and box are transformed according to `change_box in {right,left,both}` at `studies/minimal_overfit_study/overfit_water_minimal.py:417`.

This is explicitly designed for convention sweeps across coordinate-handling choices.

### 3.3 Edge-order canonicalization for targets and graph

The script canonicalizes both:

- block-matrix target edge ordering (per key) at `studies/minimal_overfit_study/overfit_water_minimal.py:448`
- graph edge ordering at `studies/minimal_overfit_study/overfit_water_minimal.py:519`.

Sorting policy comes from `canonicalize_edge_order`:

- self-edges first
- off-diagonal edges sorted by `(distance, sx, sy, sz, src, dst)`
  in `studies/minimal_overfit_study/common.py:1074`.

### 3.4 Graph feature construction (manual)

Instead of `src/data/graph_features.py`, this script builds graph features inline:

- ASE neighbor list + explicit self-edge insertion at `studies/minimal_overfit_study/overfit_water_minimal.py:498`, `studies/minimal_overfit_study/overfit_water_minimal.py:503`
- edge vectors/distances from `(positions, box, edge_shift)` at `studies/minimal_overfit_study/overfit_water_minimal.py:539`
- spherical harmonics with `normalize=False` at `studies/minimal_overfit_study/overfit_water_minimal.py:577`
- radial soft one-hot features with extra scaling `* sqrt(n_radial)` at `studies/minimal_overfit_study/overfit_water_minimal.py:586`, `studies/minimal_overfit_study/overfit_water_minimal.py:594`
- edge type indices from `mapper.edge_type2idx` at `studies/minimal_overfit_study/overfit_water_minimal.py:614`.

### 3.5 Strict edge alignment checks

Before training, the script validates graph-vs-target alignment for H/S/D:

- calls at `studies/minimal_overfit_study/overfit_water_minimal.py:634`, `studies/minimal_overfit_study/overfit_water_minimal.py:644`, `studies/minimal_overfit_study/overfit_water_minimal.py:654`
- logic in `studies/minimal_overfit_study/strict_checks.py:47`.

`require_exact=False` checks exact prefix alignment used by current truncation logic.
`require_exact=True` also checks equal counts and key sets.

## 4. Model architecture used (`MinimalNetwork`)

The main model is instantiated at `studies/minimal_overfit_study/overfit_water_minimal.py:841` from `studies/minimal_overfit_study/common.py:583`.

### 4.1 Node encoder

`MinimalNodeEncoder`:

- learned embedding over element ids
- outputs pure scalars `hidden_scalar_dim x 0e`
  at `studies/minimal_overfit_study/common.py:178`.

### 4.2 Edge encoder

`MinimalEdgeEncoder`:

1. one-hot edge type concatenated with radial basis
2. linear projection to scalar channels
3. tensor product with SH irreps (`FullyConnectedTensorProduct`)
4. parity-aware `Gate`
5. custom equivariant layer norm (`e3LayerNorm`)

implemented at `studies/minimal_overfit_study/common.py:198`.

### 4.3 Message passing block

`MinimalMessageBlock` at `studies/minimal_overfit_study/common.py:309` performs:

1. node update first:
   - aggregate edge features with `index_add_` on destinations
   - concatenate with current node features
   - `Linear -> Gate -> e3LayerNorm`
2. edge update second:
   - concat(updated src node, updated dst node, old edge)
   - tensor product with SH
   - `Gate -> e3LayerNorm`.

This is a hand-rolled message step, not the main `MessageBlock` in `src/net/layers.py`.

### 4.4 Head

`MinimalHead` at `studies/minimal_overfit_study/common.py:477`:

- per edge type, uses separate linears for diagonal and off-diagonal edges
- determines diagonal status from `(edge_shift==0 and src==dst)`
- returns dict: `pair_key -> {"vectors", "edges"}`.

The outer script wraps this into `IrrepsBlockData` and converts to blocks before loss:

- wrapping at `studies/minimal_overfit_study/overfit_water_minimal.py:986`
- conversion at `studies/minimal_overfit_study/overfit_water_minimal.py:997`.

## 5. Training objective and optimization

### 5.1 Loss target

The script trains only on Hamiltonian reconstruction:

- target = `target_H_matrix` (`BlockMatrix`) at `studies/minimal_overfit_study/overfit_water_minimal.py:476`
- total loss set as `loss = loss_H` at `studies/minimal_overfit_study/overfit_water_minimal.py:1139`.

No overlap/density reconstruction terms in the objective, and no energy/electron losses.

### 5.2 Two loss modes

1. standard matrix-block MSE (`--train-on-irrep-parts` off)
   `studies/minimal_overfit_study/overfit_water_minimal.py:1082`
2. irrep-decomposed loss (`--train-on-irrep-parts` on):
   - split pred/target by irrep
   - convert each slice back to blocks
   - accumulate per-irrep MSE
   `studies/minimal_overfit_study/overfit_water_minimal.py:1000`, `studies/minimal_overfit_study/common.py:1778`.

Both modes support `partial_train in {diag, offdiag, None}` via mask filtering.

### 5.3 Optional block normalization

If `--normalize-blocks` is enabled:

- diagonal and off-diagonal Frobenius magnitudes are estimated per key
- each selected block is divided by corresponding normalization factor during loss

at `studies/minimal_overfit_study/overfit_water_minimal.py:691` and used in `studies/minimal_overfit_study/overfit_water_minimal.py:1039`, `studies/minimal_overfit_study/overfit_water_minimal.py:1105`.

### 5.4 Optimizer and stability checks

- `Adam` optimizer at `studies/minimal_overfit_study/overfit_water_minimal.py:877`
- `ReduceLROnPlateau` stepped on training loss at `studies/minimal_overfit_study/overfit_water_minimal.py:880`, `studies/minimal_overfit_study/overfit_water_minimal.py:1143`
- gradient clipping at `studies/minimal_overfit_study/overfit_water_minimal.py:1165`
- explicit NaN/Inf checks:
  - loss-level at `studies/minimal_overfit_study/overfit_water_minimal.py:1149`
  - gradient-level at `studies/minimal_overfit_study/overfit_water_minimal.py:1171`
  - activation-level checks in `check_for_nans` used throughout model internals (`studies/minimal_overfit_study/common.py:22`).

## 6. Evaluation and artifacts

Final evaluation reconstructs predictions, applies optional partial masking, and computes:

- MAE/MSE and modified metrics with overlap-based correction `mu_H` via `compute_detailed_metrics`
  at `studies/minimal_overfit_study/common.py:700`
- per-block metrics dump at `studies/minimal_overfit_study/overfit_water_minimal.py:1530`
- distance-binned error curves at `studies/minimal_overfit_study/overfit_water_minimal.py:1571`, logic in `studies/minimal_overfit_study/common.py:778`
- optional per-irrep images/video outputs at `studies/minimal_overfit_study/overfit_water_minimal.py:1619`, `studies/minimal_overfit_study/overfit_water_minimal.py:1690`.

Saved artifacts include:

- best checkpoint: `best_model.pt`
- final checkpoint: `final_model.pt`
- distance curve JSON/PNG
- optional per-irrep images and training video.

## 7. Differences summary (vs main pipeline in `src/` + `scripts/train_silicon.py`)

### 7.1 Scope and data handling

- Minimal script: one snapshot, no train/val split, manual graph build in-script.
- Main pipeline: multi-snapshot dataset via `DatasetFactory` + `E3GNNDataset` + DataLoaders (`scripts/train_silicon.py`, `src/data/factory.py`, `src/data/gnn_dataset.py`).

### 7.2 Model/training framework

- Minimal script: plain PyTorch loop and custom `MinimalNetwork`.
- Main pipeline: PyTorch Lightning `E3GNN` with callbacks, structured logging, and scheduler integration in model class.

### 7.3 Objective

- Minimal script: Hamiltonian-only reconstruction loss.
- Main pipeline: multi-target matrix losses (`hamiltonian`, `overlap`, `density`) plus optional observable losses (`Tr(DH)`, `Tr(DS)`), and optional force/stress routes.

### 7.4 Edge and convention controls

- Minimal script adds explicit convention-sweep knobs (`xyz_permutation`, `change_box`, `box_convention`) and strict edge checks.
- Main pipeline follows standard convention handling inside `Snapshot`/dataset flow and does not expose this same sweep surface in one script.

### 7.5 Diagnostics depth

- Minimal script is debug-heavy: per-step NaN guards, per-irrep logging, distance bins, frame/video production.
- Main pipeline is production-oriented: benchmark callback, cleaner forward path, fewer per-operation debug interventions.

### 7.6 Feature-construction differences

- Minimal script uses SH with `normalize=False` and manually rescales radial embedding by `sqrt(n_radial)`.
- Main pipeline uses `compute_graph_features` with SH `normalize=True, normalization="component"` and no extra post radial scaling.

### 7.7 Head design differences

- Minimal head has separate diagonal/off-diagonal projections per edge type.
- Main `DeepHead` uses a shared trunk plus per-pair projection, without this explicit diag/offdiag split at head projection level.

### 7.8 Practical role

- Minimal script: convention/debug/overfit microscope for one structure.
- Main pipeline: reusable training system for broader datasets and experiments.
