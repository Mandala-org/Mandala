# Minimal Overfit Observables Study: Implementation Plan

## 1) What `src/` does today (ground truth for parity)

### 1.1 Data targets built per snapshot (`src/data/gnn_dataset.py`)
- For each sample, `y` includes:
  - `hamiltonian`
  - `overlap`
  - `density`
  - `energy = Tr(D·H)` via `snap.get_energy()`
  - `num_electrons = Tr(D·S)` via `snap.get_number_of_electrons()`
  - `forces`
  - `stress`
- Matrix targets can be represented as block matrices or irreps vectors depending on `train_target`.

### 1.2 Model outputs (`src/net/e3gnn.py`)
- Model has a separate head per matrix in `cfg.matrix_targets` (default `['hamiltonian', 'overlap', 'density']`).
- Forward returns predictions for each requested matrix.

### 1.3 Losses/metrics currently wired in main training step (`src/net/e3gnn.py::_shared_step`)
- Matrix losses: per-matrix MSE/MAE combined with `loss_l1_fraction`.
- Observable metrics and (optionally) losses:
  - `energy` from predicted `H` and `D`: `Tr(H_pred · D_pred)`
  - `num_electrons` from predicted `D` and `S`: `Tr(D_pred · S_pred)`
- Optional `train_observables_on_gt` path computes partial-GT observable losses:
  - `E(H_pred, D_gt)` and `E(H_gt, D_pred)`
  - `N(D_pred, S_gt)` and `N(D_gt, S_pred)`

### 1.4 Forces/stress status in main code
- Infrastructure exists:
  - dataset stores `forces`, `stress`
  - `get_forces()` and `get_stress()` are implemented (autograd via energy)
- But force/stress losses are **not** currently added in `_shared_step`.
- So for strict parity with current `src`, we should first implement H/S/D + energy + electrons, then optionally add force training as a deliberate extension.

---

## 2) New study bootstrap (exact copy requirement)

Create `studies/minimal_overfit_observables/` and copy these Python files **verbatim** from `studies/minimal_overfit_study/` as baseline:
- `overfit_water_minimal.py`
- `common.py`
- `detailed_logging.py`
- `strict_checks.py`
- `analyze_model.py`
- `analyze_data.py`
- `analyze_sweep_conventions.py`
- `e3mlp_variants.py`
- `test_e3mlp_variants.py`

Then rename entry script for clarity:
- `overfit_water_minimal.py` -> `overfit_observables_minimal.py`

Keep the original copy too during migration (safer diffing).

---

## 3) Scope and compatibility target

## 3.1 Phase-1 target (must-have)
- Train/predict matrices: Hamiltonian + Overlap + Density.
- Compute/log observable metrics: energy and electron count.
- Add optional observable losses with coefficient (`loss_coef_observables`).
- Preserve existing minimal study features (edge checks, video, irrep metrics, benchmarking, unit scaling, etc.).

## 3.2 Phase-2 target (extension)
- Add force training via autograd from predicted observables.
- Force metrics/losses configurable and separately weighted.

## 3.3 Phase-3 target (optional)
- Stress training/loss (same pattern as force path).

---

## 4) Detailed implementation steps

## Step 0: Scaffold and freeze baseline
Files:
- `studies/minimal_overfit_observables/*`

Actions:
- Copy files exactly.
- Add one smoke script that runs baseline script unchanged for 3-5 epochs.

Acceptance:
- Baseline behavior in new folder matches current minimal study.

## Step 1: Config/API extension in `overfit_observables_minimal.py`
Add argparse/config fields (mirroring `src/net/common.py` semantics where useful):
- `--matrix-targets` (comma list; default `hamiltonian,overlap,density`)
- `--train-target` (`matrix|irreps`, default `matrix`)
- `--enable-energy` (bool, default `True`)
- `--enable-num-electrons` (bool, default `True`)
- `--train-on-energy` (bool, default `True`)
- `--train-on-num-electrons` (bool, default `True`)
- `--train-observables-on-gt` (bool, default `False`)
- `--loss-coef-observables` (float, default conservative e.g. `1e-5`)
- `--enable-forces` / `--train-on-forces` / `--loss-coef-forces` (phase-2 wired)

Constraints:
- If `train_on_energy` or `train_on_num_electrons` is true and `loss_coef_observables==0`, hard error.
- If force training enabled and required matrices unavailable, hard error.

## Step 2: Multi-matrix head outputs in `common.py`
Current minimal model predicts only H.

Actions:
- Extend `MinimalNetwork` to maintain one head per matrix target (`ModuleDict`), similar to main `src`.
- Return dict:
  - `preds_irreps = {'hamiltonian': ..., 'overlap': ..., 'density': ...}`
- Preserve all existing options (magnitude factorization, scalar MLP, tensor-square options) for each head.

Acceptance:
- Forward pass works for any subset of matrix targets.

## Step 3: Target preparation for H/S/D
In `overfit_observables_minimal.py`:
- Build canonicalized targets for all requested matrices.
- Maintain exact same edge canonicalization and strict checks across all matrices.
- Ensure edge set agreement among H/S/D before training (hard error if mismatch in strict mode).

Acceptance:
- For each matrix target, predicted and target keys/edges align after canonicalization.

## Step 4: Matrix loss generalization
Refactor current H-only matrix loss block:
- Loop over `matrix_targets`.
- Compute per-matrix block loss (respect existing partial masks, irrep-part mode, normalization modes, factorization mode).
- Keep per-matrix logging:
  - `train/<name>_mae`, `train/<name>_mse`
  - `train/loss_<name>_block`

Important for parity:
- If only `hamiltonian` is selected, results should reduce to current behavior.

## Step 5: Observable computation + losses
Implement in training loop after matrix predictions are reconstructed/symmetrized:
- `E_pred = Tr(H_pred · D_pred)` when both available and energy enabled.
- `N_pred = Tr(D_pred · S_pred)` when both available and electrons enabled.

Metrics:
- `train/energy_mae`
- `train/num_electrons_mae`

Losses:
- standard mode:
  - `loss_E = coef * MSE(E_pred, E_true)`
  - `loss_N = coef * MSE(N_pred, N_true)`
- GT-observable mode (`train_observables_on_gt`):
  - `E(H_pred, D_gt)` + `E(H_gt, D_pred)` / 2
  - `N(D_pred, S_gt)` + `N(D_gt, S_pred)` / 2

Total:
- `loss_total = loss_matrix_total + loss_E + loss_N + existing optional terms`

## Step 6: Force training (phase-2)
Add explicit and isolated code path (default off):
- Compute predicted energy scalar from matrices.
- `forces_pred = -dE_pred / dR` via autograd.
- `loss_F = loss_coef_forces * MSE(forces_pred, forces_true)`.
- Add metrics: MAE/MSE of forces.

Technical requirements:
- `positions.requires_grad_(True)` before forward.
- Ensure no `.detach()` breaks computational graph before force loss.
- Guard with hard errors when force labels unavailable.

Note:
- This is an extension beyond currently wired `src/_shared_step`, but uses the same physical definition as `src/get_forces()`.

## Step 7: Logging/reporting updates
Keep WandB always-on behavior from your current minimal setup.
Add blocks:
- Matrix losses per target.
- Observable losses/metrics.
- Force losses/metrics (if enabled).
- Contribution fractions in print logs:
  - matrix vs observables vs forces.

## Step 8: Analysis script updates
Update `analyze_model.py` in new study to support multi-target evaluation:
- Matrix-level metrics for H/S/D.
- Observable errors for energy/electrons.
- Optional force evaluation if model/run includes force mode.

## Step 9: Validation matrix (required)
Prepare deterministic smoke runs:
1. H only (regression against old minimal).
2. H+S+D matrix-only.
3. H+S+D + observables loss.
4. H+S+D + observables loss + GT-observable mode.
5. H+S+D + force loss (small epoch test).

For each case check:
- no edge mismatch
- no NaN/Inf
- total loss decomposition sums correctly
- metrics have expected keys.

---

## 5) Key design decisions to lock before coding

1. **Force parity objective**
- Option A (strict `src` parity): ship without force loss first.
- Option B (requested functionality): include force loss in phase-2 default-off.

2. **Training unit for observables**
- Matrix unit scaling already exists in minimal script.
- Keep energy/electron computed from scaled matrices to avoid unit inconsistency.

3. **Partial training masks (`diag/offdiag/shifted_self`)**
- Apply only to matrix losses.
- Do not apply masks to scalar observables (energy/electrons) unless explicitly requested later.

---

## 6) Implementation order (fastest safe path)
1. Bootstrap folder + exact file copy.
2. Multi-matrix heads + matrix losses (no observables yet).
3. Add energy/electron metrics then losses.
4. Add GT-observable variant.
5. Add force loss path (default off).
6. Update analysis scripts and run smoke matrix.

This order minimizes breakage and makes regressions easy to localize.
