# ZnCuSnSeS Envelope And Radial Learning Implementation Plan

## Goal

Improve global Hamiltonian MAE on ZnCuSnSeS without sacrificing short-range accuracy, while keeping all new behavior strictly opt-in and preserving current performance when the new options are disabled.

This plan reflects the selected directions:

- Keep Slater-soft-cutoff as the fitted envelope family.
- Support two envelope usages:
  - normalize Hamiltonian targets by the fitted envelope
  - multiply model predictions by the fitted envelope
- Support two independent learned radial options:
  - pair-conditioned radial MLP
  - pair-specific distance normalization using `r / r0(pair)`
- Explore loss weighting / objective shaping.
- Skip:
  - hybrid residual on top of the envelope
  - learnable envelope parameters
  - support / sparsity modeling

## Design Constraints

### 1. Existing behavior must not change

When all new flags are left at defaults:

- dataset construction must behave exactly as today
- model forward pass must behave exactly as today
- training loss and validation metrics must behave exactly as today
- no new files should be required
- no new compute should be spent beyond negligible argument parsing / conditional checks

### 2. Performance must not regress

We should avoid:

- recomputing fitted envelope metadata inside workers
- per-batch Python-heavy pair lookups
- rebuilding pair-conditioned constants repeatedly
- branching inside inner loops more than necessary

### 3. New features should be orthogonal

The main experimental axes should be independently switchable:

- envelope target normalization
- envelope prediction multiplication
- pair-conditioned radial MLP
- pair-specific distance normalization
- loss weighting / auxiliary normalized loss

This allows a structured sweep rather than a monolithic rewrite.

## High-Level Strategy

We will treat the fitted Slater-soft-cutoff envelope as a precomputed pair-type prior. The model can then use that prior in three places:

1. target preprocessing
2. output postprocessing
3. loss weighting / auxiliary losses

Separately, we will improve the learned radial representation by:

1. conditioning the radial path on pair identity
2. rescaling distance per pair using `r / r0(pair)`

The primary optimization target remains the reconstructed physical Hamiltonian MAE.

## Precomputed Metadata

### Envelope fit artifact

We need one stable artifact, stored on disk and reusable by dataset workers, training, evaluation, and analysis:

- unordered pair key, e.g. `Zn-Se`
- fitted Slater-soft-cutoff parameters
  - `amp`
  - `rate`
  - `nu`
  - `r0`
  - `tau`
- fit quality metrics
  - `log_rmse`
  - `log_mae`
  - `raw_rmse`
  - `raw_mae`
- metadata
  - snapshot / dataset provenance
  - magnitude definition used for fitting
  - cutoff used during fitting

Suggested file:

- `artifacts/zncusnses_hamiltonian_slater_soft_cutoff.json`

### Derived per-pair constants

At load time we should derive and cache:

- pair index -> envelope parameter tensor
- pair index -> `r0(pair)` tensor for distance normalization
- pair index -> any loss-weight reference constants

These should be prepared once per process and reused.

## Required CLI / Config Surface

Add new options with safe defaults:

- `--hamiltonian-envelope-path`
  - default: `None`
- `--hamiltonian-envelope-mode`
  - values:
    - `off`
    - `normalize_target`
    - `multiply_prediction`
  - default: `off`
- `--hamiltonian-envelope-eps`
  - default small positive value for safe division
- `--pair-conditioned-radial-mlp`
  - values: `true/false`
  - default: `false`
- `--pair-distance-normalization`
  - values:
    - `off`
    - `pair_r0`
  - default: `off`
- `--loss-weighting-mode`
  - values:
    - `off`
    - `distance_short_range_bias`
    - `envelope_inverse_sqrt_clipped`
    - `envelope_inverse_clipped`
  - default: `off`
- `--loss-weight-min`
- `--loss-weight-max`
- `--loss-normalized-hamiltonian-coef`
  - default: `0.0`

Important:

- `normalize_target` and `multiply_prediction` should be mutually exclusive in the first implementation.
- If `--hamiltonian-envelope-path` is missing, all envelope-dependent modes should error clearly rather than silently falling back.

## Detailed Implementation Plan

## Phase 1: Shared envelope utilities

### 1.1 Add a reusable envelope module

Create a small utility module, likely under:

- `src/analysis/`
- or `src/data/`
- or `src/net/`

Recommended content:

- load envelope JSON
- validate pair coverage
- map unordered pair strings to pair-type indices
- evaluate Slater-soft-cutoff envelope for a tensor of distances and pair indices
- return:
  - envelope magnitude
  - `r0(pair)` if requested

Suggested functions:

- `load_pair_envelope_table(...)`
- `build_pair_envelope_tensors(...)`
- `evaluate_slater_soft_cutoff(...)`

### 1.2 Define one canonical pair-key convention

We must ensure:

- `Zn-Se` and `Se-Zn` map to the same fitted entry
- training pair type indices resolve deterministically

This same convention must be reused in:

- fitting script outputs
- dataset code
- training code
- evaluation / analysis code

### 1.3 Cache tensors

Envelope parameter tensors should be:

- built once
- kept on CPU and moved to model/device in batch-friendly form when needed
- reused for every batch

Avoid:

- repeated JSON parsing
- repeated dictionary string lookup per batch element

## Phase 2: Target normalization path

### 2.1 Apply only to Hamiltonian targets

Do not touch overlap or density behavior.

When `--hamiltonian-envelope-mode normalize_target` is enabled:

- compute `E(pair, distance)` per Hamiltonian block edge
- normalize the Hamiltonian target by `max(E, eps)`
- feed the normalized target into the loss path

Important design point:

- store both normalized and physical targets in the batch path if needed
- validation metrics must still be computed on reconstructed physical predictions

### 2.2 Reconstruct physical predictions for metrics

If training is done in normalized space:

- model output corresponds to normalized Hamiltonian
- before MAE/MSE metrics, reconstruct:
  - `H_pred_phys = H_pred_norm * E`

Primary metrics remain:

- physical Hamiltonian MAE
- physical Hamiltonian MSE if logged

### 2.3 Numerical stability

Use:

- `E_safe = max(E, eps)`

Optionally add clipping in normalized space later if needed, but do not add it in the first implementation unless instability appears.

## Phase 3: Prediction multiplication path

### 3.1 Apply envelope at the Hamiltonian head output

When `--hamiltonian-envelope-mode multiply_prediction` is enabled:

- the head predicts a scale-free Hamiltonian-like tensor
- right before comparison with the target, multiply each block by `E(pair, distance)`

This is the safest first envelope factorization because:

- the physical target stays unchanged
- the prediction is transformed in a simple multiplicative way
- current loss code can remain mostly intact once the Hamiltonian prediction tensor is replaced by the modulated one

### 3.2 Preserve logging behavior

All logged Hamiltonian outputs and validation metrics should refer to the reconstructed physical prediction, not the internal pre-envelope tensor.

If useful for debugging, optionally log additional diagnostics later:

- normalized-space MAE
- envelope magnitude statistics

But keep those off by default.

## Phase 4: Pair-specific distance normalization

### 4.1 Use `r / r0(pair)`

This option is independent of the envelope mode.

When `--pair-distance-normalization pair_r0` is enabled:

- compute edge distance as usual
- divide by fitted `r0(pair)` for that pair type

This normalized distance should be used only in the learned radial path, not for physical geometry or reporting.

### 4.2 Integration point

Most likely place:

- radial embedding / graph feature construction

We want:

- raw distance still available if existing logic needs it
- normalized distance used as the radial input to the selected components

Avoid:

- changing neighbor construction
- changing cutoff semantics
- changing physical edge distances stored for diagnostics

### 4.3 Precompute `r0(pair)`

This should come from the same envelope JSON and be cached with the pair parameter tensors.

## Phase 5: Pair-conditioned radial MLP

### 5.1 Independent from distance normalization

This option should work in four regimes:

1. off
2. on with raw distance
3. off with `r / r0(pair)`
4. on with `r / r0(pair)`

### 5.2 Minimal safe implementation

Do not redesign the full GNN.

Recommended first version:

- keep the current radial basis machinery
- inject pair conditioning into the radial MLP / radial projection path

Possible mechanisms:

- concatenate pair embedding to radial features before radial MLP
- FiLM-style modulation of radial hidden activations by pair embedding

Preferred first option:

- concatenate pair embedding

Reason:

- simpler
- easier to verify
- less risk of subtle initialization issues

### 5.3 Reuse existing pair-type embedding

If possible:

- reuse `edge_type_idx`
- reuse or lightly extend existing pair embedding infrastructure

Avoid:

- introducing a second incompatible pair-identity system

## Phase 6: Loss weighting and objective shaping

### 6.1 Primary loss remains physical

Regardless of envelope mode, the main optimization target should remain the physical Hamiltonian error after reconstruction.

This protects the project objective:

- minimize global Hamiltonian MAE

### 6.2 Add optional weighting modes

Support:

- `off`
- `distance_short_range_bias`
  - stronger weight on short edges
- `envelope_inverse_sqrt_clipped`
  - boosts small-envelope regions mildly
- `envelope_inverse_clipped`
  - stronger version, bounded by `weight_min/max`

All weighting should be bounded.

### 6.3 Add optional auxiliary normalized loss

When `--loss-normalized-hamiltonian-coef > 0`:

- compute a secondary loss in normalized space
- add it with a small coefficient

This should be auxiliary only.

Recommended formula:

- `L_total = L_phys + lambda_norm * L_norm`

where:

- `L_phys` is the main Hamiltonian loss in physical units
- `L_norm` is computed on `H / E_safe`

### 6.4 Keep MAE and MSE aligned

Any weighting or normalization that applies to one Hamiltonian loss metric should be applied in the same structural way to the other, differing only by absolute vs squared residual.

This avoids metric drift and confusion.

## Phase 7: Logging and diagnostics

Add opt-in or lightweight logging for:

- envelope mode in run summary
- pair distance normalization mode
- pair-conditioned radial MLP enabled/disabled
- loss weighting mode
- normalized auxiliary loss coefficient

Validation diagnostics worth logging:

- global Hamiltonian MAE
- MAE by distance bucket
- MAE by pair type
- optional normalized-space MAE

These are important for deciding whether an apparent global gain comes from the desired mechanism.

## Phase 8: Precompute and reuse aggressively

### 8.1 What to precompute

Precompute once and reuse:

- fitted envelope JSON
- pair index -> parameter tensors
- pair index -> `r0(pair)`
- pair index -> any derived constants needed by weighting

### 8.2 Where to cache

Recommended:

- load JSON once during dataset / model setup
- keep processed tensors attached to:
  - config helper object
  - model module
  - or dataset bundle metadata

The exact owner should minimize repeated device transfers.

### 8.3 What not to precompute prematurely

Do not try to cache per-edge envelope values globally across the whole dataset in the first pass unless profiling shows a real bottleneck.

Reason:

- edge distances are already available per sample
- evaluating a small closed-form function is cheap
- giant per-edge caches can complicate invalidation and memory use

## Verification Plan

## Unit / smoke verification

Before sweeping:

1. `off` modes reproduce current outputs bitwise or near-bitwise.
2. `normalize_target` reconstructs physical predictions correctly.
3. `multiply_prediction` applies envelope only to Hamiltonian predictions.
4. pair key lookup is order-invariant.
5. `r / r0(pair)` uses the correct fitted `r0`.
6. weighting modes produce bounded finite weights.

## Regression checks

Run one short training / validation step with:

- all defaults off
- each option enabled separately

Confirm:

- no crash
- no unexpected change in unrelated targets
- no change in overlap/density behavior

## Sweep Plan For 128 Runs

Do this in stages, not as one flat grid.

## Stage A: Mechanism screening, 32 runs

Goal:

- identify which modeling knobs help without harming short-range behavior

Axes:

- envelope mode:
  - `off`
  - `normalize_target`
  - `multiply_prediction`
- pair-conditioned radial MLP:
  - `off`
  - `on`
- pair distance normalization:
  - `off`
  - `pair_r0`

This gives `3 x 2 x 2 = 12` settings.

Run:

- 2 seeds for all 12 = 24 runs
- 8 extra runs for the most promising 4 settings with 2 additional seeds or nearby LR checks

## Stage B: Loss shaping, 48 runs

Take the best 3-4 settings from Stage A.

Sweep:

- loss weighting:
  - `off`
  - `distance_short_range_bias`
  - `envelope_inverse_sqrt_clipped`
  - `envelope_inverse_clipped`
- normalized auxiliary loss coefficient:
  - `0`
  - small positive value

Example:

- 4 selected model settings
- 4 weighting modes
- 2 aux-loss settings
- 1 seed

= 32 runs

Use the remaining 16 runs for:

- second seeds on the top candidates
- one or two LR / scheduler refinements

## Stage C: Exploitation, 48 runs

Take the strongest 4-6 candidates overall and refine:

- LR
- scheduler patience
- maybe one additional small weighting-range tweak

Use this stage to convert a good mechanism into the best trainable recipe.

## Implementation Order

Recommended order:

1. shared envelope utilities and JSON format
2. prediction multiplication mode
3. target normalization mode
4. pair distance normalization with `r / r0(pair)`
5. pair-conditioned radial MLP
6. bounded loss weighting modes
7. auxiliary normalized loss
8. logging / diagnostics polish

Reason:

- the first three are the clearest and easiest to verify
- the latter ones depend on them conceptually

## Expected Outcomes

Best-case:

- the model no longer wastes capacity learning pair-specific radial scale from scratch
- far-tail behavior becomes easier to fit
- short-range MAE stays stable or improves
- global Hamiltonian MAE drops measurably

What would count as failure:

- better relative tail behavior but worse short-range absolute MAE
- improved normalized-space metrics but unchanged physical Hamiltonian MAE
- too much complexity added before basic envelope factorization is validated

## Summary

The plan is to introduce a small, orthogonal set of opt-in mechanisms:

- fitted Slater-soft-cutoff envelope for Hamiltonian target normalization or prediction multiplication
- pair-specific distance normalization via `r / r0(pair)`
- pair-conditioned radial learning
- bounded loss shaping around the physical Hamiltonian objective

All new functionality should default to off, reuse the same precomputed envelope metadata, and leave current behavior untouched unless explicitly enabled.
