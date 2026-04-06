# Mandala `src/` Migration Guide

This document records the main missing or misaligned features between the current stable implementation under `src/` and the richer workflows in:

- `studies/minimal_silicon_study`
- `studies/minimal_overfit_study`
- `studies/minimal_overfit_observables`

It is organized by implementation tasks so it can double as a migration checklist.

## Scope and Exclusions

This guide focuses on features that should plausibly migrate into the unified framework. The following study-only options are explicitly excluded from the target merge for now:

- `convention`
- `orbital-selection`
- `sh-mode`
- `xyz-permutation`
- `change-box`
- `box-convention`

## Recommended Destination Layout

The current `src/` tree already has strong homes for core model and data abstractions, but several study features would land more cleanly with a small expansion of the package layout:

- `src/train/runner.py`
  - CLI-independent training orchestration
- `src/train/checkpointing.py`
  - save / resume / config restoration
- `src/train/logging.py`
  - WandB logging helpers and detailed metric packaging
- `src/data/preprocessing.py`
  - graph-target alignment, strict reconciliation, sample assembly
- `src/data/cache.py`
  - preprocessed-sample cache logic
- `src/data/validation.py`
  - strict graph/operator consistency checks
- `src/net/e3mlp_variants.py`
  - richer equivariant MLP family registry
- `src/utils/metrics.py`
  - aligned matrix metrics, per-irrep metrics, gauge-corrected metrics
- `src/utils/visualization.py`
  - matrix, DOS, irrep, and video-oriented visual outputs
- `src/postprocessing/electronic_structure.py`
  - DOS, generalized eigenvalues, band-structure helpers

## Task 1. Data Loading, Backend Parsing, and Snapshot Construction

### 1.1 Backend discovery and split construction

Gap:

The stable dataset factory expects explicit `(matrix_path, info_path)` pairs, while the silicon workflow already supports directory discovery, temperature-aware grouping, and split construction.

Study references:

- `studies/minimal_silicon_study/train_silicon_minimal.py:908-989`
  - snapshot-pair and temperature-aware dataset discovery
- `studies/minimal_silicon_study/train_silicon_minimal.py:2610-2658`
  - split construction from discovered pairs

Current `src/` status:

- `src/data/factory.py:36-128`
  - dataset registration and shared mapper creation
- `src/data/factory.py:134-154`
  - basic helper for train/val pair lists

Suggested destination:

- extend `src/data/factory.py`
- optionally add `src/data/discovery.py`

### 1.2 Unified snapshot construction across backends

Gap:

`src/` already has the right abstractions, but the practical multi-backend training flow is still centered on OpenMX in the dataset path.

Study references:

- `studies/minimal_silicon_study/train_silicon_minimal.py:1231-1315`
  - preprocessing assumes already-materialized `Snapshot` objects carrying `H/S/D`
- `studies/minimal_overfit_study/overfit_water_minimal.py:120-190`
  - simple single-snapshot loading and unit selection
- `studies/minimal_overfit_observables/overfit_observables_minimal.py:140-200`
  - observable-oriented single-snapshot setup

Current `src/` status:

- `src/data/gnn_dataset.py:118-159`
  - `Snapshot.from_openmx` path is directly used in dataset loading
- `src/data/snapshot.py:51-107`
  - snapshot abstraction is backend-agnostic

Suggested destination:

- keep `src/data/snapshot.py` as the canonical snapshot object
- extend `src/data/factory.py` or `src/data/discovery.py`
  - to select backend-specific loaders cleanly

### 1.3 Raw snapshot caching

Gap:

Raw snapshot caching exists, but only as one layer of a larger caching story.

Study references:

- `studies/minimal_silicon_study/train_silicon_minimal.py:818-882`
  - cache-key and cache I/O helpers

Current `src/` status:

- `src/data/gnn_dataset.py:94-159`
  - raw `Snapshot` cache only

Suggested destination:

- keep the raw snapshot cache in `src/data/gnn_dataset.py`
- optionally refactor generic cache helpers into `src/data/cache.py`

## Task 2. Data Handling, Preprocessing, and Graph/Target Alignment

### 2.1 Rich snapshot-to-sample preprocessing

Gap:

The silicon study contains the full reusable preprocessing pipeline, while `src/` currently performs a lighter conversion from `Snapshot` to `(x, y)`.

Study references:

- `studies/minimal_silicon_study/train_silicon_minimal.py:1195-1213`
  - mapper preparation from a representative sample
- `studies/minimal_silicon_study/train_silicon_minimal.py:1231-1638`
  - full `preprocess_sample` logic
- `studies/minimal_silicon_study/train_silicon_minimal.py:1771-1812`
  - edge-feature recomputation for displaced positions
- `studies/minimal_silicon_study/train_silicon_minimal.py:1815-1898`
  - device / pinned-memory helpers

Current `src/` status:

- `src/data/gnn_dataset.py:161-239`
  - basic snapshot-to-sample conversion

Suggested destination:

- new `src/data/preprocessing.py`

### 2.2 Preprocessed-sample caching

Gap:

The stable code does not yet persist graph-aligned, preprocessed samples.

Study references:

- `studies/minimal_silicon_study/train_silicon_minimal.py:1641-1765`
  - cache-aware preprocessing driver

Current `src/` status:

- no preprocessed-sample cache layer in `src/`

Suggested destination:

- new `src/data/cache.py`
- integration points in `src/data/preprocessing.py`

### 2.3 Canonical sparse-edge handling and cutoff filtering

Gap:

The studies explicitly canonicalize sparse target edges and optionally filter targets by cutoff before training and metrics.

Study references:

- `studies/minimal_silicon_study/train_silicon_minimal.py:1016-1034`
  - canonicalize sparse block matrix edges
- `studies/minimal_silicon_study/train_silicon_minimal.py:1035-1054`
  - cutoff filtering of sparse targets
- `studies/minimal_overfit_study/overfit_water_minimal.py:426-429`
  - cutoff-to-targets runtime flag
- `studies/minimal_silicon_study/train_silicon_minimal.py:467-468`
  - corresponding runtime surface

Current `src/` status:

- `src/data/snapshot.py:142-236`
  - canonical snapshot edge ordering exists
- no explicit dataset-level target cutoff and canonicalization stage with the same flexibility

Suggested destination:

- `src/data/preprocessing.py`
- helper methods can remain close to `src/data/snapshot.py`

### 2.4 Graph construction, reverse-edge repair, and exact edge matching

Gap:

The studies contain stricter graph construction than `src/`, including reverse-edge repair and exact graph-target matching.

Study references:

- `studies/minimal_silicon_study/train_silicon_minimal.py:1086-1193`
  - graph-edge and matrix-edge lookup plus reconciliation
- `studies/minimal_silicon_study/train_silicon_minimal.py:1367-1413`
  - reverse-edge repair
- `studies/minimal_silicon_study/train_silicon_minimal.py:1436-1456`
  - exact graph-target edge reconciliation
- `studies/minimal_silicon_study/train_silicon_minimal.py:468`
  - runtime switch for exact edge match

Current `src/` status:

- `src/data/graph_features.py:27-212`
  - graph construction and feature generation
- `src/data/snapshot.py:142-236`
  - canonical edge ordering

Suggested destination:

- extend `src/data/graph_features.py`
- add reconciliation helpers in `src/data/preprocessing.py`
- keep trace-alignment math in `src/core/sparse_math.py`

### 2.5 Self-edge feature hygiene and validation checks

Gap:

The studies have explicit self-edge spherical-harmonic cleanup and stronger validation checks than the stable code.

Study references:

- `studies/minimal_silicon_study/train_silicon_minimal.py:1469-1511`
  - self-edge spherical-harmonic cleanup
- `studies/minimal_overfit_study/strict_checks.py:20-107`
- `studies/minimal_overfit_observables/strict_checks.py:20-107`
- `studies/minimal_silicon_study/train_silicon_minimal.py:1531-1585`
  - strict alignment / reverse-edge checks in the silicon flow

Current `src/` status:

- `src/data/graph_features.py:96-113,178-184`
  - a few local assertions only
- `src/net/common.py:157-170`
  - `safety_checks` config exists

Suggested destination:

- extend `src/data/graph_features.py`
- new `src/data/validation.py`

## Task 3. Model Creation and Architecture Variants

### 3.1 Configuration surface for architecture search

Gap:

`src/net/common.py` already contains many knobs, but the study workflows expose a larger practical search surface.

Study references:

- `studies/minimal_overfit_study/overfit_water_minimal.py:152-186`
  - hidden dim / `l_max` / hidden irreps / depth / cutoff / radial basis
- `studies/minimal_overfit_observables/overfit_observables_minimal.py:221-260`
  - same core architecture controls for the observables study
- `studies/minimal_silicon_study/train_silicon_minimal.py:411-468`
  - expanded architecture and head options in the silicon workflow

Current `src/` status:

- `src/net/common.py:41-95,114-145`
  - strong baseline config surface already exists

Suggested destination:

- continue extending `src/net/common.py`
- reserve larger variant implementations for dedicated files

### 3.2 Edge-encoder options

Gap:

The studies expose additional edge-encoder controls not yet fully surfaced in `src/`.

Study references:

- `studies/minimal_overfit_study/overfit_water_minimal.py:319-341`
  - `e3layernorm`, SH tensor-square, radial embedding scale
- `studies/minimal_overfit_observables/overfit_observables_minimal.py:366-379`
  - same controls
- `studies/minimal_silicon_study/train_silicon_minimal.py:438-457`
  - same controls in the silicon workflow

Current `src/` status:

- `src/net/encoders.py:95-179`
  - edge encoder supports multiple styles
- `src/net/common.py:49,115-120`
  - some encoder/radial controls are already present

Suggested destination:

- extend `src/net/encoders.py`
- add missing config fields to `src/net/common.py` where needed

### 3.3 Message-passing and equivariant MLP variants

Gap:

The stable code supports a solid baseline family, but the studies already expose a much broader E3-MLP registry.

Study references:

- `studies/minimal_overfit_study/e3mlp_variants.py:353-1285`
- `studies/minimal_overfit_observables/e3mlp_variants.py:353-1285`
  - `NormActE3MLP`
  - `GateE3MLP`
  - `GateScalarsMLPE3MLP`
  - `GateMagnitudesE3MLP`
  - `ResidualE3MLP`
  - `BilinearSelfTPE3MLP`
  - `InvariantSelfAttentionE3MLP`
  - `InvariantMoEE3MLP`
  - `ScalarMagnitudeSelfTPGatedE3MLP`

Current `src/` status:

- `src/net/activations.py:61-276`
  - `NormAct`, `S2Act`, `GateScalarsMLP`, `GateMagnitudes`
- `src/net/common.py:368-436`
  - generic `E3MLP` and tensor-product support
- `src/net/layers.py:265-571`
  - message-passing blocks already have a strong modular skeleton

Suggested destination:

- new `src/net/e3mlp_variants.py`
- extend `src/net/activations.py`
- add selector/config plumbing in `src/net/common.py`

### 3.4 Head variants

Gap:

The stable `DeepHead` is simpler than the study head family.

Study references:

- `studies/minimal_overfit_study/common.py:563-918`
  - tensor-square-capable head, shifted-self handling, magnitude factorization hooks
- `studies/minimal_overfit_observables/common.py:588-855`
  - `E3GateMLP`, deeper head projections, tensor-square head input, separate shifted-self branch
- `studies/minimal_silicon_study/train_silicon_minimal.py:439-468`
  - shifted-self split, head tensor-square, self-edge node embeddings, head E3MLP depth

Current `src/` status:

- `src/net/heads.py:21-112`
  - shared trunk + per-pair projections
- `src/net/common.py:81-85`
  - some head-depth configuration

Suggested destination:

- extend `src/net/heads.py`

## Task 4. Training Routine, Possible Targets, and Losses

### 4.1 Multi-target operator supervision

Gap:

The studies expose a cleaner runtime surface for selecting operator targets and mixing them with observable guidance.

Study references:

- `studies/minimal_overfit_observables/overfit_observables_minimal.py:147-182`
  - matrix target subset and observable-loss coefficients
- `studies/minimal_silicon_study/train_silicon_minimal.py:323-363`
  - same plus density-specific loss weight

Current `src/` status:

- `src/net/common.py:128-145`
  - matrix target selection and base loss coefficients
- `src/net/e3gnn.py:299-367`
  - matrix losses are implemented

Suggested destination:

- extend `src/net/common.py`
- extend `src/net/e3gnn.py`

### 4.2 Energy and electron-count observable guidance

Gap:

The stable code already supports these observables, but the studies contain richer logging and evaluation around them.

Study references:

- `studies/minimal_overfit_observables/overfit_observables_minimal.py:153-182`
- `studies/minimal_silicon_study/train_silicon_minimal.py:329-357`
- `studies/minimal_silicon_study/train_silicon_minimal.py:1957-2092`
  - composed loss calculation including observable terms

Current `src/` status:

- `src/net/e3gnn.py:374-453`
  - energy and electron-count evaluation and training

Suggested destination:

- mostly extend `src/net/e3gnn.py`
- move reporting helpers into `src/utils/metrics.py` or `src/train/logging.py`

### 4.3 Forces and stress as training targets

Gap:

`src/` can derive forces and stress, but the full study-side training/evaluation workflow around them is richer.

Study references:

- `studies/minimal_overfit_observables/overfit_observables_minimal.py:183-200`
  - force-related runtime options
- `studies/minimal_silicon_study/train_silicon_minimal.py:365-380`
  - force-related parser surface
- `studies/minimal_silicon_study/train_silicon_minimal.py:1935-1954`
  - force computation from predicted matrices
- `studies/minimal_silicon_study/train_silicon_minimal.py:2209-2421`
  - split evaluation with force metrics

Current `src/` status:

- `src/net/e3gnn.py:577-661`
  - force and stress derivation from predicted operators
- `src/data/gnn_dataset.py:173-237`
  - force/stress targets carried through the dataset

Suggested destination:

- extend `src/net/e3gnn.py`
- add reusable reporting helpers in `src/utils/metrics.py`

### 4.4 Partial, weighted, and structured losses

Gap:

The study code exposes more structured objective variants than the stable code.

Study references:

- `studies/minimal_overfit_study/overfit_water_minimal.py:343-423`
  - partial block-class training, shifted-self split, normalization, magnitude factorization
- `studies/minimal_silicon_study/train_silicon_minimal.py:358-399`
  - density weighting and density-rescaling-related controls
- `studies/minimal_overfit_study/overfit_water_minimal.py:357-359`
  - per-irrep loss decomposition flag

Current `src/` status:

- `src/net/common.py:140-145`
  - coarse loss coefficients
- `src/net/e3gnn.py:299-453`
  - standard matrix and observable losses

Suggested destination:

- extend `src/net/common.py`
- extend `src/net/e3gnn.py`
- put reporting utilities into `src/utils/metrics.py`

## Task 5. Physics-Aware Postprocessing and Evaluation

### 5.1 Gauge-corrected Hamiltonian diagnostics (`mu_H`)

Gap:

The study code computes the optimal overlap-gauge shift `mu_H` and uses it in diagnostics. The stable code does not yet expose this as a first-class metric path.

Study references:

- `studies/minimal_overfit_study/common.py:1049-1077`
  - `compute_mu_H`
- `studies/minimal_overfit_observables/common.py:1031-1087`
  - `compute_mu_H` and aligned variant
- `studies/minimal_overfit_study/common.py:1078-1155`
  - detailed metrics using `mu_H`
- `studies/minimal_overfit_observables/common.py:1088-1226`
  - detailed metrics and aligned detailed metrics

Current `src/` status:

- `src/core/sparse_math.py:71-148`
  - trace and alignment primitives exist
- `src/net/e3gnn.py:299-453`
  - no gauge-aware metric layer yet

Suggested destination:

- `src/utils/metrics.py`
- optionally `src/core/sparse_math.py`

### 5.2 Density normalization to the correct electron count

Gap:

The silicon workflow supports post-prediction density rescaling for corrected metrics. `src/` does not yet expose that path explicitly.

Study references:

- `studies/minimal_silicon_study/train_silicon_minimal.py:393-400`
  - runtime flag for density rescaling
- `studies/minimal_silicon_study/train_silicon_minimal.py:2168-2195`
  - post-prediction density rescaling logic
- `studies/minimal_silicon_study/train_silicon_minimal.py:3289-3290,3507-3513`
  - corrected vs pre-correction metrics

Current `src/` status:

- `src/net/e3gnn.py:392-407`
  - electron-count observable and loss
- `src/data/snapshot.py:239-245`
  - `Tr(DS)` and `Tr(DH)` helpers

Suggested destination:

- `src/postprocessing/electronic_structure.py`
- or `src/data/snapshot.py`

### 5.3 Aligned metrics, per-irrep metrics, and split evaluation

Gap:

The studies already contain reusable aligned metrics, per-irrep metrics, and split-level evaluation logic.

Study references:

- `studies/minimal_overfit_observables/common.py:1166-1285`
  - aligned detailed metrics and basic matrix metrics
- `studies/minimal_overfit_study/common.py:2390-2609`
  - irrep filtering, splitting, and per-irrep metrics
- `studies/minimal_overfit_observables/common.py:2544-2642`
  - analogous irrep-aware metrics
- `studies/minimal_silicon_study/train_silicon_minimal.py:2209-2421`
  - split-level evaluation driver

Current `src/` status:

- `src/net/e3gnn.py:299-453`
  - batch-level metrics
- `src/core/sparse_math.py:92-185`
  - alignment primitives

Suggested destination:

- `src/utils/metrics.py`
- `src/train/logging.py`

### 5.4 DOS, generalized eigenvalues, and band structure

Gap:

The studies already contain DOS/eigenvalue analysis, while `src/` does not yet expose a stable postprocessing API for these electronic-structure outputs.

Study references:

- `studies/minimal_overfit_study/common.py:1483-1625`
  - generalized eigenvalues, DOS, DOS comparison plots
- `studies/minimal_overfit_observables/common.py:1613-1707`
  - corresponding observables-study implementation

Current `src/` status:

- `src/data/snapshot.py:239-245`
  - only core trace observables are exposed

Suggested destination:

- new `src/postprocessing/electronic_structure.py`

### 5.5 Visualization and figure-producing utilities

Gap:

The studies already contain matrix, irrep, DOS, and video-oriented visualization helpers that are not yet exposed from `src/`.

Study references:

- `studies/minimal_overfit_study/common.py:1538-1625,1870-2389`
  - DOS plots, Hamiltonian visualizations, frame export, video compilation
- `studies/minimal_overfit_observables/common.py:1668-1707,1996-2492`
  - analogous observables-study visualization paths

Current `src/` status:

- no dedicated visualization module under `src/`

Suggested destination:

- `src/utils/visualization.py`
- optional `src/utils/video.py`

## Task 6. Experiment Management, Logging, and Sweeps

### 6.1 Checkpointing and resume infrastructure

Gap:

The silicon workflow has robust checkpoint save/load, config restoration, and resume-from-WandB-run-id behavior that is not yet packaged under `src/`.

Study references:

- `studies/minimal_silicon_study/train_silicon_minimal.py:599-789`
  - checkpoint loading, path resolution, config restoration, change summaries
- `studies/minimal_silicon_study/train_silicon_minimal.py:2427-2491`
  - checkpoint/bootstrap logic
- `studies/minimal_silicon_study/train_silicon_minimal.py:3380-3552`
  - periodic checkpoint save and WandB summary updates
- `studies/minimal_silicon_study/train_silicon_minimal.py:3740-3784`
  - final checkpoint save and run summary

Current `src/` status:

- no dedicated `src/train/` runtime package

Suggested destination:

- `src/train/checkpointing.py`
- `src/train/runner.py`

### 6.2 Detailed logging and WandB payload builders

Gap:

The studies have a reusable detailed logging layer that is still outside `src/`.

Study references:

- `studies/minimal_overfit_study/detailed_logging.py:6-351`
- `studies/minimal_overfit_observables/detailed_logging.py:6-327`
- `studies/minimal_silicon_study/train_silicon_minimal.py:3096-3150,3477-3524`
  - integration of detailed metrics and per-irrep logs

Current `src/` status:

- `src/net/common.py:146-156`
  - logging toggles exist
- no stable detailed-logging helper package

Suggested destination:

- `src/train/logging.py`
- optionally `src/utils/logging.py`

### 6.3 Benchmark instrumentation and activation logging

Gap:

The studies contain richer benchmark timing and activation-magnitude logging than the stable code currently exposes as reusable helpers.

Study references:

- `studies/minimal_overfit_study/overfit_water_minimal.py:199-252`
  - logging and benchmark runtime flags
- `studies/minimal_overfit_observables/overfit_observables_minimal.py:280-340`
  - analogous runtime flags
- `studies/minimal_overfit_study/overfit_water_minimal.py:1415-1859`
  - benchmark accumulation and logging
- `studies/minimal_overfit_observables/overfit_observables_minimal.py:1274-1908`
  - benchmark accumulation and logging

Current `src/` status:

- `src/net/common.py:146-156`
  - basic logging and benchmark toggles only

Suggested destination:

- `src/train/logging.py`
- optional `src/utils/benchmark.py`

## Suggested Migration Order

To reduce churn, the safest order is:

1. Data preprocessing and validation
2. Metrics and postprocessing
3. Head variants and E3-MLP variants
4. Training/runtime orchestration
5. Visualization and final experiment-management polish

## Short Summary

The stable `src/` code already contains the correct core abstractions:

- `src/data/snapshot.py`
- `src/data/graph_features.py`
- `src/data/gnn_dataset.py`
- `src/core/sparse_math.py`
- `src/net/common.py`
- `src/net/encoders.py`
- `src/net/layers.py`
- `src/net/heads.py`
- `src/net/e3gnn.py`

The main migration work is therefore not a redesign. It is a consolidation task:

1. move the richer preprocessing and alignment pipeline into `src/`
2. expose the full architecture-search surface through stable modules
3. bring gauge-aware and observable-aware evaluation into reusable utilities
4. package training/runtime/logging infrastructure so the whole workflow lives in one codebase
