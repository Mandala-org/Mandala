# Mandala Workflow Inventory

This document is a consolidated inventory of the full Mandala workflow, written as if the features currently prototyped in `studies/minimal_silicon_study`, `studies/minimal_overfit_study`, and `studies/minimal_overfit_observables` have already been merged into the stable `src/` codebase.

The goal is to make the full framework easy to sketch in the paper as one coherent architecture. To support that, the inventory is organized by implementation tasks rather than by modules alone.

The following study-only options are intentionally excluded from this inventory:

- `convention`
- `orbital-selection`
- `sh-mode`
- `xyz-permutation`
- `change-box`
- `box-convention`

## One-Line Summary

Mandala is a modular E(3)-equivariant graph neural network framework for learning sparse electronic-structure operators in localized orbital bases, assembling sparse Hamiltonian, overlap, and density matrices, and supervising those predictions both at the matrix level and through observable guidance on energy, number of electrons, forces, and stress.

## Figure-Oriented Task Layout

The cleanest top-level figure for the paper is a task pipeline:

1. Data loading, handling, and preprocessing
2. Model creation and architecture variants
3. Sparse operator prediction and matrix assembly
4. Training routine, targets, and losses
5. Physics-aware postprocessing and evaluation
6. Experiment management, logging, and sweeps

The sections below expand each task into finer-grained sub-tasks.

## 1. Data Loading, Handling, and Preprocessing

### 1.1 Backend data ingestion

Mandala ingests localized-basis electronic-structure data from multiple backends and converts them into a common internal representation under `src/data/`.

Supported backend families:

- OpenMX snapshots with sparse Hamiltonian, overlap, and density matrices
- FHI-aims outputs converted into the same sparse localized-orbital representation
- PySCF-generated localized-basis data mapped into the same internal format

Each parsed configuration contributes:

- atomic species
- atomic positions
- periodic simulation box when present
- sparse Hamiltonian blocks
- sparse overlap blocks
- sparse density blocks
- derived scalar observables such as energy and electron count
- optional force targets
- optional stress targets

### 1.2 Unified snapshot representation

After parsing, all backends are represented uniformly through a `Snapshot`-level abstraction containing:

- sparse block operators indexed by atom pair and lattice shift
- shared orbital configuration metadata
- basis-conversion metadata
- force and stress targets when available

This is the point where backend-specific file formats disappear and the rest of the workflow becomes backend-agnostic.

### 1.3 Sparse operator standardization

Mandala standardizes all sparse operators before graph construction and training.

Primary operator objects:

- Hamiltonian `H`
- overlap `S`
- density `D`

Each operator is treated as a sparse block matrix with explicit edge metadata:

- lattice shift
- source atom index
- destination atom index

Standardization steps include:

1. sparse block parsing
2. optional Hamiltonian unit conversion
3. canonical edge ordering
4. optional cutoff filtering of target matrices
5. fixed basis/orbital mapping
6. block-to-irrep conversion through a shared mapper

### 1.4 Dataset discovery and split construction

The merged workflow supports:

- explicit train/validation file-pair registration
- directory-based snapshot discovery
- split construction from discovered pairs
- restart- and checkpoint-aware dataset reuse

### 1.5 Caching

Two cache levels are part of the merged workflow:

- raw `Snapshot` caching for parsed backend outputs
- preprocessed-sample caching for graph-aligned training samples

The second cache level is especially important for larger datasets and repeated architecture sweeps.

### 1.6 Graph construction for periodic systems

Mandala builds a periodic atomistic graph from each structure and aligns it with the sparse operator representation.

Graph nodes carry:

- atomic species embedding
- position and cell context for geometric feature generation

Graph edges are split into:

- off-diagonal atom-pair edges
- explicit self-edges
- explicit shifted self-edges when they are present in the sparse operators

Per-edge metadata includes:

- source and destination atom indices
- lattice-shift vector
- pair-type index
- displacement vector
- radial embedding
- spherical-harmonic features

### 1.7 Graph-target alignment and validation

This is one of the most important implementation tasks in Mandala.

The merged workflow supports:

- canonicalization of sparse matrix edges
- exact graph-to-target edge reconciliation
- reverse-edge completion where required
- trace-alignment metadata for efficient observable evaluation
- safety checks validating graph/operator consistency

This alignment step is what lets the framework move cleanly between:

- graph-native equivariant message passing
- sparse operator assembly
- trace-based observables such as energy and electron count

### 1.8 Preprocessed training sample assembly

After graph construction and reconciliation, each training sample packages:

- graph inputs
- matrix block targets or irrep targets
- observable targets
- any auxiliary alignment information needed for losses or metrics

This makes the downstream training loop independent of raw backend formats.

## 2. Model Creation and Architecture Variants

### 2.1 Representation configuration

Mandala uses an E(3)-equivariant hidden representation built from irreducible representations.

Configurable representation choices include:

- base hidden multiplicity
- maximum angular momentum `l_max`
- explicit hidden-irrep string
- parity content

### 2.2 Input encoders

The encoder stack includes:

- node encoder for atom-type initialization
- edge encoder combining geometry and pair-type information
- optional tensor-square preprocessing of spherical-harmonic features before edge encoding
- optional equivariant normalization in the encoder path

### 2.3 Message-passing trunk

The main GNN trunk supports:

- configurable number of message-passing layers
- tensor-product based edge updates
- multiple node-update variants
- multiple edge-update variants
- optional residual connections
- optional self-connections
- configurable tensor-product variants

### 2.4 Equivariant nonlinearity and E3-MLP variants

One of Mandala’s main strengths is that it exposes a broad family of equivariant nonlinear and E3-MLP choices rather than one fixed architecture.

The merged workflow includes:

- `NormAct`
- `Gate`
- `GateScalarsMLP`
- `GateMagnitudes`
- residual E3-MLP variants
- bilinear self-tensor-product variants
- invariant self-attention variants
- scalar-magnitude self-tensor-product gated variants

This family should be treated in the paper as a deliberate architecture-search space.

### 2.5 Head design and output variants

After message passing, Mandala predicts sparse operator coefficients and assembles them into sparse matrices.

The unified output layer supports any subset of:

- Hamiltonian
- overlap
- density

The merged head space includes:

- shared equivariant trunk plus per-operator projections
- configurable head depth
- configurable number of head-specific equivariant MLP layers
- optional tensor-square preprocessing before the head projection
- optional use of node embeddings for self-edge prediction
- separate handling of shifted self-edges
- optional scalar-specific branches
- optional magnitude-factorized block prediction

### 2.6 Architecture search dimensions

The full architecture search space spans:

- hidden irreps
- `l_max`
- message-passing depth
- cutoff radius
- number of radial basis functions
- radial embedding scale
- tensor-product implementation
- edge update mode
- node update mode
- residual toggles
- self-connection toggle
- head depth
- head variant
- equivariant nonlinearity family
- matrix target subset

## 3. Sparse Operator Prediction and Matrix Assembly

### 3.1 Prediction form

Mandala handles outputs in two equivalent forms:

- irrep-vector form for equivariant computation
- sparse block-matrix form for physical observables and final outputs

Conversion between the two is handled by the shared block/irrep mapper.

### 3.2 Sparse assembly

Predicted coefficients are assembled into sparse block matrices for:

- `H`
- `S`
- `D`

This assembly preserves the graph-aligned edge structure and allows downstream matrix-level metrics and trace-based observables to be evaluated exactly on sparse objects.

### 3.3 Structural postprocessing before observables

Before observables or metrics are computed, the merged workflow can apply:

- matrix symmetrization for metrics and visualization
- shifted-self-aware block handling
- trace-aligned sparse multiplication utilities

## 4. Training Routine, Targets, and Losses

### 4.1 Training targets

Mandala supports supervision on any subset of:

- Hamiltonian blocks
- overlap blocks
- density blocks
- energy observable
- number of electrons
- forces
- stress tensor

### 4.2 Matrix-level supervision

The matrix-supervision layer supports:

- full multi-target matrix training
- configurable target subset selection
- matrix loss aggregation across sparse keys
- optional density-specific loss reweighting
- symmetrized evaluation paths

### 4.3 Observable guidance

Mandala’s core methodological strength is that matrix learning can be augmented by observable guidance derived from the predicted operators.

Observable-guided supervision includes:

- energy guidance from `Tr(DH)`
- electron-count guidance from `Tr(DS)`
- force guidance from energy derivatives with respect to atomic positions
- stress guidance from energy derivatives with respect to the periodic cell

The merged workflow supports:

- pure matrix supervision
- mixed matrix plus observable guidance on any selected subset of energy, electron count, forces, and stress
- full combined training on matrices and all selected observables

The training loop combines:

- per-matrix block losses
- observable losses for all enabled guided quantities

Possible training-time controls include:

- coefficient on observable losses
- coefficient on force losses
- coefficient on stress losses
- density-specific matrix weighting
- optional use of ground-truth operators for partial observable diagnostics

### 4.4 Training loop mechanics

The merged runtime also includes:

- gradient clipping
- gradient accumulation
- learning-rate scheduling
- benchmark timing hooks
- proper graceful termination and resuming of training runs

## 5. Physics-Aware Postprocessing and Evaluation

### 5.1 Symmetrization

Mandala supports symmetrization of predicted operators before matrix-level metrics and visualization.

### 5.2 Hamiltonian gauge correction

Mandala evaluates Hamiltonian predictions up to the overlap gauge freedom:

- `H -> H - mu_H S`

where `mu_H` is chosen to minimize the mean-squared deviation with respect to the target Hamiltonian.

This belongs in the paper figure as part of the postprocessing/evaluation task rather than as a low-level implementation detail.

### 5.3 Density normalization

The merged workflow supports density normalization so that the predicted density satisfies the electron-count constraint:

- `Tr(D_pred S_pred) = N_target`

This is used as an evaluation and correction path and should appear explicitly in the workflow sketch.

### 5.4 Matrix-space evaluation

Mandala’s evaluation layer supports:

- per-matrix MAE/MSE
- aligned matrix metrics
- per-irrep metrics
- distance-resolved error curves
- gauge-corrected Hamiltonian diagnostics

### 5.5 Electronic-structure evaluation

From predicted `(H, S, D)`, the merged workflow supports:

- generalized eigenvalue recovery
- density-of-states comparison
- band-structure analysis from the predicted Hamiltonian

### 5.6 Visualization

The merged analysis layer includes:

- matrix heatmaps
- per-irrep images
- frame-by-frame training visualizations
- training-video generation from saved frames

## 6. Experiment Management, Logging, and Sweeps

### 6.1 Checkpointing and restart support

The merged workflow includes:

- latest / best / final checkpoints
- resume-from-checkpoint
- resume-from-WandB-run-id
- checkpoint-aware config restoration
- validation against incompatible resume-time overrides

### 6.2 Logging and diagnostics

Mandala supports:

- configurable console logging
- first-sample data and graph logging
- model-shape logging
- activation-magnitude logging
- per-irrep metric logging
- detailed WandB dashboards
- timing/benchmark reporting

### 6.3 Sweep and search support

The framework is built to support systematic exploration over:

- representation choices
- message-passing variants
- head variants
- observable-guidance settings
- loss-weighting settings
- data and preprocessing policies

## 7. Compact End-to-End Task Summary

The full merged workflow can therefore be summarized as the following task chain:

1. Load backend data and convert it into a unified sparse-operator `Snapshot`
2. Canonicalize, preprocess, reconcile, and cache graph-aligned samples
3. Build an E(3)-equivariant GNN with configurable encoders, message-passing blocks, nonlinearities, and heads
4. Predict sparse Hamiltonian, overlap, and density operators
5. Train with matrix losses and optional observable guidance on energy, electron count, forces, and stress
6. Apply physics-aware postprocessing such as symmetrization, gauge correction, and density normalization
7. Evaluate, visualize, and analyze the resulting operators through metrics, DOS, and band-structure access
8. Log, checkpoint, resume, and sweep over architecture and training variants

## Figure-Authoring Notes

If this document is turned into a paper figure, the most natural visual layout is:

- a left-to-right main pipeline for Tasks 1 to 5
- a side branch from predicted operators to postprocessing/evaluation
- a second side branch for logging, checkpoints, and sweeps

The three most important messages the figure should communicate are:

1. Mandala predicts sparse electronic-structure operators, not only scalar properties.
2. Those operators support both matrix reconstruction and observable-guided supervision.
3. The framework exposes a broad modular architecture-search space rather than one fixed model.
