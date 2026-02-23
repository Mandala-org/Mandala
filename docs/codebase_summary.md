# Codebase Summary (Stable Architecture Notes)

This note captures the persistent project architecture and avoids run-specific or in-progress study details.

## High-Level Flow

1. Parse electronic-structure outputs (OpenMX/FHI-aims) into sparse block matrices.
2. Store data as `Snapshot` objects containing `hamiltonian`, `overlap`, and `density`.
3. Convert blocks to irrep vectors via `BlockIrrepMapper` when needed.
4. Build graph features (neighbors, spherical harmonics, radial embeddings).
5. Run E(3)-equivariant GNN (encoders -> message passing -> heads).
6. Train on matrix/irrep losses and optional observable losses (`Tr(DH)`, `Tr(DS)`), with optional force/stress gradients.

## Core Modules (`src/core`)

- `orbital_irrep_config.py`
  - Parses/validates per-element orbital irreps.
  - Supports YAML/dict and compact orbital notation.
  - Provides element dims and merge helpers.
- `block_irrep_mapper.py`
  - Central block <-> irrep conversion using reduced tensor products.
  - Builds per-element-pair mapping tensors and edge-type indices.
- `basis_converter.py`
  - Basis conversions among OpenMX, e3nn, and FHI-aims conventions.
- `irreps_builder.py`
  - Auto-builds hidden irreps and checks tensor-product path feasibility.
- `sparse_math.py`
  - Sparse trace operations and matrix/vector conversion helpers.

## Data Modules (`src/data`)

- `openmx_info_parser.py`
  - Parses energies, occupancies, eigenvalues, dipoles, geometry, forces, and inferred box.
- `openmx_parser.py`
  - Parses sparse matrix blocks from OpenMX outputs with periodic shifts.
  - Builds `BlockMatrix` and optional convention conversion.
- `fhiaims_parser.py`
  - Parses geometry, basis indices, and ELSI CSC matrices from FHI-aims outputs.
- `block_matrix.py`
  - Sparse block container with algebra, transpose, masking/reordering, rotation, dense conversion, and serialization.
  - Includes `IrrepsBlockData` for irrep-space storage.
- `snapshot.py`
  - Aggregate container for H/S/D plus geometry metadata.
  - Physics helpers (`get_energy`, `get_number_of_electrons`), basis change, rotation, filtering, export helpers.
- `graph_features.py`
  - Neighbor-list construction, deterministic edge ordering, edge typing, SH/radial embeddings.
- `gnn_dataset.py`
  - In-memory dataset pipeline (snapshot -> model input/targets) with optional caching.
- `factory.py`
  - Builds train/val datasets with a single shared global mapper.

## Network Modules (`src/net`)

- `common.py`
  - Main `Config` dataclass and shared building blocks (`build_hidden_irreps`, `RadialMLP`, `E3MLP`, tensor-product helpers).
- `activations.py`
  - Equivariant and scalar activation factories.
- `encoders.py`
  - Node and edge feature encoders.
- `layers.py`
  - Equivariant convolution/message passing blocks (`EquiConv`, `EdgeUpdateBlock`, `NodeUpdateBlock`, `MessageBlock`).
- `heads.py`
  - Per-pair equivariant readout heads to produce output irreps.
- `e3gnn.py`
  - Full Lightning model: forward pass, losses, observables, optimizer/scheduler config, optional forces/stress prediction.
- `benchmark.py`
  - Training/validation timing and profiling callback with YAML report output.

## Utility Modules (`src/utils`)

- `summary.py`: model/dataset summary and edge statistics.
- `units.py`: unit constants (Hartree -> eV).
- `__init__.py`: utility exports.

## Design Characteristics

- Strong emphasis on explicit sparse block handling and physics-aware observables.
- Separation between data parsing/representation logic and model/training logic.
- Reusable mapper-centered design for consistent block/irrep transforms across parsing, training, and evaluation.
