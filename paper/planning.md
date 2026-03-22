# Mandala CPC Draft Plan

## Target article type

- Venue: *Computer Physics Communications*.
- Article type: *Computer Programs in Physics* (CPiC).
- Immediate objective: produce an "alpha" manuscript that is structurally close to a submission-ready program paper, while intentionally leaving numerical results and final benchmark figures for a later revision.

## Paper positioning

- The paper should present Mandala as a software framework for learning electronic-structure objects, not as a single-purpose benchmark model.
- The central scientific and software claim is that Mandala combines:
  - a modular, code-oriented architecture for sparse DFT matrix learning,
  - physically meaningful downstream supervision on observables such as total energy and atomic forces,
  - a broad and configurable E(3)-equivariant architecture search space.
- The draft should treat the capabilities currently split between `src/` and the study directories as one coherent framework, because the studies already define the intended functionality and API direction.

## Repository-derived narrative

### Stable code path in `src/`

- `Snapshot`, `BlockMatrix`, and `IrrepsBlockData` provide the central data abstraction for one structure and its block-sparse Hamiltonian, overlap, and density matrices.
- `BlockIrrepMapper` and the basis converters define the representation bridge between atomic-orbital matrix blocks and irreducible E(3) features.
- `DatasetFactory` and `E3GNNDataset` provide a reusable training data pipeline from raw electronic-structure files to graph inputs and physics targets.
- `E3GNN` together with `encoders.py`, `layers.py`, and `heads.py` provides the modular baseline training workflow using Hydra, PyTorch Lightning, and W&B.

### Extended capabilities already visible in studies

- `minimal_overfit_study` exposes a rich family of architecture variants for message passing and equivariant head design.
- `minimal_overfit_observables` extends the framework to simultaneous prediction of Hamiltonian, overlap, and density, together with observable losses on energy and electron count.
- `minimal_silicon_study` extends this further to multi-snapshot materials datasets and force supervision through autograd on predicted observables.

## Proposed manuscript structure

### 1. Introduction

- Motivate the computational cost of Kohn-Sham DFT and the need for surrogates that preserve access to electronic observables rather than only atomic energies.
- Position Mandala relative to machine-learned interatomic potentials and other ML electronic-structure surrogates.
- State the three main contributions:
  - modular sparse-matrix learning framework,
  - observable-aware and force-aware training,
  - architecture-search-ready E(3)-equivariant design.

### 2. Theoretical background

- Subsection on the electronic-structure problem and Kohn-Sham DFT.
- Subsection on E(3)-equivariant graph neural networks for atomistic modeling.
- Subsection connecting equivariant local representations to block-sparse matrix prediction and downstream observables.

### 3. Numerical implementation

- Software design goals and overall workflow.
- Data ingestion, parser layer, and basis harmonization.
- Sparse matrix representation and irrep mapping.
- Graph construction and equivariant message passing.
- Multi-head prediction of Hamiltonian, overlap, and density.
- Observable calculation and training on energy, electron count, and forces.
- Hyperparameter optimization and architecture search space.
- Verification, testing, and extensibility.

### 4. Conclusion

- Summarize the software contribution and planned scientific uses.
- Defer quantitative performance assessment to a later manuscript revision.

## Writing choices for the alpha draft

- Write the paper as if the observable-training and architecture-search features are already part of the stable public interface.
- Keep the discussion results-free: no benchmark claims that require final numbers.
- Prefer precise implementation language over speculative promises.
- Emphasize modularity repeatedly, but always tie it to concrete software abstractions.
- Mention parser support for OpenMX, FHI-aims, and PySCF-derived artifacts as evidence of extensibility.

## File layout to implement

- `paper/main.tex`
- `paper/sections/abstract.tex`
- `paper/sections/program_summary.tex`
- `paper/sections/introduction.tex`
- `paper/sections/background.tex`
- `paper/sections/implementation.tex`
- `paper/sections/conclusion.tex`
- `paper/sections/acknowledgments.tex`

## Known placeholders for later revisions

- Final author list and correspondence footnotes.
- Final repository DOI / CPC program-library metadata.
- Numerical benchmark section and figures.
- Final comparison against external methods.
- Final wording of the generative-AI disclosure statement.
