# MANDALA

[![Tests](https://github.com/Mandala-org/Mandala/actions/workflows/tests.yml/badge.svg)](https://github.com/Mandala-org/Mandala/actions/workflows/tests.yml)

**Documentation:** https://mandala-org.github.io/Mandala/

**MANDALA** is an E(3)-equivariant graph neural network framework that learns
the electronic structure of molecules and materials as sparse operators. From
atomic positions, species and lattice it predicts the atom-pair-resolved
Hamiltonian **H**, overlap **S** and density matrix **D** in a localized
atomic-orbital basis, and derives observables from them — electron counts
`Tr(DS)`, band energies `Tr(DH)`, band structures and densities of states.
Because these observables are differentiable functions of the network
output, they also guide training.

- **Operators with exact symmetry.** Blocks transform as
  `H_ij -> D_i(Q) H_ij D_j(Q)^T` under rotations and reflections, by
  construction.
- **Observable and spectral guidance.** Band energies, electron counts and
  eigenvalues near the Fermi level enter the loss next to matrix elements.
- **One typed pipeline for every architecture.** The MANDALA network, DeepH,
  DeepH-E3, MACE-H and hybrids are YAML compositions of the same validated
  stages; new layers are registered in a few lines.
- **Reproducible runs.** Frozen manifests, immutable splits and exact
  epoch-boundary continuation.

## Installation

```bash
git clone git@github.com:Mandala-org/Mandala.git mandala && cd mandala
git lfs install && git lfs pull
python3.10 -m venv mandala-venv && source mandala-venv/bin/activate
python -m pip install --upgrade pip wheel
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install torch-scatter -f https://data.pyg.org/whl/torch-2.5.1+cpu.html --no-build-isolation
python -m pip install -e '.[dev]'
```

For CUDA 12.1 use the `cu121` index URLs for `torch` and `torch-scatter`.
Commands are run from the repository root with `PYTHONPATH=src:.`.

Check the installation (about twenty seconds on a CPU):

```bash
PYTHONPATH=src:. python examples/layer_cookbook/verify.py
```

## A first look

```bash
PYTHONPATH=src:. python examples/quickstart/operators.py   # operators, traces, irreps, bands
PYTHONPATH=src:. python examples/quickstart/model.py       # build a model and predict H, S, D
PYTHONPATH=src:. python -m scripts.train --config examples/first_model/silicon.yaml \
  --output runs/silicon --accelerator cpu                   # train on bundled silicon data
```

## Documentation

The documentation covers installation, the data model, configuration,
training and inference, the mathematics of every component, the typed
pipeline, a cookbook that implements a new layer of every kind, and the
reproduction of all paper results. Build it with

```bash
source mandala-venv/bin/activate
python -m pip install -e '.[docs]'
PYTHONPATH=src:. python -m sphinx -b html docs docs/_build/html
```

and open `docs/_build/html/index.html`.

| Topic | Source |
| --- | --- |
| Installation, quickstart, first model | [`docs/getting_started`](docs/getting_started) |
| Data, configuration, training, inference | [`docs/user_guide`](docs/user_guide) |
| Symmetry, operators, architecture, objectives | [`docs/theory`](docs/theory) |
| Typed pipeline and component catalogue | [`docs/architecture`](docs/architecture) |
| Writing new layers (cookbook) | [`docs/extending`](docs/extending), [`examples/layer_cookbook`](examples/layer_cookbook) |
| Reproducing the paper | [`docs/reproduction`](docs/reproduction), [`scripts/reproduce_examples`](scripts/reproduce_examples) |

## Tests

GitHub Actions runs the unit, physics and equivariance subsets on CPU with
Python 3.10 and experimental Python 3.11 (allowed to fail while the package
requires Python <3.11), and prints line coverage in the job log. GPU, network,
large-data, slow, integration and workflow tests are excluded; data-file
exclusions are listed explicitly in [the workflow](.github/workflows/tests.yml).
Optional comparisons against external upstream source checkouts are skipped
when those checkouts are unavailable.

## Repository layout

| Path | Content |
| --- | --- |
| `src/core` | Orbital/irrep bookkeeping, basis converters, sparse algebra |
| `src/data` | Parsers, `Snapshot`, `BlockMatrix`, graphs, datasets, exporters |
| `src/net` | `E3GNN`, `FeaturePipeline`, layers, objectives |
| `src/analysis` | Evaluation reports, DOS/bands, density grids |
| `scripts` | Training, export, evaluation and benchmarking tools |
| `configs` | Reference (`mandala_paper`, `deeph`, `deeph_e3`, `mace_h`) and hybrid recipes |
| `examples` | Quickstart, first model and layer cookbook |
| `sweeps/paper_ablations` | Ablation sweeps of the paper |
| `paper` | Manuscript, figures and result data |
| `tests` | Unit, physics, equivariance and workflow tests |

## License

See [LICENSE](LICENSE).
