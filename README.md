# Mandala

An E(3)-equivariant graph neural network framework for block-sparse electronic-
structure operators: Hamiltonian **H**, overlap **S**, and density **D**.
Models, data interfaces, training, continuation, and physical evaluation share
one configurable framework.

## Modular operator architectures

Every `E3GNN` uses the same typed transformation graph. Flat `Config` settings
compile into that graph; explicit recipes select and connect its components.
The reference presets in [`configs/reference`](configs/reference) represent
DeepH, DeepH-E3, and MACE-H; [`configs/hybrids`](configs/hybrids) demonstrates
mixing their transformations with existing MANDALA attention and readouts.
Paper names select configuration files, not separate model classes.

```bash
PYTHONPATH=src:. python -m scripts.train \
  --config configs/reference/deeph_e3.yaml \
  --set data.root=/path/to/openmx_or_native_snapshots \
  --output /path/to/new_run --dry-run
```

Remove `--dry-run` to train. Continue with `--resume /path/to/new_run/latest_checkpoint.pt`.
Fitted output statistics use only the explicit training split and are saved in
the checkpoint. Native real and complex SOC blocks are supported.

See [configuration, training and inference](docs/modular_operator_models.md)
and the [reference coverage and verification matrix](docs/reference_architecture_coverage.md).
The [SiOx ROSI launchers](scripts/train_siox_models/README.md) prepare a shared
paper split and resumable DeepH, DeepH-E3, MACE-H and paper-configured MANDALA runs.
Flat `Config` settings remain supported. Old checkpoints with symmetry-breaking
coordinate-wise residual scales require explicit migration; see the
[scientific contract audit](docs/test_suite_audit.md#scientific-contract-follow-up--23-october-2026).

## Template-free OpenMX-compatible prediction export

Standalone inference requires only a loaded model, atom identities, Cartesian
positions (Å), and the lattice (Å, vectors as rows; `None` for a molecule):

```python
from data.openmx_writer import export_model_openmx

export_model_openmx(model, atoms=atoms, positions=positions,
                    lattice=lattice, output_path="predicted.matrix")
```

Or use an XYZ/extxyz file (periodic inputs must contain lattice and PBC metadata):

```bash
python scripts/export_prediction.py --checkpoint MODEL_CHECKPOINT.pt \
  --structure input.xyz --output predicted.matrix --device cpu
```

New checkpoints embed the orbital configuration. Older checkpoints without it
must be explicitly re-saved with `hyper_parameters['orbital_cfg']` set to the
loaded model's `model.mapper.orbital_cfg.to_dict()`; it cannot be inferred from
atom identities alone. No reference/template/info files are read by this path.

The model's cutoff and supplied (unwrapped) geometry determine the complete
periodic neighbor graph. `CpyCell` is the largest absolute integer shift component
in that graph (zero if only the origin is needed). This supports skewed lattices,
unwrapped atoms, and excludes cutoff equality, like the model graph. The exporter
builds the full `(2*CpyCell+1)^3` translation table exactly in OpenMX's
`Generation_ATV` order: origin at `Rn=0`, then ascending nested i/j/k loops with
k innermost, skipping the origin. `Rn` is looked up in the inverse `ratv` table,
not calculated using an arithmetic formula. Local neighbors are onsite first,
then ordered by neighbor atom and Rn, as in `Trn_System`.

Output uses OpenMX real H/S/D text-section headers and 20-decimal numeric rows.
Only predicted sections are emitted. A small `atv_ijk Rn=...` preamble provides
the translation mapping for readers (including MANDALA); position/momentum
overlaps and unpredicted matrices are **not fabricated**. This is a text dump,
not binary `.scfout`. It is not a byte-identical reconstruction of a particular
DFT calculation: OpenMX may choose a larger solver/orbital-dependent translation
table, and auxiliary quantities require additional predictors or integrals.
All supplied predictions must exactly match the cutoff graph, and exported
blocks are globally Hermitian-averaged with no density rescaling.

Density ingestion now uses **`0.5 * (D + D.T)`**, retaining native spin=0
normalization. Export writes density directly, without halving. Text and
processed-HDF5 OpenMX ingestion use the same convention; dataset caches are
invalidated to rebuild targets. Existing checkpoints and manually saved
snapshots trained/stored with doubled densities are not automatically converted.
Density-grid reconstruction and `Tr(DH)`/`Tr(DS)` use the supplied normalization;
physical spin sums, where needed, must be explicit.

## Installation

Package metadata currently declares **Python 3.10** (`>=3.10,<3.11`). Broader
Python support has not yet been certified. The commands below reproduce the
recorded PyTorch 2.5.1 CPU environment; they are a versioned installation example,
not a claim that every other PyTorch/CUDA combination is unsupported.

With Python 3.10 and platform build tools already installed:

```bash
git clone git@github.com:Mandala-org/Mandala.git mandala
cd mandala
python3.10 -m venv mandala-venv
source mandala-venv/bin/activate
python -m pip install --upgrade pip wheel
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install torch-scatter -f https://data.pyg.org/whl/torch-2.5.1+cpu.html --no-build-isolation
python -m pip install -e '.[dev]'
```

For the recorded CUDA 12.1 wheel stack, replace the PyTorch and torch-scatter
installation commands with:

```bash
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install torch-scatter -f https://data.pyg.org/whl/torch-2.5.1+cu121.html
```

Match compiled extensions to the selected PyTorch/CUDA stack and the host GPU
driver. On HPC systems use the site's project environment/profile and record
its exact runtime. Source builds also require compatible compiler/build tools.
Recorded GPU probes and their limited scope are described in the
[benchmark protocol](docs/operator_benchmark_pilots.md); they do not establish
compatibility for every GPU, Python version, or dependency combination.

## Focused verification

Run tests for the behavior being changed. A starting set for modular models is:

```bash
PYTHONPATH=src:. python -m pytest -q \
  tests/unit/net/test_pipeline_contracts.py \
  tests/unit/net/test_reference_fixtures.py \
  tests/workflow/test_modular_training.py
```

The scientific contract checks exercise trained symmetry, raw-source training
parity, periodic geometry, composable transformations, physical observables,
label isolation, and actual learning:

```bash
PYTHONPATH=src:. python -m pytest -q \
  tests/physics/test_o3_model_contracts.py \
  tests/physics/test_geometry_contracts.py \
  tests/physics/test_independent_operator_oracle.py \
  tests/unit/net/test_scientific_composition.py \
  tests/workflow/test_scientific_training_contracts.py
```

Their frozen upstream oracles and synthetic inputs run without source checkouts
or scientific datasets. Four learning experiments are marked `slow` and take
several minutes on CPU. Live source comparisons require the pinned checkouts
listed in [reference coverage](docs/reference_architecture_coverage.md).
GPU tests require `MANDALA_RUN_GPU_TESTS=1`; large-data tests require
`MANDALA_LARGE_DATA_ROOT`. Tests using repository scientific data additionally
need the relevant Git LFS objects (`git lfs install`, then `git lfs pull`).

The [test audit](docs/test_suite_audit.md) records dated verification results
and remaining limits. Its [22 October inventory](docs/test_inventory.json) is a
historical snapshot, not a current test count or measured line/branch coverage.
There is currently no tracked CI workflow; local passes do not imply CI exists.

## Workflows and demonstrations

- [Model configuration, training, continuation and inference](docs/modular_operator_models.md)
- [Original publication-example recipes](scripts/reproduce_examples/README.md)
- [Four-model SiOx comparison](scripts/train_siox_models/README.md)
- [Public benchmark pilots and metric conventions](docs/operator_benchmark_pilots.md)
- [Accuracy-investigation protocol and staged-compute policy](docs/mandala_accuracy_ablation_protocol.md)
- [Dated investigation summary](docs/mandala_accuracy_investigation.md)
- [Research execution and storage](docs/RESEARCHOPS.md)
- [Executable Python demos](demos/README.md), including checkpoint evaluation
- [Offline interactive SOC rotation lab](demos/soc_rotations/index.html)

The documentation above is maintained in this repository. A searchable API
website and an expanded compatibility matrix remain release work; neither is
implied by the existing tutorials.
