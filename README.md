# Mandala

An E(3)-equivariant Graph Neural Network implementation framework to predict
**block-sparse DFT matrices** (Hamiltonian **H** and Density **D**)
in linear time using E3NN + PyTorch. Designed for arbitrary chemistry,
wide hyperparameter optimization and distributed training on HPC clusters.

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

## Quickstart

To install run

```bash
# Clone the repository
git clone git@github.com:Mandala-org/Mandala.git mandala
cd mandala
# Get proper Python version and utils (provided for Debian-based systems)
sudo add-apt-repository -y 'ppa:deadsnakes/ppa'
sudo apt install python3.10 python3.10-venv python3.10-distutils
sudo apt install python3-dev build-essential
# Create venv
python3.10 -m venv mandala-venv
source mandala-venv/bin/activate
# Update pip
python -m pip install --upgrade pip
# Install torch if you don't have it
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cpu
# Install wheel and then torch-scatter
pip install wheel
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.5.1.html --no-build-isolation
# Install Mandala
pip install -e .[dev]
```

Installation requires gcc>=9.3.0 and cmake.
Requires Python >=3.10,<3.11.

## HPC (CUDA) Quickstart

To install run

```bash
# Clone the repository
git clone git@github.com:Mandala-org/Mandala.git mandala
cd mandala
# Create venv
python3.10 -m venv mandala-venv
source mandala-venv/bin/activate
# Update pip
python -m pip install --upgrade pip
# Install torch if you don't have it
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
# Install torch-scatter
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.5.1+cu121.html
# Install Mandala
pip install -e .[dev]
```

Installation requires gcc>=9.3.0 and cmake.
The instructions depend on having cuda12.1 installed.
Other versions of cuda or torch have not been tested.
Requires Python >=3.10,<3.11.

Before running tests make sure to download lsf-handled files (see [#7](https://github.com/Mandala-org/Mandala/issues/7) if not on debian-based system)
```bash
sudo apt install git-lfs
git lfs install
git lfs pull
```

The test suite is under active cleanup; run it as a development check

```bash
python -m pytest
```

To launch training, pick a config and run:

```bash
python scripts/train.py --config-name debug_cpu
```

For a short end-to-end inference example, see
[`demos/demo_07_checkpoint_evaluation.py`](demos/demo_07_checkpoint_evaluation.py).
Checkpoint evaluation uses the existing `Snapshot` and `E3GNN` classes together
with stateless helpers; no new checkpoint format is required.
