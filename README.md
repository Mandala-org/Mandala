# Mandala

An E(3)-equivariant Graph Neural Network implementation framework to predict
**block-sparse DFT matrices** (Hamiltonian **H** and Density **D**)
in linear time using E3NN + PyTorch. Designed for arbitrary chemistry,
hyperoptimization and distributed training on HPC clusters.

## Quickstart

To install run

```bash
# Clone the repository
git clone git@github.com:Mandala-org/Mandala.git mandala
cd mandala
# Create venv
python -m venv mandala-venv
source mandala-venv/bin/activate
# Update pip
python -m pip install --upgrade pip
# Install
pip install -e .[dev]
```

The tests should run pass

```bash
pytest
```

To launch training, pick a config and run:

```bash
python scripts/train.py --config-name debug_cpu
```
