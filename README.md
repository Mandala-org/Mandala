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

The tests should run pass

```bash
pytest
```

To launch training, pick a config and run:

```bash
python scripts/train.py --config-name debug_cpu
```
