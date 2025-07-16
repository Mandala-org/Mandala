# mandala

An E(3)-equivariant Graph Neural Network implementation framework to predict
**block-sparse DFT matrices** (Hamiltonian **H** and Density **D**)
in linear time using E3NN + PyTorch. Designed for arbitrary chemistry,
hyperoptimization and distributed training on A100/V100 clusters.

## Quickstart

To install run

```bash
git clone git@github.com:Mandala-org/Mandala.git mandala
cd mandala
python -m venv mandala-venv
source mandala-venv/bin/activate
pip install -e .[dev]
```

To launch training, pick a config and run:

```bash
python scripts/train.py --config-name debug_cpu
```
