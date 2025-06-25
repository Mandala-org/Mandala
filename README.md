# E3MatrixNet

An E(3)-equivariant Graph Neural Network that predicts **sparse DFT block
matrices** (Hamiltonian **H**, Density **D**, Overlap **S**) using E3NN +
PyTorch.  Designed for arbitrary chemistry, Metatensor IO, and distributed
training on A100/V100 clusters.

This repository is under heavy development — follow `docs/` for design notes.

## Quickstart

Hydra configuration files have been flattened. The `conf/` directory now contains four standalone config files:

- `debug_cpu.yaml`
- `small_gpu.yaml`
- `medium_gpu.yaml`
- `debug_cpu_silicon.yaml`

To launch training, pick a config and run:

```bash
python scripts/train.py --config-name debug_cpu
```
