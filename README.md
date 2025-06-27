# E3GNN4Matrix

An E(3)-equivariant Graph Neural Network implementation framework to predict
**block-sparse DFT matrices** (Hamiltonian **H** and Density **D**)
in linear time using E3NN + PyTorch. Designed for arbitrary chemistry,
hyperoptimization and distributed training on A100/V100 clusters.

## Quickstart
To launch training, pick a config and run:

```bash
python scripts/train.py --config-name debug_cpu
```
