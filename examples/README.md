# Examples

Run every example from the repository root with `PYTHONPATH=src:.`.

| Example | What it shows |
| --- | --- |
| `quickstart/operators.py` | Load an OpenMX calculation; blocks, traces `Tr(DS)` and `Tr(DH)`, Hermiticity, block-to-irrep round trip, band structure |
| `quickstart/model.py` | Build the default MANDALA E3GNN and predict H, S and D |
| `quickstart/evaluate.py` | Restore a checkpoint, report matrix and band-energy errors, plot bands |
| `first_model/silicon.yaml` | Train a compact model on the bundled silicon calculations |
| `layer_cookbook/layers.py` | A new layer of every E3GNN kind, registered with the pipeline compiler |
| `layer_cookbook/cookbook.yaml` | A complete network built only from cookbook layers |
| `layer_cookbook/verify.py` | Numerical E(3)-equivariance certificate for every layer and the full model |
| `layer_cookbook/train.py` | The standard trainer with the cookbook layers registered |

```bash
PYTHONPATH=src:. python examples/layer_cookbook/verify.py
PYTHONPATH=src:. python -m scripts.train --config examples/first_model/silicon.yaml \
  --output runs/silicon --accelerator cpu
PYTHONPATH=src:. python examples/quickstart/evaluate.py \
  --checkpoint runs/silicon/best_model.pt \
  --structure data/small/Si_perturbed_scale1_078 --plot bands.png
```

The documentation (`docs/`, built with Sphinx) explains each example in
detail.
