"""Build the default MANDALA E3GNN and predict operators for a crystal.

Run from the repository root::

    PYTHONPATH=src:. python examples/quickstart/model.py
"""

from pathlib import Path

import torch

from core.sparse_math import trace_matmul_sparse_block_matrix
from data.factory import DatasetFactory
from net.common import Config
from net.e3gnn import E3GNN

ROOT = Path(__file__).resolve().parents[2]
SILICON = ROOT / "data" / "small" / "Si"

# A deliberately small network: two message-passing layers, l <= 2 features.
cfg = Config(
    cutoff_radius=6.0,
    l_max=2,
    hidden_irreps="32x0e+8x0o+16x1o+8x1e+8x2e+4x2o+4x3o+4x3e+2x4e",
    n_radial=16,
    radial_layers=(32,),
    num_layers_gnn=2,
    node_update_message_agg="attention",
    matrix_targets=["hamiltonian", "overlap", "density"],
    verbosity=0,
)

# The factory parses the calculation, applies the cutoff, builds the periodic
# graph and returns the orbital mapper shared by data and model.
factory = DatasetFactory(cfg, convention="e3nn")
factory.add_snapshot(SILICON / "HS.out", SILICON / "Si.out", purpose="train")
train, _, mapper = factory.create()
x, y = train[0]

torch.manual_seed(0)
model = E3GNN(mapper=mapper, cfg=cfg)
print("parameters:", sum(p.numel() for p in model.parameters()))
print("pipeline stages:", len(model.pipeline.records))
print("composite blocks:", list(model.pipeline.groups))

# The forward pass returns irrep coefficients per atom pair ...
irreps = model(x)
print("Si-Si irrep vectors:", tuple(irreps["hamiltonian"]["Si-Si"].shape))

# ... and predict_matrices returns physical BlockMatrix operators.
with torch.no_grad():
    predicted = model.predict_matrices(x)
H, S, D = (predicted[name] for name in ("hamiltonian", "overlap", "density"))
print("Si-Si blocks:", tuple(H["Si-Si"].shape))
print("Tr(DS) of the untrained model:", float(trace_matmul_sparse_block_matrix(D, S)))
