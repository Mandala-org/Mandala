import torch
from omegaconf import DictConfig
import dataclasses

from net.e3gnn import E3GNN
from net.common import HyperParams
from data.factory import DatasetFactory

# 1. Create a DatasetFactory with position gradients enabled
factory = DatasetFactory(
    cutoff_gnn=3.0,
    cutoff_matrix=4.0,
    l_max_sh=2,
    n_radial=64,
    enable_forces=True,  # This is crucial for the test
)

# Add a small, real data snapshot
factory.add_snapshot(
    "data/big/silicon/900K/Si_DM",
    "data/big/silicon/900K/info.txt",
)

# Create the dataset and mapper
train_ds, _, mapper = factory.create()

# Get a sample. The dataset should have set requires_grad on positions.
x, _ = train_ds[0]

# 2. Set up the model configuration
hp = HyperParams(hidden_base_dim=16, l_max=2)
cfg = DictConfig(
    {
        "model": dataclasses.asdict(hp),
        "training": {"lr": 1e-3},
        "logging": {"pedantic": True},  # Enable pedantic checks
    }
)

model = E3GNN(mapper, train_ds.edge_types, cfg)
# 3. Predict forces and check that they are not all zero
forces = model.predict_forces(x)

assert forces.shape == x["positions"].shape
assert not torch.allclose(
    forces, torch.zeros_like(forces)
), "Forces are all zero, gradients are likely detached."
