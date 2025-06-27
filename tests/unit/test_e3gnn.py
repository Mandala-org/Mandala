import pytest
import torch

from core.orbital_irrep_config import OrbitalIrrepConfig
from net.common import HyperParams
from core.block_irrep_mapper import BlockIrrepMapper
from net.e3gnn import E3GNN
from e3nn.o3 import Irreps


@pytest.mark.unit
def test_forward_smoke():
    # ------- dummy orbital config ------------------
    cfg = OrbitalIrrepConfig.from_dict({"H": "1x0e"})
    edge_types = ["H-H"]

    hp = HyperParams(num_layers_gnn=1, num_layers_matrix=1)

    model = E3GNN(BlockIrrepMapper(cfg), edge_types, hp=hp, device="cpu")

    # ------- fake batch ----------------------------
    N, E = 4, 3
    node_one_hot = torch.eye(1)[torch.zeros(N, dtype=torch.long)]  # all H
    edge_one_hot = torch.eye(1)[torch.zeros(E, dtype=torch.long)]  # H-H
    edge_len = torch.randn(E, hp.n_radial)
    sh = Irreps.spherical_harmonics(hp.l_max)
    edge_sh = torch.randn(E, sh.dim)  # random SH features

    x_gnn = {
        "node_one_hot": node_one_hot,
        "edge_one_hot": edge_one_hot,
        "edge_length_emb": edge_len,
        "edge_sh": edge_sh,
        "edge_index": torch.tensor([[0, 1, 2], [1, 2, 3]]),
        "atoms": ("H", "H", "H", "H"),
    }
    x_mat = x_gnn  # small test: same graph
    atoms = ("H", "H", "H", "H")

    preds = model(x_gnn, x_mat)
    assert set(preds.keys()) == {"hamiltonian", "overlap", "density"}
    for v in preds.values():
        assert v.atoms == atoms
