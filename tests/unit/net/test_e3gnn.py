import pytest
import torch
from torch import nn

from core.orbital_irrep_config import OrbitalIrrepConfig
from net.common import Config
from core.block_irrep_mapper import BlockIrrepMapper
from net.e3gnn import E3GNN
from e3nn.o3 import Irreps


class MockHead(nn.Module):
    def __init__(self, mock_impl):
        super().__init__()
        self.mock_impl = mock_impl

    def forward(self, *args, **kwargs):
        return self.mock_impl(*args, **kwargs)


@pytest.mark.unit
def test_forward_smoke():
    # ------- dummy orbital config ------------------
    orb_cfg = OrbitalIrrepConfig.from_dict({"H": "1x0e"})

    cfg = Config(num_layers_gnn=1, num_layers_matrix=1, safety_checks=True)

    model = E3GNN(BlockIrrepMapper(orb_cfg), cfg)

    # ------- fake batch ----------------------------
    N, E = 4, 7
    node_type_idx = torch.zeros(N, dtype=torch.long)  # all H
    edge_type_idx = torch.zeros(E, dtype=torch.long)  # H-H
    edge_len = torch.randn(E, cfg.n_radial)
    sh = Irreps.spherical_harmonics(cfg.l_max)
    edge_sh = torch.randn(E, sh.dim)  # random SH features

    x = {
        "node_type_idx": node_type_idx,
        "edge_type_idx": edge_type_idx,
        "edge_length_emb": edge_len,
        "edge_sh": edge_sh,
        "edge_index": torch.tensor([[0, 1, 2, 3, 0, 1, 2], [0, 1, 2, 3, 1, 2, 3]]),
        "edge_shift": torch.zeros(3, E, dtype=torch.long),
        "atoms": ("H", "H", "H", "H"),
        "num_self_edges": N,
        "index_gnn_cutoff": E,
        "box": torch.eye(3),
    }
    atoms = ("H", "H", "H", "H")

    preds = model(x)
    assert set(preds.keys()) == {"hamiltonian", "overlap", "density"}
    for v in preds.values():
        assert v.atoms == atoms
