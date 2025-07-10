"""
Additional coverage for E3GNN:

We instantiate the model under five **categorically different**
HyperParams settings and run a bare `forward` pass on a dummy mini-batch.

The test does **not** back-prop or call Lightning’s training loop – it is
meant to guard against shape / device / construction regressions for the
most important architecture flags.
"""

import pytest
import torch
from e3nn.o3 import Irreps

from core.orbital_irrep_config import OrbitalIrrepConfig
from core.block_irrep_mapper import BlockIrrepMapper
from net.common import HyperParams
from net.e3gnn import E3GNN


# ──────────────────────────────────────────────────────────────────────
# helper to fabricate a minimal synthetic batch
# ──────────────────────────────────────────────────────────────────────
def make_dummy_graph(hp):
    N, E = 4, 3
    node_type_idx = torch.zeros(N, dtype=torch.long)  # all H
    edge_type_idx = torch.zeros(E, dtype=torch.long)  # H-H

    edge_len = torch.randn(E, hp.n_radial)
    sh_irreps = Irreps.spherical_harmonics(hp.l_max)
    edge_sh = torch.randn(E, sh_irreps.dim)

    edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)

    x = {
        "node_type_idx": node_type_idx,
        "edge_type_idx": edge_type_idx,
        "edge_length_emb": edge_len,
        "edge_sh": edge_sh,
        "edge_index": edge_index,
        "atoms": ("H", "H", "H", "H"),
        "index_gnn_cutoff": E,
        "overlap_vectors": torch.randn(E, 1),
    }
    return x


# ──────────────────────────────────���───────────────────────────────────
# parameter sets – each dict overrides defaults
# ──────────────────────────────────────────────────────────────────────
HP_VARIANTS = [
    {},  # default (gate, residuals, dropout 0)
    {"nonlin_kind": "normact", "batch_norm": True},
    {"nonlin_kind": "s2act", "dropout": 0.2, "use_edge_updates": False},
    {"nonlin_kind": "id", "residual_connections": False},
    {"use_self_update": False, "head_hidden_mul": 2.0, "radial_layers": (64, 32)},
]

cfg = OrbitalIrrepConfig.from_dict({"H": "1x0e"})
mapper = BlockIrrepMapper(cfg)
EDGE_TYPES = ["H-H"]


@pytest.mark.parametrize("hp_kwargs", HP_VARIANTS)
@pytest.mark.integration
def test_e3gnn_forward_variants(hp_kwargs):
    from omegaconf import OmegaConf
    from dataclasses import asdict

    hp = HyperParams(**hp_kwargs)

    # Create a minimal mock config
    mock_cfg = OmegaConf.create(
        {"model": asdict(hp), "training": {"lr": 1e-3}, "logging": {"pedantic": False}}
    )

    model = E3GNN(
        mapper,
        EDGE_TYPES,
        mock_cfg,
        device="cpu",
    )

    x = make_dummy_graph(hp)
    atoms = ("H",) * 4

    preds = model(x)

    # ------------- basic assertions -----------------------------------
    assert set(preds.keys()) == {"hamiltonian", "overlap", "density"}

    for name, block in preds.items():
        # correct atom tuple
        assert block.atoms == atoms
        # at least one pair key present
        assert "H-H" in block.pair_vectors
        # shapes coherent
        vec = block.pair_vectors["H-H"]
        edges = block.pair_edges["H-H"]
        assert vec.shape[0] == edges.shape[1]
