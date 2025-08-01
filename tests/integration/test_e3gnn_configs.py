"""
Additional coverage for E3GNN:

We instantiate the model under five **categorically different**
Config settings and run a bare `forward` pass on a dummy mini-batch.

The test does **not** back-prop or call Lightning’s training loop – it is
meant to guard against shape / device / construction regressions for the
most important architecture flags.
"""

import pytest
import torch
from e3nn.o3 import Irreps

from core.orbital_irrep_config import OrbitalIrrepConfig
from core.block_irrep_mapper import BlockIrrepMapper
from net.common import Config
from net.e3gnn import E3GNN


# ──────────────────────────────────────────────────────────────────────
# helper to fabricate a minimal synthetic batch
# ──────────────────────────────────────────────────────────────────────
def make_dummy_graph(cfg):
    N, E = 4, 7
    node_type_idx = torch.zeros(N, dtype=torch.int)  # all H
    edge_type_idx = torch.zeros(E, dtype=torch.int)  # H-H

    edge_len = torch.randn(E, cfg.n_radial)
    sh_irreps = Irreps.spherical_harmonics(cfg.l_max_gnn)
    edge_sh = torch.randn(E, sh_irreps.dim)

    edge_index = torch.tensor(
        [[0, 1, 2, 3, 0, 1, 2], [0, 1, 2, 3, 1, 2, 3]], dtype=torch.int
    )

    x = {
        "node_type_idx": node_type_idx,
        "edge_type_idx": edge_type_idx,
        "edge_length_emb": edge_len,
        "edge_sh": edge_sh,
        "edge_index": edge_index,
        "atoms": ("H", "H", "H", "H"),
        "index_gnn_cutoff": E,
        "num_self_edges": N,
        "overlap_vectors": torch.randn(E, 1),
    }
    return x


# ─────────────────────────────────────────────────────────────────────
# parameter sets – each dict overrides defaults
# ──────────────────────────────────────────────────────────────────────
HP_VARIANTS = [
    {},  # default
    {"nonlin_kind": "gate_scalars_mlp"},
    {"edge_update_node_combine": "sum", "edge_update": "concat"},
    {"node_update_message_agg": "sum", "node_update": "replace"},
    {
        "edge_update": "replace",
        "node_update": "sum",
        "edge_update_residual": False,
        "node_update_residual": False,
    },
    {"l_max_gnn": 1, "l_max_matrix": 3, "hidden_base_dim": 32},
]

orb_cfg = OrbitalIrrepConfig.from_dict({"H": "1x0e"})
mapper = BlockIrrepMapper(orb_cfg)


@pytest.mark.parametrize("hp_kwargs", HP_VARIANTS)
@pytest.mark.integration
def test_e3gnn_forward_variants(hp_kwargs):

    cfg = Config(**hp_kwargs)

    model = E3GNN(
        mapper,
        cfg,
    )

    x = make_dummy_graph(cfg)
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
