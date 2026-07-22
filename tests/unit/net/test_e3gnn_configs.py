"""
Additional coverage for E3GNN:

We instantiate the model under five **categorically different**
Config settings and run a bare `forward` pass on a dummy mini-batch.

The test does **not** back-prop or call Lightning’s training loop - it is
meant to guard against shape / device / construction regressions for the
most important architecture flags.
"""

import pytest
import torch
from e3nn.o3 import Irreps

from core.orbital_irrep_config import OrbitalIrrepConfig
from core.block_irrep_mapper import BlockIrrepMapper
from data.edge_alignment import build_prediction_edge_metadata
from net.common import Config as ProductionConfig, resolve_hidden_irreps
from net.e3gnn import E3GNN


def Config(**overrides):
    """Construct a compact model while retaining the variant under test."""
    values = dict(
        l_max=2,
        hidden_base_dim=4,
        hidden_irreps=None,
        n_radial=8,
        radial_layers=[8],
        num_layers_gnn=1,
        neck_depth=1,
        internal_e3mlp_layers=1,
        head_e3mlp_layers=1,
        matrix_targets=["hamiltonian", "overlap", "density"],
        verbosity=0,
    )
    values.update(overrides)
    return ProductionConfig(**values)


# ──────────────────────────────────────────────────────────────────────
# helper to fabricate a minimal synthetic batch
# ──────────────────────────────────────────────────────────────────────
def make_dummy_graph(cfg, mapper):
    N = 4
    node_type_idx = torch.zeros(N, dtype=torch.long)  # all H

    edge_index = torch.tensor(
        [[0, 1, 2, 3, 0, 1, 1, 2], [0, 1, 2, 3, 1, 0, 2, 1]], dtype=torch.long
    )
    E = edge_index.shape[1]
    edge_type_idx = torch.zeros(E, dtype=torch.long)  # H-H
    edge_shift = torch.zeros(3, E, dtype=torch.long)
    edge_len = torch.randn(E, cfg.n_radial)
    sh_irreps = Irreps.spherical_harmonics(cfg.l_max)
    edge_sh = torch.randn(E, sh_irreps.dim)
    pred_meta = build_prediction_edge_metadata(
        edge_index=edge_index,
        edge_shift=edge_shift,
        edge_type_idx=edge_type_idx,
        atoms=("H", "H", "H", "H"),
        edge_types=mapper.edge_types,
        edge_type2idx=mapper.edge_type2idx,
        separate_shifted_self=bool(cfg.separate_shifted_self),
    )
    node_one_hot = torch.nn.functional.one_hot(node_type_idx, num_classes=1).to(
        dtype=cfg.dtype
    )
    edge_one_hot = torch.nn.functional.one_hot(edge_type_idx, num_classes=1).to(
        dtype=cfg.dtype
    )

    x = {
        "node_type_idx": node_type_idx,
        "node_one_hot": node_one_hot,
        "edge_type_idx": edge_type_idx,
        "edge_one_hot": edge_one_hot,
        "edge_length_emb": edge_len,
        "edge_sh": edge_sh,
        "edge_index": edge_index,
        "edge_shift": edge_shift,
        "atoms": ("H", "H", "H", "H"),
        "atoms_tuple": ("H", "H", "H", "H"),
        "atom_counts": {"H": 4},
        "num_self_edges": N,
        "positions": torch.zeros(N, 3, dtype=cfg.dtype),
        "box": torch.eye(3, dtype=cfg.dtype),
        **pred_meta,
    }
    return x


# ─────────────────────────────────────────────────────────────────────
# parameter sets - each dict overrides defaults
# ──────────────────────────────────────────────────────────────────────
HP_VARIANTS = [
    {},  # default
    {"nonlin_kind": "gate_scalars_mlp"},
    {"edge_encoder_use_sh_tensor_square": True},
    {"head_use_tensor_square": True},
    {"edge_update_node_combine": "sum"},
    {"edge_update_node_combine": "tensor_product"},
    {"node_update_message_agg": "average"},
    {"node_update_message_agg": "attention", "node_update_attention_heads": 2},
    {"edge_update_residual": False, "node_update_residual": False},
    {"l_max": 3, "hidden_base_dim": 8},
    {"hidden_irreps": "8x0e+8x0o+4x1e+4x1o", "e3layernorm": True},
]

orb_cfg = OrbitalIrrepConfig.from_dict({"H": "1x0e"})
mapper = BlockIrrepMapper(orb_cfg)


@pytest.mark.parametrize("hp_kwargs", HP_VARIANTS)
@pytest.mark.unit
def test_e3gnn_forward_variants(hp_kwargs):

    cfg = Config(**hp_kwargs, safety_checks=True)

    model = E3GNN(
        mapper,
        cfg,
    )

    x = make_dummy_graph(cfg, mapper)
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


@pytest.mark.unit
def test_resolve_hidden_irreps_respects_explicit_override():
    cfg = Config(hidden_irreps="8x0e+8x0o+4x1e+4x1o")
    hidden = resolve_hidden_irreps(cfg)
    assert str(hidden) == "8x0e+8x0o+4x1e+4x1o"


@pytest.mark.unit
def test_init_weights_factor_scales_model_parameters():
    orb_cfg = OrbitalIrrepConfig.from_dict({"H": "1x0e"})
    mapper = BlockIrrepMapper(orb_cfg)

    torch.manual_seed(1234)
    model_ref = E3GNN(mapper, Config(init_weights_factor=1.0, verbosity=0))

    torch.manual_seed(1234)
    model_scaled = E3GNN(mapper, Config(init_weights_factor=0.1, verbosity=0))

    ref_params = dict(model_ref.named_parameters())
    scaled_params = dict(model_scaled.named_parameters())
    assert ref_params.keys() == scaled_params.keys()

    checked_any = False
    for name in ref_params:
        ref = ref_params[name]
        scaled = scaled_params[name]
        if not ref.is_floating_point():
            continue
        assert torch.allclose(scaled, ref * 0.1, atol=1e-7, rtol=1e-6), name
        checked_any = True

    assert checked_any
