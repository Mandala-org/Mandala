import pytest
import torch
from torch import nn
from collections import Counter

from core.orbital_irrep_config import OrbitalIrrepConfig
from net.common import Config
from core.block_irrep_mapper import BlockIrrepMapper
from net.e3gnn import E3GNN
from e3nn.o3 import Irreps
from data.factory import DatasetFactory
from data.block_matrix import IrrepsBlockData


class MockHead(nn.Module):
    def __init__(self, mock_impl):
        super().__init__()
        self.mock_impl = mock_impl

    def forward(self, *args, **kwargs):
        return self.mock_impl(*args, **kwargs)


@pytest.mark.parametrize("edge_encoder_style", ["mandala", "deeph_e3"])
@pytest.mark.unit
def test_forward_smoke(edge_encoder_style):
    # ------- dummy orbital config ------------------
    orb_cfg = OrbitalIrrepConfig.from_dict({"H": "1x0e"})

    cfg = Config(
        num_layers_gnn=1,
        edge_encoder_style=edge_encoder_style,
        safety_checks=True,
    )

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


@pytest.mark.unit
def test_forward_smoke_onthefly_deeph_e3():
    orb_cfg = OrbitalIrrepConfig.from_dict({"H": "1x0e"})
    cfg = Config(
        num_layers_gnn=1,
        edge_encoder_style="deeph_e3",
        precompute_edge_features=False,
        cutoff_radius=3.0,
        safety_checks=True,
        verbosity=0,
    )
    model = E3GNN(BlockIrrepMapper(orb_cfg), cfg)

    x = {
        "positions": torch.tensor(
            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.8], [0.0, 0.7, 0.0]],
            dtype=cfg.dtype,
        ),
        "box": None,
        "atoms": ("H", "H", "H"),
        "node_type_idx": torch.zeros(3, dtype=torch.long),
    }

    preds = model(x)
    assert set(preds.keys()) == {"hamiltonian", "overlap", "density"}


@pytest.mark.integration
def test_forward_h2o_cutoff5_with_split_head():
    cfg = Config(
        cutoff_radius=5.0,
        separate_shifted_self=True,
        head_use_node_embeddings_for_self_edges=True,
        safety_checks=True,
        verbosity=0,
    )
    factory = DatasetFactory(cfg)
    factory.add_snapshot(
        "data/small/H2O/original/H2O.matrix",
        "data/small/H2O/original/H2O.info.out",
    )
    train_ds, _, mapper = factory.create()
    model = E3GNN(mapper, cfg)
    x, _ = train_ds[0]

    preds = model(x)

    assert set(preds.keys()) == {"hamiltonian", "overlap", "density"}


@pytest.mark.parametrize(
    ("partial_train", "separate_shifted_self", "expected"),
    [
        ("diag", False, [True, False, False]),
        ("shifted_self", False, [False, True, False]),
        ("offdiag", False, [False, True, True]),
        ("offdiag", True, [False, False, True]),
    ],
)
@pytest.mark.unit
def test_partial_train_mask(partial_train, separate_shifted_self, expected):
    orb_cfg = OrbitalIrrepConfig.from_dict({"H": "1x0e"})
    cfg = Config(
        num_layers_gnn=1,
        partial_train=partial_train,
        separate_shifted_self=separate_shifted_self,
        safety_checks=True,
        verbosity=0,
    )
    model = E3GNN(BlockIrrepMapper(orb_cfg), cfg)
    edges_5d = torch.tensor(
        [
            [0, 1, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, 1, 0],
            [0, 1, 1],
        ],
        dtype=torch.long,
    )

    mask = model._partial_train_mask(edges_5d)

    assert mask.tolist() == expected


@pytest.mark.unit
def test_irrep_part_loss_zero_on_matching_target():
    orb_cfg = OrbitalIrrepConfig.from_dict({"H": "1x0e + 1x1o"})
    mapper = BlockIrrepMapper(orb_cfg)
    cfg = Config(
        matrix_targets=["hamiltonian"],
        enable_energy=False,
        enable_num_electrons=False,
        train_on_energy=False,
        train_on_num_electrons=False,
        train_on_irrep_parts=True,
        log_per_irrep_metrics=True,
        safety_checks=True,
        verbosity=0,
    )
    model = E3GNN(mapper, cfg)

    edges_5d = torch.tensor(
        [
            [0, 1, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, 1, 0],
            [0, 1, 1],
        ],
        dtype=torch.long,
    )
    key = "H-H"
    pair_vectors = {key: torch.randn(edges_5d.shape[1], mapper.vector_dim(key))}
    pair_edges = {key: edges_5d}
    lookup = {tuple(edge.tolist()): (key, idx) for idx, edge in enumerate(edges_5d.t())}
    target_irreps = IrrepsBlockData(
        atoms=("H", "H"),
        atom_counts=Counter(("H", "H")),
        pair_vectors=pair_vectors,
        pair_edges=pair_edges,
        lookup=lookup,
        orbital_cfg=mapper.orbital_cfg,
    )
    x = {
        "node_type_idx": torch.zeros(2, dtype=torch.long),
        "edge_index": edges_5d[3:],
        "edge_shift": edges_5d[:3],
    }
    y = {
        "hamiltonian": target_irreps.to_blocks(mapper),
        "energy": torch.tensor(0.0),
        "num_electrons": torch.tensor(0.0),
        "forces": None,
        "stress": None,
    }

    logged_metrics = {}
    model.forward = lambda batch_x: {"hamiltonian": target_irreps}
    model.log_dict = lambda metrics, **kwargs: logged_metrics.update(metrics)

    loss = model._shared_step((x, y), batch_idx=0, stage="train")

    assert loss.item() == pytest.approx(0.0, abs=1e-7)
    assert "train/edges_diag" in logged_metrics
    assert any(
        key.startswith("train/hamiltonian_irrep_") and key.endswith("_mae")
        for key in logged_metrics
    )
    assert any(
        key.startswith("train/hamiltonian_irrep_") and key.endswith("_loss")
        for key in logged_metrics
    )
