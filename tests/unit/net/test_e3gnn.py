import pytest
import torch
from torch import nn
from collections import Counter

from core.orbital_irrep_config import OrbitalIrrepConfig
from data.edge_alignment import build_prediction_edge_metadata
from net.common import Config as ProductionConfig
from core.block_irrep_mapper import BlockIrrepMapper
from net.e3gnn import E3GNN
from e3nn.o3 import Irreps
from data.factory import DatasetFactory
from data.block_matrix import IrrepsBlockData
from data.graph_features import compute_graph_features


def Config(**overrides):
    """Construct a small model while preserving each test's explicit options."""
    values = dict(
        l_max=1,
        hidden_base_dim=2,
        hidden_irreps="2x0e+2x0o+1x1e+1x1o",
        n_radial=4,
        radial_layers=[4],
        num_layers_gnn=1,
        neck_depth=1,
        internal_e3mlp_layers=1,
        head_e3mlp_layers=1,
        verbosity=0,
    )
    values.update(overrides)
    return ProductionConfig(**values)


def _build_static_graph_x(
    *,
    node_type_idx: torch.Tensor,
    edge_index: torch.Tensor,
    edge_shift: torch.Tensor,
    edge_type_idx: torch.Tensor,
    atoms: tuple[str, ...],
    mapper: BlockIrrepMapper,
    cfg: Config,
    edge_length_emb: torch.Tensor | None = None,
    edge_sh: torch.Tensor | None = None,
    positions: torch.Tensor | None = None,
    box: torch.Tensor | None = None,
) -> dict:
    num_species = len(mapper.orbital_cfg.elements())
    metadata = build_prediction_edge_metadata(
        edge_index=edge_index,
        edge_shift=edge_shift,
        edge_type_idx=edge_type_idx,
        atoms=atoms,
        edge_types=mapper.edge_types,
        edge_type2idx=mapper.edge_type2idx,
        separate_shifted_self=bool(cfg.separate_shifted_self),
    )
    x = {
        "node_type_idx": node_type_idx,
        "node_one_hot": torch.nn.functional.one_hot(
            node_type_idx, num_classes=num_species
        ).to(dtype=cfg.dtype),
        "edge_index": edge_index,
        "edge_shift": edge_shift,
        "edge_type_idx": edge_type_idx,
        "edge_one_hot": torch.nn.functional.one_hot(
            node_type_idx[edge_index[0]] * num_species + node_type_idx[edge_index[1]],
            num_classes=num_species * num_species,
        ).to(dtype=cfg.dtype),
        "atoms": atoms,
        "atoms_tuple": atoms,
        "atom_counts": Counter(atoms),
        "num_self_edges": int(
            (
                ((edge_index[0] == edge_index[1]) & (edge_shift == 0).all(dim=0)).sum()
            ).item()
        ),
        "positions": positions,
        "box": box,
        **metadata,
    }
    if edge_length_emb is not None:
        x["edge_length_emb"] = edge_length_emb
    if edge_sh is not None:
        x["edge_sh"] = edge_sh
    return x


class MockHead(nn.Module):
    def __init__(self, mock_impl):
        super().__init__()
        self.mock_impl = mock_impl

    def forward(self, *args, **kwargs):
        return self.mock_impl(*args, **kwargs)


@pytest.mark.parametrize("edge_encoder_style", ["rich", "distance"])
@pytest.mark.unit
def test_forward_smoke(edge_encoder_style):
    # ------- dummy orbital config ------------------
    orb_cfg = OrbitalIrrepConfig.from_dict({"H": "1x0e"})

    cfg = Config(
        num_layers_gnn=1,
        edge_encoder_style=edge_encoder_style,
        matrix_targets=["hamiltonian", "overlap", "density"],
        safety_checks=True,
    )

    model = E3GNN(BlockIrrepMapper(orb_cfg), cfg)

    # ------- fake batch ----------------------------
    N = 4
    node_type_idx = torch.zeros(N, dtype=torch.long)  # all H
    edge_index = torch.tensor(
        [[0, 1, 2, 3, 0, 1, 1, 2], [0, 1, 2, 3, 1, 0, 2, 1]], dtype=torch.long
    )
    E = edge_index.shape[1]
    edge_type_idx = torch.zeros(E, dtype=torch.long)  # H-H
    edge_len = torch.randn(E, cfg.n_radial)
    sh = Irreps.spherical_harmonics(cfg.l_max)
    edge_sh = torch.randn(E, sh.dim)  # random SH features
    edge_shift = torch.zeros(3, E, dtype=torch.long)
    atoms = ("H", "H", "H", "H")

    x = _build_static_graph_x(
        node_type_idx=node_type_idx,
        edge_index=edge_index,
        edge_shift=edge_shift,
        edge_type_idx=edge_type_idx,
        atoms=atoms,
        mapper=model.mapper,
        cfg=cfg,
        edge_length_emb=edge_len,
        edge_sh=edge_sh,
        positions=torch.zeros(N, 3, dtype=cfg.dtype),
        box=torch.eye(3, dtype=cfg.dtype),
    )

    preds = model(x)
    assert set(preds.keys()) == {"hamiltonian", "overlap", "density"}
    for v in preds.values():
        assert v.atoms == atoms


@pytest.mark.unit
def test_forward_smoke_onthefly_deeph_e3():
    orb_cfg = OrbitalIrrepConfig.from_dict({"H": "1x0e"})
    cfg = Config(
        num_layers_gnn=1,
        edge_encoder_style="distance",
        precompute_edge_features=False,
        cutoff_radius=3.0,
        matrix_targets=["hamiltonian", "overlap", "density"],
        safety_checks=True,
        verbosity=0,
    )
    model = E3GNN(BlockIrrepMapper(orb_cfg), cfg)

    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [0.0, 0.0, 0.8], [0.0, 0.7, 0.0]],
        dtype=cfg.dtype,
    )
    edge_index, edge_shift, edge_type_idx, _, _, num_self_edges, _ = (
        compute_graph_features(
            positions=positions,
            box=None,
            atoms=("H", "H", "H"),
            cfg=cfg,
            sh_irreps=Irreps.spherical_harmonics(cfg.l_max),
            edge_type2idx=model.mapper.edge_type2idx,
        )
    )
    x = {
        **_build_static_graph_x(
            node_type_idx=torch.zeros(3, dtype=torch.long),
            edge_index=edge_index,
            edge_shift=edge_shift,
            edge_type_idx=edge_type_idx,
            atoms=("H", "H", "H"),
            mapper=model.mapper,
            cfg=cfg,
            positions=positions,
            box=None,
        ),
        "num_self_edges": num_self_edges,
    }

    preds = model(x)
    assert set(preds.keys()) == {"hamiltonian", "overlap", "density"}


@pytest.mark.integration
def test_forward_h2o_cutoff5_with_split_head():
    cfg = Config(
        cutoff_radius=5.0,
        matrix_targets=["hamiltonian", "overlap", "density"],
        separate_shifted_self=True,
        head_use_node_embeddings_for_self_edges=True,
        allow_openmx_positions_box_from_out=True,
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
        symmetrize_output=False,
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
        "pred_trace_alignment": {key: (key, torch.arange(edges_5d.shape[1]))},
    }
    y = {
        "hamiltonian": target_irreps.to_blocks(mapper),
        "forces": None,
        "stress": None,
    }

    logged_metrics = {}
    model.forward = lambda batch_x: {"hamiltonian": target_irreps}
    model.log_dict = lambda metrics, **kwargs: logged_metrics.update(metrics)

    loss = model._shared_step((x, y), batch_idx=0, stage="train")

    assert loss.item() == pytest.approx(0.0, abs=1e-7)
    assert any(
        key.startswith("train/hamiltonian_irrep_") and key.endswith("_mae")
        for key in logged_metrics
    )
    assert any(
        key.startswith("train/hamiltonian_irrep_") and key.endswith("_loss")
        for key in logged_metrics
    )


@pytest.mark.unit
def test_irrep_metrics_logged_for_all_matrix_targets():
    orb_cfg = OrbitalIrrepConfig.from_dict({"H": "1x0e + 1x1o"})
    mapper = BlockIrrepMapper(orb_cfg)
    cfg = Config(
        matrix_targets=["hamiltonian", "overlap", "density"],
        enable_energy=False,
        enable_num_electrons=False,
        train_on_energy=False,
        train_on_num_electrons=False,
        log_per_irrep_metrics=True,
        log_train_metrics=True,
        symmetrize_output=False,
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
        "pred_trace_alignment": {key: (key, torch.arange(edges_5d.shape[1]))},
    }
    y = {
        "hamiltonian": target_irreps.to_blocks(mapper),
        "overlap": target_irreps.to_blocks(mapper),
        "density": target_irreps.to_blocks(mapper),
        "forces": None,
        "stress": None,
    }

    logged_metrics = {}
    model.forward = lambda batch_x: {
        "hamiltonian": target_irreps,
        "overlap": target_irreps,
        "density": target_irreps,
    }
    model.log_dict = lambda metrics, **kwargs: logged_metrics.update(metrics)

    loss = model._shared_step((x, y), batch_idx=0, stage="train")

    assert loss.item() == pytest.approx(0.0, abs=1e-7)
    assert any(
        key.startswith("train/hamiltonian_irrep_") and key.endswith("_l2_block_abs")
        for key in logged_metrics
    )
    assert any(
        key.startswith("train/overlap_irrep_") and key.endswith("_l2_block_abs")
        for key in logged_metrics
    )
    assert any(
        key.startswith("train/density_irrep_") and key.endswith("_l2_block_abs")
        for key in logged_metrics
    )


@pytest.mark.unit
def test_energy_mae_gt_hamiltonian_logged():
    orb_cfg = OrbitalIrrepConfig.from_dict({"H": "1x0e"})
    mapper = BlockIrrepMapper(orb_cfg)
    cfg = Config(
        matrix_targets=["density"],
        enable_energy=True,
        enable_num_electrons=False,
        train_on_energy=False,
        train_on_num_electrons=False,
        log_partial_gt_observables=True,
        log_per_irrep_metrics=False,
        symmetrize_output=False,
        safety_checks=True,
        verbosity=0,
    )
    model = E3GNN(mapper, cfg)

    block = IrrepsBlockData(
        atoms=("H",),
        atom_counts=Counter(("H",)),
        pair_vectors={"H-H": torch.tensor([[1.0]])},
        pair_edges={"H-H": torch.tensor([[0], [0], [0], [0], [0]], dtype=torch.long)},
        lookup={(0, 0, 0, 0, 0): ("H-H", 0)},
        orbital_cfg=mapper.orbital_cfg,
    )
    x = {
        "node_type_idx": torch.zeros(1, dtype=torch.long),
        "edge_index": torch.tensor([[0], [0]], dtype=torch.long),
        "edge_shift": torch.zeros(3, 1, dtype=torch.long),
        "pred_trace_alignment": build_prediction_edge_metadata(
            edge_index=torch.tensor([[0], [0]], dtype=torch.long),
            edge_shift=torch.zeros(3, 1, dtype=torch.long),
            edge_type_idx=torch.zeros(1, dtype=torch.long),
            atoms=("H",),
            edge_types=mapper.edge_types,
            edge_type2idx=mapper.edge_type2idx,
            separate_shifted_self=False,
        )["pred_trace_alignment"],
    }
    y = {
        "hamiltonian": block.to_blocks(mapper),
        "overlap": block.to_blocks(mapper),
        "density": block.to_blocks(mapper),
        "energy": torch.tensor(1.0),
        "forces": None,
        "stress": None,
    }

    logged_metrics = {}
    model.forward = lambda batch_x: {"density": block}
    model.log_dict = lambda metrics, **kwargs: logged_metrics.update(metrics)

    loss = model._shared_step((x, y), batch_idx=0, stage="train")

    assert loss.item() == pytest.approx(0.0, abs=1e-7)
    assert "train/energy_mae_gt_hamiltonian" in logged_metrics
    assert logged_metrics["train/energy_mae_gt_hamiltonian"].item() == pytest.approx(
        0.0, abs=1e-7
    )


@pytest.mark.unit
def test_energy_mae_gt_density_logged():
    orb_cfg = OrbitalIrrepConfig.from_dict({"H": "1x0e"})
    mapper = BlockIrrepMapper(orb_cfg)
    cfg = Config(
        matrix_targets=["hamiltonian"],
        enable_energy=True,
        enable_num_electrons=False,
        train_on_energy=False,
        train_on_num_electrons=False,
        log_partial_gt_observables=True,
        log_per_irrep_metrics=False,
        symmetrize_output=False,
        safety_checks=True,
        verbosity=0,
    )
    model = E3GNN(mapper, cfg)

    block = IrrepsBlockData(
        atoms=("H",),
        atom_counts=Counter(("H",)),
        pair_vectors={"H-H": torch.tensor([[1.0]])},
        pair_edges={"H-H": torch.tensor([[0], [0], [0], [0], [0]], dtype=torch.long)},
        lookup={(0, 0, 0, 0, 0): ("H-H", 0)},
        orbital_cfg=mapper.orbital_cfg,
    )
    x = {
        "node_type_idx": torch.zeros(1, dtype=torch.long),
        "edge_index": torch.tensor([[0], [0]], dtype=torch.long),
        "edge_shift": torch.zeros(3, 1, dtype=torch.long),
        "pred_trace_alignment": build_prediction_edge_metadata(
            edge_index=torch.tensor([[0], [0]], dtype=torch.long),
            edge_shift=torch.zeros(3, 1, dtype=torch.long),
            edge_type_idx=torch.zeros(1, dtype=torch.long),
            atoms=("H",),
            edge_types=mapper.edge_types,
            edge_type2idx=mapper.edge_type2idx,
            separate_shifted_self=False,
        )["pred_trace_alignment"],
    }
    y = {
        "hamiltonian": block.to_blocks(mapper),
        "overlap": block.to_blocks(mapper),
        "density": block.to_blocks(mapper),
        "energy": torch.tensor(1.0),
        "forces": None,
        "stress": None,
    }

    logged_metrics = {}
    model.forward = lambda batch_x: {"hamiltonian": block}
    model.log_dict = lambda metrics, **kwargs: logged_metrics.update(metrics)

    loss = model._shared_step((x, y), batch_idx=0, stage="train")

    assert loss.item() == pytest.approx(0.0, abs=1e-7)
    assert "train/energy_mae_gt_density" in logged_metrics
    assert logged_metrics["train/energy_mae_gt_density"].item() == pytest.approx(
        0.0, abs=1e-7
    )


@pytest.mark.unit
def test_num_electrons_half_gt_metrics_logged():
    orb_cfg = OrbitalIrrepConfig.from_dict({"H": "1x0e"})
    mapper = BlockIrrepMapper(orb_cfg)
    x = {
        "node_type_idx": torch.zeros(1, dtype=torch.long),
        "edge_index": torch.tensor([[0], [0]], dtype=torch.long),
        "edge_shift": torch.zeros(3, 1, dtype=torch.long),
        "pred_trace_alignment": build_prediction_edge_metadata(
            edge_index=torch.tensor([[0], [0]], dtype=torch.long),
            edge_shift=torch.zeros(3, 1, dtype=torch.long),
            edge_type_idx=torch.zeros(1, dtype=torch.long),
            atoms=("H",),
            edge_types=mapper.edge_types,
            edge_type2idx=mapper.edge_type2idx,
            separate_shifted_self=False,
        )["pred_trace_alignment"],
    }
    block = IrrepsBlockData(
        atoms=("H",),
        atom_counts=Counter(("H",)),
        pair_vectors={"H-H": torch.tensor([[1.0]])},
        pair_edges={"H-H": torch.tensor([[0], [0], [0], [0], [0]], dtype=torch.long)},
        lookup={(0, 0, 0, 0, 0): ("H-H", 0)},
        orbital_cfg=mapper.orbital_cfg,
    )
    y = {
        "hamiltonian": block.to_blocks(mapper),
        "overlap": block.to_blocks(mapper),
        "density": block.to_blocks(mapper),
        "num_electrons": torch.tensor(1.0),
        "forces": None,
        "stress": None,
    }

    cfg_density = Config(
        matrix_targets=["density"],
        enable_energy=False,
        enable_num_electrons=True,
        train_on_energy=False,
        train_on_num_electrons=False,
        log_partial_gt_observables=True,
        verbosity=0,
    )
    model_density = E3GNN(mapper, cfg_density)
    logged_density = {}
    model_density.forward = lambda batch_x: {"density": block}
    model_density.log_dict = lambda metrics, **kwargs: logged_density.update(metrics)
    model_density._shared_step((x, y), batch_idx=0, stage="train")
    assert "train/num_electrons_mae_gt_overlap" in logged_density

    cfg_overlap = Config(
        matrix_targets=["overlap"],
        enable_energy=False,
        enable_num_electrons=True,
        train_on_energy=False,
        train_on_num_electrons=False,
        log_partial_gt_observables=True,
        verbosity=0,
    )
    model_overlap = E3GNN(mapper, cfg_overlap)
    logged_overlap = {}
    model_overlap.forward = lambda batch_x: {"overlap": block}
    model_overlap.log_dict = lambda metrics, **kwargs: logged_overlap.update(metrics)
    model_overlap._shared_step((x, y), batch_idx=0, stage="train")
    assert "train/num_electrons_mae_gt_density" in logged_overlap


@pytest.mark.unit
def test_invalid_observable_training_configs_raise():
    orb_cfg = OrbitalIrrepConfig.from_dict({"H": "1x0e"})
    mapper = BlockIrrepMapper(orb_cfg)

    with pytest.raises(
        ValueError, match="train_on_energy=True requires predicting both"
    ):
        E3GNN(
            mapper,
            Config(
                matrix_targets=["density"],
                train_on_energy=True,
                train_observables_on_gt=False,
                loss_coef_observables=1.0,
                verbosity=0,
            ),
        )

    with pytest.raises(
        ValueError,
        match="train_on_num_electrons=True with train_observables_on_gt=True",
    ):
        E3GNN(
            mapper,
            Config(
                matrix_targets=["hamiltonian"],
                train_on_num_electrons=True,
                train_observables_on_gt=True,
                loss_coef_observables=1.0,
                verbosity=0,
            ),
        )


@pytest.mark.unit
def test_zero_weight_energy_guidance_control_is_valid():
    orb_cfg = OrbitalIrrepConfig.from_dict({"H": "1x0e"})
    mapper = BlockIrrepMapper(orb_cfg)

    with pytest.raises(ValueError, match="allow_zero_observable_loss_control"):
        E3GNN(
            mapper,
            Config(
                matrix_targets=["hamiltonian"],
                enable_energy=True,
                train_on_energy=True,
                train_observables_on_gt=True,
                loss_coef_observables=0.0,
                verbosity=0,
            ),
        )

    model = E3GNN(
        mapper,
        Config(
            matrix_targets=["hamiltonian"],
            enable_energy=True,
            train_on_energy=True,
            train_observables_on_gt=True,
            loss_coef_observables=0.0,
            allow_zero_observable_loss_control=True,
            verbosity=0,
        ),
    )

    assert model.cfg.train_on_energy is True
    assert model.cfg.loss_coef_observables == 0.0


@pytest.mark.unit
def test_energy_half_gt_training_accepts_length_one_energy_target():
    orb_cfg = OrbitalIrrepConfig.from_dict({"H": "1x0e"})
    mapper = BlockIrrepMapper(orb_cfg)
    cfg = Config(
        matrix_targets=["hamiltonian"],
        enable_energy=True,
        enable_num_electrons=False,
        train_on_energy=True,
        train_on_num_electrons=False,
        train_observables_on_gt=True,
        log_partial_gt_observables=True,
        loss_coef_observables=1.0,
        verbosity=0,
    )
    model = E3GNN(mapper, cfg)

    block = IrrepsBlockData(
        atoms=("H",),
        atom_counts=Counter(("H",)),
        pair_vectors={"H-H": torch.tensor([[1.0]])},
        pair_edges={"H-H": torch.tensor([[0], [0], [0], [0], [0]], dtype=torch.long)},
        lookup={(0, 0, 0, 0, 0): ("H-H", 0)},
        orbital_cfg=mapper.orbital_cfg,
    )
    x = {
        "node_type_idx": torch.zeros(1, dtype=torch.long),
        "edge_index": torch.tensor([[0], [0]], dtype=torch.long),
        "edge_shift": torch.zeros(3, 1, dtype=torch.long),
        "pred_trace_alignment": build_prediction_edge_metadata(
            edge_index=torch.tensor([[0], [0]], dtype=torch.long),
            edge_shift=torch.zeros(3, 1, dtype=torch.long),
            edge_type_idx=torch.zeros(1, dtype=torch.long),
            atoms=("H",),
            edge_types=mapper.edge_types,
            edge_type2idx=mapper.edge_type2idx,
            separate_shifted_self=False,
        )["pred_trace_alignment"],
    }
    y = {
        "hamiltonian": block.to_blocks(mapper),
        "overlap": block.to_blocks(mapper),
        "density": block.to_blocks(mapper),
        "energy": torch.tensor([1.0]),
        "forces": None,
        "stress": None,
    }

    logged_metrics = {}
    model.forward = lambda batch_x: {"hamiltonian": block}
    model.log_dict = lambda metrics, **kwargs: logged_metrics.update(metrics)

    loss = model._shared_step((x, y), batch_idx=0, stage="val")

    assert torch.isfinite(loss)
    assert "val/energy_mae_gt_density" in logged_metrics
