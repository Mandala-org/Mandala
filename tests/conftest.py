import pytest
import os
from collections import Counter
from pathlib import Path
import torch

from core.orbital_irrep_config import OrbitalIrrepConfig
from data.openmx_parser import parse_openmx_scfout
from data.snapshot import Snapshot
from data.block_matrix import BlockMatrix
from data.factory import DatasetFactory
from core.block_irrep_mapper import BlockIrrepMapper
from data.gnn_dataset import E3GNNDataset
from net.common import Config


_CATEGORY_BY_DIRECTORY = {
    "unit": "unit",
    "integration": "integration",
    "physics": "physics",
    "equivariance": "equivariance",
    "workflow": "workflow",
    "analysis": "integration",
    "gpu": "gpu",
    "large_data": "large_data",
}


def pytest_collection_modifyitems(config, items):
    """Classify every test and gate suites requiring special infrastructure."""
    for item in items:
        relative = Path(str(item.path)).parts
        try:
            tests_index = relative.index("tests")
            directory = relative[tests_index + 1]
        except (ValueError, IndexError):
            continue
        category = _CATEGORY_BY_DIRECTORY.get(directory)
        if category is not None and item.get_closest_marker(category) is None:
            item.add_marker(getattr(pytest.mark, category))

        slow = item.get_closest_marker("slow")
        if slow is not None and not (slow.args or slow.kwargs.get("reason")):
            raise pytest.UsageError(
                f"{item.nodeid}: @pytest.mark.slow requires a justification."
            )
        if item.get_closest_marker("gpu") and os.getenv("MANDALA_RUN_GPU_TESTS") != "1":
            item.add_marker(
                pytest.mark.skip(reason="GPU suite requires MANDALA_RUN_GPU_TESTS=1.")
            )
        if item.get_closest_marker("large_data") and not os.getenv(
            "MANDALA_LARGE_DATA_ROOT"
        ):
            item.add_marker(
                pytest.mark.skip(
                    reason="Large-data suite requires MANDALA_LARGE_DATA_ROOT."
                )
            )


@pytest.fixture(scope="session")
def h2o_orbital_cfg():
    return OrbitalIrrepConfig.from_dict({"H": "3s2p", "O": "3s3p2d"})


@pytest.fixture(scope="session")
def small_angular_snapshot_e3nn():
    """Tiny deterministic snapshot with scalar and vector orbital channels."""
    orbital_cfg = OrbitalIrrepConfig.from_dict({"H": "1s1p"})
    atoms = ("H", "H")
    atom_counts = Counter(atoms)
    edges = torch.tensor(
        [
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 1, 0, 1],
            [0, 1, 1, 0],
        ],
        dtype=torch.long,
    )
    lookup = {
        tuple(int(value) for value in edges[:, idx]): ("H-H", idx)
        for idx in range(edges.shape[1])
    }
    dim, _ = orbital_cfg.block_dims("H-H")
    generator = torch.Generator().manual_seed(20260722)

    def matrix() -> BlockMatrix:
        return BlockMatrix(
            atoms=atoms,
            atom_counts=atom_counts,
            pair_blocks={
                "H-H": torch.randn(edges.shape[1], dim, dim, generator=generator)
            },
            pair_edges={"H-H": edges.clone()},
            lookup=dict(lookup),
            orbital_cfg=orbital_cfg,
            basis="e3nn",
        )

    return Snapshot(
        matrix(),
        matrix(),
        matrix(),
        positions=torch.tensor([[0.0, 0.0, 0.0], [4.0, 0.0, 0.0]]),
        box=torch.eye(3) * 10.0,
    )


@pytest.fixture(scope="session")
def small_multispecies_snapshot_e3nn():
    """Tiny complete graph spanning the H/O/Si mapper-union contract."""
    orbital_cfg = OrbitalIrrepConfig.from_dict({"H": "1s", "O": "1s", "Si": "1s"})
    atoms = ("H", "O", "Si")
    atom_counts = Counter(atoms)
    edge_tuples = [(0, 0, 0, i, j) for i in range(3) for j in range(3)]
    edges = torch.tensor(edge_tuples, dtype=torch.long).T
    generator = torch.Generator().manual_seed(20260723)

    def matrix() -> BlockMatrix:
        pair_blocks = {}
        pair_edges = {}
        lookup = {}
        for edge_idx, edge in enumerate(edge_tuples):
            i, j = edge[3], edge[4]
            key = f"{atoms[i]}-{atoms[j]}"
            pair_blocks[key] = torch.randn(1, 1, 1, generator=generator)
            pair_edges[key] = edges[:, edge_idx : edge_idx + 1].clone()
            lookup[edge] = (key, 0)
        return BlockMatrix(
            atoms=atoms,
            atom_counts=atom_counts,
            pair_blocks=pair_blocks,
            pair_edges=pair_edges,
            lookup=lookup,
            orbital_cfg=orbital_cfg,
            basis="e3nn",
        )

    return Snapshot(
        matrix(),
        matrix(),
        matrix(),
        positions=torch.tensor([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0], [6.0, 0.0, 0.0]]),
        box=torch.eye(3) * 20.0,
    )


@pytest.fixture(scope="session")
def small_angular_dataset_e3nn(small_angular_snapshot_e3nn):
    """Preprocessed tiny H dataset with all matrix targets enabled."""
    cfg = Config(
        cutoff_radius=5.0,
        l_max=1,
        hidden_base_dim=2,
        hidden_irreps="2x0e+2x0o+1x1e+1x1o",
        n_radial=4,
        radial_layers=[4],
        num_layers_gnn=1,
        neck_depth=1,
        internal_e3mlp_layers=1,
        head_e3mlp_layers=1,
        matrix_targets=["hamiltonian", "overlap", "density"],
        safety_checks=True,
        verbosity=0,
    )
    mapper = BlockIrrepMapper(small_angular_snapshot_e3nn.hamiltonian.orbital_cfg)
    patcher = pytest.MonkeyPatch()
    patcher.setattr(
        Snapshot,
        "from_openmx",
        lambda *args, **kwargs: small_angular_snapshot_e3nn,
    )
    try:
        dataset = E3GNNDataset(
            [(Path("synthetic.matrix"), Path("synthetic.out"))], mapper, cfg=cfg
        )
    finally:
        patcher.undo()
    return dataset, mapper, cfg


@pytest.fixture(scope="session")
def h2o_snapshot(h2o_orbital_cfg):
    sample = Path("./data/small/H2O/original/H2O.matrix")
    atoms = list("HHHHOO")
    return parse_openmx_scfout(sample, atoms, h2o_orbital_cfg, convention="openmx")


@pytest.fixture(scope="session")
def h2o_rotation_pair():
    base = Path("data/small/H2O")
    cfg = Config(allow_openmx_positions_box_from_out=True)
    original = Snapshot.from_openmx(
        base / "original/H2O.matrix",
        base / "original/H2O.info.out",
        cfg=cfg,
        convention="openmx",
    )
    rotated = Snapshot.from_openmx(
        base / "rotated/H2O.matrix",
        base / "rotated/H2O.info.out",
        cfg=cfg,
        convention="openmx",
    )
    return (
        original.reduce_orbitals("1s1p").to_e3nn(),
        rotated.reduce_orbitals("1s1p").to_e3nn(),
    )


@pytest.fixture(scope="session")
def si_snapshot():
    base = Path("./data/big/silicon/300K")
    matrix_path = base / "HS.out"
    info_path = base / "Si.out"
    cfg = Config(cutoff_radius=15.0)
    return Snapshot.from_openmx(
        str(matrix_path),
        str(info_path),
        convention="openmx",
        cfg=cfg,
    )


@pytest.fixture(scope="session")
def factory_results(small_multispecies_snapshot_e3nn):
    patcher = pytest.MonkeyPatch()
    patcher.setattr(
        Snapshot,
        "from_openmx",
        lambda *args, **kwargs: small_multispecies_snapshot_e3nn,
    )
    patcher.setattr(
        DatasetFactory,
        "_load_info",
        lambda self, *paths: type(
            "Info", (), {"orbital_set": {"H": "1s", "O": "1s", "Si": "1s"}}
        )(),
    )
    cfg = Config(
        cutoff_radius=7.0,
        l_max=1,
        n_radial=4,
        radial_layers=[4],
        device="cpu",
        train_target="matrix",
        matrix_targets=["hamiltonian", "overlap", "density"],
        dtype=torch.float32,
        safety_checks=True,
    )
    fac = DatasetFactory(cfg)
    fac.add_snapshot("synthetic-train-1.matrix", "synthetic.info", purpose="train")
    fac.add_snapshot("synthetic-train-2.matrix", "synthetic.info", purpose="train")
    fac.add_snapshot("synthetic-val.matrix", "synthetic.info", purpose="val")
    try:
        yield fac.create()
    finally:
        patcher.undo()
