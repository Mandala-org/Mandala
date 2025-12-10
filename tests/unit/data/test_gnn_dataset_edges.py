import torch
from collections import Counter
from data.gnn_dataset import E3GNNDataset
from data.snapshot import Snapshot
from data.block_matrix import BlockMatrix
from core.orbital_irrep_config import OrbitalIrrepConfig
from net.common import Config
from e3nn.o3 import Irreps


def test_process_snapshot_to_sample():
    # Mock dataset instance (partial)
    class MockDataset(E3GNNDataset):
        def __init__(self):
            self.cfg = Config(
                cutoff_matrix=3.0,
                cutoff_gnn=3.0,
                n_radial=5,
                precompute_edge_features=True,
                train_target="matrix",
                matrix_targets=["density"],
                enable_forces=False,
                enable_stress=False,
                train_on_forces=False,
                train_on_stress=False,
            )
            self.sh_irreps = Irreps("1x0e")
            self.mapper = type("MockMapper", (), {"edge_type2idx": {"H-H": 0}})()
            self.orbital_cfg = OrbitalIrrepConfig.from_dict({"H": "1s"})

        def _process_snapshot_to_sample(self, snap):
            return E3GNNDataset._process_snapshot_to_sample(self, snap)

    ds = MockDataset()

    # Mock Snapshot
    atoms = ("H", "H")
    atom_counts = Counter(atoms)
    orbital_cfg = OrbitalIrrepConfig.from_dict({"H": "1s"})

    # BlockMatrix edges:
    # 0. (0,0,0, 0, 1) - Off-diag
    # 1. (0,0,0, 1, 0) - Off-diag (symmetric to 0)
    # 2. (1,0,0, 0, 1) - Off-diag, shifted
    # 3. (-1,0,0, 1, 0) - Off-diag, shifted (symmetric to 2)
    # 4. (0,0,0, 0, 0) - Diag
    # 5. (0,0,0, 1, 1) - Diag

    edges = torch.tensor(
        [
            [0, 0, 0, 0, 1],
            [0, 0, 0, 1, 0],
            [1, 0, 0, 0, 1],
            [-1, 0, 0, 1, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 1, 1],
        ]
    ).t()
    blocks = torch.randn(6, 1, 1)

    pair_blocks = {"H-H": blocks}
    pair_edges = {"H-H": edges}
    lookup = {tuple(e): ("H-H", i) for i, e in enumerate(edges.t().tolist())}

    bm = BlockMatrix(
        atoms, atom_counts, pair_blocks, pair_edges, lookup, orbital_cfg, "e3nn"
    )

    positions = torch.tensor([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    box = torch.eye(3) * 10.0

    snap = Snapshot(bm, bm, bm, positions=positions, box=box)

    # Run processing
    x, y = ds._process_snapshot_to_sample(snap)

    # Check target_index_map
    tim = y["target_index_map"]["H-H"]

    # GNN edges (computed by compute_graph_features):
    # Pos: 0.0, 2.0. Box 10.0. Cutoff 3.0.
    # Finds:
    # (0,0,0,0,0) -> Index 4
    # (0,0,0,1,1) -> Index 5
    # (0,0,0,0,1) -> Index 0
    # (0,0,0,1,0) -> Index 1

    # Edges 2 and 3 are distant (dist=8.0 or 12.0) so not found by GNN.

    assert tim.shape[0] == 4
    # Order: Self edges first, then off-diag sorted.
    # 0: (0,0,0,0,0) -> 4
    # 1: (0,0,0,1,1) -> 5
    # 2: (0,0,0,0,1) -> 0
    # 3: (0,0,0,1,0) -> 1

    assert tim[0] == 4
    assert tim[1] == 5
    assert tim[2] == 0
    assert tim[3] == 1
