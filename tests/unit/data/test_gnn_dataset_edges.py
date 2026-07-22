import torch
from pathlib import Path
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
                cutoff_radius=3.0,
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
            self.dtype = self.cfg.dtype
            self.device = torch.device("cpu")
            self.hamiltonian_envelope_mode = "off"
            self.pair_distance_normalization = "off"
            self.loss_weighting_mode = "off"
            self.envelope_table = None
            self.edge_type_r0 = None
            self.spectral_loss_enabled = False
            self.mapper = type(
                "MockMapper",
                (),
                {"edge_type2idx": {"H-H": 0}, "edge_types": ["H-H"]},
            )()
            self.orbital_cfg = OrbitalIrrepConfig.from_dict({"H": "1s"})

        def _process_snapshot_to_sample(self, snap):
            return E3GNNDataset._process_snapshot_to_sample(
                self,
                snap,
                matrix_path=Path("synthetic.matrix"),
                info_path=Path("synthetic.out"),
            )

    ds = MockDataset()

    # Mock Snapshot
    atoms = ("H", "H")
    atom_counts = Counter(atoms)
    orbital_cfg = OrbitalIrrepConfig.from_dict({"H": "1s"})

    # BlockMatrix edges:
    # Match the graph ordering used by compute_graph_features:
    # 0. (0,0,0, 0, 0) - Diag
    # 1. (0,0,0, 1, 1) - Diag
    # 2. (0,0,0, 0, 1) - Off-diag
    # 3. (0,0,0, 1, 0) - Off-diag (symmetric to 2)

    edges = torch.tensor(
        [
            [0, 0, 0, 0, 0],
            [0, 0, 0, 1, 1],
            [0, 0, 0, 0, 1],
            [0, 0, 0, 1, 0],
        ]
    ).t()
    blocks = torch.randn(4, 1, 1)

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

    assert "pred_trace_alignment" in x
    assert "pred_pair_edges_static" in x
    assert "pred_lookup_static" in x
    assert "edge_partitions" in x
    assert "node_one_hot" in x
    assert "edge_one_hot" in x
    assert x["pred_pair_edges_static"]["H-H"].shape[0] == 5
    assert x["node_one_hot"].shape == (2, 1)
    assert x["edge_one_hot"].shape[0] == x["edge_index"].shape[1]
    assert x["pred_trace_alignment"]["H-H"][0] == "H-H"
    assert torch.equal(
        x["pred_pair_edges_static"]["H-H"],
        torch.cat([x["edge_shift"], x["edge_index"]], dim=0),
    )
