import torch
from net.e3gnn import E3GNN
from data.block_matrix import IrrepsBlockData
from core.orbital_irrep_config import OrbitalIrrepConfig


def test_wrap_head_output():
    # Test _wrap_head_output logic (rebuilding lookup)
    # Mock E3GNN
    class MockE3GNN:
        def __init__(self):
            self.mapper = type(
                "MockMapper",
                (),
                {"orbital_cfg": OrbitalIrrepConfig.from_dict({"H": "1s"})},
            )()

        _wrap_head_output = E3GNN._wrap_head_output

    model = MockE3GNN()

    # Raw output from head
    # Key "H-H"
    # Edges: (0,0,0,0,1)
    edges = torch.tensor([[0, 0, 0, 0, 1]]).t()
    vectors = torch.randn(1, 1)

    raw = {"H-H": vectors}
    atoms = ("H", "H")
    x = {
        "atoms_tuple": atoms,
        "atom_counts": {"H": 2},
        "pred_pair_edges_static": {"H-H": edges},
        "pred_lookup_static": {(0, 0, 0, 0, 1): ("H-H", 0)},
    }

    ibd = model._wrap_head_output(raw, x)

    assert isinstance(ibd, IrrepsBlockData)
    assert (0, 0, 0, 0, 1) in ibd.lookup
    assert ibd.lookup[(0, 0, 0, 0, 1)] == ("H-H", 0)
