import pytest
import torch
from ase import Atoms
from net.common import Config
from data.graph_features import compute_graph_features
from e3nn.o3 import Irreps
from core.orbital_irrep_config import OrbitalIrrepConfig
from core.block_irrep_mapper import BlockIrrepMapper


@pytest.mark.unit
def test_is_closest_edge_computation():
    # Setup a simple system with PBC
    atoms = Atoms(
        "H2", positions=[[0, 0, 0], [0.8, 0, 0]], cell=[1.5, 1.5, 1.5], pbc=True
    )

    # Config to ensure we get multiple images
    cfg = Config(cutoff_matrix=1.0)

    # Dummy irreps and mapper for edge_type2idx
    sh_irreps = Irreps.spherical_harmonics(cfg.l_max_gnn)
    orb_cfg = OrbitalIrrepConfig.from_dict({"H": "1x0e"})
    mapper = BlockIrrepMapper(orb_cfg)

    # Call the function to be tested
    edge_index, _, _, _, _, num_self_edges, is_closest_edge = compute_graph_features(
        positions=torch.tensor(atoms.get_positions(), dtype=torch.float32),
        box=torch.tensor(atoms.get_cell().array, dtype=torch.float32),
        atoms=tuple(atoms.get_chemical_symbols()),
        cfg=cfg,
        sh_irreps=sh_irreps,
        edge_type2idx=mapper.edge_type2idx,
    )

    # Assertions
    assert is_closest_edge.dtype == torch.bool
    assert len(is_closest_edge) == edge_index.shape[1]

    # Check self-edges
    assert torch.all(is_closest_edge[:num_self_edges])

    # Check off-diagonal edges
    offdiag_edges = edge_index[:, num_self_edges:]
    offdiag_is_closest = is_closest_edge[num_self_edges:]

    # Create canonical pair IDs
    num_atoms = len(atoms)
    pair_ids = torch.minimum(
        offdiag_edges[0], offdiag_edges[1]
    ) * num_atoms + torch.maximum(offdiag_edges[0], offdiag_edges[1])

    unique_pairs = torch.unique(pair_ids)

    for pair in unique_pairs:
        pair_mask = pair_ids == pair
        # Assert that exactly one edge is marked as closest for this pair
        assert torch.sum(offdiag_is_closest[pair_mask]) == 1
