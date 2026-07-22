import torch
from e3nn.o3 import Irreps
from net.common import Config
from data.graph_features import compute_graph_features, _minimal_disp


def test_minimal_disp():
    # Test _minimal_disp logic
    # 1D case for simplicity (embedded in 3D)
    # Cell size 10.0 along x
    box = torch.eye(3) * 10.0
    inv_box = torch.inverse(box)

    # Atom at 1.0, Atom at 9.0
    # Distance should be 2.0 (wrapping around)
    # Vector from 1.0 to 9.0 is +8.0. Minimal image is -2.0.

    pos = torch.tensor([[1.0, 0.0, 0.0], [9.0, 0.0, 0.0]])

    # Edge 0->1
    # _minimal_disp expects edges tensor where indices 3 and 4 are src and dst
    edges = torch.tensor([0, 0, 0, 0, 1])  # (sx, sy, sz, src, dst)

    disp = _minimal_disp(pos, edges, box, inv_box)

    assert torch.allclose(disp, torch.tensor([-2.0, 0.0, 0.0]))


def test_compute_graph_features():
    # Test full graph construction
    # 2 atoms in a box
    pos = torch.tensor([[1.0, 0.0, 0.0], [9.0, 0.0, 0.0]])
    box = torch.eye(3) * 10.0
    atoms = ("H", "H")

    cfg = Config(
        cutoff_radius=3.0,  # Should catch wrapped edge (dist 2.0)
        n_radial=5,
    )
    sh_irreps = Irreps("1x0e + 1x1o")
    edge_type2idx = {"H-H": 0}

    # Expected edges:
    # Self edges: (0,0), (1,1)
    # Wrapped edges: (0,1) with shift (-1,0,0) -> dist 2.0
    #                (1,0) with shift (1,0,0) -> dist 2.0

    (
        edge_index,
        edge_shift,
        edge_type_idx,
        edge_length_emb,
        edge_sh,
        num_self_edges,
        edge_lengths,
    ) = compute_graph_features(pos, box, atoms, cfg, sh_irreps, edge_type2idx)

    # Check self edges
    assert num_self_edges == 2
    # First 2 edges should be self edges (sorted by src)
    # (0,0) and (1,1)
    assert edge_index[0, 0] == 0 and edge_index[1, 0] == 0
    assert edge_index[0, 1] == 1 and edge_index[1, 1] == 1

    # Check other edges
    # Should have 2 more edges
    assert edge_index.shape[1] == 4
    assert torch.allclose(edge_lengths[:num_self_edges], torch.zeros(2))
    assert torch.allclose(edge_lengths[num_self_edges:], torch.full((2,), 2.0))

    # Edge 0->1
    # Find edge where src=0, dst=1
    mask = (edge_index[0] == 0) & (edge_index[1] == 1)
    assert mask.sum() == 1
    idx = torch.where(mask)[0][0]

    # Check shift
    # 0->1 is +8.0. Minimal is -2.0.
    # delta = pos[1] - pos[0] + shift@cell
    # -2.0 = 8.0 + shift*10 -> shift = -1
    assert torch.allclose(edge_shift[:, idx].float(), torch.tensor([-1.0, 0.0, 0.0]))

    # Edge 1->0
    # Find edge where src=1, dst=0
    mask = (edge_index[0] == 1) & (edge_index[1] == 0)
    assert mask.sum() == 1
    idx = torch.where(mask)[0][0]

    # Check shift
    # 1->0 is -8.0. Minimal is +2.0.
    # 2.0 = -8.0 + shift*10 -> shift = +1
    assert torch.allclose(edge_shift[:, idx].float(), torch.tensor([1.0, 0.0, 0.0]))

    # Zero-shift self edges should keep only the scalar SH channel.
    assert torch.allclose(
        edge_sh[:num_self_edges, 1:], torch.zeros_like(edge_sh[:num_self_edges, 1:])
    )


def test_compute_graph_features_is_invariant_under_atom_image_relabeling():
    cfg = Config(
        cutoff_radius=3.0,
        n_radial=5,
        safety_checks=True,
    )
    sh_irreps = Irreps("1x0e + 1x1o")
    edge_type2idx = {"H-H": 0}
    atoms = ("H", "H")
    box = torch.eye(3) * 10.0

    pos_ref = torch.tensor([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    pos_shifted = torch.tensor([[0.0, 0.0, 0.0], [12.0, 0.0, 0.0]])
    atom_image_shifts = torch.tensor([[0, 0, 0], [1, 0, 0]], dtype=torch.long)

    out_ref = compute_graph_features(pos_ref, box, atoms, cfg, sh_irreps, edge_type2idx)
    out_shifted = compute_graph_features(
        pos_shifted, box, atoms, cfg, sh_irreps, edge_type2idx
    )

    (
        edge_index_ref,
        edge_shift_ref,
        edge_type_idx_ref,
        edge_length_emb_ref,
        edge_sh_ref,
        num_self_edges_ref,
        edge_lengths_ref,
    ) = out_ref
    (
        edge_index_shifted,
        edge_shift_shifted,
        edge_type_idx_shifted,
        edge_length_emb_shifted,
        edge_sh_shifted,
        num_self_edges_shifted,
        edge_lengths_shifted,
    ) = out_shifted

    # Geometry and ordering should be identical.
    assert torch.equal(edge_index_ref, edge_index_shifted)
    assert torch.equal(edge_type_idx_ref, edge_type_idx_shifted)
    assert num_self_edges_ref == num_self_edges_shifted
    assert torch.allclose(edge_length_emb_ref, edge_length_emb_shifted)
    assert torch.allclose(edge_sh_ref, edge_sh_shifted)
    assert torch.allclose(edge_lengths_ref, edge_lengths_shifted)

    # Edge shifts change exactly by the atom-image relabeling rule:
    # new_shift = old_shift + shift(src) - shift(dst)
    src = edge_index_ref[0]
    dst = edge_index_ref[1]
    expected_edge_shift = (
        edge_shift_ref + atom_image_shifts[src].T - atom_image_shifts[dst].T
    )
    assert torch.equal(edge_shift_shifted, expected_edge_shift)
