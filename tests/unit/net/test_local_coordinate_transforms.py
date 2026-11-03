"""Original local-coordinate scalar primitives against independent DeepH code.

DeepH's crystal-graph and local-coordinate layers are blocks of generic stages
(library ``deeph``); they are checked against the former monolithic layers in
``tests/physics/test_reference_decomposition.py`` and against DeepH-pack
through ``tests/fixtures/reference_operators``.
"""

import importlib
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from e3nn import o3

from data.bond_geometry import bond_frames, bond_neighborhoods, tesseral_harmonics
from net.transforms.builtin import ScalarMLP
from net.transforms.local_coordinates import ChannelLinear, IrrepToBondFrame


@pytest.fixture(autouse=True)
def double_precision():
    old = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(old)


@pytest.fixture
def source(monkeypatch):
    path = os.getenv("MANDALA_DEEPH_REFERENCE")
    if not path or not (Path(path) / "deeph/model.py").exists():
        pytest.skip(
            "Set MANDALA_DEEPH_REFERENCE to enable independent source comparison"
        )
    monkeypatch.syspath_prepend(path)
    return importlib.import_module("deeph.model"), importlib.import_module(
        "deeph.preprocess.get_rc"
    )


def graph():
    positions = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.1, 0.0], [0.2, 1.3, 0.4]])
    edges = torch.tensor([(i, j) for i in range(3) for j in range(3)]).T
    vectors = positions[edges[1]] - positions[edges[0]]
    return edges, vectors


def test_bond_relations_have_two_endpoint_segments():
    edges, vectors = graph()
    frames = bond_frames(edges, vectors)
    relation = bond_neighborhoods(edges, vectors, frames, num_l=5)
    assert relation["sub_atom_idx"].shape == (54, 2)
    assert relation["sub_edge_ang"].shape == (54, 25)
    for output in range(9):
        for endpoint in range(2):
            selected = relation["sub_index"] == 2 * output + endpoint
            assert selected.sum() == 3
            assert (
                relation["sub_atom_idx"][selected, 0] == edges[endpoint, output]
            ).all()
    torch.testing.assert_close(
        frames.transpose(-1, -2) @ frames, torch.eye(3).expand_as(frames)
    )


def test_collinear_frames_fail_explicitly():
    edges = torch.tensor([[0, 0], [0, 1]])
    with pytest.raises(ValueError, match="independent"):
        bond_frames(edges, torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]))


@pytest.mark.parametrize("axes", [(0, 1, 2), (1, 2, 0)])
def test_irrep_frame_bridge_is_invariant_with_explicit_coordinate_convention(axes):
    edges, vectors = graph()
    frames = bond_frames(edges, vectors)
    irreps = o3.Irreps("2x0e + 1x1o + 1x2e")
    nodes = irreps.randn(3, -1)
    spatial = o3.rand_matrix(dtype=torch.float64)
    permutation = torch.tensor(axes)
    representation_rotation = spatial.index_select(-2, permutation).index_select(
        -1, permutation
    )
    rotated_nodes = nodes @ irreps.D_from_matrix(representation_rotation).T
    rotated_frames = spatial @ frames
    bridge = IrrepToBondFrame(irreps, axis_permutation=axes)
    torch.testing.assert_close(
        bridge(rotated_nodes, edges, rotated_frames),
        bridge(nodes, edges, frames),
        rtol=3e-12,
        atol=3e-12,
    )


@pytest.mark.parametrize("num_l", [1, 3, 5])
def test_tesseral_angles_and_default_frames_match_source(source, num_l):
    model, geometry = source
    edges, vectors = graph()
    frames = bond_frames(edges, vectors)
    for e, atom in enumerate(edges[0]):
        selected = torch.where(edges[0] == atom)[0]
        selected = selected[torch.sort(vectors[selected].norm(dim=-1)).indices]
        neighbors = SimpleNamespace(
            eijs=vectors[selected],
            indices=edges[1, selected],
            Rs=torch.zeros(len(selected), 3, dtype=torch.long),
        )
        expected, _ = geometry._get_local_coordinate(vectors[e], neighbors)
        torch.testing.assert_close(frames[e], expected)
    relations = bond_neighborhoods(edges, vectors, frames, num_l=num_l)
    # Independently use upstream polar-axis conversion and spherical harmonics.
    for e in range(edges.shape[1]):
        local = -vectors @ frames[e]
        spherical = model.get_spherical_from_cartesian(local)
        expected = torch.cat(
            [
                model.SphericalHarmonics().get(l, spherical[:, 0], spherical[:, 1])
                for l in range(num_l)
            ],
            dim=-1,
        )
        torch.testing.assert_close(
            tesseral_harmonics(local, num_l), expected, rtol=1e-13, atol=1e-13
        )
        for endpoint in range(2):
            selected = relations["sub_index"] == 2 * e + endpoint
            torch.testing.assert_close(
                relations["sub_edge_ang"][selected],
                expected[relations["sub_edge_idx"][selected]],
                rtol=1e-13,
                atol=1e-13,
            )


def test_edge_and_channel_readouts_match_source(source):
    model, _ = source
    node, edge = torch.randn(3, 4), torch.randn(5, 3)
    indices = torch.tensor([[0, 0, 1, 1, 2], [1, 2, 0, 2, 0]])
    pairs = torch.cat([node[indices[0]], node[indices[1]], edge], -1)
    ours = ScalarMLP(11, [128], 7, "silu", activate_last=True, bias=True)
    reference = model.MPLayer(4, 3, 7, True, True, "LayerNorm", "CGConv", 6).e_lin
    ours.layers.load_state_dict(reference.state_dict())
    torch.testing.assert_close(ours(pairs), reference(pairs))
    ours_channel = ChannelLinear(7, 4, 2)
    reference_channel = model.MultipleLinear(7, 4, 2)
    ours_channel.load_state_dict(reference_channel.state_dict())
    torch.testing.assert_close(ours_channel(node), reference_channel(node, None))
