"""Vectorised host-side paths reproduce the former per-edge loops exactly.

The reference functions below are the implementations these paths replaced
(develop a066506e), kept verbatim apart from their names. On CUDA tensors the
loops synchronised the device once per edge; the replacements must return
identical values, dtypes, orderings and containers.
"""

from __future__ import annotations

from collections import Counter
import types

import pytest
import torch

from core.sparse_math import build_trace_alignment_from_pair_edges
from data.block_matrix import IrrepsBlockData
from data.bond_geometry import bond_frames, bond_neighborhood_relations


# --------------------------------------------------------------------------
# Former implementations (references).
# --------------------------------------------------------------------------
def reference_bond_frames(
    edge_index, vectors, *, tolerance=1e-6, neighbor_selection=None
):
    """First bond and shortest independent neighbor define an oriented frame.

    Stored edge order resolves distance ties exactly through torch.sort. A
    collinear environment is rejected rather than assigning an arbitrary gauge.
    """
    frames = []
    for edge, source in enumerate(edge_index[0].tolist()):
        neighbors = torch.where(edge_index[0] == source)[0]
        if neighbor_selection is not None:
            neighbors = neighbors[torch.isin(neighbors, neighbor_selection)]
        neighbors = neighbors[torch.sort(vectors[neighbors].norm(dim=-1)).indices]
        nonzero = neighbors[vectors[neighbors].norm(dim=-1) > 1e-8]
        if not len(nonzero):
            raise ValueError(f"Cannot construct bond frame for isolated atom {source}")
        first = vectors[edge]
        if torch.allclose(first.detach(), torch.zeros_like(first)):
            first = vectors[nonzero[0]]
        second = None
        for neighbor in nonzero:
            candidate = vectors[neighbor]
            normal = torch.linalg.cross(first, candidate)
            if normal.norm() > tolerance:
                second = normal / normal.norm()
                break
        if second is None:
            raise ValueError(f"No linearly independent chemical bond for atom {source}")
        axis = first / first.norm()
        frames.append(torch.stack([axis, second, torch.linalg.cross(axis, second)], -1))
    return torch.stack(frames) if frames else vectors.new_empty((0, 3, 3))


def reference_bond_neighborhood_relations(edge_index, *, neighbor_selection=None):
    """Prepare bond-rooted index relations; no geometry or harmonics involved."""
    atoms, edges, segments = [], [], []
    for output_edge, endpoints in enumerate(edge_index.T):
        for endpoint, center in enumerate(endpoints):
            selected = torch.where(edge_index[0] == center)[0]
            if neighbor_selection is not None:
                selected = selected[torch.isin(selected, neighbor_selection)]
            atoms.append(
                torch.stack([center.expand(len(selected)), edge_index[1, selected]], -1)
            )
            edges.append(selected)
            segments.append(torch.full_like(selected, 2 * output_edge + endpoint))
    return {
        "sub_atom_idx": torch.cat(atoms) if atoms else edge_index.new_empty((0, 2)),
        "sub_edge_idx": torch.cat(edges) if edges else edge_index.new_empty(0),
        "sub_index": torch.cat(segments) if segments else edge_index.new_empty(0),
    }


def reference_trace_alignment(
    pair_edges_a,
    pair_edges_b=None,
    *,
    device=None,
):
    """
    Precompute reverse-edge alignment for vectorized sparse traces.

    For each key ``A-B`` in ``pair_edges_a``, the returned mapping stores:
    - reverse key ``B-A`` in ``pair_edges_b``
    - a 1-D LongTensor of indices aligning reverse edges in ``pair_edges_b`` to
      the order of edges in ``pair_edges_a``
    """
    if pair_edges_b is None:
        pair_edges_b = pair_edges_a

    alignment = {}
    for key, edges_a in pair_edges_a.items():
        el_a, el_b = key.split("-")
        rev_key = f"{el_b}-{el_a}"
        if rev_key not in pair_edges_b:
            raise ValueError(
                f"Key {rev_key} missing in second edge set (reverse of {key})."
            )

        edges_b_rev = pair_edges_b[rev_key]
        mapping = {
            tuple(map(int, edge.tolist())): idx
            for idx, edge in enumerate(edges_b_rev.t())
        }

        indices_b = []
        for edge in edges_a.t():
            sx, sy, sz, i, j = map(int, edge.tolist())
            rev_edge = (-sx, -sy, -sz, j, i)
            if rev_edge not in mapping:
                raise ValueError(
                    f"Edge {rev_edge} (reverse of {(sx, sy, sz, i, j)}) missing in second edge set."
                )
            indices_b.append(mapping[rev_edge])

        idx_device = device if device is not None else edges_a.device
        alignment[key] = (
            rev_key,
            torch.tensor(indices_b, device=idx_device, dtype=torch.long),
        )

    return alignment


def reference_from_packed(
    cls, values, x, mapper, layout, *, frame="global", operator=None
):
    """Represent packed orbital channels through MANDALA's public container."""
    from collections import Counter

    species = tuple(layout.species_shells)
    atoms = tuple(species[int(index)] for index in x["node_type_idx"])
    edges = x.get("edge_keys")
    if edges is None:
        shifts = x.get(
            "edge_shift", x["edge_index"].new_zeros((3, x["edge_index"].shape[1]))
        )
        edges = torch.cat((shifts, x["edge_index"]), dim=0)
    pair_vectors, pair_edges, lookup = {}, {}, {}
    for left in species:
        for right in species:
            selected = torch.tensor(
                [
                    index
                    for index, edge in enumerate(edges.T)
                    if atoms[int(edge[3])] == left and atoms[int(edge[4])] == right
                ],
                device=values.device,
                dtype=torch.long,
            )
            if not len(selected):
                continue
            pair = f"{left}-{right}"
            pair_vectors[pair] = layout.encode(values.index_select(0, selected))
            pair_edges[pair] = edges.index_select(1, selected)
            for local, global_index in enumerate(selected.tolist()):
                lookup[tuple(edges[:, global_index].tolist())] = (pair, local)
    frames = x.get("bond_frames") if frame == "bond" else None
    if frame == "bond" and frames is None:
        raise ValueError("Bond-frame output requires explicit frames")
    return cls(
        atoms,
        Counter(atoms),
        pair_vectors,
        pair_edges,
        lookup,
        mapper.orbital_cfg,
        "openmx",
        layout,
        values,
        edges,
        frames,
        operator,
    )


# --------------------------------------------------------------------------
# Inputs.
# --------------------------------------------------------------------------
def _graph(atoms, neighbors, dtype, seed):
    generator = torch.Generator().manual_seed(seed)
    positions = torch.rand(atoms, 3, generator=generator, dtype=dtype) * 6
    source = torch.arange(atoms).repeat_interleave(neighbors)
    target = torch.randint(0, atoms, (atoms * neighbors,), generator=generator)
    shift = torch.randint(-1, 2, (atoms * neighbors, 3), generator=generator)
    vectors = positions[target] + shift.to(dtype) * 6 - positions[source]
    vectors[0] = 0.0  # a zero-length periodic self image uses the fallback bond
    return torch.stack([source, target]), vectors


def _symmetric_keys(atoms, edges, seed):
    generator = torch.Generator().manual_seed(seed)
    i = torch.randint(0, atoms, (edges,), generator=generator)
    j = torch.randint(0, atoms, (edges,), generator=generator)
    s = torch.randint(-2, 3, (3, edges), generator=generator)
    forward = torch.cat((s, i[None], j[None]))
    reverse = torch.cat((-s, j[None], i[None]))
    return torch.unique(torch.cat((forward, reverse), 1), dim=1)


def _selection(edges, seed):
    generator = torch.Generator().manual_seed(seed)
    return torch.randperm(edges, generator=generator)[: 3 * edges // 4].sort().values


# --------------------------------------------------------------------------
# Tests.
# --------------------------------------------------------------------------
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("select", [False, True])
def test_bond_frames_match_reference(dtype, select):
    edge_index, vectors = _graph(24, 12, dtype, seed=1)
    selection = _selection(edge_index.shape[1], 2) if select else None
    expected = reference_bond_frames(edge_index, vectors, neighbor_selection=selection)
    actual = bond_frames(edge_index, vectors, neighbor_selection=selection)
    assert actual.dtype == expected.dtype
    assert torch.equal(actual, expected)


def test_bond_frames_empty_and_isolated():
    empty = torch.empty((2, 0), dtype=torch.long)
    assert bond_frames(empty, torch.empty((0, 3))).shape == (0, 3, 3)
    collinear = torch.tensor([[0, 0], [1, 1]])
    vectors = torch.tensor([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    with pytest.raises(ValueError, match="No linearly independent"):
        bond_frames(collinear, vectors)


@pytest.mark.parametrize("select", [False, True])
def test_bond_neighborhood_relations_match_reference(select):
    edge_index, _ = _graph(24, 12, torch.float64, seed=3)
    selection = _selection(edge_index.shape[1], 4) if select else None
    expected = reference_bond_neighborhood_relations(
        edge_index, neighbor_selection=selection
    )
    actual = bond_neighborhood_relations(edge_index, neighbor_selection=selection)
    assert actual.keys() == expected.keys()
    for key in expected:
        assert actual[key].dtype == expected[key].dtype
        assert torch.equal(actual[key], expected[key]), key


def test_bond_neighborhood_relations_empty():
    empty = torch.empty((2, 0), dtype=torch.long)
    for key, value in bond_neighborhood_relations(empty).items():
        assert value.numel() == 0, key


@pytest.mark.parametrize("species", [("C",), ("Mo", "S")])
def test_trace_alignment_matches_reference(species):
    keys = _symmetric_keys(30, 400, seed=5)
    types_ = torch.arange(30) % len(species)
    pair_edges = {}
    for a, left in enumerate(species):
        for b, right in enumerate(species):
            mask = (types_[keys[3]] == a) & (types_[keys[4]] == b)
            if mask.any():
                pair_edges[f"{left}-{right}"] = keys[:, mask]
    expected = reference_trace_alignment(pair_edges)
    actual = build_trace_alignment_from_pair_edges(pair_edges)
    assert actual.keys() == expected.keys()
    for key in expected:
        assert actual[key][0] == expected[key][0]
        assert torch.equal(actual[key][1], expected[key][1])


def test_trace_alignment_reports_missing_reverse_edge():
    pair_edges = {"C-C": torch.tensor([[0], [0], [1], [0], [1]])}
    with pytest.raises(ValueError, match="missing in second edge set"):
        build_trace_alignment_from_pair_edges(pair_edges)


@pytest.mark.parametrize("species", [("C",), ("Mo", "S")])
@pytest.mark.parametrize("use_edge_keys", [False, True])
def test_from_packed_matches_reference(species, use_edge_keys):
    keys = _symmetric_keys(30, 400, seed=7)
    layout = types.SimpleNamespace(
        species_shells={name: None for name in species}, encode=lambda v: v * 2.0
    )
    mapper = types.SimpleNamespace(orbital_cfg="cfg")
    x = {
        "node_type_idx": torch.arange(30) % len(species),
        "edge_index": keys[3:],
        "edge_shift": keys[:3],
    }
    if use_edge_keys:
        x["edge_keys"] = keys
    values = torch.randn(keys.shape[1], 17, generator=torch.Generator().manual_seed(8))
    expected = reference_from_packed(
        IrrepsBlockData, values, x, mapper, layout, operator="hamiltonian"
    )
    actual = IrrepsBlockData.from_packed(
        values, x, mapper, layout, operator="hamiltonian"
    )
    assert actual.atoms == expected.atoms
    assert actual.atom_counts == expected.atom_counts == Counter(expected.atoms)
    assert list(actual.lookup.items()) == list(expected.lookup.items())
    assert actual.pair_vectors.keys() == expected.pair_vectors.keys()
    for pair in expected.pair_vectors:
        assert torch.equal(actual.pair_vectors[pair], expected.pair_vectors[pair])
        assert torch.equal(actual.pair_edges[pair], expected.pair_edges[pair])
    assert torch.equal(actual.packed_edges, expected.packed_edges)
