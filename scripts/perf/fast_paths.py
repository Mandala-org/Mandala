"""Vectorised drop-in replacements for per-edge Python loops in the framework.

Every function here must return results identical to the function it
replaces (same values, dtypes, ordering and container layout). The originals
loop over edges and, for CUDA tensors, synchronise the device on every
element access. These versions move each index tensor to the host at most a
few times per call and do the per-edge work with tensor operations.

They are used by ``scripts/perf/optimize_reference_inference.py`` as
switchable patches; nothing in ``src`` imports them.
"""

from __future__ import annotations

from collections import Counter, defaultdict

import torch


# ---------------------------------------------------------------------------
# data.block_matrix.IrrepsBlockData.from_packed
# ---------------------------------------------------------------------------
def from_packed(cls, values, x, mapper, layout, *, frame="global", operator=None):
    species = tuple(layout.species_shells)
    types = x["node_type_idx"]
    atoms = tuple(species[int(i)] for i in types.tolist())
    edges = x.get("edge_keys")
    if edges is None:
        shifts = x.get(
            "edge_shift", x["edge_index"].new_zeros((3, x["edge_index"].shape[1]))
        )
        edges = torch.cat((shifts, x["edge_index"]), dim=0)
    count = len(species)
    types = types.to(edges.device).long()
    codes = (types[edges[3]] * count + types[edges[4]]).tolist()
    edge_rows = edges.T.tolist()
    buckets = defaultdict(list)
    for index, code in enumerate(codes):
        buckets[code].append(index)
    pair_vectors, pair_edges, lookup = {}, {}, {}
    for li, left in enumerate(species):
        for ri, right in enumerate(species):
            indices = buckets.get(li * count + ri)
            if not indices:
                continue
            selected = torch.tensor(indices, device=values.device, dtype=torch.long)
            pair = f"{left}-{right}"
            pair_vectors[pair] = layout.encode(values.index_select(0, selected))
            pair_edges[pair] = edges.index_select(1, selected.to(edges.device))
            for local, global_index in enumerate(indices):
                lookup[tuple(edge_rows[global_index])] = (pair, local)
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


# ---------------------------------------------------------------------------
# core.sparse_math.build_trace_alignment_from_pair_edges
# ---------------------------------------------------------------------------
def trace_alignment(pair_edges_a, pair_edges_b=None, *, device=None):
    if pair_edges_b is None:
        pair_edges_b = pair_edges_a
    alignment = {}
    host_b = {}
    for key, edges_a in pair_edges_a.items():
        el_a, el_b = key.split("-")
        rev_key = f"{el_b}-{el_a}"
        if rev_key not in pair_edges_b:
            raise ValueError(
                f"Key {rev_key} missing in second edge set (reverse of {key})."
            )
        if rev_key not in host_b:
            host_b[rev_key] = {
                tuple(edge): idx
                for idx, edge in enumerate(pair_edges_b[rev_key].t().tolist())
            }
        mapping = host_b[rev_key]
        indices_b = []
        for sx, sy, sz, i, j in edges_a.t().tolist():
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


# ---------------------------------------------------------------------------
# data.block_matrix.IrrepsBlockData.to_blocks
# ---------------------------------------------------------------------------
def to_blocks(self, mapper):
    from data.block_matrix import BlockMatrix

    pair_blk = {}
    global_rows = None
    for key, vec in self.pair_vectors.items():
        pair_blk[key] = mapper.vectors_to_blocks(key, vec, layout=self.layout)
        if self.frames is not None:
            if global_rows is None:
                global_rows = {
                    tuple(edge): index
                    for index, edge in enumerate(self.packed_edges.T.tolist())
                }
            rows = torch.tensor(
                [global_rows[tuple(edge)] for edge in self.pair_edges[key].T.tolist()],
                dtype=torch.long,
                device=self.frames.device,
            )
            frames = self.frames.index_select(0, rows)
            left, right = key.split("-")
            pair_blk[key] = self.layout.rotation.rotate_blocks(
                pair_blk[key],
                frames.transpose(-1, -2),
                self.layout.species_shells[left],
                self.layout.species_shells[right],
            )
        if self.layout is not None and self.operator == "hamiltonian":
            from utils.units import HARTREE_TO_EV

            pair_blk[key] = pair_blk[key] / HARTREE_TO_EV
    result = BlockMatrix(
        self.atoms,
        self.atom_counts,
        pair_blk,
        self.pair_edges,
        self.lookup,
        self.orbital_cfg,
        self.basis,
    )
    if self.layout is not None:
        result._coverage_masks = {
            pair: self.layout.unpack_blocks(
                tuple(pair.split("-")),
                self.layout.packed_mask(tuple(pair.split("-")), device=block.device),
            ).expand_as(block)
            for pair, block in pair_blk.items()
        }
    return result


# ---------------------------------------------------------------------------
# data.bond_geometry.bond_neighborhood_relations
# ---------------------------------------------------------------------------
def _eligible_by_source(edge_index, neighbor_selection):
    """Ascending edge indices grouped by source atom, as CSR (order, ptr)."""
    sources = edge_index[0]
    eligible = torch.arange(sources.numel(), device=sources.device)
    if neighbor_selection is not None:
        eligible = eligible[torch.isin(eligible, neighbor_selection)]
    order = eligible[torch.sort(sources[eligible], stable=True).indices]
    num_atoms = int(edge_index.max()) + 1 if edge_index.numel() else 0
    counts = torch.bincount(sources[order], minlength=num_atoms)
    ptr = torch.cumsum(counts, 0) - counts
    return order, ptr, counts


def bond_neighborhood_relations(edge_index, *, neighbor_selection=None):
    if edge_index.shape[1] == 0:
        return {
            "sub_atom_idx": edge_index.new_empty((0, 2)),
            "sub_edge_idx": edge_index.new_empty(0),
            "sub_index": edge_index.new_empty(0),
        }
    order, ptr, counts = _eligible_by_source(edge_index, neighbor_selection)
    centers = edge_index.T.reshape(-1)  # edge 0 source, edge 0 target, edge 1 ...
    sizes = counts[centers]
    total = int(sizes.sum())
    segment = torch.repeat_interleave(
        torch.arange(centers.numel(), device=edge_index.device), sizes
    )
    starts = torch.cumsum(sizes, 0) - sizes
    offset = torch.arange(total, device=edge_index.device) - starts[segment]
    selected = order[ptr[centers[segment]] + offset]
    return {
        "sub_atom_idx": torch.stack([centers[segment], edge_index[1, selected]], -1),
        "sub_edge_idx": selected,
        "sub_index": segment,
    }


# ---------------------------------------------------------------------------
# data.bond_geometry.bond_frames
# ---------------------------------------------------------------------------
def bond_frames(edge_index, vectors, *, tolerance=1e-6, neighbor_selection=None):
    sources = edge_index[0]
    count = sources.numel()
    if not count:
        return vectors.new_empty((0, 3, 3))
    # The neighbour ordering depends only on the source atom: compute it once
    # per atom with exactly the original sequence of operations.
    candidates = {}
    for source in sorted(set(sources.tolist())):
        neighbors = torch.where(sources == source)[0]
        if neighbor_selection is not None:
            neighbors = neighbors[torch.isin(neighbors, neighbor_selection)]
        neighbors = neighbors[torch.sort(vectors[neighbors].norm(dim=-1)).indices]
        nonzero = neighbors[vectors[neighbors].norm(dim=-1) > 1e-8]
        if not len(nonzero):
            raise ValueError(f"Cannot construct bond frame for isolated atom {source}")
        candidates[source] = nonzero
    width = max(len(v) for v in candidates.values())
    table = torch.full(
        (int(max(candidates)) + 1, width), -1, dtype=torch.long, device=vectors.device
    )
    for source, nonzero in candidates.items():
        table[source, : len(nonzero)] = nonzero
    rows = table[sources]  # (E, K)
    valid = rows >= 0
    first = vectors
    zero = (first.detach().abs() <= 1e-8).all(-1)  # torch.allclose(first, 0)
    first = torch.where(zero[:, None], vectors[rows[:, 0]], first)
    candidate = vectors[rows.clamp_min(0)]  # (E, K, 3)
    normal = torch.linalg.cross(first[:, None, :].expand_as(candidate), candidate)
    lengths = normal.norm(dim=-1)
    usable = valid & (lengths > tolerance)
    if not bool(usable.any(-1).all()):
        bad = int(sources[~usable.any(-1)][0])
        raise ValueError(f"No linearly independent chemical bond for atom {bad}")
    pick = usable.to(torch.int8).argmax(-1)
    arange = torch.arange(count, device=vectors.device)
    second = normal[arange, pick] / lengths[arange, pick, None]
    axis = first / first.norm(dim=-1, keepdim=True)
    return torch.stack([axis, second, torch.linalg.cross(axis, second)], -1)


# ---------------------------------------------------------------------------
# torch_scatter.scatter replacement that torch.compile handles correctly.
# ---------------------------------------------------------------------------
_TORCH_SCATTER = None


def compile_safe_scatter(src, index, dim=-1, out=None, dim_size=None, reduce="sum"):
    """Same results as torch_scatter.scatter for 1-D indices, in plain torch.

    torch 2.7.1's torch.compile miscompiles torch_scatter's ``reduce="mean"``
    (errors up to ~40% of the result magnitude on CUDA; ``add`` is fine).
    The cause is the element count: compiled ``index_add_`` of a ones tensor
    into a single slot rounds the count up to a multiple of 128 (72 -> 128).
    Counts here are one-hot sums, which compile correctly. Summation order
    differs from torch_scatter's atomics, so results agree to floating-point
    noise, not bitwise.
    """
    if out is not None or index.dim() != 1:
        return _TORCH_SCATTER(src, index, dim, out, dim_size, reduce)
    dim = dim % src.dim()
    if dim_size is None:
        dim_size = int(index.max()) + 1 if index.numel() else 0
    shape = list(src.shape)
    shape[dim] = dim_size
    if reduce in ("sum", "add", "mean"):
        total = src.new_zeros(shape).index_add_(dim, index, src)
        if reduce != "mean":
            return total
        count = segment_counts(index, dim_size).to(src.dtype)
        view = [1] * src.dim()
        view[dim] = dim_size
        return total / count.clamp_min(1).view(view)
    if reduce == "max":
        expanded = index.view(
            [-1 if d == dim else 1 for d in range(src.dim())]
        ).expand_as(src)
        return src.new_zeros(shape).scatter_reduce_(
            dim, expanded, src, "amax", include_self=False
        )
    return _TORCH_SCATTER(src, index, dim, out, dim_size, reduce)


def segment_counts(index, dim_size):
    """Occurrences of each segment id; correct under torch.compile."""
    if index.numel() * max(dim_size, 1) <= 50_000_000:
        arange = torch.arange(dim_size, device=index.device)
        return (index[:, None] == arange).sum(0)
    return torch.bincount(index, minlength=dim_size)


def scatter_aliases():
    """(module, attribute) pairs currently bound to torch_scatter.scatter."""
    import sys

    import torch_scatter

    global _TORCH_SCATTER
    _TORCH_SCATTER = torch_scatter.scatter
    found = []
    for module in list(sys.modules.values()):
        try:
            if getattr(module, "scatter", None) is torch_scatter.scatter:
                found.append((module, "scatter"))
        except Exception:
            continue
    return found


# ---------------------------------------------------------------------------
# torch_geometric.utils.degree replacement (same compile bug: a float
# scatter-add of ones into one slot counts 72 as 128). PyG's graph-mode
# LayerNorm uses it, which breaks DeepH's node updates under torch.compile.
# ---------------------------------------------------------------------------
_PYG_DEGREE = None


def compile_safe_degree(index, num_nodes=None, dtype=None):
    if num_nodes is None:
        num_nodes = int(index.max()) + 1 if index.numel() else 0
    counts = segment_counts(index, num_nodes)
    return counts.to(dtype if dtype is not None else torch.long)


def degree_aliases():
    """(module, attribute) pairs currently bound to torch_geometric's degree."""
    import sys

    from torch_geometric import utils

    global _PYG_DEGREE
    _PYG_DEGREE = utils.degree
    found = []
    for module in list(sys.modules.values()):
        try:
            if getattr(module, "degree", None) is utils.degree:
                found.append((module, "degree"))
        except Exception:
            continue
    return found
