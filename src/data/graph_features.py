import torch
from e3nn.o3 import Irreps, spherical_harmonics
from e3nn.math import soft_one_hot_linspace
from net.common import Config
from typing import Tuple, Dict

from ase import Atoms
from ase.neighborlist import neighbor_list


def _non_scalar_sh_slices(sh_irreps: Irreps) -> list[slice]:
    return [
        sh_irreps.slices()[idx] for idx, (_, ir) in enumerate(sh_irreps) if ir.l > 0
    ]


def _minimal_disp(
    pos: torch.Tensor,
    edges: torch.Tensor,
    box: torch.Tensor | None,
    inv_box: torch.Tensor | None = None,
) -> torch.Tensor:
    if box is None:
        return pos[edges[4]] - pos[edges[3]]
    if inv_box is None:
        inv_box = torch.inverse(box)
    delta = pos[edges[4]] - pos[edges[3]]  # cart
    frac = delta @ inv_box
    frac = frac - torch.round(frac)
    return frac @ box


def compute_edge_geometry_from_static_edges(
    *,
    positions: torch.Tensor,
    box: torch.Tensor | None,
    edge_index: torch.Tensor,
    edge_shift: torch.Tensor,
    sh_irreps: Irreps,
    cutoff_radius: float,
    n_radial: int,
    radial_embedding_scale: str,
    radial_lengths: torch.Tensor | None = None,
    radial_basis_end: float | None = None,
    edge_batch: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if box is not None:
        shift_float = edge_shift.T.to(dtype=positions.dtype)
        edge_disp = (
            positions[edge_index[1]]
            - positions[edge_index[0]]
            + (
                torch.einsum("ei,eij->ej", shift_float, box[edge_batch])
                if box.ndim == 3
                else shift_float @ box
            )
        )
    else:
        edge_disp = positions[edge_index[1]] - positions[edge_index[0]]

    edge_lengths = torch.linalg.norm(edge_disp, dim=-1)
    edge_sh = spherical_harmonics(
        sh_irreps, edge_disp, normalize=True, normalization="component"
    )
    is_zero_shift_self_edge = (edge_index[0] == edge_index[1]) & (edge_shift == 0).all(
        dim=0
    )
    sh_non_scalar_slices = _non_scalar_sh_slices(sh_irreps)
    if is_zero_shift_self_edge.any() and sh_non_scalar_slices:
        edge_sh = edge_sh.clone()
        for slc in sh_non_scalar_slices:
            edge_sh[is_zero_shift_self_edge, slc] = 0.0
    if radial_lengths is None:
        radial_lengths = edge_lengths
    else:
        radial_lengths = radial_lengths.to(
            device=edge_lengths.device, dtype=edge_lengths.dtype
        )
    if radial_basis_end is None:
        radial_basis_end = cutoff_radius

    edge_length_emb = soft_one_hot_linspace(
        radial_lengths,
        start=0.0,
        end=radial_basis_end,
        number=n_radial,
        basis="gaussian",
        cutoff=False,
    )
    if radial_embedding_scale == "sqrt_n_radial":
        edge_length_emb = edge_length_emb * (n_radial**0.5)
    return edge_length_emb, edge_sh, edge_lengths


def compute_graph_features(
    positions: torch.Tensor,
    box: torch.Tensor | None,
    atoms: Tuple[str, ...],
    cfg: Config,
    sh_irreps: Irreps,
    edge_type2idx: Dict[str, int],
    edge_type_r0: torch.Tensor | None = None,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    int,
    torch.Tensor,
]:
    """
    Returns
    -------
    edge_index : (2, E_total) long
    edge_shift : (3, E_total) long
    edge_type_idx : (E_total,) long
    edge_length_emb : (E_total, cfg.n_radial) float
    edge_sh : (E_total, sh_dim) float
    num_self_edges : int
    edge_lengths : (E_total,) float
    """

    # 1. Create ase.Atoms object
    ase_atoms = Atoms(
        symbols=atoms,
        positions=positions.detach().cpu().numpy(),
        cell=box.detach().cpu().numpy() if box is not None else None,
        pbc=box is not None,
    )

    if box is None:
        print("Warning: box is None, periodic boundary conditions are disabled.")

    # 2. Use ase.neighborlist to get edges and offsets
    # 'i' is the source atom index, 'j' is the destination atom index,
    # 'S' is the offset vector in lattice coordinates.
    src, dst, offsets = neighbor_list(
        "ijS", ase_atoms, cfg.cutoff_radius, self_interaction=False
    )

    # 3. Handle self-edges explicitly
    num_atoms = len(atoms)
    self_edge_src = torch.arange(num_atoms, dtype=torch.long, device=positions.device)
    self_edge_dst = torch.arange(num_atoms, dtype=torch.long, device=positions.device)
    self_edge_shift = torch.zeros(
        (3, num_atoms), dtype=torch.long, device=positions.device
    )

    # 4. Combine self-edges and off-diagonal edges
    offdiag_edge_src_unsorted = torch.from_numpy(src).to(positions.device)
    offdiag_edge_dst_unsorted = torch.from_numpy(dst).to(positions.device)
    offdiag_edge_shift_unsorted = (
        torch.from_numpy(offsets).to(positions.device).to(torch.long).T
    )

    # 5. Calculate displacement vectors using the offsets
    # disp = pos[j] - pos[i] + S @ box
    # Match the calculation in Snapshot._edge_displacements exactly (order and dtype)
    if box is not None:
        shift_float = offdiag_edge_shift_unsorted.T.to(positions.device).to(
            positions.dtype
        )
        offdiag_disp_unsorted = (
            positions[offdiag_edge_dst_unsorted]
            - positions[offdiag_edge_src_unsorted]
            + shift_float @ box.to(positions.device)
        )
    else:
        offdiag_disp_unsorted = (
            positions[offdiag_edge_dst_unsorted] - positions[offdiag_edge_src_unsorted]
        )

    if cfg.safety_checks:
        # Check periodic equivalence of the reconstructed destination positions.
        dst_residual = (
            positions[offdiag_edge_src_unsorted] + offdiag_disp_unsorted
        ) - positions[offdiag_edge_dst_unsorted]
        if box is not None:
            inv_box = torch.linalg.inv(box.to(positions.device))
            frac_residual = dst_residual @ inv_box
            frac_residual = frac_residual - torch.round(frac_residual)
            cart_residual = frac_residual @ box.to(positions.device)
        else:
            cart_residual = dst_residual
        assert torch.all(
            torch.linalg.norm(cart_residual, dim=-1) < 1e-4
        ), "Displacement vectors do not lead to periodic-equivalent destination positions."

    # 6. Sort off-diagonal edges
    # Match the sorting logic in Snapshot.canonicalize_edges:
    # Primary key: distance
    # Tie-breaker: sx, sy, sz, src, dst

    offdiag_lengths_unsorted = torch.linalg.norm(offdiag_disp_unsorted, dim=-1)

    # Move to CPU for sorting
    od_d_cpu = offdiag_lengths_unsorted.cpu().tolist()
    od_src_cpu = offdiag_edge_src_unsorted.cpu().tolist()
    od_dst_cpu = offdiag_edge_dst_unsorted.cpu().tolist()
    od_shift_cpu = (
        offdiag_edge_shift_unsorted.t().cpu().tolist()
    )  # List of [sx, sy, sz]

    # Combine into a list of tuples
    # (dist, sx, sy, sz, src, dst, original_idx)
    to_sort = []
    for i in range(len(od_d_cpu)):
        shift = od_shift_cpu[i]
        to_sort.append(
            (
                od_d_cpu[i],
                shift[0],
                shift[1],
                shift[2],
                od_src_cpu[i],
                od_dst_cpu[i],
                i,
            )
        )

    to_sort.sort()

    sorted_indices = torch.tensor(
        [x[-1] for x in to_sort], device=positions.device, dtype=torch.long
    )

    offdiag_edge_src = offdiag_edge_src_unsorted[sorted_indices]
    offdiag_edge_dst = offdiag_edge_dst_unsorted[sorted_indices]
    offdiag_edge_shift = offdiag_edge_shift_unsorted[:, sorted_indices]

    # 7. Combine all edges and features
    edge_src = torch.cat([self_edge_src, offdiag_edge_src])
    edge_dst = torch.cat([self_edge_dst, offdiag_edge_dst])
    edge_shift = torch.cat([self_edge_shift, offdiag_edge_shift], dim=1)
    edge_index = torch.stack([edge_src, edge_dst]).to(positions.device)

    # 8. Create edge_type_idx
    self_edge_keys = [f"{atoms[i]}-{atoms[i]}" for i in range(num_atoms)]
    offdiag_edge_keys = [
        f"{atoms[i]}-{atoms[j]}" for i, j in zip(offdiag_edge_src, offdiag_edge_dst)
    ]
    all_edge_keys = self_edge_keys + offdiag_edge_keys

    if cfg.safety_checks:
        all_edge_keys_test = [
            f"{atoms[i]}-{atoms[j]}" for i, j in zip(edge_src, edge_dst)
        ]
        # compare with all_edge_keys
        for k1, k2 in zip(all_edge_keys, all_edge_keys_test):
            assert k1 == k2, "Edge keys do not match!"

    edge_type_idx = torch.tensor(
        [edge_type2idx[key] for key in all_edge_keys],
        dtype=torch.long,
        device=positions.device,
    )

    if box is not None:
        shift_float = edge_shift.T.to(dtype=positions.dtype)
        edge_disp = positions[edge_dst] - positions[edge_src] + shift_float @ box
    else:
        edge_disp = positions[edge_dst] - positions[edge_src]
    edge_lengths = torch.linalg.norm(edge_disp, dim=-1)

    # 9. Calculate geometric features for the final edge order
    radial_lengths = None
    radial_basis_end = None
    if edge_type_r0 is not None:
        radial_lengths = edge_lengths / edge_type_r0.index_select(0, edge_type_idx)
        radial_basis_end = 1.0

    edge_length_emb, edge_sh, _ = compute_edge_geometry_from_static_edges(
        positions=positions,
        box=box,
        edge_index=edge_index,
        edge_shift=edge_shift,
        sh_irreps=sh_irreps,
        cutoff_radius=cfg.cutoff_radius,
        n_radial=cfg.n_radial,
        radial_embedding_scale=cfg.radial_embedding_scale,
        radial_lengths=radial_lengths,
        radial_basis_end=radial_basis_end,
    )

    return (
        edge_index,
        edge_shift,
        edge_type_idx,
        edge_length_emb,
        edge_sh,
        len(self_edge_src),
        edge_lengths,
    )


def compute_graph_features_from_target_edges(
    *,
    positions: torch.Tensor,
    box: torch.Tensor | None,
    atoms: Tuple[str, ...],
    pair_edges: Dict[str, torch.Tensor],
    cfg: Config,
    sh_irreps: Irreps,
    edge_type2idx: Dict[str, int],
    edge_type_r0: torch.Tensor | None = None,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    int,
    torch.Tensor,
]:
    """Build graph features on the exact target support supplied by DeepH-E3."""
    edge_rows = [edges.T for edges in pair_edges.values()]
    if not edge_rows:
        raise ValueError("Cannot build a graph from empty target edge support.")
    edges = torch.cat(edge_rows, dim=0).to(device=positions.device)
    if torch.unique(edges, dim=0).shape[0] != edges.shape[0]:
        raise ValueError("Target support contains duplicate edges.")

    shifts = edges[:, :3]
    src = edges[:, 3]
    dst = edges[:, 4]
    if box is None:
        displacement = positions[dst] - positions[src]
    else:
        displacement = (
            positions[dst] - positions[src] + shifts.to(dtype=positions.dtype) @ box
        )
    lengths = torch.linalg.norm(displacement, dim=-1)
    is_diag = (src == dst) & (shifts == 0).all(dim=1)
    diag_indices = torch.nonzero(is_diag, as_tuple=False).flatten().tolist()
    offdiag_indices = torch.nonzero(~is_diag, as_tuple=False).flatten().tolist()
    diag_indices.sort(key=lambda idx: int(src[idx].item()))
    offdiag_indices.sort(
        key=lambda idx: (
            float(lengths[idx].item()),
            *(int(value) for value in edges[idx].tolist()),
        )
    )
    order = torch.tensor(
        diag_indices + offdiag_indices, dtype=torch.long, device=positions.device
    )
    edges = edges.index_select(0, order)
    edge_shift = edges[:, :3].T.contiguous()
    edge_index = edges[:, 3:5].T.contiguous()
    edge_pairs = edge_index.T.detach().cpu().tolist()
    edge_type_idx = torch.tensor(
        [edge_type2idx[f"{atoms[src]}-{atoms[dst]}"] for src, dst in edge_pairs],
        dtype=torch.long,
        device=positions.device,
    )

    radial_lengths = None
    radial_basis_end = None
    if edge_type_r0 is not None:
        raw_lengths = lengths.index_select(0, order)
        radial_lengths = raw_lengths / edge_type_r0.index_select(0, edge_type_idx)
        radial_basis_end = 1.0
    edge_length_emb, edge_sh, edge_lengths = compute_edge_geometry_from_static_edges(
        positions=positions,
        box=box,
        edge_index=edge_index,
        edge_shift=edge_shift,
        sh_irreps=sh_irreps,
        cutoff_radius=cfg.cutoff_radius,
        n_radial=cfg.n_radial,
        radial_embedding_scale=cfg.radial_embedding_scale,
        radial_lengths=radial_lengths,
        radial_basis_end=radial_basis_end,
    )
    if len(diag_indices) != len(atoms):
        raise ValueError(
            "Target graph must contain exactly one zero-shift self edge per atom; "
            f"found {len(diag_indices)} for {len(atoms)} atoms."
        )
    return (
        edge_index,
        edge_shift,
        edge_type_idx,
        edge_length_emb,
        edge_sh,
        len(diag_indices),
        edge_lengths,
    )


def complete_graph_metadata(
    inputs, mapper, *, species_order=None, separate_shifted_self=True
):
    """Supply common sparse-operator metadata for any declared graph support."""
    from collections import Counter
    from data.edge_alignment import build_prediction_edge_metadata

    if "pred_pair_edges_static" in inputs:
        return inputs
    species = tuple(species_order or mapper.orbital_cfg.elements())
    atoms = tuple(species[int(index)] for index in inputs["node_type_idx"])
    index = inputs["edge_index"]
    shifts = inputs.get("edge_shift", index.new_zeros((3, index.shape[1])))
    pairs = [f"{left}-{right}" for left in species for right in species]
    pair_to_index = {pair: i for i, pair in enumerate(pairs)}
    inputs.update(
        edge_shift=shifts,
        atoms=atoms,
        atoms_tuple=atoms,
        atom_counts=dict(Counter(atoms)),
    )
    inputs.update(
        build_prediction_edge_metadata(
            edge_index=index,
            edge_shift=shifts,
            edge_type_idx=inputs["edge_type_idx"],
            atoms=atoms,
            edge_types=pairs,
            edge_type2idx=pair_to_index,
            separate_shifted_self=separate_shifted_self,
        )
    )
    return inputs


def evaluate_graph_geometry(
    inputs,
    *,
    local_coordinates=False,
    num_l=5,
    require_frames=False,
    recompute_geometry=False,
    derive_frames=True,
    neighborhood_support="output",
):
    """Evaluate geometry on prepared supports without selecting any edges.

    Recalculation is required for coordinate derivatives. Explicit relations,
    including periodic images and neighborhood membership, remain unchanged.
    """
    from data.graph_topology import edge_displacements

    index, shifts = inputs["edge_index"], inputs["edge_shift"]
    if "edge_vectors" not in inputs or recompute_geometry:
        if "positions" not in inputs:
            if "edge_vectors" not in inputs:
                raise ValueError("Geometry needs positions or explicit edge vectors")
        else:
            inputs["edge_vectors"] = edge_displacements(inputs, index, shifts)
    vectors = inputs["edge_vectors"]
    lengths = vectors.norm(dim=-1)
    inputs.update(
        edge_lengths=lengths,
        edge_length=lengths,
        edge_keys=torch.cat((shifts, index)),
        selfloop_edge=(index[0] == index[1]) & (shifts == 0).all(0),
    )
    inputs.setdefault("edge_batch", inputs["node_batch"][index[0]])
    for name in inputs.get("_support_policies", {"message": {}}):
        relation = inputs[f"{name}_edge_index"]
        if "positions" in inputs:
            value = edge_displacements(inputs, relation, inputs[f"{name}_shift"])
        else:
            mapping = inputs[f"{name}_to_output"]
            if (mapping < 0).any():
                raise ValueError("Independent supports need positions for geometry")
            value = vectors[mapping]
        inputs[f"{name}_vectors"] = value
        inputs[f"{name}_lengths"] = value.norm(dim=-1)
    if require_frames or local_coordinates:
        from data.bond_geometry import (
            bond_frames,
            bond_neighborhood_relations,
            tesseral_harmonics,
        )

        selection = None
        if neighborhood_support != "output":
            selection = inputs[f"{neighborhood_support}_to_output"]
            if (selection < 0).any():
                raise ValueError(
                    "Bond neighborhoods must reference output-edge features"
                )
        if "bond_frames" not in inputs or (recompute_geometry and derive_frames):
            inputs["bond_frames"] = bond_frames(
                index, vectors, neighbor_selection=selection
            )
        if local_coordinates:
            if "sub_index" not in inputs:
                inputs.update(
                    bond_neighborhood_relations(index, neighbor_selection=selection)
                )
            local = -torch.einsum(
                "ei,eij->ej",
                vectors[inputs["sub_edge_idx"]],
                inputs["bond_frames"][inputs["sub_index"] // 2],
            )
            inputs["sub_edge_ang"] = tesseral_harmonics(local, num_l)
    return inputs


def add_graph_views(
    inputs,
    *,
    local_coordinates=False,
    num_l=5,
    require_frames=False,
    refresh=False,
    derive_frames=True,
):
    """Compatibility helper: prepare missing relations, then evaluate geometry.

    New workflows explicitly prepare topology and evaluate geometry separately.
    ``refresh`` only recomputes geometric values and never replaces supports.
    """
    from data.graph_topology import prepare_graph_topology

    prepare_graph_topology(inputs)
    return evaluate_graph_geometry(
        inputs,
        local_coordinates=local_coordinates,
        num_l=num_l,
        require_frames=require_frames,
        recompute_geometry=refresh,
        derive_frames=derive_frames,
    )
