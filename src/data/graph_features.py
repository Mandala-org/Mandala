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


def compute_graph_features(
    positions: torch.Tensor,
    box: torch.Tensor | None,
    atoms: Tuple[str, ...],
    cfg: Config,
    sh_irreps: Irreps,
    edge_type2idx: Dict[str, int],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """
    Returns
    -------
    edge_index : (2, E_total) long
    edge_shift : (3, E_total) long
    edge_type_idx : (E_total,) long
    edge_length_emb : (E_total, cfg.n_radial) float
    edge_sh : (E_total, sh_dim) float
    num_self_edges : int
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
        # check if displacements lead to correct destinations
        positions_dst_reconstructed = (
            positions[offdiag_edge_src_unsorted] + offdiag_disp_unsorted
        )
        # Due to periodic boundaries, we need to map positions back into the unit cell
        if box is not None:
            positions_dst_reconstructed = (
                positions_dst_reconstructed + 1e-4
            )  # avoid edge cases
            inv_box = torch.linalg.pinv(box.to(positions.device))
            frac_coords = positions_dst_reconstructed @ inv_box
            frac_coords = frac_coords - torch.floor(frac_coords)
            positions_dst_reconstructed = frac_coords @ box.to(positions.device) - 1e-4
        diffs = positions_dst_reconstructed - positions[offdiag_edge_dst_unsorted]
        assert torch.all(
            torch.linalg.norm(diffs, dim=-1) < 1e-4
        ), "Displacement vectors do not lead to correct destination positions."

    self_disp = torch.zeros((num_atoms, 3), device=positions.device)

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
    offdiag_disp = offdiag_disp_unsorted[sorted_indices]
    offdiag_lengths = offdiag_lengths_unsorted[sorted_indices]

    # 7. Combine all edges and features
    edge_src = torch.cat([self_edge_src, offdiag_edge_src])
    edge_dst = torch.cat([self_edge_dst, offdiag_edge_dst])
    edge_shift = torch.cat([self_edge_shift, offdiag_edge_shift], dim=1)
    edge_index = torch.stack([edge_src, edge_dst]).to(positions.device)
    edge_disp = torch.cat([self_disp, offdiag_disp], dim=0)
    edge_lengths = torch.cat(
        [torch.zeros(num_atoms, device=positions.device), offdiag_lengths]
    )

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

    # 9. Calculate geometric features for the final edge order
    edge_sh = spherical_harmonics(
        sh_irreps, edge_disp, normalize=True, normalization="component"
    )
    is_zero_shift_self_edge = (edge_src == edge_dst) & (edge_shift == 0).all(dim=0)
    sh_non_scalar_slices = _non_scalar_sh_slices(sh_irreps)
    if is_zero_shift_self_edge.any() and sh_non_scalar_slices:
        edge_sh = edge_sh.clone()
        for slc in sh_non_scalar_slices:
            edge_sh[is_zero_shift_self_edge, slc] = 0.0
    edge_length_emb = soft_one_hot_linspace(
        edge_lengths,
        start=0.0,
        end=cfg.cutoff_radius,
        number=cfg.n_radial,
        basis="gaussian",
        cutoff=False,
    )
    if cfg.radial_embedding_scale == "sqrt_n_radial":
        edge_length_emb = edge_length_emb * (cfg.n_radial**0.5)

    return (
        edge_index,
        edge_shift,
        edge_type_idx,
        edge_length_emb,
        edge_sh,
        len(self_edge_src),
    )
