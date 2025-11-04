import torch
from e3nn.o3 import Irreps, spherical_harmonics
from e3nn.math import soft_one_hot_linspace
from net.common import Config
from typing import Tuple, Dict

from ase import Atoms
from ase.neighborlist import neighbor_list


def _minimal_disp(
    pos: torch.Tensor,
    edges: torch.Tensor,
    box: torch.Tensor | None,
    inv_box: torch.Tensor | None = None,
) -> torch.Tensor:
    if box is None:
        return pos[edges[1]] - pos[edges[0]]
    if inv_box is None:
        inv_box = torch.inverse(box)
    delta = pos[edges[1]] - pos[edges[0]]  # cart
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
) -> Tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int, int, torch.Tensor
]:
    """
    Returns
    -------
    edge_index : (2, E_total) long
    edge_type_idx : (E_total,) long
    edge_length_emb : (E_total, cfg.n_radial) float
    edge_sh : (E_total, sh_dim) float
    index_gnn_cutoff : int
    is_closest_edge: (E_total,) bool
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
        "ijS", ase_atoms, cfg.cutoff_matrix, self_interaction=False
    )

    # 3. Handle self-edges explicitly
    num_atoms = len(atoms)
    self_edge_src = torch.arange(num_atoms, dtype=torch.long, device=positions.device)
    self_edge_dst = torch.arange(num_atoms, dtype=torch.long, device=positions.device)
    self_edge_offsets = torch.zeros(
        (num_atoms, 3), dtype=torch.long, device=positions.device
    )

    # 4. Combine self-edges and off-diagonal edges
    offdiag_edge_src_unsorted = torch.from_numpy(src).to(positions.device)
    offdiag_edge_dst_unsorted = torch.from_numpy(dst).to(positions.device)
    offdiag_edge_offsets = torch.from_numpy(offsets).to(positions.device).to(torch.long)

    # 5. Calculate displacement vectors using the offsets
    # disp = pos[j] + S @ box - pos[i]
    offdiag_disp_unsorted = (
        positions[offdiag_edge_dst_unsorted]
        + torch.matmul(
            offdiag_edge_offsets.to(positions.device), box.to(positions.device)
        )
        - positions[offdiag_edge_src_unsorted]
    )

    if cfg.pedantic:
        # check if displacements lead to correct destinations
        positions_dst_reconstructed = (
            positions[offdiag_edge_src_unsorted] + offdiag_disp_unsorted
        )
        # Due to periodic boundaries, we need to map positions back into the unit cell
        if box is not None:
            inv_box = torch.inverse(box.to(positions.device))
            frac_coords = positions_dst_reconstructed @ inv_box
            frac_coords = frac_coords - torch.floor(frac_coords)
            positions_dst_reconstructed = frac_coords @ box.to(positions.device)
        diffs = positions_dst_reconstructed - positions[offdiag_edge_dst_unsorted]
        assert torch.all(
            torch.linalg.norm(diffs, dim=-1) < 1e-4
        ), "Displacement vectors do not lead to correct destination positions."

    self_disp = torch.zeros((num_atoms, 3), device=positions.device)

    # 6. Calculate lengths and sort off-diagonal edges

    offdiag_lengths_unsorted = torch.linalg.norm(offdiag_disp_unsorted, dim=-1)

    sorted_indices = torch.argsort(offdiag_lengths_unsorted)

    offdiag_edge_src = offdiag_edge_src_unsorted[sorted_indices]
    offdiag_edge_dst = offdiag_edge_dst_unsorted[sorted_indices]
    offdiag_edge_offsets = offdiag_edge_offsets[sorted_indices]
    offdiag_disp = offdiag_disp_unsorted[sorted_indices]
    offdiag_lengths = offdiag_lengths_unsorted[sorted_indices]

    # 7. Combine all edges and features
    edge_src = torch.cat([self_edge_src, offdiag_edge_src])
    edge_dst = torch.cat([self_edge_dst, offdiag_edge_dst])
    edge_offsets = torch.cat([self_edge_offsets, offdiag_edge_offsets])
    edge_index = torch.cat([edge_src, edge_dst]).to(positions.device)
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

    if cfg.pedantic:
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
    edge_length_emb = soft_one_hot_linspace(
        edge_lengths,
        start=0.0,
        end=cfg.cutoff_matrix,
        number=cfg.n_radial,
        basis="gaussian",
        cutoff=False,
    )

    # 10. Determine the GNN cfg.cutoff_gnn index
    index_gnn_cutoff = torch.sum(edge_lengths <= cfg.cutoff_gnn).item()

    return (
        edge_index,
        edge_offsets,
        edge_type_idx,
        edge_length_emb,
        edge_sh,
        index_gnn_cutoff,
        len(self_edge_src),
    )
