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
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int, int]:
    """
    Returns
    -------
    edge_index : (2, E_total) long
    edge_type_idx : (E_total,) long
    edge_length_emb : (E_total, cfg.n_radial) float
    edge_sh : (E_total, sh_dim) float
    index_gnn_cutoff : int
    """

    # 1. Create ase.Atoms object
    ase_atoms = Atoms(
        symbols=atoms,
        positions=positions.detach().cpu().numpy(),
        cell=box.detach().cpu().numpy() if box is not None else None,
        pbc=box is not None,
    )

    # 2. Use ase.neighborlist to get edges and offsets
    # 'i' is the source atom index, 'j' is the destination atom index,
    # 'S' is the offset vector in lattice coordinates.
    src, dst, offsets = neighbor_list(
        "ijS", ase_atoms, cfg.cutoff_matrix, self_interaction=False
    )

    # 3. Handle self-edges explicitly
    num_atoms = len(atoms)
    self_edge_src = torch.arange(num_atoms, dtype=torch.long)
    self_edge_dst = torch.arange(num_atoms, dtype=torch.long)

    # 4. Combine self-edges and off-diagonal edges
    offdiag_edge_src = torch.from_numpy(src)
    offdiag_edge_dst = torch.from_numpy(dst)
    offdiag_edge_offsets = torch.from_numpy(offsets).to(torch.float32)

    # 5. Calculate displacement vectors using the offsets
    # disp = pos[j] + S @ box - pos[i]
    offdiag_disp = (
        positions[offdiag_edge_dst]
        + torch.matmul(
            offdiag_edge_offsets.to(positions.device), box.to(positions.device)
        )
        - positions[offdiag_edge_src]
    )
    self_disp = torch.zeros((num_atoms, 3), device=positions.device)

    # 6. Calculate lengths and sort off-diagonal edges
    offdiag_lengths = torch.linalg.norm(offdiag_disp, dim=-1)
    sorted_indices = torch.argsort(offdiag_lengths)

    offdiag_edge_src = offdiag_edge_src[sorted_indices]
    offdiag_edge_dst = offdiag_edge_dst[sorted_indices]
    offdiag_disp = offdiag_disp[sorted_indices]
    offdiag_lengths = offdiag_lengths[sorted_indices]

    # 7. Combine all edges and features
    edge_src = torch.cat([self_edge_src, offdiag_edge_src])
    edge_dst = torch.cat([self_edge_dst, offdiag_edge_dst])
    edge_index = torch.stack([edge_src, edge_dst]).to(positions.device)

    disp = torch.cat([self_disp, offdiag_disp])
    lengths = torch.linalg.norm(disp, dim=-1)

    # 8. Create edge_type_idx
    self_edge_keys = [f"{atoms[i]}-{atoms[i]}" for i in range(num_atoms)]
    offdiag_edge_keys = [
        f"{atoms[i]}-{atoms[j]}" for i, j in zip(offdiag_edge_src, offdiag_edge_dst)
    ]
    all_edge_keys = self_edge_keys + offdiag_edge_keys
    edge_type_idx = torch.tensor(
        [edge_type2idx[key] for key in all_edge_keys],
        dtype=torch.long,
        device=positions.device,
    )

    # 9. Calculate geometric features for the final edge order
    edge_sh = spherical_harmonics(
        sh_irreps, disp, normalize=True, normalization="component"
    )
    edge_length_emb = soft_one_hot_linspace(
        lengths,
        start=0.0,
        end=cfg.cutoff_matrix,
        number=cfg.n_radial,
        basis="gaussian",
        cutoff=False,
    )

    # 10. Determine the GNN cfg.cutoff_gnn index
    index_gnn_cutoff = torch.sum(lengths <= cfg.cutoff_gnn).item()

    return (
        edge_index,
        edge_type_idx,
        edge_length_emb,
        edge_sh,
        index_gnn_cutoff,
        len(self_edge_src),
    )
