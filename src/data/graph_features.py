import torch
from e3nn.o3 import Irreps, spherical_harmonics
from e3nn.math import soft_one_hot_linspace
from net.common import Config
from typing import Tuple, Dict


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

    # 1. Collect all edges and their properties
    edges = []
    for i, el_i in enumerate(atoms):
        for j, el_j in enumerate(atoms):

            key = f"{el_i}-{el_j}"
            edges.append(
                {
                    "src": i,
                    "dst": j,
                    "key": key,
                }
            )

    # 2. Separate self-edges and off-diagonal edges
    self_edges = sorted(
        [e for e in edges if e["src"] == e["dst"]], key=lambda e: e["src"]
    )
    offdiag_edges = [e for e in edges if e["src"] != e["dst"]]

    # 3. Calculate lengths for off-diagonal edges and sort them

    offdiag_edge_index = torch.tensor(
        [[e["src"] for e in offdiag_edges], [e["dst"] for e in offdiag_edges]],
        dtype=torch.long,
        device=positions.device,  # Use positions device
    )
    disp = _minimal_disp(
        positions,
        offdiag_edge_index,
        box,
    )
    lengths = torch.linalg.norm(disp, dim=-1)
    sorted_indices = torch.argsort(lengths)
    offdiag_edges = [offdiag_edges[i] for i in sorted_indices]
    lengths = lengths[sorted_indices]
    offdiag_edges = [
        e for i, e in enumerate(offdiag_edges) if lengths[i] <= cfg.cutoff_matrix
    ]
    lengths = lengths[lengths <= cfg.cutoff_matrix]

    # 4. Combine edges in the specified order
    all_edges = self_edges + offdiag_edges
    edge_index = torch.tensor(
        [[e["src"] for e in all_edges], [e["dst"] for e in all_edges]],
        dtype=torch.long,
        device=positions.device,
    )
    edge_type_idx = torch.tensor(
        [edge_type2idx[e["key"]] for e in all_edges],
        dtype=torch.long,
        device=positions.device,
    )

    # 5. Calculate geometric features for the final edge order
    disp = _minimal_disp(positions, edge_index, box)
    lengths = torch.linalg.norm(disp, dim=-1)
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

    # 6. Determine the GNN cfg.cutoff_gnn index
    index_gnn_cutoff = torch.sum(lengths <= cfg.cutoff_gnn).item()

    return (
        edge_index,
        edge_type_idx,
        edge_length_emb,
        edge_sh,
        index_gnn_cutoff,
        len(self_edges),
    )
