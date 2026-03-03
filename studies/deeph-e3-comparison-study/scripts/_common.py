from __future__ import annotations

import io
import random
import importlib.util
from contextlib import redirect_stdout
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from ase import Atoms
from ase.neighborlist import neighbor_list
from e3nn.math import soft_one_hot_linspace
from e3nn.o3 import Irreps, spherical_harmonics

# Project imports
from core.block_irrep_mapper import BlockIrrepMapper
from data.block_matrix import IrrepsBlockData
from data.snapshot import Snapshot
from net.common import build_hidden_irreps

# Minimal study imports
_ROOT = Path(__file__).resolve().parents[3]
_MINIMAL_COMMON_PATH = _ROOT / "studies" / "minimal_overfit_study" / "common.py"
_SPEC = importlib.util.spec_from_file_location(
    "minimal_overfit_study_common_runtime", _MINIMAL_COMMON_PATH
)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(
        f"Could not load minimal study common module from {_MINIMAL_COMMON_PATH}"
    )
_MINIMAL_COMMON = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MINIMAL_COMMON)
MinimalNetwork = _MINIMAL_COMMON.MinimalNetwork
canonicalize_edge_order = _MINIMAL_COMMON.canonicalize_edge_order


@dataclass
class GraphInputs:
    node_type_idx: torch.Tensor
    edge_type_idx: torch.Tensor
    edge_index: torch.Tensor
    edge_shift: torch.Tensor
    edge_length_emb: torch.Tensor
    edge_sh_current: torch.Tensor
    edge_sh_aligned: torch.Tensor
    edge_dist: torch.Tensor


def set_seed(seed: int = 0) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def canonicalize_block_matrix_edges(
    block_matrix, positions: torch.Tensor, box: torch.Tensor | None
):
    """Apply the same per-key canonical ordering used for graph edges."""
    order_dict = {}
    for key, edges_5d in block_matrix.pair_edges.items():
        edge_shift_local = edges_5d[:3]
        edge_index_local = edges_5d[3:]
        _, _, perm_local = canonicalize_edge_order(
            edge_index=edge_index_local,
            edge_shift=edge_shift_local,
            positions=positions,
            box=box,
        )
        order_dict[key] = perm_local
    return block_matrix.reorder_edges(order_dict)


def load_water_snapshot(
    *,
    data_path: Path,
    info_path: Path,
    device: torch.device,
    convention: str = "e3nn",
    cutoff_radius: float | None = None,
):
    snapshot = Snapshot.from_openmx(
        matrix_path=data_path,
        info_path=info_path,
        convention=convention,
        symmetrize_density=True,
        cutoff_radius=cutoff_radius,
        dtype=torch.float32,
    )

    positions = snapshot.positions.to(device)
    box = snapshot.box.to(device) if snapshot.box is not None else None
    atoms_list = list(snapshot.hamiltonian.atoms)

    hamiltonian = canonicalize_block_matrix_edges(
        snapshot.hamiltonian.to(device), positions, box
    )
    overlap = canonicalize_block_matrix_edges(
        snapshot.overlap.to(device), positions, box
    )
    density = canonicalize_block_matrix_edges(
        snapshot.density.to(device), positions, box
    )

    # Match minimal training target policy.
    hamiltonian = (hamiltonian + hamiltonian.transpose()) * 0.5

    mapper = BlockIrrepMapper(
        snapshot.hamiltonian.orbital_cfg, device=device, dtype=torch.float32
    )

    return snapshot, atoms_list, positions, box, hamiltonian, overlap, density, mapper


def build_minimal_graph_inputs(
    *,
    atoms_list: list[str],
    positions: torch.Tensor,
    box: torch.Tensor | None,
    mapper: BlockIrrepMapper,
    cutoff_radius: float,
    n_radial: int,
    l_max: int,
    device: torch.device,
) -> GraphInputs:
    num_atoms = len(atoms_list)

    ase_atoms = Atoms(
        symbols=atoms_list,
        positions=positions.detach().cpu().numpy(),
        cell=box.detach().cpu().numpy() if box is not None else None,
        pbc=box is not None,
    )

    src, dst, offsets = neighbor_list(
        "ijS", ase_atoms, cutoff_radius, self_interaction=False
    )

    self_src = torch.arange(num_atoms, dtype=torch.long, device=device)
    self_dst = torch.arange(num_atoms, dtype=torch.long, device=device)
    self_offsets = torch.zeros((num_atoms, 3), dtype=torch.long, device=device)

    all_src = torch.cat([self_src, torch.from_numpy(src).to(device)])
    all_dst = torch.cat([self_dst, torch.from_numpy(dst).to(device)])
    all_offsets = torch.cat([self_offsets, torch.from_numpy(offsets).to(device).long()])

    edge_index = torch.stack([all_src, all_dst], dim=0)
    edge_shift = all_offsets.T

    edge_index, edge_shift, _ = canonicalize_edge_order(
        edge_index=edge_index,
        edge_shift=edge_shift,
        positions=positions,
        box=box,
    )

    if box is not None:
        edge_vec = (
            positions[edge_index[1]]
            - positions[edge_index[0]]
            + edge_shift.T.float() @ box
        )
    else:
        edge_vec = positions[edge_index[1]] - positions[edge_index[0]]

    edge_dist = torch.linalg.norm(edge_vec, dim=1)
    edge_vec_norm = edge_vec.clone()
    non_zero_mask = edge_dist > 1e-6
    edge_vec_norm[non_zero_mask] = edge_vec[non_zero_mask] / edge_dist[
        non_zero_mask
    ].unsqueeze(-1)

    sh_irreps = Irreps.spherical_harmonics(l_max)
    edge_sh_current = spherical_harmonics(sh_irreps, edge_vec_norm, normalize=False)
    edge_sh_aligned = spherical_harmonics(
        sh_irreps,
        edge_vec,
        normalize=True,
        normalization="component",
    )

    edge_length_emb = soft_one_hot_linspace(
        edge_dist,
        start=0.0,
        end=cutoff_radius,
        number=n_radial,
        basis="gaussian",
        cutoff=False,
    )

    # Match minimal script scaling behavior.
    edge_length_emb = edge_length_emb * (n_radial**0.5)

    element_to_idx = {
        elem: idx for idx, elem in enumerate(mapper.orbital_cfg.elements())
    }
    node_type_idx = torch.tensor([element_to_idx[a] for a in atoms_list], device=device)

    edge_type_strs = [
        f"{atoms_list[edge_index[0, i].item()]}-{atoms_list[edge_index[1, i].item()]}"
        for i in range(edge_index.shape[1])
    ]
    edge_type_idx = torch.tensor(
        [mapper.edge_type2idx[k] for k in edge_type_strs], device=device
    )

    return GraphInputs(
        node_type_idx=node_type_idx,
        edge_type_idx=edge_type_idx,
        edge_index=edge_index,
        edge_shift=edge_shift,
        edge_length_emb=edge_length_emb,
        edge_sh_current=edge_sh_current,
        edge_sh_aligned=edge_sh_aligned,
        edge_dist=edge_dist,
    )


def make_network(
    *,
    mapper: BlockIrrepMapper,
    n_radial: int,
    l_max: int,
    hidden_dim: int,
    num_layers: int,
    device: torch.device,
):
    hidden_irreps = build_hidden_irreps(
        l_max=l_max, base_dim=hidden_dim, use_odd_features=True
    )
    sh_irreps = Irreps.spherical_harmonics(l_max)
    num_elements = len(mapper.orbital_cfg.elements())
    num_edge_types = num_elements**2

    # Suppress very verbose construction prints.
    with redirect_stdout(io.StringIO()):
        network = MinimalNetwork(
            num_elements=num_elements,
            n_radial=n_radial,
            num_edge_types=num_edge_types,
            hidden_irreps=hidden_irreps,
            sh_irreps=sh_irreps,
            num_layers=num_layers,
            mapper=mapper,
            magnitude_factorization=False,
            head_mlp_for_scalars=False,
            head_use_tensor_square=False,
            separate_shifted_self=False,
        ).to(device)

    return network


def forward_to_blocks(
    *,
    network: MinimalNetwork,
    graph: GraphInputs,
    atoms_list: list[str],
    mapper: BlockIrrepMapper,
    basis: str,
    orbital_cfg,
    use_aligned_sh: bool,
) -> Tuple[IrrepsBlockData, object]:
    num_atoms = len(atoms_list)
    batch_node = torch.zeros(
        num_atoms, dtype=torch.long, device=graph.node_type_idx.device
    )
    batch_edge = torch.zeros(
        graph.edge_index.shape[1], dtype=torch.long, device=graph.node_type_idx.device
    )

    edge_sh = graph.edge_sh_aligned if use_aligned_sh else graph.edge_sh_current

    with redirect_stdout(io.StringIO()):
        pred_raw = network(
            graph.node_type_idx,
            graph.edge_type_idx,
            graph.edge_index,
            graph.edge_shift,
            graph.edge_length_emb,
            edge_sh,
            batch_node,
            batch_edge,
            log_to_wandb=False,
        )

    pair_vec = {}
    pair_edges = {}
    lookup = {}

    for key, payload in pred_raw.items():
        pair_vec[key] = payload["vectors"]
        pair_edges[key] = payload["edges"]
        for idx, edge_5d in enumerate(payload["edges"].t()):
            sx, sy, sz, i, j = edge_5d.tolist()
            lookup[(int(sx), int(sy), int(sz), int(i), int(j))] = (key, idx)

    pred_irreps = IrrepsBlockData(
        atoms=tuple(atoms_list),
        atom_counts={k: atoms_list.count(k) for k in set(atoms_list)},
        pair_vectors=pair_vec,
        pair_edges=pair_edges,
        lookup=lookup,
        orbital_cfg=orbital_cfg,
        basis=basis,
    )

    pred_blocks = pred_irreps.to_blocks(mapper)
    return pred_irreps, pred_blocks


def loss_per_key_sum_mse(pred_matrix, target_matrix) -> torch.Tensor:
    loss = torch.tensor(0.0, device=next(iter(pred_matrix.pair_blocks.values())).device)
    for key in target_matrix.pair_blocks.keys():
        if key not in pred_matrix.pair_blocks:
            continue
        pred = pred_matrix.pair_blocks[key]
        targ = target_matrix.pair_blocks[key]
        min_n = min(pred.shape[0], targ.shape[0])
        if min_n == 0:
            continue
        loss = loss + F.mse_loss(pred[:min_n], targ[:min_n])
    return loss


def loss_global_element_mse(pred_matrix, target_matrix) -> torch.Tensor:
    sum_sq = torch.tensor(
        0.0, device=next(iter(pred_matrix.pair_blocks.values())).device
    )
    n_elem = 0
    for key in target_matrix.pair_blocks.keys():
        if key not in pred_matrix.pair_blocks:
            continue
        pred = pred_matrix.pair_blocks[key]
        targ = target_matrix.pair_blocks[key]
        min_n = min(pred.shape[0], targ.shape[0])
        if min_n == 0:
            continue
        diff = pred[:min_n] - targ[:min_n]
        sum_sq = sum_sq + (diff**2).sum()
        n_elem += diff.numel()
    if n_elem == 0:
        return torch.tensor(0.0, device=sum_sq.device)
    return sum_sq / n_elem


def flatten_gradients(model: torch.nn.Module) -> torch.Tensor:
    chunks = []
    for p in model.parameters():
        if p.grad is not None:
            chunks.append(p.grad.detach().reshape(-1))
    if not chunks:
        return torch.zeros(0)
    return torch.cat(chunks)


def edge_tuples(edge_index: torch.Tensor, edge_shift: torch.Tensor):
    out = []
    for e in range(edge_index.shape[1]):
        out.append(
            (
                int(edge_shift[0, e].item()),
                int(edge_shift[1, e].item()),
                int(edge_shift[2, e].item()),
                int(edge_index[0, e].item()),
                int(edge_index[1, e].item()),
            )
        )
    return out


def metric_stats(a: torch.Tensor, b: torch.Tensor) -> Dict[str, float]:
    if a.shape != b.shape:
        raise ValueError(f"Shape mismatch: {a.shape} vs {b.shape}")
    diff = (a - b).abs()
    rmse = torch.sqrt(torch.mean((a - b) ** 2))
    denom = torch.clamp(a.abs(), min=1e-12)
    rel = diff / denom
    return {
        "max_abs": float(diff.max().item()),
        "mean_abs": float(diff.mean().item()),
        "rmse": float(rmse.item()),
        "max_rel": float(rel.max().item()),
        "mean_rel": float(rel.mean().item()),
    }
