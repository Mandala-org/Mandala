from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from ase.io import read as ase_read
from e3nn.o3 import Irreps

from core.block_irrep_mapper import BlockIrrepMapper
from core.orbital_irrep_config import OrbitalIrrepConfig
from data.edge_alignment import (
    build_prediction_edge_metadata,
    strict_reverse_edge_check,
)
from data.graph_features import compute_graph_features
from data.openmx_info_parser import parse_info_out
from net.common import Config, get_torch_dtype


def load_orbital_cfg_from_reference_info(
    info_path: str | Path,
    *,
    dtype: torch.dtype = torch.float32,
) -> OrbitalIrrepConfig:
    info = parse_info_out(info_path, dtype=dtype)
    return OrbitalIrrepConfig.from_dict(info.orbital_set)


def load_structure_from_cif(
    cif_path: str | Path,
    *,
    dtype: torch.dtype = torch.float32,
    device: str | torch.device = "cpu",
) -> tuple[tuple[str, ...], torch.Tensor, torch.Tensor]:
    atoms = ase_read(str(cif_path))
    if atoms.cell is None or atoms.cell.rank < 3:
        raise ValueError(
            f"CIF structure does not define a full periodic cell: {cif_path}"
        )
    atoms_tuple = tuple(str(sym) for sym in atoms.get_chemical_symbols())
    positions = torch.tensor(atoms.get_positions(), dtype=dtype, device=device)
    box = torch.tensor(atoms.cell.array, dtype=dtype, device=device)
    return atoms_tuple, positions, box


def build_model_input_from_structure(
    *,
    atoms: tuple[str, ...],
    positions: torch.Tensor,
    box: torch.Tensor | None,
    cfg: Config,
    mapper: BlockIrrepMapper,
) -> dict[str, Any]:
    dtype = get_torch_dtype(cfg.dtype)
    sh_irreps = Irreps.spherical_harmonics(cfg.l_max)

    (
        edge_index,
        edge_shift,
        edge_type_idx,
        edge_length_emb,
        edge_sh,
        num_self_edges,
    ) = compute_graph_features(
        positions=positions,
        box=box,
        atoms=atoms,
        cfg=cfg,
        sh_irreps=sh_irreps,
        edge_type2idx=mapper.edge_type2idx,
    )
    strict_reverse_edge_check(edge_index, edge_shift, edge_set_name="graph")

    elem2idx = {el: i for i, el in enumerate(mapper.orbital_cfg.elements())}
    try:
        node_type_idx = torch.tensor(
            [elem2idx[el] for el in atoms],
            dtype=torch.long,
            device=positions.device,
        )
    except KeyError as exc:
        raise ValueError(
            f"Structure contains element {exc.args[0]!r} that is missing from the orbital config."
        ) from exc

    num_species = len(mapper.orbital_cfg.elements())
    node_one_hot = F.one_hot(node_type_idx, num_classes=num_species).to(dtype=dtype)
    src_type = node_type_idx[edge_index[0]]
    dst_type = node_type_idx[edge_index[1]]
    edge_one_hot = F.one_hot(
        src_type * num_species + dst_type,
        num_classes=num_species * num_species,
    ).to(dtype=dtype)

    pred_metadata = build_prediction_edge_metadata(
        edge_index=edge_index,
        edge_shift=edge_shift,
        edge_type_idx=edge_type_idx,
        atoms=atoms,
        edge_types=mapper.edge_types,
        edge_type2idx=mapper.edge_type2idx,
        separate_shifted_self=bool(cfg.separate_shifted_self),
    )

    x: dict[str, Any] = {
        "node_type_idx": node_type_idx,
        "node_one_hot": node_one_hot,
        "positions": positions,
        "box": box,
        "atoms": atoms,
        "atoms_tuple": atoms,
        "atom_counts": dict(Counter(atoms)),
        "edge_index": edge_index,
        "edge_shift": edge_shift,
        "edge_type_idx": edge_type_idx,
        "edge_one_hot": edge_one_hot,
        "num_self_edges": num_self_edges,
        **pred_metadata,
    }
    if cfg.precompute_edge_features:
        x["edge_length_emb"] = edge_length_emb
        x["edge_sh"] = edge_sh
    return x
