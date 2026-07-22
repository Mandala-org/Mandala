from __future__ import annotations

from pathlib import Path

import torch

from core.block_irrep_mapper import BlockIrrepMapper
from data.structure_inference import (
    build_model_input_from_structure,
    load_orbital_cfg_from_reference_info,
    load_structure_from_cif,
)
from net.common import Config


def test_load_structure_from_cif_and_build_model_input():
    repo_root = Path(__file__).resolve().parents[3]
    cif_path = repo_root / "data" / "structure" / "Silicon.cif"
    info_path = repo_root / "data" / "big" / "silicon" / "300K" / "Si.out"

    cfg = Config(
        cutoff_radius=7.5,
        l_max=4,
        n_radial=64,
        precompute_edge_features=True,
        separate_shifted_self=True,
        dtype=torch.float32,
    )
    orbital_cfg = load_orbital_cfg_from_reference_info(info_path, dtype=cfg.dtype)
    mapper = BlockIrrepMapper(
        orbital_cfg,
        diagonal=False,
        device="cpu",
        dtype=cfg.dtype,
    )

    atoms, positions, box = load_structure_from_cif(cif_path, dtype=cfg.dtype)
    x = build_model_input_from_structure(
        atoms=atoms,
        positions=positions,
        box=box,
        cfg=cfg,
        mapper=mapper,
    )

    assert atoms == ("Si",) * 8
    assert positions.shape == (8, 3)
    assert box.shape == (3, 3)
    assert x["edge_index"].shape[0] == 2
    assert x["edge_shift"].shape[0] == 3
    assert "edge_length_emb" in x
    assert "edge_sh" in x
    assert "pred_pair_edges_static" in x
    assert "pred_trace_alignment" in x
    assert "edge_partitions" in x
