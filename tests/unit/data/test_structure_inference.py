from __future__ import annotations

from pathlib import Path

import pytest
import torch

from core.block_irrep_mapper import BlockIrrepMapper
from data.structure_inference import (
    build_model_input_from_structure,
    load_orbital_cfg_from_reference_info,
    load_structure_from_cif,
)
from net.common import Config


@pytest.mark.large_data
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


def test_structure_inputs_use_frozen_species_order():
    from core.orbital_irrep_config import OrbitalIrrepConfig
    from core.block_irrep_mapper import BlockIrrepMapper
    from net.common import Config
    from data.structure_inference import build_model_input_from_structure

    mapper = BlockIrrepMapper(
        OrbitalIrrepConfig.from_dict({"C": ["1x0e"], "H": ["1x0e"]})
    )
    cfg = Config(species={"order": ["H", "C"]}, cutoff_radius=2.0, l_max=0, n_radial=2)
    inputs = build_model_input_from_structure(
        atoms=("C", "H"),
        positions=torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        box=None,
        cfg=cfg,
        mapper=mapper,
    )
    assert inputs["node_type_idx"].tolist() == [1, 0]
    for index, (left, right) in enumerate(inputs["edge_index"].T):
        assert (
            inputs["edge_type_idx"][index]
            == inputs["node_type_idx"][left] * 2 + inputs["node_type_idx"][right]
        )


def test_structure_inputs_accept_paper_sio2_envelope_without_r0():
    from core.orbital_irrep_config import OrbitalIrrepConfig

    envelope = Path(__file__).resolve().parents[3] / "scripts/reproduce_examples/sio2_envelope.json"
    mapper = BlockIrrepMapper(
        OrbitalIrrepConfig.from_dict({"O": ["1x0e"], "Si": ["1x0e"]})
    )
    cfg = Config(cutoff_radius=3.0, l_max=0, n_radial=2,
                 hamiltonian_envelope_mode="multiply_prediction",
                 hamiltonian_envelope_path=str(envelope),
                 pair_distance_normalization="off")
    inputs = build_model_input_from_structure(
        atoms=("O", "Si"), positions=torch.tensor([[0., 0., 0.], [1.6, 0., 0.]]),
        box=None, cfg=cfg, mapper=mapper,
    )
    assert inputs["edge_envelope"].shape == inputs["edge_length"].shape
    assert torch.isfinite(inputs["edge_envelope"]).all()
    assert "edge_r0" not in inputs


@pytest.mark.parametrize("normalization", ["off", "pair_r0"])
def test_structure_inputs_keep_r0_for_supported_envelopes(tmp_path, normalization):
    import json
    from core.orbital_irrep_config import OrbitalIrrepConfig

    envelope = tmp_path / "envelope.json"
    envelope.write_text(json.dumps({"family": "slater_soft_cutoff", "pairs": {
        "H-H": {"theta": [0., 0., 0., 2., 0.]}
    }}))
    mapper = BlockIrrepMapper(OrbitalIrrepConfig.from_dict({"H": ["1x0e"]}))
    cfg = Config(cutoff_radius=3.0, l_max=0, n_radial=2,
                 hamiltonian_envelope_path=str(envelope),
                 pair_distance_normalization=normalization)
    inputs = build_model_input_from_structure(
        atoms=("H", "H"), positions=torch.tensor([[0., 0., 0.], [1., 0., 0.]]),
        box=None, cfg=cfg, mapper=mapper,
    )
    torch.testing.assert_close(inputs["edge_r0"], torch.full_like(inputs["edge_length"], 2.))
