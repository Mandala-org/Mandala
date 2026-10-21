"""Native block export and unlabeled geometric inference through E3GNN."""

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from core.block_irrep_mapper import BlockIrrepMapper
from core.orbital_irrep_config import OrbitalIrrepConfig
from data.operator_dataset import load_operator_sample, reconstruct_operator_blocks
from net.common import Config
from net.configuration import resolve_configuration
from net.e3gnn import E3GNN


ROOT = Path(__file__).resolve().parents[3]


def geometry(directory):
    np.savetxt(directory / "element.dat", [1, 1, 1], fmt="%d")
    np.savetxt(
        directory / "site_positions.dat",
        np.array([[0.0, 0.0, 0.0], [1.0, 0.2, 0.0], [0.1, 1.0, 0.3]]).T,
    )
    np.savetxt(directory / "lat.dat", np.eye(3) * 8)
    (directory / "orbital_types.dat").write_text("0\n0\n0\n")
    (directory / "info.json").write_text(json.dumps({"isspinful": False}))


@pytest.mark.parametrize("kind", ["deeph", "deeph_e3"])
def test_geometry_only_native_export_moves_input_dtype_and_retains_directed_support(
    tmp_path, kind
):
    geometry(tmp_path)
    parameters = (
        {
            "node_width": 3,
            "edge_width": 7,
            "post_edge_width": 3,
            "num_l": 2,
            "angular_dim": 4,
        }
        if kind == "deeph"
        else {
            "embedding_width": 3,
            "radial_width": 3,
            "node_irreps": "3x0e+2x1o",
            "edge_irreps": "3x0e+2x1o",
            "angular_irreps": "1x0e+1x1o",
        }
    )
    config, codec = resolve_configuration(
        ROOT / f"configs/reference/{kind}.yaml",
        species_shells={"H": [0]},
        overrides={"parameters": parameters},
    )
    mapper = BlockIrrepMapper(OrbitalIrrepConfig.from_dict({"H": ["1x0e"]}))
    model = E3GNN(mapper, Config(pipeline=config["pipeline"], verbosity=0))
    sample = load_operator_sample(
        tmp_path,
        codec,
        species=["H"],
        matrix_files={},
        graph_source="radius",
        graph_radius=2.0,
        local_coordinates=kind == "deeph",
        num_l=2,
        target_frame=config["data"]["target_frame"],
        dtype=torch.float64,
    )
    assert sample["targets"] == {}
    model.train()
    result = model.predict_operator_blocks(sample, codec, return_masks=True)
    assert model.training
    blocks, coverage = result["hamiltonian"]
    assert set(blocks) == set(sample["keys"])
    assert len(blocks) == 9
    assert all(block.dtype == torch.float32 for block in blocks.values())
    assert all(mask.all() for mask in coverage.values())
    prepared = dict(
        sample,
        inputs={
            key: (
                value.float()
                if isinstance(value, torch.Tensor) and value.is_floating_point()
                else value
            )
            for key, value in sample["inputs"].items()
        },
    )
    with torch.no_grad():
        direct = model(prepared["inputs"])["hamiltonian"]
        expected = reconstruct_operator_blocks(
            prepared, codec, direct, output_frame="global"
        )
    for key in blocks:
        torch.testing.assert_close(blocks[key], expected[key])
    wrong_codec = type(codec).from_species_shells({"H": [0, 1]})
    with pytest.raises(ValueError, match="orbital layout"):
        model.predict_operator_blocks(sample, wrong_codec)


def test_radius_graph_rejects_implicit_or_invalid_cutoff(tmp_path):
    geometry(tmp_path)
    _, codec = resolve_configuration(
        ROOT / "configs/reference/deeph_e3.yaml", species_shells={"H": [0]}
    )
    for radius in (None, 0, -1, float("nan")):
        with pytest.raises(ValueError, match="finite positive"):
            load_operator_sample(
                tmp_path,
                codec,
                species=["H"],
                matrix_files={},
                graph_source="radius",
                graph_radius=radius,
            )


def test_export_rejects_reassigned_same_angular_momentum_shells(tmp_path):
    geometry(tmp_path)
    (tmp_path / "orbital_types.dat").write_text("0 0\n0 0\n0 0\n")
    config, codec = resolve_configuration(
        ROOT / "configs/reference/deeph_e3.yaml",
        species_shells={"H": [0, 0]},
        overrides={
            "parameters": {
                "embedding_width": 3,
                "radial_width": 3,
                "node_irreps": "3x0e",
                "edge_irreps": "3x0e",
                "angular_irreps": "1x0e",
            }
        },
    )
    mapper = BlockIrrepMapper(OrbitalIrrepConfig.from_dict({"H": ["1x0e", "1x0e"]}))
    model = E3GNN(mapper, Config(pipeline=config["pipeline"], verbosity=0))
    sample = load_operator_sample(
        tmp_path,
        codec,
        species=["H"],
        matrix_files={},
        graph_source="radius",
        graph_radius=2.0,
    )
    wrong = type(codec).from_species_shells(
        {"H": [0, 0]}, channel_assignments=list(reversed(codec.channel_assignments))
    )
    assert codec.shell_pairs == wrong.shell_pairs
    assert codec.packed_dim == wrong.packed_dim
    with pytest.raises(ValueError, match="shell-channel assignment"):
        model.predict_operator_blocks(sample, wrong)
    with pytest.raises(ValueError, match="Frozen pipeline operator layout"):
        resolve_configuration(
            config,
            species_shells={"H": [0, 0]},
            overrides={
                "codec": {
                    "channel_assignments": [
                        {"H H": list(next(iter(value.values())))}
                        for value in reversed(codec.channel_assignments)
                    ]
                }
            },
        )
