"""Configuration contracts and full-topology differential model checks.

Tiny models retain every stage of the default three-layer architectures while
reducing channel counts and angular degree. Optional source oracles require
MANDALA_DEEPHE3_REFERENCE and MANDALA_MACEH_REFERENCE_ROOT respectively.
"""

from __future__ import annotations

import hashlib
import importlib
import os
from pathlib import Path
import sys
import types

import pytest
import torch
from e3nn.o3 import Irreps

from net.configuration import resolve_configuration
from net.pipeline import FeaturePipeline
from net.training_policies import masked_error


CONFIG_ROOT = Path(__file__).resolve().parents[3] / "configs" / "reference"


@pytest.fixture(autouse=True)
def _double_precision():
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(previous)


def _tiny_configuration(kind, *, spinful=False):
    parameters = dict(
        embedding_width=3,
        radial_width=4,
        angular_irreps="1x0e+1x1o+1x2e",
        node_irreps="3x0e+2x1o+1x2e",
        edge_irreps="3x0e+2x1o+1x2e",
    )
    if kind == "mace_h":
        parameters.update(
            hidden_irreps="2x0e+2x1o+2x2e",
            density_lmax=2,
            density_radial_width=4,
            radial_layers=[4, 4, 4],
        )
    return resolve_configuration(
        CONFIG_ROOT / f"{kind}.yaml",
        species_shells={"Si": [0, 1]},
        spinful=spinful,
        overrides={"parameters": parameters},
    )


def _inputs():
    positions = torch.tensor([[0.0, 0, 0], [1.2, 0.3, 0.2], [-0.4, 1.3, 0.7]])
    edges = torch.tensor([(i, j) for i in range(3) for j in range(3)]).T
    vectors = positions[edges[1]] - positions[edges[0]]
    lengths = vectors.norm(dim=-1)
    onsite = edges[0] == edges[1]
    return {
        "node_type_idx": torch.zeros(3, dtype=torch.long),
        "node_one_hot": torch.ones(3, 1),
        "edge_type_idx": torch.zeros(9, dtype=torch.long),
        "edge_one_hot": torch.ones(9, 1),
        "edge_index": edges,
        "edge_vectors": vectors,
        "edge_lengths": lengths,
        "node_batch": torch.zeros(3, dtype=torch.long),
        "edge_batch": torch.zeros(9, dtype=torch.long),
        "selfloop_edge": onsite,
        "message_edge_index": edges[:, ~onsite],
        "message_vectors": vectors[~onsite],
        "message_lengths": lengths[~onsite],
        "message_to_output": torch.where(~onsite)[0],
    }


def _load_oracle(kind):
    if kind == "deeph_e3":
        location = os.getenv("MANDALA_DEEPHE3_REFERENCE")
        package_dir = Path(location) / "deephe3" if location else None
        hashes = {
            "model.py": "e27fbb04ade27371aa798da1a47ed4c4adc2bec629ef46dfa968c1e0734f8297"
        }
        entry = "model"
    else:
        location = os.getenv("MANDALA_MACEH_REFERENCE_ROOT")
        package_dir = Path(location) / "maceh" if location else None
        hashes = {
            "maceh.py": "689b32347a3df902c52db0d76d16fa8b5eb1a5ff4d787932045724ddec04e17c",
            "macehmodules.py": "18193eef191041a9644bac7a0e8eb3c39c3d1dc33967341b014db672e0cae197",
            "statistics.py": "348a9e0ef79ddc0696192dbb586e683169de33e1d121abd3a1380d80294d3000",
        }
        entry = "maceh"
    if package_dir is None:
        pytest.skip(f"Pinned {kind} source oracle is not configured.")
    for filename, expected in hashes.items():
        path = package_dir / filename
        assert path.is_file(), f"Missing pinned oracle: {path}"
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected
    name = f"_mandala_full_pipeline_{kind}_oracle"
    if name not in sys.modules:
        package = types.ModuleType(name)
        package.__path__ = [str(package_dir)]
        sys.modules[name] = package
    return (
        importlib.import_module(f"{name}.{entry}"),
        importlib.import_module(f"{name}.e3modules"),
    )


def _reference_model(kind, configuration, oracle):
    model, _ = oracle
    p, layout = configuration["parameters"], configuration["layout"]
    common = dict(
        num_species=configuration["species"]["count"],
        irreps_edge_init=f"{p['embedding_width']}x0e",
        irreps_sh=p["angular_irreps"],
        irreps_post_node=layout["post_node_irreps"],
        irreps_mid_edge=p["edge_irreps"],
        irreps_post_edge=layout["post_irreps"],
        irreps_out_edge=layout["coupled_irreps"],
        r_max=p["cutoff_radius"],
        num_basis=p["radial_width"],
    )
    if kind == "deeph_e3":
        return model.Net(
            **common,
            irreps_embed_node=f"{p['embedding_width']}x0e",
            irreps_mid_node=p["node_irreps"],
            irreps_out_node="1x0e",
            num_block=3,
            use_sbf=False,
            selftp=p["terminal_self_product"],
        )
    return model.Net(
        **common,
        num_blocks=3,
        max_ell=p["density_lmax"],
        hidden_irreps=p["hidden_irreps"],
        avg_num_neighbors=p["avg_num_neighbors"],
        atomic_numbers=[14],
        correlation=p["correlation"],
        radial_MLP=p["radial_layers"],
        mace_norm="e3LayerNorm",
        expand_norm="e3LayerNorm",
        basis_func="Gaussian",
        num_gaussian=p["density_radial_width"],
        shift_scale=True,
    )


def _source_parameter_name(kind, name):
    _, stage, tail = name.split(".", 2)
    if kind == "deeph_e3":
        prefix = {
            "embedding": "embedding",
            "node_readout": "lin_node",
            "edge_readout": "lin_edge",
        }.get(stage)
        if stage.startswith("node_update_"):
            prefix = f"node_update_blocks.{stage.rsplit('_', 1)[-1]}"
        if stage.startswith("edge_update_"):
            prefix = f"edge_update_blocks.{stage.rsplit('_', 1)[-1]}"
    else:
        prefix = {
            "node_embedding": "mace.node_embedding.linear",
            "density_radial_basis": "mace.radial_embedding",
            "edge_readout": "edge_update.lin_edge",
        }.get(stage)
        if stage.startswith("density_") and stage.rsplit("_", 1)[-1].isdigit():
            prefix = f"mace.interactions.{stage.rsplit('_', 1)[-1]}"
        if stage.startswith("product_"):
            prefix = f"mace.products.{stage.rsplit('_', 1)[-1]}"
        if stage.startswith("angular_expansion_"):
            prefix = f"expansion.node_order_expansion_blocks.{stage.rsplit('_', 1)[-1]}"
        if stage.startswith("edge_update_"):
            prefix = f"edge_update.edge_update_blocks.{stage.rsplit('_', 1)[-1]}"
    if prefix is None:
        raise AssertionError(f"No source parameter coordinate for {name}")
    return f"{prefix}.{tail}"


def _assert_parameter_parity(kind, actual, expected, *, gradients=False):
    source = dict(expected.named_parameters())
    mapped = set()
    for name, parameter in actual.named_parameters():
        source_name = _source_parameter_name(kind, name)
        mapped.add(source_name)
        reference = source[source_name]
        if gradients:
            parameter, reference = parameter.grad, reference.grad
            if parameter is None or reference is None:
                assert parameter is reference
                continue
        torch.testing.assert_close(parameter, reference, rtol=2e-9, atol=2e-10)
    assert mapped == source.keys()


@pytest.mark.parametrize("kind", ["deeph_e3", "mace_h"])
def test_presets_describe_complete_default_architectures(kind):
    config, codec = resolve_configuration(
        CONFIG_ROOT / f"{kind}.yaml", species_shells={"Si": [0, 1]}
    )
    assert config["parameters"]["embedding_width"] == 64
    assert config["parameters"]["radial_width"] == 128
    assert config["layout"]["packed_dim"] == codec.packed_dim == 16
    assert config["layout"]["real_packed_dim"] == 16
    stages = config["pipeline"]["stages"]
    assert sum(s["transform"] == "endpoint_tensor_product_edge" for s in stages) == 3
    if kind == "deeph_e3":
        assert (
            sum(s["transform"] == "endpoint_tensor_product_node" for s in stages) == 3
        )
        assert config["parameters"]["terminal_self_product"] is False
    else:
        for name in (
            "neighbor_density",
            "symmetric_product",
            "angular_degree_expansion",
        ):
            assert sum(s["transform"] == name for s in stages) == 3
        assert config["parameters"]["correlation"] == 3
        assert (
            next(s for s in stages if s["id"] == "output_affine")["options"][
                "gradient_mode"
            ]
            == "identity"
        )


def test_generic_resolution_uses_explicit_species_and_no_architecture_dispatch():
    source = {
        "schema_version": 1,
        "codec": {"spinful": True},
        "parameters": {"width": 3},
        "pipeline": {
            "stages": [{"id": "custom", "options": {"width": "${parameters.width}"}}],
            "outputs": {},
        },
    }
    result, codec = resolve_configuration(
        source,
        species_shells={"O": [0, 1], "H": [0]},
        overrides={"parameters": {"width": 5}},
    )
    assert result["species"]["order"] == ["H", "O"]
    assert result["pipeline"]["stages"][0]["options"]["width"] == 5
    assert result["layout"]["real_packed_dim"] == 8 * codec.packed_dim
    assert source["parameters"]["width"] == 3
    with pytest.raises(ValueError, match="exactly once"):
        resolve_configuration(
            source,
            species_shells={"H": [0]},
            overrides={"species": {"order": ["H", "H"]}},
        )


@pytest.mark.parametrize("kind", ["deeph_e3", "mace_h"])
@pytest.mark.parametrize("spinful", [False, True])
def test_resolved_layout_matches_source_automatic_target_analysis(kind, spinful):
    oracle = _load_oracle(kind)
    package = oracle[0].__package__
    utility = importlib.import_module(f"{package}.utils")
    config, _ = resolve_configuration(
        CONFIG_ROOT / f"{kind}.yaml",
        species_shells={"Si": [0, 0, 1], "H": [0]},
        spinful=spinful,
    )
    # Upstream dataset vocabulary sorts atomic numbers before orbital analysis.
    _, coupled, post = utility.orbital_analysis(
        {"1": [0], "14": [0, 0, 1]}, "a", spinful
    )
    assert Irreps(config["layout"]["coupled_irreps"]) == coupled
    assert Irreps(config["layout"]["post_irreps"]) == post
    parameters = config["parameters"]
    node = parameters["node_irreps"]
    if spinful:
        node = utility.refine_post_node(
            node,
            parameters["node_irreps"],
            parameters["edge_irreps"],
            parameters["angular_irreps"],
            post,
        )
    assert Irreps(config["layout"]["post_node_irreps"]) == Irreps(node)


@pytest.mark.parametrize("kind", ["deeph_e3", "mace_h"])
def test_tiny_complete_pipeline_builds_without_oracle(kind):
    config, codec = _tiny_configuration(kind)
    pipeline = FeaturePipeline(config["pipeline"], num_species=1)
    if kind == "mace_h":
        pipeline.transforms["output_affine"].fitted.fill_(True)
    outputs, trace = pipeline(_inputs(), return_trace=True)
    assert outputs["hamiltonian"].shape == (9, codec.packed_dim)
    assert trace["edge_3"].shape[0] == 9
    assert torch.isfinite(outputs["hamiltonian"]).all()


@pytest.mark.parametrize("kind", ["deeph_e3", "mace_h"])
@pytest.mark.parametrize("spinful", [False, True])
def test_full_topology_native_predictions_gradients_and_three_adam_steps(kind, spinful):
    oracle = _load_oracle(kind)
    config, codec = _tiny_configuration(kind, spinful=spinful)
    torch.manual_seed(337)
    expected = _reference_model(kind, config, oracle)
    torch.manual_seed(337)
    actual = FeaturePipeline(config["pipeline"], num_species=1)
    _assert_parameter_parity(kind, actual, expected)
    inputs = _inputs()
    source_data = types.SimpleNamespace(
        x=inputs["node_type_idx"],
        edge_index=inputs["edge_index"],
        edge_attr=torch.cat(
            [inputs["edge_lengths"][:, None], inputs["edge_vectors"]], dim=-1
        ),
        batch=inputs["node_batch"],
    )

    # Upstream forward uses both mapping and attribute access.
    class SourceGraph(dict):
        __getattr__ = dict.__getitem__

    source_data = SourceGraph(vars(source_data))
    if kind == "mace_h":
        affine = actual.transforms["output_affine"]
        with torch.no_grad():
            means = torch.linspace(-0.2, 0.2, codec.irreps_out.dim)
            scales = torch.linspace(0.3, 1.7, codec.irreps_out.dim)
            # Identical frozen, nontrivial train-only statistics isolate the
            # forward/backward formula; fitting-population tests live elsewhere.
            affine.mean.copy_(means[None])
            affine.std.copy_(scales[None])
            affine.fitted.fill_(True)
            expected.mean_tensor.copy_(means[None, None])
            expected.std_tensor.copy_(scales[None, None])
    source_codec = oracle[1].e3TensorDecomp(
        Irreps(config["layout"]["coupled_irreps"]),
        [tuple(pair) for pair in config["layout"]["shell_pairs"]],
        torch.get_default_dtype(),
        spinful=spinful,
    )
    generator = torch.Generator().manual_seed(199)
    desired = torch.randn(9, codec.irreps_out.dim, generator=generator)
    target = source_codec.get_H(desired).detach()
    mask = torch.ones_like(target, dtype=torch.bool)
    mask[0] = False
    optimizer_args = dict(lr=0.002, betas=(0.9, 0.999), eps=1e-8)
    ours_optimizer = torch.optim.Adam(actual.parameters(), **optimizer_args)
    source_optimizer = torch.optim.Adam(expected.parameters(), **optimizer_args)
    for _ in range(3):
        ours_optimizer.zero_grad()
        source_optimizer.zero_grad()
        _, source_coefficients = expected(source_data)
        source_prediction = source_codec.get_H(source_coefficients)
        prediction = actual(inputs)["hamiltonian"]
        torch.testing.assert_close(prediction, source_prediction, rtol=2e-9, atol=2e-10)
        ours_loss = masked_error(prediction, target, mask, kind="mse")
        source_loss = (source_prediction - target).abs().square()[mask].mean()
        torch.testing.assert_close(ours_loss, source_loss, rtol=2e-10, atol=2e-11)
        ours_loss.backward()
        source_loss.backward()
        _assert_parameter_parity(kind, actual, expected, gradients=True)
        ours_optimizer.step()
        source_optimizer.step()
        _assert_parameter_parity(kind, actual, expected)
