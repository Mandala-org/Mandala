"""Decomposed DeepH / DeepH-E3 / MACE-H blocks against their monolithic layers.

The fixtures (``tests/fixtures/legacy_reference_blocks``) were captured from
the monolithic layer modules before their decomposition into pipeline stages.
Each block of the reference recipes must draw the same seeded parameters and,
from them, reproduce the outputs and every gradient.
"""

from pathlib import Path

import pytest
import torch

from net.block_library import library_blocks
from net.pipeline import FeaturePipeline

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "tests/fixtures/legacy_reference_blocks"


BLOCKS = library_blocks(["deeph", "deeph_e3", "mace"])


@pytest.fixture(autouse=True)
def _double():
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(previous)


def _irreps(irreps):
    from e3nn import o3

    return {"irreps": irreps, "width": o3.Irreps(irreps).dim}


def _deeph_node(options):
    spec = {
        "inputs": {
            "nodes": {"domain": "node", **_irreps("4x0e")},
            "edges": {"domain": "edge", **_irreps("3x0e")},
        },
        "stages": [
            {
                "id": "layer",
                "transform": "crystal_graph_update",
                "inputs": [
                    "nodes",
                    "edges",
                    "edge_index",
                    "node_batch",
                    "edge_lengths",
                ],
                "outputs": ["out"],
                "options": {"width": 4, "normalization": options["normalization"]},
            }
        ],
    }
    names = {"message": "", "norm": "norm"}
    return spec, ["nodes", "edge_index", "edges", "node_batch", "edge_lengths"], names


def _deeph_edge(options):
    spec = {
        "inputs": {
            "nodes": {"domain": "node", **_irreps("4x0e")},
            "edges": {"domain": "edge", **_irreps("3x0e")},
        },
        "stages": [
            {
                "id": "layer",
                "transform": "endpoint_edge_update",
                "inputs": ["nodes", "edges", "edge_index"],
                "outputs": ["out"],
                "options": {"width": 7, "activate_last": options["final_activation"]},
            }
        ],
    }
    return spec, ["nodes", "edge_index", "edges"], {"mlp": "e_lin"}


def _deeph_local(options):
    spec = {
        "inputs": {
            "nodes": {"domain": "node", **_irreps("3x0e")},
            "edges": {"domain": "edge", **_irreps("5x0e")},
            "sub_edge_ang": {
                "domain": "bond_neighbor",
                "representation": "local_components",
                "frame": "bond",
                "width": 9,
            },
        },
        "stages": [
            {
                "id": "layer",
                "transform": "local_coordinate_readout",
                "inputs": [
                    "nodes",
                    "edges",
                    "sub_atom_idx",
                    "sub_edge_idx",
                    "sub_edge_ang",
                    "sub_index",
                    "edge_lengths",
                ],
                "outputs": ["out"],
                "options": {
                    "width": 3,
                    "out_features": 7,
                    "normalization": options["normalization"],
                },
            }
        ],
    }
    slots = [
        "nodes",
        "edges",
        "sub_atom_idx",
        "sub_edge_idx",
        "sub_edge_ang",
        "sub_index",
        "edge_lengths",
    ]
    return spec, slots, {"message": "", "norm": "bn", "readout": "e_lin"}


_E3_NAMES = {
    "pre": "lin_pre",
    "product": "conv",
    "radial": "conv",
    "post": "lin_post",
    "connection": "sc",
    "norm": "norm",
    "residual": "skip_connect.sc",
    "square": "self_tp",
}


def _e3(options, edge):
    irreps_out = options.get("irreps_out_edge" if edge else "irreps_out_node")
    block_options = {
        "irreps_out": irreps_out,
        "radial_hidden": [64, 64],
        "self_connection": options.get("use_sc", True),
        "sort_irreps": options.get("if_sort_irreps", False),
        "self_product": options.get("use_selftp", False),
    }
    spec = {
        "inputs": {
            "nodes": {"domain": "node", **_irreps("3x0e+2x1o")},
            "edges": {"domain": "edge", **_irreps("2x0e+1x1o+1x2e")},
            "species": {"domain": "node", **_irreps("2x0e")},
            "pair_species": {"domain": "edge", **_irreps("4x0e")},
            "harmonics": {"domain": "edge", **_irreps("0e+1o+2e")},
            "radial": {"domain": "edge", **_irreps("4x0e")},
        },
        "stages": [
            {
                "id": "layer",
                "transform": "e3_edge_update" if edge else "e3_node_update",
                "inputs": [
                    "nodes",
                    "pair_species" if edge else "species",
                    "harmonics",
                    "edges",
                    "radial",
                    "edge_index",
                    "edge_batch" if edge else "node_batch",
                ],
                "outputs": ["out"],
                "options": block_options,
            }
        ],
    }
    slots = [
        "nodes",
        "pair_species" if edge else "species",
        "harmonics",
        "edges",
        "radial",
        "edge_index",
        "edge_batch" if edge else "node_batch",
    ]
    return spec, slots, _E3_NAMES


def _mace_density(options):
    graph = options["normalization"] == "graph_component"
    stages = [
        {
            "id": "density",
            "transform": "density_interaction",
            "inputs": ["hidden", "harmonics", "radial", "edge_index", "node_batch"],
            "outputs": ["density", "message"],
            "options": {
                "irreps_out": "2x0e+2x1o",
                "radial_layers": [4, 4],
                "graph_norm": graph,
                "degree_scale": not graph,
                "avg_num_neighbors": 2.0,
            },
        }
    ]
    outputs = {"density": "density"}
    if options["emit_expansion"]:
        stages.append(
            {
                "id": "expansion",
                "transform": "irrep_linear",
                "inputs": ["message"],
                "outputs": ["expansion_raw"],
                "options": {"irreps_out": "2x0e+2x1o"},
            }
        )
        stages.append(
            {
                "id": "expansion_degree",
                "transform": "constant_scale",
                "inputs": ["expansion_raw"],
                "outputs": ["expansion"],
                "options": {"divisor": 2.0},
                "enabled": not graph,
            }
        )
    stages.append(
        {
            "id": "skip",
            "transform": "species_tensor_product",
            "inputs": ["hidden", "species"],
            "outputs": ["skip"],
            "options": {"irreps_out": "2x0e+2x1o"},
        }
    )
    outputs["skip"] = "skip"
    if options["emit_expansion"]:
        outputs["expansion"] = "expansion"
    spec = {
        "inputs": {
            "species": {"domain": "node", **_irreps("2x0e")},
            "hidden": {"domain": "node", **_irreps("2x0e+2x1o")},
            "harmonics": {"domain": "edge", **_irreps("0e+1o")},
            "radial": {"domain": "edge", **_irreps("3x0e")},
        },
        "stages": stages,
        "outputs": outputs,
    }
    slots = ["species", "hidden", "harmonics", "radial", "edge_index", "node_batch"]
    names = {
        "density/lift": "linear_up",
        "density/convolution": "",
        "density/project": "linear",
        "density/norm": "norm",
        "expansion": "linear_up_expansion",
        "skip": "skip_tp",
    }
    return spec, slots, names


def _mace_product(options):
    spec = {
        "inputs": {
            "density": {
                "domain": "node",
                "representation": "channel_irreps",
                "irreps": "1x0e+1x1o",
                "width": 2,
            },
            "species": {"domain": "node", **_irreps("2x0e")},
            "skip": {"domain": "node", **_irreps("2x0e+2x1o")},
        },
        "stages": [
            {
                "id": "layer",
                "transform": "symmetric_product",
                "inputs": ["density", "species", "skip"],
                "outputs": ["out"],
                "options": {"irreps_out": "2x0e+2x1o", "correlation": 3},
            }
        ],
    }
    names = {"contraction": "symmetric_contractions", "mix": "linear"}
    return spec, ["density", "species", "skip"], names


def _mace_expansion(options):
    spec = {
        "inputs": {
            "hidden": {"domain": "node", **_irreps("2x0e+2x1o")},
            "angular": {"domain": "node", **_irreps("2x0e+2x1o")},
            "species": {"domain": "node", **_irreps("2x0e")},
        },
        "stages": [
            {
                "id": "layer",
                "transform": "angular_expansion",
                "inputs": ["hidden", "angular", "species", "node_batch"],
                "outputs": ["out"],
                "options": {
                    "irreps_out": "2x0e+2x1o+1x2e",
                    "gate": options["nonlinearity"],
                    "graph_norm": options["normalization"] == "graph_component",
                },
            }
        ],
    }
    names = {
        "hidden_projection": "hidden_linear",
        "angular_projection": "up_linear",
        "product": "",
        "post": "lin_post",
        "connection": "sc",
        "norm": "norm",
        "skip": "skip_connect",
    }
    return spec, ["hidden", "angular", "species", "node_batch"], names


CASES = {
    "deeph_node": _deeph_node,
    "deeph_edge": _deeph_edge,
    "deeph_local": _deeph_local,
    "e3_node": lambda options: _e3(options, edge=False),
    "e3_edge": lambda options: _e3(options, edge=True),
    "mace_density": _mace_density,
    "mace_product": _mace_product,
    "mace_expansion": _mace_expansion,
}


def _case(path):
    for prefix, builder in CASES.items():
        if path.stem.startswith(prefix):
            return builder
    raise AssertionError(path)


def _monolith_name(name, names):
    """Monolithic parameter name of a decomposed pipeline parameter."""
    stage, tail = name.removeprefix("transforms.").split(".", 1)
    inner = stage.split("/", 1)[1] if stage.startswith("layer/") else stage
    source = names[inner]
    if inner in {"connection", "residual", "square", "skip"} or (
        source == "skip_connect"
    ):
        for prefix in ("tp.", "project."):
            if tail.startswith(prefix) and inner != "square":
                tail = tail[len(prefix) :]
        if inner == "square":
            tail = tail.removeprefix("tp.")
    if inner == "norm" and source in {"norm", "bn"} and tail.startswith("norm."):
        tail = tail[len("norm.") :]
    if source == "e_lin":
        tail = tail.removeprefix("layers.")
    return f"{source}.{tail}" if source else tail


@pytest.mark.parametrize(
    "path", sorted(FIXTURES.glob("*.pt")), ids=lambda path: path.stem
)
def test_decomposed_block_matches_monolithic_layer(path):
    record = torch.load(path, weights_only=False)
    spec, slots, names = _case(path)(record["options"])
    spec = {"blocks": BLOCKS, "outputs": {"out": "out"}, **spec}
    torch.manual_seed(record["seed"])
    model = FeaturePipeline(spec, num_species=2)

    parameters = dict(model.named_parameters())
    mapped = {_monolith_name(name, names): p for name, p in parameters.items()}
    assert len(mapped) == len(parameters)
    monolith = set(record["parameter_names"])
    assert mapped.keys() <= monolith
    # The monolithic local layer owns a batch normalization even when unused.
    unused = {n for n, g in record["parameter_grads"].items() if g is None}
    assert monolith - mapped.keys() <= unused
    for name, parameter in mapped.items():
        torch.testing.assert_close(
            parameter.detach(), record["state"][name], rtol=0, atol=0, msg=name
        )

    leaves = [x.clone().requires_grad_(x.is_floating_point()) for x in record["inputs"]]
    inputs = dict(zip(slots, leaves))
    if "edge_batch" in slots:
        # The monolithic edge update received node graphs and indexed them.
        inputs["edge_batch"] = inputs["edge_batch"][inputs["edge_index"][0]]
    outputs = list(model(inputs).values())
    assert len(outputs) == len(record["outputs"])
    for actual, expected in zip(outputs, record["outputs"]):
        torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
    sum(x.square().mean() + 0.031 * x.sum() for x in outputs).backward()
    for leaf, expected in zip(leaves, record["input_grads"]):
        if expected is not None:
            torch.testing.assert_close(leaf.grad, expected, rtol=1e-10, atol=1e-12)
    for name, parameter in mapped.items():
        torch.testing.assert_close(
            parameter.grad,
            record["parameter_grads"][name],
            rtol=1e-10,
            atol=1e-12,
            msg=name,
        )
