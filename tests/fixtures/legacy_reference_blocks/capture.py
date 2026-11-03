"""Capture the monolithic DeepH / DeepH-E3 / MACE-H layer modules.

Run once on the commit before their decomposition into pipeline stages. Each
variant stores the seed, the module's initial state, its inputs, outputs and
the gradients of a fixed loss. The decomposed blocks must reproduce the seeded
initialization (parameter for parameter) and, from those weights, the outputs
and gradients (see ``tests/physics/test_reference_decomposition.py``).
"""

from pathlib import Path

import torch
from e3nn import o3

HERE = Path(__file__).resolve().parent
NODE_EDGES = torch.tensor([[0, 1, 2, 0, 3, 4, 3, 1], [1, 2, 0, 0, 4, 3, 3, 0]])
NODE_BATCH = torch.tensor([0, 0, 0, 1, 1])


def _loss(values):
    values = values if isinstance(values, (tuple, list)) else (values,)
    return sum(x.square().mean() + 0.031 * x.sum() for x in values)


def _record(name, seed, build, inputs, options):
    torch.manual_seed(seed)
    module = build()
    state = {k: v.detach().clone() for k, v in module.state_dict().items()}
    leaves = [x.detach().clone().requires_grad_(x.is_floating_point()) for x in inputs]
    output = module(*leaves)
    if isinstance(output, dict):
        output = tuple(output.values())
    _loss(output).backward()
    outputs = output if isinstance(output, tuple) else (output,)
    torch.save(
        {
            "seed": seed,
            "options": options,
            "state": state,
            "parameter_names": [n for n, _ in module.named_parameters()],
            "inputs": [x.detach() for x in inputs],
            "outputs": [x.detach() for x in outputs],
            "input_grads": [None if x.grad is None else x.grad for x in leaves],
            "parameter_grads": {
                n: None if p.grad is None else p.grad.detach()
                for n, p in module.named_parameters()
            },
        },
        HERE / f"{name}.pt",
    )


def _deeph():
    from data.bond_geometry import bond_frames, bond_neighborhoods
    from net.transforms.local_coordinates import (
        EndpointScalarEdgeUpdate,
        LocalCoordinateMessage,
        ScalarGatedNodeUpdate,
    )

    generator = torch.Generator().manual_seed(5)
    node = torch.randn(5, 4, generator=generator)
    edge = torch.randn(8, 3, generator=generator)
    distance = torch.rand(8, generator=generator) + 0.5
    for norm in ("layer", "batch", "none"):
        _record(
            f"deeph_node_{norm}",
            881,
            lambda: ScalarGatedNodeUpdate(4, 3, normalization=norm),
            [node, NODE_EDGES, edge, NODE_BATCH, distance],
            {"normalization": norm},
        )
    for final in (True, False):
        _record(
            f"deeph_edge_{'act' if final else 'linear'}",
            883,
            lambda: EndpointScalarEdgeUpdate(4, 3, 7, final_activation=final),
            [node, NODE_EDGES, edge],
            {"final_activation": final},
        )
    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.1, 0.0], [0.2, 1.3, 0.4], [1.1, 1.2, 0.9]]
    )
    edges = torch.tensor([(i, j) for i in range(4) for j in range(4) if i != j]).T
    vectors = positions[edges[1]] - positions[edges[0]]
    relations = bond_neighborhoods(edges, vectors, bond_frames(edges, vectors), num_l=3)
    lcmp_node = torch.randn(4, 3, generator=generator)
    lcmp_edge = torch.randn(edges.shape[1], 5, generator=generator)
    for norm in ("none", "batch"):
        _record(
            f"deeph_local_{norm}",
            891,
            lambda: LocalCoordinateMessage(3, 5, 9, 7, normalization=norm),
            [
                lcmp_node,
                lcmp_edge,
                relations["sub_atom_idx"],
                relations["sub_edge_idx"],
                relations["sub_edge_ang"],
                relations["sub_index"],
                vectors.norm(dim=-1),
            ],
            {"normalization": norm},
        )


def _e3_inputs(node_irreps, edge_irreps, generator):
    node = torch.randn(5, o3.Irreps(node_irreps).dim, generator=generator)
    edge = torch.randn(8, o3.Irreps(edge_irreps).dim, generator=generator)
    vectors = torch.randn(8, 3, generator=generator)
    harmonics = o3.spherical_harmonics(
        "0e+1o+2e", vectors, normalize=True, normalization="component"
    )
    radial = torch.randn(8, 4, generator=generator)
    node_one_hot = torch.nn.functional.one_hot(torch.tensor([0, 1, 0, 1, 1]), 2)
    edge_type = node_one_hot.argmax(-1)
    edge_one_hot = torch.nn.functional.one_hot(
        edge_type[NODE_EDGES[0]] * 2 + edge_type[NODE_EDGES[1]], 4
    )
    return node, edge, harmonics, radial, node_one_hot.double(), edge_one_hot.double()


def _deeph_e3():
    from net.transforms.equivariant import (
        EndpointTensorProductEdgeUpdate,
        EndpointTensorProductNodeUpdate,
    )

    generator = torch.Generator().manual_seed(17)
    node_irreps, edge_irreps = "3x0e+2x1o", "2x0e+1x1o+1x2e"
    node, edge, sh, radial, node_oh, edge_oh = _e3_inputs(
        node_irreps, edge_irreps, generator
    )
    common = dict(num_species=2, fc_len_in=4, irreps_sh="0e+1o+2e")
    node_variants = {
        "default": dict(irreps_out_node=node_irreps),
        "project": dict(irreps_out_node="2x0e+2x1o+1x2e"),
        "selftp": dict(irreps_out_node=node_irreps, use_selftp=True),
        "nosc": dict(irreps_out_node=node_irreps, use_sc=False),
        "sorted": dict(irreps_out_node=node_irreps, if_sort_irreps=True),
    }
    for name, options in node_variants.items():
        _record(
            f"e3_node_{name}",
            331,
            lambda: EndpointTensorProductNodeUpdate(
                irreps_in_node=node_irreps,
                irreps_in_edge=edge_irreps,
                **common,
                **options,
            ),
            [node, node_oh, sh, edge, radial, NODE_EDGES, NODE_BATCH],
            options,
        )
    edge_variants = {
        "default": dict(irreps_out_edge=edge_irreps),
        "project": dict(irreps_out_edge="3x0e+1x1o+1x2e"),
        "selftp": dict(irreps_out_edge=edge_irreps, use_selftp=True),
    }
    for name, options in edge_variants.items():
        _record(
            f"e3_edge_{name}",
            337,
            lambda: EndpointTensorProductEdgeUpdate(
                irreps_in_node=node_irreps,
                irreps_in_edge=edge_irreps,
                **common,
                **options,
            ),
            [node, edge_oh, sh, edge, radial, NODE_EDGES, NODE_BATCH],
            options,
        )


def _mace_h():
    from net.transforms.cluster import (
        AngularDegreeExpansion,
        NeighborDensityInteraction,
        SymmetricProductContraction,
        angular_expansion_projection_irreps,
    )

    generator = torch.Generator().manual_seed(23)
    attrs = torch.nn.functional.one_hot(torch.tensor([0, 1, 0, 1, 1]), 2).double()
    hidden = torch.randn(5, o3.Irreps("2x0e+2x1o").dim, generator=generator)
    vectors = torch.randn(8, 3, generator=generator)
    sh = o3.spherical_harmonics(
        "0e+1o", vectors, normalize=True, normalization="component"
    )
    radial = torch.randn(8, 3, generator=generator)
    density = dict(
        node_attrs_irreps="2x0e",
        node_feats_irreps="2x0e+2x1o",
        edge_attrs_irreps="0e+1o",
        edge_feats_irreps="3x0e",
        target_irreps="2x0e+2x1o",
        hidden_irreps="2x0e+2x1o",
        avg_num_neighbors=2.0,
        radial_layers=[4, 4],
    )
    for norm in ("none", "graph_component"):
        for emit in (False, True):
            _record(
                f"mace_density_{norm}_{'expansion' if emit else 'plain'}",
                735,
                lambda: NeighborDensityInteraction(
                    **density, normalization=norm, emit_expansion=emit
                ),
                [attrs, hidden, sh, radial, NODE_EDGES, NODE_BATCH],
                {"normalization": norm, "emit_expansion": emit},
            )
    channel_density = torch.randn(5, 2, 4, generator=generator)
    skip = torch.randn(5, o3.Irreps("2x0e+2x1o").dim, generator=generator)
    _record(
        "mace_product",
        739,
        lambda: SymmetricProductContraction("2x0e+2x1o", "2x0e+2x1o", 3, 2),
        [channel_density, attrs, skip],
        {"correlation": 3},
    )
    expansion = torch.randn(5, o3.Irreps("2x0e+2x1o").dim, generator=generator)
    out = "2x0e+2x1o+1x2e"
    for nonlin in (True, False):
        for norm in ("graph_component", "none"):
            _record(
                f"mace_expansion_{'gate' if nonlin else 'linear'}_{norm}",
                743,
                lambda: AngularDegreeExpansion(
                    "2x0e+2x1o",
                    angular_expansion_projection_irreps("2x0e+2x1o", out),
                    "2x0e+2x1o",
                    angular_expansion_projection_irreps("2x0e+2x1o", out),
                    out,
                    2,
                    nonlinearity=nonlin,
                    normalization=norm,
                ),
                [hidden, expansion, attrs, NODE_BATCH],
                {"nonlinearity": nonlin, "normalization": norm},
            )


if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)
    _deeph()
    _deeph_e3()
    _mace_h()
