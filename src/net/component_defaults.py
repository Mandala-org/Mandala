"""Lower flat Config values to the same graph used by explicit configurations.

This is a configuration compatibility translator. Its output contains ordinary
registered transformations and may be edited or mixed like any other graph.
"""

from dataclasses import asdict
from e3nn.o3 import Irreps
from net.common import resolve_hidden_irreps


def default_graph(cfg, num_species):
    options = asdict(cfg)
    options["pipeline"] = None
    hidden = str(resolve_hidden_irreps(cfg))
    stages = [
        {
            "id": "node_encoding",
            "transform": "species_irrep_embedding",
            "inputs": ["node_type_idx"],
            "outputs": ["node_0"],
            "options": {"num_species": num_species, "config": options},
            "role": "encoder",
        },
        {
            "id": "edge_encoding",
            "transform": "pair_radial_angular_embedding",
            "inputs": ["edge_type_idx", "edge_length_emb", "edge_sh"],
            "outputs": ["edge_0"],
            "options": {
                "num_pairs": num_species**2,
                "irreps_out": hidden,
                "config": options,
            },
            "role": "encoder",
        },
    ]
    for index in range(cfg.num_layers_gnn):
        common = {
            "num_species": num_species,
            "irreps_out": hidden,
            "config": options,
            "info": {"layer": index},
        }
        stages.append(
            {
                "id": f"node_message_{index}",
                "transform": "endpoint_irrep_node",
                "inputs": [
                    f"node_{index}",
                    f"edge_{index}",
                    "edge_index",
                    "edge_sh",
                    "edge_length_emb",
                    "node_one_hot",
                    "edge_type_idx",
                ],
                "outputs": [f"node_{index+1}"],
                "options": common,
                "role": "message",
            }
        )
        stages.append(
            {
                "id": f"edge_message_{index}",
                "transform": "endpoint_irrep_edge",
                "inputs": [
                    f"node_{index+1}",
                    f"edge_{index}",
                    "edge_index",
                    "edge_sh",
                    "edge_length_emb",
                    "edge_one_hot",
                    "edge_type_idx",
                ],
                "outputs": [f"edge_{index+1}"],
                "options": common,
                "role": "message",
            }
        )
    outputs = {}
    for operator in cfg.matrix_targets:
        slot = f"{operator}_vectors"
        stages.append(
            {
                "id": f"readout_{operator}",
                "transform": "partitioned_irrep_readout",
                "inputs": [
                    f"node_{cfg.num_layers_gnn}",
                    f"edge_{cfg.num_layers_gnn}",
                    "pred_pair_edges_static",
                    "edge_partitions",
                ],
                "outputs": [slot],
                "options": {
                    "irreps_neck": hidden,
                    "config": options,
                    "info": {"matrix": operator},
                },
                "role": "readout",
            }
        )
        outputs[operator] = slot
    return {
        "inputs": {
            "edge_length_emb": {"domain": "edge", "irreps": f"{cfg.n_radial}x0e"},
            "edge_sh": {
                "domain": "edge",
                "representation": "irreps",
                "irreps": str(Irreps.spherical_harmonics(cfg.l_max)),
            },
            "pred_pair_edges_static": {
                "domain": "relation",
                "representation": "object",
            },
            "edge_partitions": {"domain": "relation", "representation": "object"},
        },
        "stages": stages,
        "outputs": outputs,
    }


def register_components(register):
    from e3nn.o3 import Irreps
    from net.common import Config
    from net.encoders import NodeEncoder, EdgeEncoder
    from net.layers import NodeUpdateBlock, EdgeUpdateBlock
    from net.heads import DeepHead
    from net.pipeline import (
        FeatureSpec,
        _global_irreps,
        _same_support,
        _same_angular_basis,
    )

    def canonical(*specs):
        basis = _same_angular_basis(*specs)
        if basis not in {None, "xyz"}:
            raise ValueError(
                "Shared components require canonical xyz features at the representation boundary"
            )

    @register("species_irrep_embedding")
    def node(p, inputs, mapper):
        p = dict(p)
        cfg = Config.from_mapping(p.pop("config"))
        module = NodeEncoder(p.pop("num_species"), cfg, info={"name": "node_encoding"})
        if p or len(inputs) != 1 or inputs[0].representation != "index":
            raise ValueError("Species embedding expects one integer node index")
        module.accepts_activation_mags = True
        return module, [FeatureSpec("node", "irreps", str(module.irreps_out))]

    @register("pair_radial_angular_embedding")
    def edge(p, inputs, mapper):
        p = dict(p)
        cfg = Config.from_mapping(p.pop("config"))
        module = EdgeEncoder(
            p.pop("num_pairs"),
            Irreps(p.pop("irreps_out")),
            cfg,
            info={"name": "edge_encoding"},
        )
        if p or len(inputs) != 3:
            raise ValueError(
                "Pair embedding expects pair index, radial and angular features"
            )
        _same_support(*inputs)
        canonical(inputs[2])
        module.accepts_activation_mags = True
        return module, [
            FeatureSpec(
                "edge", "irreps", str(module.irreps_out), support=inputs[0].support
            )
        ]

    def endpoint(p, inputs, domain):
        p = dict(p)
        cfg = Config.from_mapping(p.pop("config"))
        out = Irreps(p.pop("irreps_out"))
        n = p.pop("num_species")
        cls = NodeUpdateBlock if domain == "node" else EdgeUpdateBlock
        kwargs = {f"{domain}_irreps_out": out}
        module = cls(
            _global_irreps(inputs[0], "node"),
            _global_irreps(inputs[1], "edge"),
            n,
            cfg,
            n_edge_types=n * n,
            info=p.pop("info", None),
            **kwargs,
        )
        if p or len(inputs) != 7:
            raise ValueError("Endpoint update expects seven typed graph features")
        _same_support(inputs[1], inputs[2], inputs[3], inputs[4], inputs[6])
        canonical(inputs[0], inputs[1], inputs[3])
        return module, [
            FeatureSpec(
                domain,
                "irreps",
                str(module.irreps_out),
                support=inputs[0 if domain == "node" else 1].support,
            )
        ]

    @register("endpoint_irrep_node")
    def node_update(p, inputs, mapper):
        return endpoint(p, inputs, "node")

    @register("endpoint_irrep_edge")
    def edge_update(p, inputs, mapper):
        return endpoint(p, inputs, "edge")

    @register("partitioned_irrep_readout")
    def readout(p, inputs, mapper):
        if mapper is None:
            raise ValueError("Orbital readout requires an orbital mapper")
        p = dict(p)
        cfg = Config.from_mapping(p.pop("config"))
        node, edge = _global_irreps(inputs[0], "node"), _global_irreps(
            inputs[1], "edge"
        )
        canonical(inputs[0], inputs[1])
        _same_support(*inputs[1:])
        if inputs[1].support != "output":
            raise ValueError("Partitioned orbital readout requires output-edge support")
        if any(spec.representation != "object" for spec in inputs[2:]):
            raise ValueError("Partitioned readout requires explicit edge metadata")
        module = DeepHead(
            node if cfg.head_use_node_embeddings_for_self_edges else edge,
            edge,
            Irreps(p.pop("irreps_neck")),
            list(mapper.edge_types),
            mapper,
            cfg,
            info=p.pop("info", None),
        )
        if p or len(inputs) != 4:
            raise ValueError(
                "Partitioned readout expects node, edge, pair edges and partitions"
            )
        return module, [FeatureSpec("edge", "block_vectors")]
