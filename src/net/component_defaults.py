"""Lower the flat MANDALA configuration to an explicit pipeline recipe.

The result is an ordinary recipe of registered stages, grouped into composite
blocks (``edge_encoder``, ``node_update``, ``edge_update``, ``readout``). It can
be exported, edited and recombined like any hand-written recipe; nothing in
the compiler or the trainer depends on these block names.
"""

from __future__ import annotations

import math

from e3nn.o3 import Irreps

from net.common import resolve_hidden_irreps


def _stage(id, transform, inputs, outputs, options=None, **extra):
    """A stage whose options omit values equal to the transform's defaults.

    The compiled stage then reports exactly the choices this configuration
    makes as explicit; everything else shows as default or inferred.
    """
    from net.registry import REQUIRED, registered_transforms

    info = registered_transforms().get(transform)
    if info is not None and options:
        defaults = {
            o.name: o.default for o in info.options if o.default is not REQUIRED
        }
        options = {
            key: value
            for key, value in options.items()
            if key not in defaults or _plain(value) != _plain(defaults[key])
        }
    stage = {
        "id": id,
        "transform": transform,
        "inputs": list(inputs),
        "outputs": list(outputs),
    }
    if options:
        stage["options"] = options
    stage.update(extra)
    return stage


def _plain(value):
    return list(value) if isinstance(value, tuple) else value


def _mlp(cfg, variant, **options):
    from net.transforms.settings import stage_settings

    return {"variant": variant, **stage_settings(cfg), **options}


def _nonlinearity(cfg):
    from net.transforms.settings import stage_settings

    settings = stage_settings(cfg)
    return {
        key: settings[key]
        for key in (
            "nonlinearity",
            "scalar_activation",
            "gate_activation",
            "norm",
            "s2_resolution",
        )
    }


def _convolution(cfg, pair_conditioned):
    options = {
        "irreps_out": "$irreps_out",
        "tensor_product": cfg.tp_type,
        "radial_layers": list(cfg.radial_layers),
        "radial_weight_granularity": cfg.radial_weight_granularity,
    }
    if pair_conditioned:
        options["pair_embedding_dim"] = cfg.edge_type_emb_dim
    return options


def _update_tail(cfg, stages, update, previous, attributes, residual):
    """Shared end of the node and edge updates: self-connection ... residual."""
    internal = cfg.internal_e3mlp_variant or cfg.e3mlp_variant
    current = update
    if cfg.use_self_connection:
        stages.append(
            _stage(
                "self_connection",
                "self_connection",
                [previous, attributes, current],
                ["connected"],
            )
        )
        current = "connected"
    stages.append(
        _stage(
            "activation",
            "irrep_nonlinearity",
            [current],
            ["activated"],
            _nonlinearity(cfg),
        )
    )
    current = "activated"
    if cfg.e3layernorm:
        stages.append(_stage("norm", "layer_norm", [current], ["normalized"]))
        current = "normalized"
    if cfg.dropout > 0.0:
        stages.append(
            _stage(
                "dropout", "irrep_dropout", [current], ["dropped"], {"p": cfg.dropout}
            )
        )
        current = "dropped"
    if cfg.internal_e3mlp_layers > 0:
        stages.append(
            _stage(
                "refine",
                "e3mlp",
                [current],
                ["refined"],
                _mlp(
                    cfg,
                    internal,
                    layers=cfg.internal_e3mlp_layers,
                    irreps_out="$irreps_out",
                    irreps_hidden="$irreps_out",
                ),
            )
        )
        current = "refined"
    if residual:
        stages.append(_stage("residual", "residual", [previous, current], ["out"]))
    else:
        stages[-1]["outputs"] = ["out"]
    return stages


def node_update_block(cfg, pair_conditioned):
    inputs = ["node", "edge", "edge_index", "harmonics", "radial"]
    if cfg.use_self_connection:
        inputs.append("node_attributes")
    if pair_conditioned:
        inputs.append("pair_type")
    convolution_inputs = ["gathered", "harmonics", "radial"] + (
        ["pair_type"] if pair_conditioned else []
    )
    stages = [
        _stage("pre", "e3mlp", ["node"], ["pre"], _mlp(cfg, cfg.e3mlp_variant)),
        _stage(
            "gather",
            "endpoint_concat",
            ["pre", "edge", "edge_index"],
            ["gathered"],
            {"combine": "concat"},
        ),
        _stage(
            "convolution",
            "equivariant_convolution",
            convolution_inputs,
            ["messages"],
            _convolution(cfg, pair_conditioned),
        ),
    ]
    aggregation = str(cfg.node_update_message_agg).lower()
    if aggregation == "attention":
        stages.append(
            _stage(
                "aggregate",
                "projected_attention",
                ["messages", "pre", "radial", "edge_index"],
                ["aggregated"],
                _mlp(
                    cfg,
                    cfg.e3mlp_variant,
                    scalar_dim=cfg.node_update_attention_scalar_dim,
                    heads=cfg.node_update_attention_heads,
                    receiver=1,
                ),
            )
        )
    elif aggregation in {"sum", "average"}:
        stages.append(
            _stage(
                "aggregate",
                "indexed_aggregation",
                ["messages", "edge_index", "node"],
                ["aggregated"],
                {"reduction": "sum" if aggregation == "sum" else "mean", "receiver": 1},
            )
        )
    else:
        raise ValueError("node_update_message_agg must be sum, average or attention")
    stages.append(
        _stage(
            "post", "e3mlp", ["aggregated"], ["update"], _mlp(cfg, cfg.e3mlp_variant)
        )
    )
    _update_tail(
        cfg, stages, "update", "node", "node_attributes", cfg.node_update_residual
    )
    return {
        "summary": "MANDALA node update: gated convolution of endpoint features, aggregation, refinement",
        "inputs": inputs,
        "outputs": ["out"],
        "options": {"irreps_out": None},
        "stages": stages,
    }


def edge_update_block(cfg, pair_conditioned):
    combine = str(cfg.edge_update_node_combine).lower()
    if combine not in {"concat", "sum", "tensor_product"}:
        raise ValueError(
            "edge_update_node_combine must be concat, sum or tensor_product"
        )
    inputs = ["node", "edge", "edge_index", "harmonics", "radial"]
    if cfg.use_self_connection:
        inputs.append("edge_attributes")
    if pair_conditioned:
        inputs.append("pair_type")
    convolution_inputs = ["gathered", "harmonics", "radial"] + (
        ["pair_type"] if pair_conditioned else []
    )
    stages = [
        _stage("pre", "e3mlp", ["edge"], ["pre"], _mlp(cfg, cfg.e3mlp_variant)),
        _stage(
            "gather",
            "endpoint_concat",
            ["node", "pre", "edge_index"],
            ["gathered"],
            {"combine": combine},
        ),
        _stage(
            "convolution",
            "equivariant_convolution",
            convolution_inputs,
            ["messages"],
            _convolution(cfg, pair_conditioned),
        ),
        _stage("post", "e3mlp", ["messages"], ["update"], _mlp(cfg, cfg.e3mlp_variant)),
    ]
    _update_tail(
        cfg, stages, "update", "edge", "edge_attributes", cfg.edge_update_residual
    )
    return {
        "summary": "MANDALA edge update: gated convolution of both endpoints and the edge",
        "inputs": inputs,
        "outputs": ["out"],
        "options": {"irreps_out": None},
        "stages": stages,
    }


def edge_encoder_block(cfg, hidden):
    style = cfg.edge_encoder_style.lower()
    if style == "distance":
        projected = "projected" if cfg.e3layernorm else "out"
        stages = [
            _stage(
                "project",
                "scalar_linear",
                ["radial"],
                [projected],
                {"out_features": cfg.hidden_base_dim},
            )
        ]
        if cfg.e3layernorm:
            stages.append(_stage("norm", "layer_norm", ["projected"], ["out"]))
        return {
            "summary": "Edge features from the radial basis only",
            "inputs": ["pair_type", "radial", "harmonics"],
            "outputs": ["out"],
            "stages": stages,
        }
    if style != "rich":
        raise ValueError("edge_encoder_style must be 'rich' or 'distance'")
    stages = [
        _stage(
            "pair_embedding",
            "embedding",
            ["pair_type"],
            ["pair_embedding"],
            {"width": cfg.edge_type_emb_dim},
            initialization={"kind": "normal", "std": 0.2},
        )
    ]
    angular = "harmonics"
    if cfg.edge_encoder_use_sh_tensor_square:
        stages.append(
            _stage(
                "square",
                "tensor_square",
                ["harmonics"],
                ["squared"],
                {"irreps_out": hidden},
            )
        )
        angular = "squared"
    stages.append(
        _stage(
            "geometry", "concat", ["radial", angular], ["geometry"], {"simplify": True}
        )
    )
    out = "out" if not cfg.e3layernorm else "mixed"
    stages.append(
        _stage(
            "mix",
            "fc_tensor_product",
            ["pair_embedding", "geometry"],
            [out],
            {"irreps_out": hidden},
        )
    )
    if cfg.e3layernorm:
        stages.append(_stage("norm", "layer_norm", ["mixed"], ["out"]))
    return {
        "summary": "Edge features: pair embedding (x) [radial basis, harmonics (x) harmonics]",
        "inputs": ["pair_type", "radial", "harmonics"],
        "outputs": ["out"],
        "stages": stages,
    }


def readout_block(cfg, hidden):
    diag, offdiag = cfg.head_diag_output_scale, cfg.head_offdiag_output_scale
    scales = {"onsite": diag, "shifted_self": diag, "offsite": offdiag}
    trunk = _mlp(
        cfg,
        cfg.e3mlp_variant,
        layers=cfg.neck_depth,
        irreps_out=hidden,
        activate_last=True,
    )
    if cfg.head_use_node_embeddings_for_self_edges:
        # Onsite blocks read their atom's features: one onsite edge per atom.
        stages = [
            _stage("onsite_trunk", "e3mlp", ["node"], ["onsite_trunk"], trunk),
            _stage(
                "trunk",
                "e3mlp",
                ["edge", "roles"],
                ["edge_trunk"],
                {**trunk, "inactive_routes": ["onsite"]},
            ),
            _stage(
                "merge",
                "onsite_from_nodes",
                ["edge_trunk", "onsite_trunk", "edge_index", "roles"],
                ["trunk"],
            ),
        ]
    else:
        stages = [_stage("trunk", "e3mlp", ["edge", "roles"], ["trunk"], trunk)]
    hidden_slot = "trunk"
    if cfg.head_use_tensor_square:
        stages.append(
            _stage(
                "square",
                "tensor_square",
                ["trunk", "roles"],
                ["squared"],
                {"irreps_out": hidden},
            )
        )
        hidden_slot = "squared"
    projection = _mlp(
        cfg,
        cfg.head_e3mlp_variant or cfg.e3mlp_variant,
        mode=str(getattr(cfg, "head_pair_mode", "split")).lower(),
        layers=cfg.head_e3mlp_layers,
        route_scales=scales,
        log_scale=bool(cfg.head_use_mlp_log_scale),
        log_scale_layers=cfg.head_log_scale_mlp_n_layers,
        log_scale_variant=cfg.e3mlp_variant,
    )
    stages.append(
        _stage(
            "project",
            "pair_projection",
            [hidden_slot, "partitions", "roles"],
            ["out"],
            projection,
        )
    )
    return {
        "summary": "Per-role trunk MLP and per species-pair projection to orbital irreps",
        "inputs": ["node", "edge", "edge_index", "partitions", "roles"],
        "outputs": ["out"],
        "stages": stages,
    }


ENVELOPE_WEIGHTING = frozenset(
    {"envelope_inverse_sqrt_clipped", "envelope_inverse_clipped"}
)


def _envelope_stage(cfg, emit_r0):
    """The pair-envelope stage, when anything in ``cfg`` uses the envelope."""
    path = getattr(cfg, "hamiltonian_envelope_path", None)
    mode = str(getattr(cfg, "hamiltonian_envelope_mode", "off")).lower()
    if mode not in {"off", "normalize_target", "multiply_prediction"}:
        raise ValueError(
            "hamiltonian_envelope_mode must be one of 'off', "
            "'normalize_target', or 'multiply_prediction'."
        )
    weighting = str(getattr(cfg, "loss_weighting_mode", "off")).lower()
    if not (mode != "off" or emit_r0 or weighting in ENVELOPE_WEIGHTING):
        return None
    if path is None:
        raise ValueError(
            "hamiltonian_envelope_path is required when envelope-based modes "
            "are enabled."
        )
    options = {"path": str(path), "emit_r0": emit_r0}
    order = (getattr(cfg, "species", None) or {}).get("order")
    if order:
        options["species"] = list(order)
    return _stage(
        "envelope",
        "pair_envelope",
        ["edge_lengths", "edge_type_idx"],
        ["edge_envelope", "edge_r0"] if emit_r0 else ["edge_envelope"],
        options,
        role="geometry",
    )


def _scale_by_envelope(graph, cfg):
    """Scale predicted operators by the envelope and expose the envelope.

    With an envelope mode the network predicts blocks relative to the fitted
    envelope; loss weighting and target normalization read ``edge_envelope``.
    """
    mode = str(getattr(cfg, "hamiltonian_envelope_mode", "off")).lower()
    outputs = graph["outputs"]
    if mode != "off":
        for operator in ("hamiltonian", "overlap"):
            if operator not in outputs:
                continue
            slot = outputs[operator]
            graph["stages"].append(
                _stage(
                    f"envelope_{operator}",
                    "envelope_scale",
                    [slot, "edge_envelope", "edge_partitions"],
                    [f"{slot}_physical"],
                    role="readout",
                )
            )
            outputs[operator] = f"{slot}_physical"
    outputs["edge_envelope"] = "edge_envelope"


def with_envelope(pipeline, cfg):
    """``pipeline`` with the stages the flat envelope settings ask for.

    A pipeline that already evaluates ``pair_envelope`` is left as it is.
    """
    from net.block_library import merged_blocks

    blocks = merged_blocks(pipeline)
    stages = [
        *pipeline.get("stages", ()),
        *(s for b in blocks.values() for s in b["stages"]),
    ]
    if any(s.get("transform") == "pair_envelope" for s in stages):
        return pipeline
    envelope = _envelope_stage(cfg, emit_r0=False)
    if envelope is None:
        return pipeline
    graph = {
        **pipeline,
        "stages": [envelope, *pipeline["stages"]],
        "outputs": dict(pipeline["outputs"]),
    }
    _scale_by_envelope(graph, cfg)
    return graph


def default_graph(cfg, num_species):
    """The explicit recipe of the default MANDALA E3GNN for ``cfg``."""
    normalization = str(getattr(cfg, "pair_distance_normalization", "off")).lower()
    if normalization not in {"off", "pair_r0"}:
        raise ValueError("pair_distance_normalization must be 'off' or 'pair_r0'")
    hidden = str(resolve_hidden_irreps(cfg))
    pair_conditioned = bool(getattr(cfg, "pair_conditioned_radial_mlp", False))
    scale = (
        math.sqrt(cfg.n_radial)
        if cfg.radial_embedding_scale == "sqrt_n_radial"
        else 1.0
    )
    blocks = {
        "edge_encoder": edge_encoder_block(cfg, hidden),
        "node_update": node_update_block(cfg, pair_conditioned),
        "edge_update": edge_update_block(cfg, pair_conditioned),
        "readout": readout_block(cfg, hidden),
    }
    lengths, end, inputs = "edge_lengths", cfg.cutoff_radius, {}
    stages = []
    envelope = _envelope_stage(cfg, normalization == "pair_r0")
    if envelope is not None:
        stages.append(envelope)
    if normalization == "pair_r0":
        # Distances in units of the fitted pair radius r0 of the envelope table.
        stages.append(
            _stage(
                "relative_lengths",
                "length_ratio",
                ["edge_lengths", "edge_r0"],
                ["relative_lengths"],
                role="geometry",
            )
        )
        lengths, end = "relative_lengths", 1.0
    stages += [
        _stage(
            "radial",
            "soft_one_hot_basis",
            [lengths],
            ["radial"],
            {
                "number": cfg.n_radial,
                "end": end,
                "basis": "gaussian",
                "cutoff": False,
                "scale": scale,
            },
            role="geometry",
        ),
        _stage(
            "harmonics",
            "spherical_harmonics",
            ["edge_vectors"],
            ["harmonics"],
            {"irreps": str(Irreps.spherical_harmonics(cfg.l_max))},
            role="geometry",
        ),
        _stage(
            "roles",
            "edge_roles",
            ["edge_index", "edge_shift"],
            ["roles"],
            {"separate_shifted_self": bool(cfg.separate_shifted_self)},
            role="geometry",
        ),
        _stage(
            "node_encoder",
            "embedding",
            ["node_type_idx"],
            ["node_0"],
            {"width": cfg.hidden_base_dim},
            role="encoder",
            initialization={"kind": "normal", "std": 0.2},
        ),
        _stage(
            "edge_encoder",
            "edge_encoder",
            ["edge_type_idx", "radial", "harmonics"],
            ["edge_0"],
            role="encoder",
        ),
    ]
    for layer in range(cfg.num_layers_gnn):
        extra_node = ["node_one_hot"] if cfg.use_self_connection else []
        extra_edge = ["edge_one_hot"] if cfg.use_self_connection else []
        pair = ["edge_type_idx"] if pair_conditioned else []
        geometry = ["edge_index", "harmonics", "radial"]
        stages.append(
            _stage(
                f"node_update_{layer}",
                "node_update",
                [f"node_{layer}", f"edge_{layer}", *geometry, *extra_node, *pair],
                [f"node_{layer + 1}"],
                {"irreps_out": hidden},
                role="message",
            )
        )
        stages.append(
            _stage(
                f"edge_update_{layer}",
                "edge_update",
                [f"node_{layer + 1}", f"edge_{layer}", *geometry, *extra_edge, *pair],
                [f"edge_{layer + 1}"],
                {"irreps_out": hidden},
                role="message",
            )
        )
    final = cfg.num_layers_gnn
    outputs = {}
    for operator in cfg.matrix_targets:
        slot = f"{operator}_vectors"
        stages.append(
            _stage(
                f"readout_{operator}",
                "readout",
                [
                    f"node_{final}",
                    f"edge_{final}",
                    "edge_index",
                    "edge_partitions",
                    "roles",
                ],
                [slot],
                role="readout",
            )
        )
        outputs[operator] = slot
    graph = {"blocks": blocks, "stages": stages, "outputs": outputs}
    if envelope is not None:
        _scale_by_envelope(graph, cfg)
    if inputs:
        graph["inputs"] = inputs
    return graph


# Transforms of the monolithic lowering stored in configurations saved before
# the modular decomposition; such a pipeline is re-lowered from the flat fields.
MONOLITHIC_TRANSFORMS = frozenset(
    {
        "species_irrep_embedding",
        "pair_radial_angular_embedding",
        "endpoint_irrep_node",
        "endpoint_irrep_edge",
        "partitioned_irrep_readout",
    }
)


def _is_monolithic_lowering(pipeline):
    return any(
        stage.get("transform") in MONOLITHIC_TRANSFORMS
        for stage in pipeline.get("stages", ())
    )


def pipeline_specification(cfg, mapper):
    """The pipeline a configuration describes, with layout variables resolved.

    Loads the configuration's plugins, lowers a flat configuration and
    resolves packed-shell layout variables against ``mapper``'s species.
    """
    import torch

    from net.common import Config
    from net.plugins import load_plugins

    load_plugins(cfg.plugins)
    pipeline = cfg.pipeline
    if pipeline is None or _is_monolithic_lowering(pipeline):
        pipeline = default_graph(cfg, len(mapper.orbital_cfg.elements()))
    else:
        pipeline = with_envelope(pipeline, cfg)
    if cfg.data.get("representation") == "packed_shells" and not pipeline.get(
        "operator_layout"
    ):
        shells = {
            species: tuple(ir.l for mul, ir in irreps for _ in range(mul))
            for species, irreps in mapper.orbital_cfg.element_to_irreps.items()
        }
        with torch.random.fork_rng(devices=[]):
            resolved, _ = Config.resolve(
                {**cfg.to_mapping(), "pipeline": pipeline},
                species_shells=shells,
                spinful=cfg.layout.get("spinful", False),
            )
        pipeline = resolved["pipeline"]
    return pipeline


def static_feature_pipeline(cfg, mapper):
    """The structure-only stages of ``cfg``'s pipeline, for per-sample caching."""
    from net.pipeline import FeaturePipeline

    return FeaturePipeline(
        pipeline_specification(cfg, mapper),
        num_species=len(mapper.orbital_cfg.elements()),
        mapper=mapper,
        static_only=True,
    )
