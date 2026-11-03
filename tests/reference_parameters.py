"""Upstream parameter names of the decomposed reference-model pipelines.

The DeepH, DeepH-E3 and MACE-H recipes are compositions of library blocks; the
pinned upstream implementations name the same parameters by module path. These
helpers translate pipeline parameter names for the replay tests.
"""

# Inner stages of the decomposed endpoint updates -> source attribute paths.
_E3_UPDATE = {
    "pre": "lin_pre",
    "product": "conv",
    "radial": "conv",
    "post": "lin_post",
    "connection": "sc",
    "norm": "norm",
    "residual": "skip_connect.sc",
    "square": "self_tp",
}


def _strip(tail, *prefixes):
    for prefix in prefixes:
        if tail.startswith(prefix + "."):
            return tail[len(prefix) + 1 :]
    return tail


def source_name(kind, name):
    """Source parameter name of a decomposed pipeline parameter."""
    _, stage, tail = name.split(".", 2)
    instance, _, inner = stage.partition("/")
    index = instance.rsplit("_", 1)[-1]
    if kind == "deeph":
        if stage == "species_embedding":
            return "embed." + tail
        tail = tail.replace("layers.", "") if inner in {"mlp", "readout"} else tail
        if inner == "norm":
            tail = "ln." + _strip(tail, "norm")
        prefix = (
            "lcmp"
            if instance == "local_coordinate_readout"
            else f"mp{index}" + (".cgconv" if inner in {"message", "norm"} else "")
        )
        if inner in {"mlp", "readout"}:
            return f"{prefix}.e_lin.{tail}"
        return f"{prefix}.{tail}"
    if inner in _E3_UPDATE:
        source = _E3_UPDATE[inner]
        if inner in {"connection", "residual", "square"}:
            tail = _strip(tail, "tp", "project")
        if kind == "deeph_e3":
            block = (
                "node_update_blocks"
                if instance.startswith("node")
                else "edge_update_blocks"
            )
        else:
            block = "edge_update.edge_update_blocks"
        if instance.startswith(("node_update_", "edge_update_")):
            return f"{block}.{index}.{source}.{tail}"
    if kind == "deeph_e3":
        prefix = {
            "embedding": "embedding",
            "node_readout": "lin_node",
            "edge_readout": "lin_edge",
        }[stage]
        return f"{prefix}.{tail}"
    if stage in {"node_embedding", "density_radial_basis", "edge_readout"}:
        prefix = {
            "node_embedding": "mace.node_embedding.linear",
            "density_radial_basis": "mace.radial_embedding",
            "edge_readout": "edge_update.lin_edge",
        }[stage]
        return f"{prefix}.{tail}"
    if stage == "first_expansion":
        return f"mace.interactions.0.linear_up_expansion.{tail}"
    if instance.startswith("skip_"):
        return f"mace.interactions.{index}.skip_tp.{tail}"
    if instance.startswith("density_"):
        source = {
            "lift": "linear_up",
            "convolution": "",
            "project": "linear",
            "norm": "norm",
        }[inner]
        return f"mace.interactions.{index}." + (f"{source}." if source else "") + tail
    if instance.startswith("product_"):
        source = {"contraction": "symmetric_contractions", "mix": "linear"}[inner]
        return f"mace.products.{index}.{source}.{tail}"
    if instance.startswith("angular_expansion_"):
        source = {
            "hidden_projection": "hidden_linear",
            "angular_projection": "up_linear",
            "product": "",
            "post": "lin_post",
            "connection": "sc",
            "norm": "norm",
            "skip": "skip_connect",
        }[inner]
        tail = (
            _strip(tail, "tp", "project") if inner in {"connection", "skip"} else tail
        )
        prefix = f"expansion.node_order_expansion_blocks.{index}."
        return prefix + (f"{source}." if source else "") + tail
    raise AssertionError(f"Unmapped parameter {name}")


def unused_source_parameters(record):
    """Source parameters that never receive a gradient (no counterpart needed)."""
    return {
        name
        for name in record["initial_parameters"]
        if all(step["gradients"][name] is None for step in record["steps"])
    }
