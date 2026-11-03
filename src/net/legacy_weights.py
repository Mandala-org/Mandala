"""Translate state dicts of the monolithic MANDALA modules to stage weights.

Before the modular decomposition the default model consisted of five
monolithic stages (``node_encoding``, ``edge_encoding``,
``node_message_<i>``, ``edge_message_<i>``, ``readout_<operator>``). Their
parameters map one-to-one onto the stages produced by
``net.component_defaults.default_graph``; this module performs that renaming so
checkpoints trained before the decomposition load into the modular model.
"""

from __future__ import annotations

import re

PREFIX = "pipeline.transforms."

_ENCODER = {
    "node_encoding.elem_emb.": "node_encoder.",
    "edge_encoding.edge_emb.": "edge_encoder/pair_embedding.",
    "edge_encoding.sh_tensor_square.": "edge_encoder/square.",
    "edge_encoding.tp.": "edge_encoder/mix.",
    "edge_encoding.norm.": "edge_encoder/norm.",
    "edge_encoding.distance_proj.": "edge_encoder/project.",
}
_UPDATE = {
    "lin_pre.": "pre.",
    "conv.": "convolution.",
    "lin_post.": "post.",
    "sc.": "self_connection.tp.",
    "norm_act.": "activation.",
    "layer_norm.": "norm.",
    "refine.": "refine.",
    "query_proj.": "aggregate.query_proj.",
    "key_proj.": "aggregate.key_proj.",
    "value_projs.": "aggregate.value_projs.",
    "node_pair_tp.": "gather.pair_tp.",
}
_ROLES = ("diag", "shifted_self", "offdiag")
_MONOLITHIC_STAGE = re.compile(
    r"node_encoding|edge_encoding|(node|edge)_message_\d+|readout_\w+"
)


def _sanitize(key):
    return key.replace("-", "_")


def _readout(rest, separate_shifted_self, onsite_from_nodes):
    route = {"diag": 0, "shifted_self": 1, "offdiag": 2 if separate_shifted_self else 1}
    if onsite_from_nodes and rest.startswith("diag_trunk."):
        return "onsite_trunk." + rest[len("diag_trunk.") :]
    for role in _ROLES:
        for kind, target in (
            ("trunk", "trunk.routes"),
            ("tensor_square", "square.routes"),
        ):
            head = f"{role}_{kind}."
            if rest.startswith(head):
                return f"{target}.{route[role]}.{rest[len(head):]}"
        head = f"shared_{role}_proj."
        if rest.startswith(head):
            return f"project.shared.{route[role]}.{rest[len(head):]}"
        for kind, target in (("projs", "projections"), ("log_scales", "log_scales")):
            match = re.match(rf"{role}_{kind}\.([^.]+)\.(.*)", rest)
            if match:
                key, tail = match.groups()
                return f"project.{target}.{route[role]}_{_sanitize(key)}.{tail}"
    return None


def translate_monolithic_state_dict(
    state_dict, *, separate_shifted_self=True, onsite_from_nodes=True
):
    """Return ``state_dict`` with monolithic MANDALA keys renamed to stage keys.

    The flags are the configuration's ``separate_shifted_self`` and
    ``head_use_node_embeddings_for_self_edges``. Keys that do not belong to the
    monolithic modules are kept unchanged; unknown monolithic keys raise
    ``KeyError``.
    """
    result = {}
    for key, value in state_dict.items():
        if not key.startswith(PREFIX):
            result[key] = value
            continue
        rest = key[len(PREFIX) :]
        if not _MONOLITHIC_STAGE.fullmatch(rest.split(".", 1)[0]):
            result[key] = value
            continue
        renamed = None
        for old, new in _ENCODER.items():
            if rest.startswith(old):
                renamed = new + rest[len(old) :]
        match = re.match(r"(node|edge)_message_(\d+)\.(.*)", rest)
        if match:
            kind, layer, tail = match.groups()
            for old, new in _UPDATE.items():
                if tail.startswith(old):
                    renamed = f"{kind}_update_{layer}/{new}{tail[len(old):]}"
                    break
        match = re.match(r"readout_([A-Za-z0-9_]+?)\.(.*)", rest)
        if match:
            operator, tail = match.groups()
            translated = _readout(tail, separate_shifted_self, onsite_from_nodes)
            if translated is not None:
                renamed = f"readout_{operator}/{translated}"
        if renamed is None:
            raise KeyError(f"No modular counterpart for monolithic weight {key}")
        result[PREFIX + renamed] = value
    return result
