"""Fitted per species-pair distance envelopes of operator blocks.

``pair_envelope`` evaluates an envelope table (``data.envelope``) on every
edge; ``envelope_scale`` multiplies predicted operator blocks by it, so the
network predicts the operator relative to its typical distance decay::

    - {id: envelope, transform: pair_envelope, inputs: [edge_lengths, edge_type_idx],
       outputs: [edge_envelope], options: {path: envelopes.json}}
    - {id: physical, transform: envelope_scale,
       inputs: [blocks, edge_envelope, edge_partitions], outputs: [hamiltonian]}

The envelope depends on the structure alone and is cached with the sample.
"""

from __future__ import annotations

import torch
from torch import nn

from net.features import FeatureSpec, index_spec, same_support
from net.registry import Option, Port, register_transform


class PairEnvelope(nn.Module):
    def __init__(self, path, species, emit_r0=False):
        super().__init__()
        from data.envelope import load_pair_envelope_table

        pairs = [f"{left}-{right}" for left in species for right in species]
        table = load_pair_envelope_table(
            path, pair_order=pairs, dtype=torch.float64, device="cpu"
        )
        self.family = table.family
        self.r0_index = table.r0_parameter_index
        if emit_r0 and self.r0_index is None:
            raise ValueError(f"Envelope family {self.family!r} defines no r0")
        self.emit_r0 = bool(emit_r0)
        # Not part of checkpoints: the table is configuration, not a weight.
        self.register_buffer("parameters_", table.parameters, persistent=False)
        self.register_buffer("reference_x_max", table.reference_x_max, persistent=False)

    def forward(self, lengths, pair_type):
        from data.envelope import PairEnvelopeTable, build_edge_envelope

        lengths = lengths.reshape(-1)
        table = PairEnvelopeTable(
            family=self.family,
            pair_keys=(),
            parameters=self.parameters_.to(lengths.dtype),
            reference_x_max=self.reference_x_max.to(lengths.dtype),
            r0_parameter_index=self.r0_index,
        )
        envelope = build_edge_envelope(lengths, pair_type, table)
        if not self.emit_r0:
            return envelope
        return envelope, table.r0.index_select(0, pair_type)


class EnvelopeScale(nn.Module):
    """Multiply every edge's block by its envelope value."""

    def forward(self, blocks, envelope, partitions=None):
        envelope = envelope.reshape(-1)
        if isinstance(blocks, dict):
            return {
                key: value
                * envelope.index_select(0, partitions[key]["global_idx"]).reshape(
                    -1, *([1] * (value.ndim - 1))
                )
                for key, value in blocks.items()
            }
        return blocks * envelope.reshape(-1, *([1] * (blocks.ndim - 1)))


def _species(inputs, mapper, options):
    if mapper is None:
        return None
    return list(mapper.orbital_cfg.elements())


@register_transform(
    "pair_envelope",
    category="envelope",
    summary="Fitted per species-pair distance envelope of the operator blocks",
    inputs=[Port("lengths", "edge"), Port("pair_type", "edge", "index")],
    outputs=(
        Port("envelope", "edge", "invariant"),
        Port("r0", "edge", "invariant", optional=True, doc="Only with emit_r0"),
    ),
    options=[
        Option("path", "str", key=True, doc="Envelope table (JSON)"),
        Option(
            "species",
            "list",
            infer=_species,
            infer_doc="species of the orbital configuration, in index order",
            doc="Species in the order of the species indices (edge_type_idx); "
            "a recipe should give ${species.order}: the inferred default is the "
            "orbital configuration's alphabetical order",
        ),
        Option("emit_r0", "bool", False, doc="Also output each pair's r0 distance"),
    ],
)
def _pair_envelope(p, inputs, mapper):
    lengths, pair_type = inputs
    if lengths.domain != "edge" or lengths.dim != 1:
        raise ValueError("pair_envelope needs one length per edge")
    index_spec(pair_type, "edge")
    support = same_support(*inputs)
    count = len(p["species"]) ** 2
    if pair_type.width not in (None, count):
        raise ValueError(
            f"pair_type has {pair_type.width} categories; {count} species pairs expected"
        )
    module = PairEnvelope(p["path"], p["species"], p["emit_r0"])
    spec = FeatureSpec("edge", irreps="1x0e", width=1, support=support)
    return module, [spec, spec] if p["emit_r0"] else [spec]


@register_transform(
    "envelope_scale",
    category="envelope",
    summary="Multiply each edge's operator block by its envelope value",
    inputs=[
        Port("blocks", "edge"),
        Port("envelope", "edge", "invariant"),
        Port(
            "partitions", "relation", "object", optional=True, doc="For block vectors"
        ),
    ],
)
def _envelope_scale(p, inputs, mapper):
    blocks, envelope = inputs[:2]
    if envelope.dim != 1 or not envelope.invariant:
        raise ValueError("The envelope must be one invariant value per edge")
    if blocks.domain != "edge" or blocks.representation not in {
        "matrix",
        "block_vectors",
        "irreps",
        "scalar",
    }:
        raise ValueError("envelope_scale multiplies edge operator blocks")
    if blocks.representation == "block_vectors" and len(inputs) < 3:
        raise ValueError("Block vectors per species pair need the edge partitions")
    same_support(blocks, envelope)
    return EnvelopeScale(), [blocks]
