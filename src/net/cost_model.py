"""Asymptotic cost estimates for pipeline stages.

A stage costs ``entities x operations-per-entity`` multiply-adds, where the
entity count is the largest index set the stage reads or writes, and the
per-entity count is read off the e3nn/torch submodules it owns. Measured
seconds-per-operation constants (``cost_constants.json``, written by
``scripts/benchmarks/pipeline_benchmark.py``) turn operations into time.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from torch import nn

CONSTANTS_PATH = Path(__file__).with_name("cost_constants.json")


@dataclass(frozen=True)
class SystemSize:
    """A representative structure for cost estimates."""

    atoms: int = 64
    neighbors: float = 40.0
    message_neighbors: float | None = None
    graphs: int = 1

    def entities(self, spec):
        edges = self.atoms * self.neighbors
        counts = {
            "node": self.atoms,
            "graph": self.graphs,
            "edge": edges,
            "relation": edges,
            "output": edges,
            "message": self.atoms * (self.message_neighbors or self.neighbors),
            "bond_neighbor": edges * self.neighbors,
        }
        support = spec.support if spec.domain != "node" else None
        if support and support not in counts:
            return edges
        return counts[support or spec.domain]


def _tensor_product_operations(tp):
    total = 0
    for ins in tp.instructions:
        mul_1, ir_1 = tp.irreps_in1[ins.i_in1]
        mul_2, ir_2 = tp.irreps_in2[ins.i_in2]
        mul_out, ir_out = tp.irreps_out[ins.i_out]
        coupling = ir_1.dim * ir_2.dim * ir_out.dim
        if ins.connection_mode == "uvw":
            total += mul_1 * mul_2 * coupling + mul_1 * mul_2 * mul_out * ir_out.dim
        elif ins.connection_mode in {"uvu", "uvv"}:
            total += mul_1 * mul_2 * coupling
        else:
            total += mul_1 * coupling
    return total


def _linear_operations(linear):
    return sum(
        linear.irreps_in[ins.i_in].mul
        * linear.irreps_out[ins.i_out].mul
        * linear.irreps_out[ins.i_out].ir.dim
        for ins in linear.instructions
    )


def operations_per_entity(module):
    """Multiply-adds per entity for ``module`` and its children."""
    from e3nn import o3

    if hasattr(module, "operations_per_entity"):
        return float(module.operations_per_entity())
    if isinstance(module, o3.TensorProduct):
        return float(_tensor_product_operations(module))
    if isinstance(module, o3.Linear):
        return float(_linear_operations(module))
    if isinstance(module, o3.SphericalHarmonics):
        return float(4 * module.irreps_out.dim)
    if isinstance(module, nn.Linear):
        return float(module.in_features * module.out_features)
    if isinstance(module, nn.Embedding):
        return float(module.embedding_dim)
    own = sum(p.numel() for p in module.parameters(recurse=False))
    return float(own) + sum(operations_per_entity(child) for child in module.children())


def load_constants(path=CONSTANTS_PATH):
    try:
        return json.loads(Path(path).read_text())
    except FileNotFoundError:
        return {
            "default": {"seconds_per_operation": 2.0e-10, "overhead_seconds": 2.0e-5}
        }


def stage_cost(transform, module, inputs, outputs, size, constants=None):
    """``(operations, seconds)`` for one stage on a structure of ``size``."""
    from net.registry import get_transform

    specs = [*inputs, *outputs]
    entities = max((size.entities(spec) for spec in specs), default=1)
    try:
        info = get_transform(transform)
    except ValueError:  # a disabled block instance is a bypass
        info = None
    if info is not None and info.cost is not None:
        operations = float(info.cost(module, inputs, outputs, size))
    else:
        operations = entities * operations_per_entity(module)
    constants = constants or load_constants()
    rate = constants.get("transforms", {}).get(transform, constants["default"])
    seconds = rate.get("overhead_seconds", constants["default"]["overhead_seconds"]) + (
        operations
        * rate.get(
            "seconds_per_operation", constants["default"]["seconds_per_operation"]
        )
    )
    return operations, seconds
