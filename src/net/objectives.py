"""Stateless, differentiable additions to the per-structure training loss.

Import a plugin before constructing an E3GNN to register its objective terms.
Existing matrix, observable and spectral objectives keep their usual settings.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import partial
import math
import re
from typing import Callable

import torch


_OBJECTIVE_TERMS = {}


def register_objective_term(name):
    """Register a function of keyword arguments ``predictions, targets, inputs``.

    Predictions and targets map operator names to physical ``BlockMatrix``
    objects in the same orbital basis (Hamiltonians in Hartree). ``inputs``
    contains one structure's graph metadata. Additional keyword arguments come
    from the term's YAML ``options``. Return one real scalar tensor, retaining
    the prediction's autograd graph. Terms have no learnable parameters.
    """
    if not isinstance(name, str) or re.fullmatch(r"[a-z][a-z0-9_]*", name) is None:
        raise ValueError("Objective term names must match [a-z][a-z0-9_]*")

    def decorator(function):
        if name in _OBJECTIVE_TERMS:
            raise ValueError(f"Objective term already registered: {name}")
        _OBJECTIVE_TERMS[name] = function
        return function

    return decorator


@dataclass(frozen=True)
class ObjectiveTerm:
    name: str
    weight: float
    function: Callable

    def __call__(self, *, predictions, targets, inputs):
        value = self.function(predictions=predictions, targets=targets, inputs=inputs)
        if (
            not isinstance(value, torch.Tensor)
            or value.ndim != 0
            or not value.is_floating_point()
        ):
            raise ValueError(
                f"Objective term {self.name!r} must return a real scalar tensor"
            )
        if not torch.isfinite(value):
            raise ValueError(f"Objective term {self.name!r} returned a nonfinite loss")
        return value


def build_objective_terms(objective):
    """Validate ``objective.terms`` and bind each registered function's options."""
    specifications = (objective or {}).get("terms", [])
    if not isinstance(specifications, Sequence) or isinstance(specifications, str):
        raise ValueError("objective.terms must be a list")
    terms, names = [], set()
    for specification in specifications:
        if not isinstance(specification, Mapping):
            raise ValueError("Each objective term must be a mapping")
        if set(specification) - {"name", "weight", "options"}:
            raise ValueError("Unknown objective term options")
        name = specification.get("name")
        if not isinstance(name, str) or name not in _OBJECTIVE_TERMS:
            raise ValueError(
                f"Unknown objective term {name!r}; import its registration module first"
            )
        if name in names:
            raise ValueError(f"Duplicate objective term: {name}")
        weight = float(specification.get("weight", 1.0))
        if not math.isfinite(weight) or weight < 0:
            raise ValueError(
                f"Objective term {name!r} weight must be finite and nonnegative"
            )
        options = specification.get("options", {})
        if not isinstance(options, Mapping):
            raise ValueError(f"Objective term {name!r} options must be a mapping")
        if set(options) & {"predictions", "targets", "inputs"}:
            raise ValueError(
                "Objective term options cannot replace its input arguments"
            )
        terms.append(
            ObjectiveTerm(name, weight, partial(_OBJECTIVE_TERMS[name], **options))
        )
        names.add(name)
    return terms
