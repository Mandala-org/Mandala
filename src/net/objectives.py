"""Training objectives as symbolic expressions over registered terms.

A recipe states the loss as named terms and an expression combining them::

    objective:
      terms:
        H: {term: matrix_error, operator: hamiltonian, l1_fraction: 1.0}
        S: {term: matrix_error, operator: overlap}
        E: {term: band_energy}
      expression: "H + 0.5 * S + 1e-3 * E"

A term is a function of an :class:`ObjectiveContext` (predictions, targets,
observables and model of one batch) registered with
:func:`register_objective_term`. The expression language contains numbers,
term labels, ``+ - * /``, unary minus, parentheses and the functions
``sqrt log exp abs min max``. Flat configurations lower to the same form
(:func:`default_objective`), so the trainer has a single loss path.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

import torch

from net.registry import REQUIRED, Option


@dataclass(frozen=True)
class ObjectiveTerm:
    name: str
    function: Callable
    summary: str = ""
    category: str = "custom"
    options: tuple = ()
    operators: tuple = ()
    requires: tuple = ()

    def operator_inputs(self, options):
        """``{option name: operator}`` of the predicted operators the term reads."""
        return {name: options[name] for name in self.operators if options.get(name)}

    def resolve(self, given):
        given = dict(given)
        known = {option.name for option in self.options}
        unknown = set(given) - known
        if unknown:
            raise ValueError(
                f"Unknown options for objective term {self.name}: "
                + ", ".join(sorted(unknown))
            )
        resolved = {}
        for option in self.options:
            if option.name in given:
                if option.choices and given[option.name] not in option.choices:
                    raise ValueError(
                        f"{self.name}.{option.name} must be one of {list(option.choices)}"
                    )
                resolved[option.name] = given[option.name]
            elif option.default is REQUIRED:
                raise ValueError(
                    f"Objective term {self.name} requires option {option.name!r}"
                )
            else:
                resolved[option.name] = option.default
        return resolved

    def to_json(self):
        return {
            "name": self.name,
            "summary": self.summary,
            "category": self.category,
            "options": [option.to_json() for option in self.options],
            "operators": list(self.operators),
            "requires": list(self.requires),
            # Settings that make the run compute what the term reads.
            "settings": sorted(
                {
                    switch
                    for quantity in self.requires
                    for switch in REQUIREMENTS.get(quantity, ())
                }
            ),
        }


_TERMS: dict[str, ObjectiveTerm] = {}


def register_objective_term(
    name,
    *,
    summary="",
    category="custom",
    options=(),
    operators=(),
    requires=(),
    override=False,
):
    """Register ``function(context, **options) -> scalar tensor`` as a term.

    ``operators`` names the options that select a predicted operator (the
    term's input ports, e.g. ``("operator",)``). ``requires`` names
    quantities the trainer must compute for the term (keys of
    ``REQUIREMENTS``); models check them when they are built.
    """
    operators = tuple(operators)
    declared = {option.name for option in options}
    if set(operators) - declared:
        raise ValueError(
            f"Objective term {name}: operator ports {sorted(set(operators) - declared)} "
            "must be declared options"
        )

    def decorator(function):
        if name in _TERMS and not override:
            raise ValueError(f"Objective term already registered: {name}")
        _TERMS[name] = ObjectiveTerm(
            name,
            function,
            summary or (function.__doc__ or "").strip().split("\n")[0],
            category,
            tuple(options),
            operators,
            tuple(requires),
        )
        return function

    return decorator


# Quantity -> configuration switch that makes the trainer compute it.
REQUIREMENTS = {
    "energy": ("enable_energy",),
    "num_electrons": ("enable_num_electrons",),
    "forces": ("enable_forces",),
    "stress": ("enable_stress",),
    "spectral": ("spectral_loss_enabled", "train_on_spectral"),
}


def objective_catalog():
    return [
        term.to_json()
        for term in sorted(_TERMS.values(), key=lambda t: (t.category, t.name))
    ]


@dataclass
class ObjectiveContext:
    """Everything a term may read for one batch.

    ``components`` holds quantities the trainer computes anyway (matrix MSE and
    MAE per operator, observable errors, forces, stress, spectral loss), so
    built-in terms cost nothing extra. ``outputs`` are the raw pipeline outputs.
    """

    model: Any
    x: Mapping
    y: Mapping
    stage: str
    predictions: Mapping = field(default_factory=dict)
    matrices: Mapping = field(default_factory=dict)
    targets: Mapping = field(default_factory=dict)
    outputs: Mapping = field(default_factory=dict)
    components: Mapping = field(default_factory=dict)

    def component(self, key, term):
        if key not in self.components:
            raise ValueError(
                f"Objective term {term} needs {key!r}, which this run does not compute"
            )
        return self.components[key]

    def zero(self):
        return torch.zeros((), device=self.model.device)


_FUNCTIONS = {
    "sqrt": torch.sqrt,
    "log": torch.log,
    "exp": torch.exp,
    "abs": torch.abs,
    "min": torch.minimum,
    "max": torch.maximum,
}
_BINARY = {
    ast.Add: torch.add,
    ast.Sub: torch.sub,
    ast.Mult: torch.mul,
    ast.Div: torch.div,
}


def parse_expression(expression):
    """Parse and validate; returns ``(tree, referenced labels)``."""
    tree = ast.parse(str(expression), mode="eval")
    labels = set()

    def visit(node):
        if isinstance(node, ast.Expression):
            return visit(node.body)
        if isinstance(node, ast.BinOp) and type(node.op) in _BINARY:
            visit(node.left)
            visit(node.right)
        elif isinstance(node, ast.UnaryOp) and isinstance(
            node.op, (ast.USub, ast.UAdd)
        ):
            visit(node.operand)
        elif isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            pass
        elif isinstance(node, ast.Name):
            labels.add(node.id)
        elif (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in _FUNCTIONS
            and not node.keywords
        ):
            arity = 2 if node.func.id in {"min", "max"} else 1
            if len(node.args) != arity:
                raise ValueError(f"{node.func.id} takes {arity} argument(s)")
            for argument in node.args:
                visit(argument)
        else:
            raise ValueError(
                f"Unsupported objective expression element: {ast.dump(node)}"
            )

    visit(tree)
    return tree, labels


def evaluate_expression(tree, values, reference):
    def scalar(value):
        return torch.as_tensor(
            float(value), dtype=reference.dtype, device=reference.device
        )

    def visit(node):
        if isinstance(node, ast.Expression):
            return visit(node.body)
        if isinstance(node, ast.BinOp):
            return _BINARY[type(node.op)](visit(node.left), visit(node.right))
        if isinstance(node, ast.UnaryOp):
            value = visit(node.operand)
            return -value if isinstance(node.op, ast.USub) else value
        if isinstance(node, ast.Constant):
            return scalar(node.value)
        if isinstance(node, ast.Name):
            return values[node.id]
        return _FUNCTIONS[node.func.id](*(visit(argument) for argument in node.args))

    return visit(tree)


class Objective:
    """A compiled objective: resolved terms and a parsed expression."""

    def __init__(self, specification):
        specification = dict(specification)
        terms = dict(specification.get("terms") or {})
        if not terms:
            raise ValueError("objective.terms must define at least one term")
        self.terms = {}
        for label, value in terms.items():
            if not str(label).isidentifier():
                raise ValueError(
                    f"Objective term label {label!r} must be an identifier"
                )
            value = dict(value)
            name = value.pop("term", None)
            if name not in _TERMS:
                raise ValueError(f"Unknown objective term {name!r} (label {label})")
            self.terms[label] = (_TERMS[name], _TERMS[name].resolve(value))
        expression = specification.get("expression") or " + ".join(self.terms)
        self.expression = str(expression)
        self.tree, labels = parse_expression(self.expression)
        undefined = labels - self.terms.keys()
        if undefined:
            raise ValueError(
                "Objective expression uses "
                + ", ".join(sorted(undefined))
                + ", not a term of the objective (terms: "
                + (", ".join(self.terms) or "none")
                + ")"
            )

    def requirements(self):
        return {
            quantity for term, _ in self.terms.values() for quantity in term.requires
        }

    def check_requirements(self, cfg):
        """Reject objectives whose terms read quantities the run does not compute."""
        missing = sorted(
            f"{switch} (for {quantity})"
            for quantity in self.requirements()
            for switch in REQUIREMENTS[quantity]
            if not getattr(cfg, switch, False)
        )
        if missing:
            raise ValueError(
                "The objective needs " + ", ".join(missing) + " to be enabled"
            )

    def evaluate(self, context):
        """``(total loss, {label: term value})``."""
        values = {
            label: term.function(context, **options)
            for label, (term, options) in self.terms.items()
        }
        reference = next(iter(values.values()))
        return evaluate_expression(self.tree, values, reference), values


def default_objective(cfg):
    """The objective of a flat configuration as terms and an expression."""
    options = cfg.objective or {}
    if "terms" in options:
        return {"terms": options["terms"], "expression": options.get("expression")}
    weights = options.get("weights", {})
    terms, parts = {}, []
    for operator in cfg.matrix_targets:
        label = f"matrix_{operator}"
        terms[label] = {"term": "matrix_error", "operator": operator}
        parts.append(f"{float(weights.get(operator, 1.0))!r} * {label}")
    coefficient = float(cfg.loss_coef_observables)
    if cfg.train_on_energy and coefficient:
        terms["energy"] = {"term": "band_energy"}
        parts.append(f"{coefficient!r} * energy")
    if cfg.train_on_num_electrons and coefficient:
        terms["electrons"] = {"term": "electron_count"}
        parts.append(f"{coefficient!r} * electrons")
    if cfg.enable_forces and cfg.train_on_forces and cfg.loss_coef_forces:
        terms["forces"] = {"term": "forces"}
        parts.append(f"{float(cfg.loss_coef_forces)!r} * forces")
    if cfg.enable_stress and cfg.train_on_stress and cfg.loss_coef_stress:
        terms["stress"] = {"term": "stress"}
        parts.append(f"{float(cfg.loss_coef_stress)!r} * stress")
    if getattr(cfg, "spectral_loss_enabled", False) and getattr(
        cfg, "train_on_spectral", True
    ):
        if float(cfg.spectral_loss_coef):
            terms["spectral"] = {"term": "spectral"}
            parts.append(f"{float(cfg.spectral_loss_coef)!r} * spectral")
    if cfg.l1_reg_coef > 0:
        terms["l1"] = {"term": "parameter_norm", "power": 1}
        parts.append(f"{float(cfg.l1_reg_coef)!r} * l1")
    if cfg.l2_reg_coef > 0:
        terms["l2"] = {"term": "parameter_norm", "power": 2}
        parts.append(f"{float(cfg.l2_reg_coef)!r} * l2")
    return {"terms": terms, "expression": " + ".join(parts)}


def _optional(context, key, term):
    value = context.components.get(key)
    return context.zero() if value is None else value


@register_objective_term(
    "matrix_error",
    category="operator",
    summary="Element error of a predicted operator: (1-f) MSE + f MAE",
    operators=("operator",),
    options=[
        Option("operator", "str", "hamiltonian", key=True),
        Option(
            "l1_fraction",
            "float",
            None,
            doc="Weight f of the MAE part (default: the configuration's loss_l1_fraction)",
        ),
    ],
)
def _matrix_error(context, operator, l1_fraction):
    fraction = (
        context.model.objective_l1_fraction()
        if l1_fraction is None
        else float(l1_fraction)
    )
    mse = context.component(f"matrix_mse:{operator}", "matrix_error")
    mae = context.component(f"matrix_mae:{operator}", "matrix_error")
    return (1 - fraction) * mse + fraction * mae


@register_objective_term(
    "band_energy",
    category="observable",
    requires=("energy",),
    summary="Error of the band energy Tr(DH) against the reference energy",
)
def _band_energy(context):
    return _optional(context, "energy", "band_energy")


@register_objective_term(
    "electron_count",
    category="observable",
    requires=("num_electrons",),
    summary="Error of the electron count Tr(DS) against the reference",
)
def _electron_count(context):
    return _optional(context, "num_electrons", "electron_count")


@register_objective_term(
    "forces",
    category="observable",
    summary="MSE of Hellmann-Feynman forces",
    requires=("forces",),
)
def _forces(context):
    return _optional(context, "forces", "forces")


@register_objective_term(
    "stress",
    category="observable",
    summary="MSE of the stress tensor",
    requires=("stress",),
)
def _stress(context):
    return _optional(context, "stress", "stress")


@register_objective_term(
    "spectral",
    category="observable",
    requires=("spectral",),
    summary="Tapered eigenvalue loss near the Fermi level (needs spectral_loss_enabled)",
)
def _spectral(context):
    return context.component("spectral", "spectral")


@register_objective_term(
    "parameter_norm",
    category="regularization",
    summary="Sum of the p-th power of the absolute value of every parameter",
    options=[Option("power", "choice", 2, key=True, choices=(1, 2))],
)
def _parameter_norm(context, power):
    parameters = list(context.model.parameters())
    if power == 1:
        return sum(p.abs().sum() for p in parameters)
    return sum(p.pow(2).sum() for p in parameters)


@register_objective_term(
    "output_error",
    category="output",
    summary="Error between a raw pipeline output and a batch target of the same shape",
    options=[
        Option("output", "str", key=True),
        Option("target", "str", key=True),
        Option("kind", "choice", "mse", key=True, choices=("mse", "mae")),
    ],
)
def _output_error(context, output, target, kind):
    if output not in context.outputs:
        raise ValueError(f"output_error: the pipeline has no output {output!r}")
    if context.y.get(target) is None:
        return context.zero()
    prediction, reference = context.outputs[output], context.y[target].to(
        context.outputs[output]
    )
    difference = prediction.reshape(reference.shape) - reference
    return difference.abs().mean() if kind == "mae" else difference.pow(2).mean()


def check_objective(specification):
    """Validate a recipe objective without a model (for tools and resolution)."""
    Objective(specification)
    return True


__all__ = [
    "Objective",
    "ObjectiveContext",
    "default_objective",
    "objective_catalog",
    "parse_expression",
    "register_objective_term",
]
