"""Self-describing registry of pipeline transformations.

Every transformation is a factory ``(options, inputs, mapper) -> (module,
output specs)`` registered under a name. Registration may also describe the
transformation — its category, ports, option schema and cost model — which the
compiler uses to complete options and which tools use to present it. A bare
``@register_transform("name")`` remains valid for quick plugins.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

from net.features import FeatureSpec


class _Required:
    def __repr__(self):
        return "REQUIRED"


REQUIRED = _Required()

CATEGORIES = (
    "input",
    "radial_basis",
    "angular_basis",
    "embedding",
    "encoder",
    "geometry",
    "convolution",
    "message",
    "aggregation",
    "nonlinearity",
    "normalization",
    "mlp",
    "linear",
    "residual",
    "combine",
    "product",
    "routing",
    "readout",
    "output",
    "envelope",
    "frame",
    "statistics",
    "custom",
)

PORT_KINDS = (
    "any",
    "irreps",
    "invariant",
    "index",
    "geometry",
    "object",
    "local",
    "matrix",
    "channel",
    "block_vectors",
)


@dataclass(frozen=True)
class Port:
    """One positional input or output of a transformation.

    ``domain`` names the entity set (``None`` accepts any); ``kind`` the
    representation family used for early connection checks. ``variadic``
    marks the last input as repeatable.
    """

    name: str
    domain: str | tuple[str, ...] | None = None
    kind: str = "any"
    optional: bool = False
    variadic: bool = False
    doc: str = ""

    def __post_init__(self):
        if self.kind not in PORT_KINDS:
            raise ValueError(f"Unknown port kind {self.kind!r}")

    def to_json(self):
        domain = self.domain
        return {
            "name": self.name,
            "domain": list(domain) if isinstance(domain, tuple) else domain,
            "kind": self.kind,
            "optional": self.optional,
            "variadic": self.variadic,
            "doc": self.doc,
        }


@dataclass(frozen=True)
class Option:
    """A configurable value of a transformation.

    ``key`` options are shown by default in tools; the rest are advanced.
    ``infer(inputs, mapper, options)`` supplies a value from the wiring when the
    recipe gives none; ``infer_doc`` says where it comes from.
    """

    name: str
    type: str = "any"
    default: Any = REQUIRED
    key: bool = False
    choices: tuple = ()
    infer: Callable | None = None
    infer_doc: str = ""
    doc: str = ""

    def to_json(self):
        default = None if self.default is REQUIRED else self.default
        if isinstance(default, tuple):
            default = list(default)
        return {
            "name": self.name,
            "type": self.type,
            "default": default,
            "required": self.default is REQUIRED and self.infer is None,
            "key": self.key,
            "choices": list(self.choices),
            "inferred": self.infer is not None,
            "infer_doc": self.infer_doc,
            "doc": self.doc,
        }


@dataclass
class TransformResult:
    """What a factory builds; ``refined_inputs`` tightens input contracts."""

    module: Any
    outputs: list
    refined_inputs: dict = field(default_factory=dict)


@dataclass
class BuiltStage:
    module: Any
    outputs: list
    options: dict
    explicit: frozenset
    inferred: frozenset
    refined_inputs: dict


@dataclass(frozen=True)
class TransformInfo:
    name: str
    factory: Callable
    category: str = "custom"
    summary: str = ""
    inputs: tuple = ()
    outputs: tuple = ("output",)
    options: tuple = ()
    cost: Callable | None = None
    pure: bool = True
    strict: bool = False
    bypass: int = 0

    @property
    def output_ports(self):
        return tuple(
            port if isinstance(port, Port) else Port(port) for port in self.outputs
        )

    @property
    def described(self):
        return bool(self.inputs or self.options)

    def option(self, name):
        return next((option for option in self.options if option.name == name), None)

    def resolve_options(self, given, inputs, mapper):
        """Complete ``given`` from inference rules and defaults.

        Returns ``(options, explicit, inferred)``. Explicit values always win;
        inference runs in declaration order and may read earlier values.
        """
        given = dict(given or {})
        unknown = set(given) - {option.name for option in self.options}
        if self.strict and unknown:
            raise ValueError(f"Unknown options for {self.name}: {sorted(unknown)}")
        resolved = dict(given)
        inferred = set()
        for option in self.options:
            if option.name in given:
                if option.choices and given[option.name] not in option.choices:
                    raise ValueError(
                        f"{self.name}.{option.name} must be one of {list(option.choices)}"
                    )
                continue
            if option.infer is not None:
                value = option.infer(inputs, mapper, resolved)
                if value is not None:
                    resolved[option.name] = value
                    inferred.add(option.name)
                    continue
            if option.default is REQUIRED:
                raise ValueError(f"{self.name} requires option {option.name!r}")
            resolved[option.name] = option.default
        return resolved, frozenset(given), frozenset(inferred)

    def check_arity(self, count):
        if not self.inputs:
            return
        required = sum(1 for port in self.inputs if not port.optional)
        variadic = bool(self.inputs) and self.inputs[-1].variadic
        if count < required or (not variadic and count > len(self.inputs)):
            names = ", ".join(port.name for port in self.inputs)
            raise ValueError(f"{self.name} expects inputs ({names}); received {count}")

    def to_json(self):
        return {
            "name": self.name,
            "category": self.category,
            "summary": self.summary,
            "inputs": [port.to_json() for port in self.inputs],
            "outputs": [port.name for port in self.output_ports],
            "output_ports": [port.to_json() for port in self.output_ports],
            "options": [option.to_json() for option in self.options],
            "pure": self.pure,
            "described": self.described,
            "bypass": self.bypass,
        }


_TRANSFORMS: dict[str, TransformInfo] = {}


def register_transform(
    name: str,
    *,
    category: str = "custom",
    summary: str = "",
    inputs: Sequence[Port] = (),
    outputs: Sequence[str] = ("output",),
    options: Sequence[Option] = (),
    cost: Callable | None = None,
    pure: bool = True,
    strict: bool | None = None,
    override: bool = False,
    bypass: int = 0,
):
    """Register a factory ``(options, inputs, mapper) -> (module, specs)``.

    ``pure`` declares that a parameter-free instance is a deterministic
    function of its inputs (eligible for caching when its inputs depend on the
    structure alone). Options are checked strictly whenever ports or options
    are declared; a bare registration accepts any options.
    ``bypass`` is the input a disabled instance passes through (the update of
    a residual or self-connection, say).
    """

    if category not in CATEGORIES:
        raise ValueError(f"Unknown transform category {category!r}")

    def decorator(factory):
        _ensure_builtins()
        if name in _TRANSFORMS and not override:
            raise ValueError(f"Transformation already registered: {name}")
        _TRANSFORMS[name] = TransformInfo(
            name=name,
            factory=factory,
            category=category,
            summary=summary or (factory.__doc__ or "").strip().split("\n")[0],
            inputs=tuple(inputs),
            outputs=tuple(outputs),
            options=tuple(options),
            cost=cost,
            pure=pure,
            strict=bool(options or inputs) if strict is None else strict,
            bypass=bypass,
        )
        return factory

    return decorator


def get_transform(name: str) -> TransformInfo:
    _ensure_builtins()
    try:
        return _TRANSFORMS[name]
    except KeyError:
        raise ValueError(f"Unknown transformation {name!r}") from None


def registered_transforms() -> Mapping[str, TransformInfo]:
    _ensure_builtins()
    return dict(_TRANSFORMS)


def build_transform(name, options, inputs, *, mapper=None, detailed=False):
    """Build one stage. Inputs are positional, never inferred by name.

    Returns ``(module, output specs)``, or a :class:`BuiltStage` with the
    completed options when ``detailed`` is set.
    """
    info = get_transform(name)
    info.check_arity(len(inputs))
    resolved, explicit, inferred = info.resolve_options(options, inputs, mapper)
    result = info.factory(dict(resolved), list(inputs), mapper)
    if not isinstance(result, TransformResult):
        module, outputs = result
        result = TransformResult(module, list(outputs))
    for spec in result.outputs:
        if not isinstance(spec, FeatureSpec):
            raise TypeError(f"{name} must return FeatureSpec outputs")
    if not detailed:
        return result.module, result.outputs
    return BuiltStage(
        result.module,
        result.outputs,
        resolved,
        explicit,
        inferred,
        dict(result.refined_inputs),
    )


def transform_catalog():
    """JSON description of every registered transformation."""
    return [
        info.to_json()
        for info in sorted(
            registered_transforms().values(), key=lambda i: (i.category, i.name)
        )
    ]


_LOADED = False


def _ensure_builtins():
    global _LOADED
    if not _LOADED:
        _LOADED = True
        import net.transforms  # noqa: F401  (registers the built-in library)


def infer_from(port_index, attribute="irreps"):
    """Inference rule: copy an attribute of an input feature."""

    def rule(inputs, mapper, options):
        if port_index < len(inputs):
            return getattr(inputs[port_index], attribute)
        return None

    return rule
