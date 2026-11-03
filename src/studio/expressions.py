"""A small typed, equivariant expression language for new transformations.

A block declares typed inputs and a few assignments::

    inputs:  x: node 8x0e+4x1o      s: node 4x0e
    h   = linear(x, "8x0e+4x1o")
    out = h * sigmoid(norm(h)) + 0.5 * x

Every value has an entity domain and irreps. The checker admits only
equivariant operations:

* ``+``/``-`` of values with equal irreps; adding a number needs an invariant
  (all ``0e``) value;
* ``*``/``/`` by a number, by a learned ``param(name, init)``, by an invariant
  with one channel (broadcast), by an invariant with one channel per irrep
  copy (``gate``), or elementwise between equal invariants;
* elementwise functions (``silu sigmoid tanh relu gelu softplus exp log sqrt
  abs``) of invariants only;
* ``linear(x, irreps)``, ``tp(x, y, irreps)``, ``norm(x)`` (copy norms),
  ``scalars(x)`` (the ``0e`` channels), ``concat(a, b, ...)``,
  ``mlp(s, [hidden, ...], width)`` on invariants, ``gate(x, s)``.

``check`` returns the type of every variable; ``generate`` writes a plugin
module that registers the block as a MANDALA transformation.
"""

from __future__ import annotations

import ast
import keyword
import re
from dataclasses import dataclass, field

from e3nn import o3
from torch import nn

DOMAINS = ("node", "edge", "graph")
ELEMENTWISE = {
    "silu": "torch.nn.functional.silu({})",
    "sigmoid": "torch.sigmoid({})",
    "tanh": "torch.tanh({})",
    "relu": "torch.relu({})",
    "gelu": "torch.nn.functional.gelu({})",
    "softplus": "torch.nn.functional.softplus({})",
    "exp": "torch.exp({})",
    "log": "torch.log({})",
    "sqrt": "torch.sqrt({})",
    "abs": "torch.abs({})",
}
CATEGORIES = (
    "custom",
    "nonlinearity",
    "message",
    "mlp",
    "linear",
    "combine",
    "residual",
    "normalization",
    "readout",
    "envelope",
)


class ExpressionError(ValueError):
    def __init__(self, message, line=None, column=None):
        super().__init__(message)
        self.line, self.column = line, column

    def to_json(self):
        return {"message": str(self), "line": self.line, "column": self.column}


@dataclass
class Value:
    """A typed value: ``kind`` is tensor, number or param."""

    kind: str
    code: str
    irreps: o3.Irreps | None = None
    domain: str | None = None
    number: float | None = None

    @property
    def invariant(self):
        if self.kind in {"number", "param"}:
            return True
        return all(ir.l == 0 and ir.p == 1 for _, ir in self.irreps)

    @property
    def width(self):
        return 1 if self.kind in {"number", "param"} else self.irreps.dim

    def describe(self):
        if self.kind == "number":
            return {"kind": "number", "value": self.number}
        if self.kind == "param":
            return {
                "kind": "param",
                "irreps": "1x0e",
                "domain": None,
                "invariant": True,
            }
        return {
            "kind": "tensor",
            "irreps": str(self.irreps),
            "domain": self.domain,
            "invariant": self.invariant,
            "dim": self.irreps.dim,
        }


@dataclass
class Program:
    name: str
    inputs: list
    outputs: list
    source: str
    summary: str = ""
    category: str = "custom"
    modules: list = field(default_factory=list)
    buffers: list = field(default_factory=list)
    params: list = field(default_factory=list)
    lines: list = field(default_factory=list)
    types: dict = field(default_factory=dict)


def _irreps(text, node):
    try:
        return o3.Irreps(text)
    except Exception as error:
        raise ExpressionError(
            f"Invalid irreps {text!r}: {error}", node.lineno, node.col_offset
        ) from None


def _copy_index(irreps):
    index, copy = [], 0
    for mul, ir in irreps:
        for _ in range(mul):
            index.extend([copy] * ir.dim)
            copy += 1
    return index


def _scalar_index(irreps):
    index, offset = [], 0
    for mul, ir in irreps:
        if ir.l == 0 and ir.p == 1:
            index.extend(range(offset, offset + mul))
        offset += mul * ir.dim
    return index


class _Checker:
    def __init__(self, program):
        self.program = program
        self.scope = {}
        self.counter = 0

    def fresh(self, stem):
        self.counter += 1
        return f"{stem}_{self.counter}"

    def emit(self, stem, expression, value):
        name = self.fresh(stem)
        self.program.lines.append(f"{name} = {expression}")
        value.code = name
        return value

    def module(self, stem, constructor):
        name = self.fresh(stem)
        self.program.modules.append((name, constructor))
        return f"self.{name}"

    def buffer(self, stem, values):
        name = self.fresh(stem)
        self.program.buffers.append((name, values))
        return f"self.{name}"

    def error(self, message, node):
        raise ExpressionError(
            message, getattr(node, "lineno", None), getattr(node, "col_offset", None)
        )

    def tensor(self, value, node, what="This operation"):
        if value.kind != "tensor":
            self.error(f"{what} needs a feature, not a number", node)
        return value

    def same_domain(self, values, node):
        domains = {value.domain for value in values if value.kind == "tensor"}
        if len(domains) > 1:
            self.error(f"Values live on different entities: {sorted(domains)}", node)
        return next(iter(domains)) if domains else None

    def visit(self, node):
        method = getattr(self, f"visit_{type(node).__name__}", None)
        if method is None:
            self.error(f"Unsupported syntax: {type(node).__name__}", node)
        return method(node)

    def visit_Constant(self, node):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            self.error("Only numbers and irreps strings are allowed as constants", node)
        return Value("number", repr(float(node.value)), number=float(node.value))

    def visit_Name(self, node):
        if node.id not in self.scope:
            self.error(f"Unknown name {node.id!r}", node)
        return self.scope[node.id]

    def visit_UnaryOp(self, node):
        operand = self.visit(node.operand)
        if isinstance(node.op, ast.UAdd):
            return operand
        if not isinstance(node.op, ast.USub):
            self.error("Only unary minus is supported", node)
        if operand.kind == "number":
            return Value("number", repr(-operand.number), number=-operand.number)
        result = Value(operand.kind, "", operand.irreps, operand.domain)
        return self.emit("neg", f"-{operand.code}", result)

    def visit_BinOp(self, node):
        left, right = self.visit(node.left), self.visit(node.right)
        op = type(node.op)
        symbol = {ast.Add: "+", ast.Sub: "-", ast.Mult: "*", ast.Div: "/"}.get(op)
        if symbol is None:
            self.error("Only + - * / are supported", node)
        if left.kind == "number" and right.kind == "number":
            if symbol == "/" and right.number == 0:
                self.error("Division by zero", node)
            value = {
                "+": left.number + right.number,
                "-": left.number - right.number,
                "*": left.number * right.number,
            }.get(symbol)
            value = left.number / right.number if value is None else value
            return Value("number", repr(value), number=value)
        if symbol in "+-":
            return self.additive(left, right, symbol, node)
        return self.multiplicative(left, right, symbol, node)

    def additive(self, left, right, symbol, node):
        tensors = [v for v in (left, right) if v.kind == "tensor"]
        if len(tensors) == 2:
            if left.irreps.simplify() != right.irreps.simplify():
                self.error(
                    f"Cannot {'add' if symbol == '+' else 'subtract'} {left.irreps} and {right.irreps}",
                    node,
                )
            domain = self.same_domain(tensors, node)
            return self.emit(
                "sum",
                f"{left.code} {symbol} {right.code}",
                Value("tensor", "", left.irreps, domain),
            )
        tensor = tensors[0] if tensors else None
        if tensor is None:  # number and param
            return self.emit(
                "sum", f"{left.code} {symbol} {right.code}", Value("param", "")
            )
        if not tensor.invariant:
            self.error(
                "Adding a constant breaks equivariance unless the feature is invariant (0e)",
                node,
            )
        return self.emit(
            "sum",
            f"{left.code} {symbol} {right.code}",
            Value("tensor", "", tensor.irreps, tensor.domain),
        )

    def multiplicative(self, left, right, symbol, node):
        if symbol == "/" and right.kind == "tensor" and not right.invariant:
            self.error("Division is only defined by invariant features", node)
        if left.kind != "tensor" or right.kind != "tensor":
            tensor = left if left.kind == "tensor" else right
            if tensor.kind != "tensor":
                return self.emit(
                    "product", f"{left.code} {symbol} {right.code}", Value("param", "")
                )
            if symbol == "/" and tensor is right:
                if not right.invariant:
                    self.error("Division is only defined by invariant features", node)
            return self.emit(
                "product",
                f"{left.code} {symbol} {right.code}",
                Value("tensor", "", tensor.irreps, tensor.domain),
            )
        domain = self.same_domain([left, right], node)
        if left.invariant and right.invariant and left.width == right.width:
            return self.emit(
                "product",
                f"{left.code} {symbol} {right.code}",
                Value("tensor", "", left.irreps, domain),
            )
        for feature, scale in ((left, right), (right, left)):
            if symbol == "/" and scale is left:
                continue
            if scale.invariant and scale.width == 1:
                return self.emit(
                    "product",
                    f"{left.code} {symbol} {right.code}",
                    Value("tensor", "", feature.irreps, domain),
                )
            if scale.invariant and scale.width == feature.irreps.num_irreps:
                index = self.buffer("copies", _copy_index(feature.irreps))
                expanded = f"{scale.code}.index_select(-1, {index})"
                code = (
                    f"{feature.code} {symbol} {expanded}"
                    if feature is left
                    else f"{expanded} {symbol} {feature.code}"
                )
                return self.emit(
                    "gated", code, Value("tensor", "", feature.irreps, domain)
                )
        self.error(
            f"Cannot multiply {left.irreps} by {right.irreps}: scale by an invariant with 1 or "
            f"{max(left.irreps.num_irreps, right.irreps.num_irreps)} channels, or use tp()",
            node,
        )

    def irreps_argument(self, node, call):
        if not isinstance(node, ast.Constant) or not isinstance(node.value, str):
            self.error(f'{call} expects an irreps string such as "8x0e+4x1o"', node)
        irreps = _irreps(node.value, node).simplify()
        if irreps.dim == 0 or any(mul == 0 for mul, _ in irreps):
            self.error(f"{call}: irreps must be non-empty", node)
        if irreps.dim > MAX_WIDTH:
            self.error(f"{call}: at most {MAX_WIDTH} components", node)
        return irreps

    def visit_Call(self, node):
        if not isinstance(node.func, ast.Name) or node.keywords:
            self.error("Call a built-in function with positional arguments", node)
        name, args = node.func.id, node.args
        if name in ELEMENTWISE:
            if len(args) != 1:
                self.error(f"{name} takes one argument", node)
            value = self.visit(args[0])
            if value.kind == "tensor" and not value.invariant:
                self.error(
                    f"{name} acts elementwise and needs an invariant (0e) feature; use norm() or gate()",
                    node,
                )
            template = ELEMENTWISE[name]
            kind = "tensor" if value.kind == "tensor" else "param"
            return self.emit(
                name,
                template.format(value.code),
                Value(kind, "", value.irreps, value.domain),
            )
        handler = getattr(self, f"call_{name}", None)
        if handler is None:
            self.error(f"Unknown function {name!r}", node)
        return handler(node, args)

    def call_linear(self, node, args):
        if len(args) != 2:
            self.error("linear(x, irreps)", node)
        value = self.tensor(self.visit(args[0]), node, "linear")
        out = self.irreps_argument(args[1], "linear")
        unreachable = [
            str(ir) for _, ir in out if ir not in {i for _, i in value.irreps}
        ]
        if unreachable:
            self.error(
                f"linear maps copies of the same irrep; {', '.join(unreachable)} is not in {value.irreps}",
                node,
            )
        module = self.module(
            "linear", f"o3.Linear({str(value.irreps)!r}, {str(out)!r})"
        )
        return self.emit(
            "linear", f"{module}({value.code})", Value("tensor", "", out, value.domain)
        )

    def call_tp(self, node, args):
        if len(args) != 3:
            self.error("tp(x, y, irreps)", node)
        left = self.tensor(self.visit(args[0]), node, "tp")
        right = self.tensor(self.visit(args[1]), node, "tp")
        domain = self.same_domain([left, right], node)
        out = self.irreps_argument(args[2], "tp")
        reachable = o3.Irreps(
            [
                (mul, ir)
                for mul, ir in out
                if any(ir in a * b for _, a in left.irreps for _, b in right.irreps)
            ]
        )
        if reachable.simplify() != out.simplify():
            self.error(
                f"{left.irreps} (x) {right.irreps} cannot produce all of {out}", node
            )
        module = self.module(
            "tp",
            f"o3.FullyConnectedTensorProduct({str(left.irreps)!r}, {str(right.irreps)!r}, {str(out)!r})",
        )
        return self.emit(
            "tp",
            f"{module}({left.code}, {right.code})",
            Value("tensor", "", out, domain),
        )

    def call_norm(self, node, args):
        if len(args) != 1:
            self.error("norm(x)", node)
        value = self.tensor(self.visit(args[0]), node, "norm")
        module = self.module("norm", f"o3.Norm({str(value.irreps)!r})")
        out = o3.Irreps(f"{value.irreps.num_irreps}x0e")
        return self.emit(
            "norm", f"{module}({value.code})", Value("tensor", "", out, value.domain)
        )

    def call_scalars(self, node, args):
        if len(args) != 1:
            self.error("scalars(x)", node)
        value = self.tensor(self.visit(args[0]), node, "scalars")
        index = _scalar_index(value.irreps)
        if not index:
            self.error(f"{value.irreps} has no even scalar channels", node)
        buffer = self.buffer("scalar_channels", index)
        out = o3.Irreps(f"{len(index)}x0e")
        return self.emit(
            "scalars",
            f"{value.code}.index_select(-1, {buffer})",
            Value("tensor", "", out, value.domain),
        )

    def call_concat(self, node, args):
        if len(args) < 2:
            self.error("concat(a, b, ...) needs at least two features", node)
        values = [self.tensor(self.visit(arg), node, "concat") for arg in args]
        domain = self.same_domain(values, node)
        out = sum((value.irreps for value in values), o3.Irreps())
        codes = ", ".join(value.code for value in values)
        return self.emit(
            "concat", f"torch.cat([{codes}], dim=-1)", Value("tensor", "", out, domain)
        )

    def call_gate(self, node, args):
        if len(args) != 2:
            self.error("gate(x, s)", node)
        return self.multiplicative(self.visit(args[0]), self.visit(args[1]), "*", node)

    def call_mlp(self, node, args):
        if len(args) != 3 or not isinstance(args[1], ast.List):
            self.error("mlp(s, [hidden, ...], width)", node)
        value = self.tensor(self.visit(args[0]), node, "mlp")
        if not value.invariant:
            self.error(
                "mlp() acts on invariant (0e) features; use norm() or scalars() first",
                node,
            )
        widths = []
        for item in [*args[1].elts, args[2]]:
            if (
                not isinstance(item, ast.Constant)
                or isinstance(item.value, bool)
                or not isinstance(item.value, int)
                or not 1 <= item.value <= MAX_WIDTH
            ):
                self.error(
                    f"mlp widths must be integers between 1 and {MAX_WIDTH}", item
                )
            widths.append(item.value)
        layers, previous = [], value.width
        for index, width in enumerate(widths):
            layers.append(f"nn.Linear({previous}, {width})")
            if index < len(widths) - 1:
                layers.append("nn.SiLU()")
            previous = width
        module = self.module("mlp", "nn.Sequential(" + ", ".join(layers) + ")")
        out = o3.Irreps(f"{widths[-1]}x0e")
        return self.emit(
            "mlp", f"{module}({value.code})", Value("tensor", "", out, value.domain)
        )

    def call_param(self, node, args):
        if (
            len(args) != 2
            or not isinstance(args[0], ast.Constant)
            or not isinstance(args[0].value, str)
        ):
            self.error('param("name", initial_value)', node)
        name = args[0].value
        if not name.isidentifier() or keyword.iskeyword(name):
            self.error("Parameter names must be identifiers", node)
        if name in RESERVED or name.startswith(
            ("_", "linear_", "tp_", "mlp_", "index_")
        ):
            self.error(f"Parameter name {name!r} is reserved", node)
        initial = self.visit(args[1])
        if initial.kind != "number":
            self.error("A parameter's initial value is a number", node)
        if name not in {p for p, _ in self.program.params}:
            self.program.params.append((name, initial.number))
        return Value("param", f"self.{name}")


MAX_WIDTH = 4096
# Attributes of torch.nn.Module and names the generated class uses itself.
RESERVED = frozenset(dir(nn.Module())) | {"forward", "self", "torch", "nn", "o3"}


def _parse_inputs(inputs):
    parsed = []
    if not isinstance(inputs, list) or not all(
        isinstance(item, dict) for item in inputs
    ):
        raise ExpressionError("inputs must be a list of {name, domain, irreps} objects")
    for item in inputs:
        name = str(item.get("name", "")).strip()
        if not name.isidentifier() or keyword.iskeyword(name):
            raise ExpressionError(f"Input name {name!r} is not an identifier")
        domain = item.get("domain", "node")
        if domain not in DOMAINS:
            raise ExpressionError(f"Input {name}: domain must be one of {DOMAINS}")
        try:
            irreps = o3.Irreps(item.get("irreps", ""))
        except Exception as error:
            raise ExpressionError(f"Input {name}: invalid irreps ({error})") from None
        if irreps.dim == 0:
            raise ExpressionError(f"Input {name}: irreps must not be empty")
        parsed.append({"name": name, "domain": domain, "irreps": irreps})
    if not parsed:
        raise ExpressionError("Declare at least one input")
    if len({item["name"] for item in parsed}) != len(parsed):
        raise ExpressionError("Input names must be unique")
    return parsed


def compile_block(
    name, inputs, source, outputs=("out",), summary="", category="custom"
):
    """Type-check a block; returns the :class:`Program` with generated code lines."""
    if not re.fullmatch(r"[a-z][a-z0-9_]*", name or ""):
        raise ExpressionError(
            "Block names use lowercase letters, digits and underscores"
        )
    if category not in CATEGORIES:
        raise ExpressionError(f"Category must be one of {CATEGORIES}")
    program = Program(
        name, _parse_inputs(inputs), list(outputs), source, summary, category
    )
    checker = _Checker(program)
    for item in program.inputs:
        checker.scope[item["name"]] = Value(
            "tensor", item["name"], item["irreps"], item["domain"]
        )
    try:
        tree = ast.parse(source or "", mode="exec")
    except SyntaxError as error:
        raise ExpressionError(
            f"Syntax error: {error.msg}", error.lineno, error.offset
        ) from None
    for statement in tree.body:
        if (
            not isinstance(statement, ast.Assign)
            or len(statement.targets) != 1
            or not isinstance(statement.targets[0], ast.Name)
        ):
            raise ExpressionError(
                "Each line is one assignment: name = expression",
                statement.lineno,
                statement.col_offset,
            )
        target = statement.targets[0].id
        if target in {item["name"] for item in program.inputs}:
            raise ExpressionError(
                f"{target} is an input and cannot be reassigned",
                statement.lineno,
                statement.col_offset,
            )
        value = checker.visit(statement.value)
        if value.kind == "number":
            raise ExpressionError(
                f"{target} is a constant; use it inside an expression",
                statement.lineno,
                statement.col_offset,
            )
        last = program.lines[-1] if program.lines else ""
        if value.code not in checker.scope and last.startswith(f"{value.code} = "):
            # Name the statement's final operation after its target.
            program.lines[-1] = f"{target} = {last[len(value.code) + 3:]}"
        else:
            program.lines.append(f"{target} = {value.code}")
        checker.scope[target] = Value(
            value.kind, target, value.irreps, value.domain, value.number
        )
        program.types[target] = checker.scope[target].describe()
    if not program.outputs:
        raise ExpressionError("Declare at least one output")
    for output in program.outputs:
        value = checker.scope.get(output)
        if value is None:
            raise ExpressionError(f"Output {output!r} is never assigned")
        if value.kind != "tensor":
            raise ExpressionError(
                f"Output {output!r} must be a feature, not a parameter"
            )
    return program


_ALLOWED_TOP_LEVEL = (
    ast.Expr,
    ast.Import,
    ast.ImportFrom,
    ast.Assign,
    ast.ClassDef,
    ast.FunctionDef,
)


def validate_generated(code):
    """Reject generated code whose module body is not the expected plugin shape."""
    tree = ast.parse(code)
    for index, node in enumerate(tree.body):
        if not isinstance(node, _ALLOWED_TOP_LEVEL):
            raise ExpressionError(
                f"Generated code has an unexpected top-level {type(node).__name__}"
            )
        if isinstance(node, ast.Expr) and not (
            index == 0 and isinstance(node.value, ast.Constant)
        ):
            raise ExpressionError(
                "Generated code has an unexpected top-level expression"
            )
    compile(tree, "<plugin>", "exec")
    return code


def check(name, inputs, source, outputs=("out",), summary="", category="custom"):
    """JSON result for the editor: variable types, errors and generated code."""
    try:
        program = compile_block(name, inputs, source, outputs, summary, category)
        code = validate_generated(generate(program))
    except ExpressionError as error:
        return {"ok": False, "error": error.to_json(), "types": {}}
    return {
        "ok": True,
        "types": program.types,
        "outputs": {o: program.types[o] for o in program.outputs},
        "code": code,
    }


def _class_name(name):
    return "".join(part.capitalize() for part in name.split("_")) or "Block"


def generate(program):
    """Source of a plugin module registering ``program`` as a transformation."""
    cls = _class_name(program.name)
    signature = ", ".join(item["name"] for item in program.inputs)
    declared = "\n".join(
        f"    {item['name']}: {item['domain']} {item['irreps']}"
        for item in program.inputs
    )
    body = "\n".join(
        f"#     {line}".rstrip() for line in program.source.strip().splitlines()
    )
    init = ["        super().__init__()"]
    for name, constructor in program.modules:
        init.append(f"        self.{name} = {constructor}")
    for name, values in program.buffers:
        init.append(
            f'        self.register_buffer("{name}", torch.tensor({values}, dtype=torch.long), persistent=False)'
        )
    for name, value in program.params:
        init.append(f"        self.{name} = nn.Parameter(torch.tensor({value!r}))")
    forward = [f"        {line}" for line in program.lines]
    returned = ", ".join(program.outputs)
    forward.append(f"        return {returned}")
    inputs_table = ",\n".join(
        f'    "{item["name"]}": ("{item["domain"]}", "{item["irreps"]}")'
        for item in program.inputs
    )
    out_specs = ",\n        ".join(
        f'FeatureSpec(_DOMAINS["{o}"], "irreps", "{program.types[o]["irreps"]}", support=support, '
        f'angular_basis="xyz" if o3.Irreps("{program.types[o]["irreps"]}").lmax > 0 else None)'
        for o in program.outputs
    )
    out_domains = ", ".join(
        f'"{o}": "{program.types[o]["domain"]}"' for o in program.outputs
    )
    ports = ",\n        ".join(
        f'Port("{item["name"]}", "{item["domain"]}", "{"invariant" if all(ir.l == 0 and ir.p == 1 for _, ir in item["irreps"]) else "irreps"}", doc="{item["irreps"]}")'
        for item in program.inputs
    )
    summary = repr(
        " ".join((program.summary or f"Expression block {program.name}").split())
    )
    outputs_tuple = ", ".join(f'"{o}"' for o in program.outputs) + (
        "," if len(program.outputs) == 1 else ""
    )
    declared = "\n".join(f"#{line}" for line in declared.splitlines())
    return f'''"""Generated by MANDALA Studio from an expression block."""

# Inputs:
{declared}
#
# Definition:
{body}

import torch
from e3nn import o3
from torch import nn
from torch import nn

from net.features import FeatureSpec, global_irreps, require_canonical, same_support
from net.registry import Port, register_transform

INPUTS = {{
{inputs_table}
}}
_DOMAINS = {{{out_domains}}}


class {cls}(nn.Module):
    def __init__(self):
{chr(10).join(init)}

    def forward(self, {signature}):
{chr(10).join(forward)}


@register_transform(
    "{program.name}",
    category="{program.category}",
    summary={summary},
    inputs=[
        {ports},
    ],
    outputs=({outputs_tuple}),
    override=True,
)
def build(options, inputs, mapper):
    for (name, (domain, irreps)), spec in zip(INPUTS.items(), inputs):
        if spec.domain != domain:
            raise ValueError(f"{{name}} must be a {{domain}} feature, received {{spec.domain}}")
        if global_irreps(spec).simplify() != o3.Irreps(irreps).simplify():
            raise ValueError(f"{{name}} must have irreps {{irreps}}, received {{spec.irreps}}")
    require_canonical(*inputs)
    for domain in set(_DOMAINS.values()) | {{d for d, _ in INPUTS.values()}}:
        same_support(*[spec for spec in inputs if spec.domain == domain])
    support = next((spec.support for spec in inputs if spec.domain in _DOMAINS.values()), None)
    return {cls}(), [
        {out_specs}
    ]
'''
