"""Composite blocks: named, reusable subgraphs of pipeline stages.

A recipe declares blocks under ``pipeline.blocks``::

    blocks:
      gated_update:
        inputs: [features, radial]
        outputs: [updated]
        options: {irreps_out: null, hidden: 16}
        stages:
          - {id: mix, transform: irrep_linear, inputs: [features], outputs: [mixed],
             options: {irreps_out: $irreps_out}}
          - ...

and instantiates one like a transformation::

    - {id: layer0, transform: gated_update, inputs: [node_0, radial],
       outputs: [node_1], options: {hidden: 32}}

Expansion happens before compilation. Inner stages become ``layer0/mix``;
inner slots become ``layer0/mixed``; block inputs and outputs are renamed to
the instance's slots. An option value ``$name`` is replaced by the instance's
option (a ``null`` value removes the inner option so it is inferred), and
``$(name)`` inside a longer string is substituted textually, e.g.
``"$(width)x0e"``. Blocks may instantiate other blocks.

``initialization_order`` optionally lists the block's parameterized stages in
the order their parameters are drawn from the random generator, when that
differs from the order of computation (as in some reference implementations).
A repeated id draws that stage again; the last draw is kept. An instance with
``enabled: false`` is bypassed as a whole: its single output is its first input,
or the block input named by its ``bypass`` key.
"""

from __future__ import annotations

import copy
import re

_PARAMETER = re.compile(r"\$([A-Za-z_][A-Za-z0-9_]*)")
_INLINE = re.compile(r"\$\(([A-Za-z_][A-Za-z0-9_]*)\)")
_DROP = object()


def _substitute(value, parameters, where):
    if isinstance(value, str):
        whole = _PARAMETER.fullmatch(value)
        if whole:
            name = whole.group(1)
            if name not in parameters:
                raise ValueError(f"{where}: unknown block option ${name}")
            result = parameters[name]
            return _DROP if result is None else copy.deepcopy(result)

        def replace(match):
            name = match.group(1)
            if name not in parameters:
                raise ValueError(f"{where}: unknown block option ${name}")
            return str(parameters[name])

        return _INLINE.sub(replace, value)
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            item = _substitute(item, parameters, where)
            if item is _DROP and key == "enabled":
                raise ValueError(f"{where}: enabled must be true or false, not null")
            if item is not _DROP:
                result[key] = item
        return result
    if isinstance(value, (list, tuple)):
        items = [_substitute(item, parameters, where) for item in value]
        return [item for item in items if item is not _DROP]
    return value


def validate_block(name, block):
    unknown = set(block) - {
        "inputs",
        "outputs",
        "options",
        "stages",
        "summary",
        "category",
        "initialization_order",
        "library",
    }
    if unknown:
        raise ValueError(f"Block {name}: unknown keys {sorted(unknown)}")
    for key in ("inputs", "outputs", "stages"):
        if key not in block:
            raise ValueError(f"Block {name} must declare {key}")
    produced = set(block["inputs"])
    for stage in block["stages"]:
        missing = set(stage["inputs"]) - produced
        if missing:
            raise ValueError(
                f"Block {name}, stage {stage['id']} reads {sorted(missing)}, "
                "which are neither block inputs nor earlier outputs"
            )
        produced.update(stage["outputs"])
    missing = set(block["outputs"]) - produced
    if missing:
        raise ValueError(f"Block {name} never produces outputs {sorted(missing)}")
    ids = {stage["id"] for stage in block["stages"]}
    unknown = set(block.get("initialization_order") or ()) - ids
    if unknown:
        raise ValueError(
            f"Block {name}: initialization_order names unknown stages {sorted(unknown)}"
        )


def expand_blocks(stages, blocks, *, prefix="", groups=None, depth=0):
    """Return ``(flat stages, groups)`` with every block instance expanded.

    ``groups`` maps an instance path (``layer0``, ``layer0/inner``) to the
    block name, its options and the stage ids it expanded into.
    """
    if depth > 16:
        raise ValueError("Composite blocks nest deeper than 16 levels (recursion?)")
    groups = {} if groups is None else groups
    flat = []
    for stage in stages:
        name = stage["transform"]
        if name not in blocks:
            flat.append(stage)
            continue
        block = blocks[name]
        validate_block(name, block)
        instance = stage["id"]
        path = f"{prefix}{instance}"
        if len(stage["inputs"]) != len(block["inputs"]):
            raise ValueError(
                f"Block instance {path} ({name}) expects inputs {block['inputs']}"
            )
        if len(stage["outputs"]) != len(block["outputs"]):
            raise ValueError(
                f"Block instance {path} ({name}) declares outputs {block['outputs']}"
            )
        extra = set(stage) - {
            "id",
            "transform",
            "inputs",
            "outputs",
            "options",
            "role",
            "enabled",
            "bypass",
        }
        if extra:
            raise ValueError(f"Block instance {path}: unsupported keys {sorted(extra)}")
        if stage.get("enabled", True) is False:
            disabled = dict(stage)
            chosen = disabled.get("bypass")
            if isinstance(chosen, str):
                if chosen not in block["inputs"]:
                    raise ValueError(
                        f"Block instance {path}: bypass {chosen!r} is not an input "
                        f"of {name} ({', '.join(block['inputs'])})"
                    )
                disabled["bypass"] = block["inputs"].index(chosen)
            flat.append(disabled)
            continue
        defaults = dict(block.get("options") or {})
        given = dict(stage.get("options") or {})
        unknown = set(given) - set(defaults)
        if unknown:
            raise ValueError(
                f"Block instance {path} ({name}): unknown options {sorted(unknown)}"
            )
        parameters = {**defaults, **given}
        rename = dict(zip(block["inputs"], stage["inputs"]))
        rename.update(zip(block["outputs"], stage["outputs"]))

        def slot(local):
            return rename.get(local, f"{instance}/{local}")

        inner = []
        for child in block["stages"]:
            child = _substitute(
                copy.deepcopy(child), parameters, f"{path}/{child['id']}"
            )
            child["id"] = f"{instance}/{child['id']}"
            child["inputs"] = [slot(item) for item in child["inputs"]]
            child["outputs"] = [slot(item) for item in child["outputs"]]
            if "role" in stage and "role" not in child:
                child["role"] = stage["role"]
            inner.append(child)
        start = len(groups)
        expanded, groups = expand_blocks(
            inner, blocks, prefix=prefix, groups=groups, depth=depth + 1
        )
        groups[path] = {
            "block": name,
            "options": parameters,
            "explicit": sorted(given),
            "inputs": list(stage["inputs"]),
            "outputs": list(stage["outputs"]),
            "stages": [child["id"] for child in expanded],
            "order": start,
        }
        if block.get("initialization_order"):
            groups[path]["initialization_order"] = [
                f"{instance}/{entry}" for entry in block["initialization_order"]
            ]
        flat.extend(expanded)
    return flat, groups


def _combined_port(name, readers):
    """One input port accepted by every inner reader ``[(stage id, port)]``."""
    domains = None
    for _, port in readers:
        accepted = port.get("domain")
        if accepted is None:
            continue
        accepted = {accepted} if isinstance(accepted, str) else set(accepted)
        domains = accepted if domains is None else domains & accepted
    if domains is not None and not domains:
        # Readers disagree: the block cannot be wired correctly; keep the
        # first constraint so the conflict is reported at the first reader.
        first = next(p["domain"] for _, p in readers if p.get("domain") is not None)
        domains = {first} if isinstance(first, str) else set(first)
    kinds = [p.get("kind") for _, p in readers if p.get("kind") not in (None, "any")]
    first = readers[0][1]
    return {
        **first,
        "domain": (
            None
            if domains is None
            else (next(iter(domains)) if len(domains) == 1 else sorted(domains))
        ),
        "kind": kinds[0] if kinds else first.get("kind", "any"),
        "doc": "read by " + ", ".join(dict.fromkeys(stage for stage, _ in readers)),
    }


def block_ports(blocks):
    """Port contracts of block definitions, as the compiler sees them.

    A block input combines the ports of every inner stage that reads it
    (recursively through nested blocks): the entity sets they accept are
    intersected and the most specific kind is kept. An output takes the output
    port of the inner stage producing it. Returns ``{block: {"inputs",
    "outputs"}}`` with ports in the catalog's JSON form.
    """
    from net.registry import Port, get_transform

    def unknown(name):
        return Port(name).to_json()

    def contract(name, depth=0):
        block = blocks[name]
        if depth > 16:
            raise ValueError("Composite blocks nest deeper than 16 levels")

        def stage_ports(stage):
            if stage["transform"] in blocks:
                inner = contract(stage["transform"], depth + 1)
                return inner["inputs"], inner["outputs"]
            try:
                info = get_transform(stage["transform"])
            except ValueError:
                return [], []
            return [p.to_json() for p in info.inputs], [
                p.to_json() for p in info.output_ports
            ]

        inputs, outputs = [], []
        for name_in in block["inputs"]:
            readers = []
            for stage in block["stages"]:
                ports = stage_ports(stage)[0]
                for index, slot in enumerate(stage["inputs"]):
                    if slot == name_in and ports:
                        readers.append((stage["id"], ports[min(index, len(ports) - 1)]))
            port = _combined_port(name_in, readers) if readers else unknown(name_in)
            inputs.append(
                {**port, "name": name_in, "optional": False, "variadic": False}
            )
        for name_out in block["outputs"]:
            port = unknown(name_out)
            for stage in block["stages"]:
                if name_out in stage["outputs"]:
                    ports = stage_ports(stage)[1]
                    index = stage["outputs"].index(name_out)
                    if index < len(ports):
                        port = ports[index]
                    break
            outputs.append({**port, "name": name_out, "optional": False})
        return {"inputs": inputs, "outputs": outputs}

    return {name: contract(name) for name in blocks}
