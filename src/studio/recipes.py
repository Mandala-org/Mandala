"""Recipe presets, chemical systems and YAML import/export."""

from __future__ import annotations

import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]

# Editor state (layout, expanded blocks, orientation). The framework accepts and
# ignores it; analysis leaves it out so layout changes never invalidate caches.
STUDIO_KEYS = ("studio",)

SYSTEMS = {
    "H (s, p)": {"H": [0, 1]},
    "Si s2p2d1": {"Si": [0, 0, 1, 1, 2]},
    "SiO2 s2p2d1": {"O": [0, 0, 1, 1, 2], "Si": [0, 0, 1, 1, 2]},
    "H2O": {"H": [0, 0, 1], "O": [0, 0, 1, 1, 2]},
    "Graphene C s2p2d1": {"C": [0, 0, 1, 1, 2]},
    "MoS2": {"Mo": [0, 0, 0, 1, 1, 2, 2], "S": [0, 0, 1, 1, 2]},
}

# Default basis per element when a recipe names its species.
ELEMENT_SHELLS = {"H": [0, 0, 1], "Mo": [0, 0, 0, 1, 1, 2, 2]}
DEFAULT_SHELLS = [0, 0, 1, 1, 2]

PRESET_DIRECTORIES = (
    ("Reference", ROOT / "configs" / "reference"),
    ("Hybrids", ROOT / "configs" / "hybrids"),
    ("Pilots", ROOT / "configs" / "pilots"),
)
EXTRA_PRESETS = (
    ("Examples", ROOT / "examples" / "first_model" / "silicon.yaml", None),
    (
        "Examples",
        ROOT / "examples" / "layer_cookbook" / "cookbook.yaml",
        "examples/layer_cookbook/layers.py",
    ),
)
# One line per shipped recipe for the Open dialog (a recipe's own first comment
# line is used otherwise).
PRESET_SUMMARIES = {
    "mandala_paper_blocks": "MANDALA paper model as a stage recipe with reusable blocks",
    "mandala_paper": "MANDALA paper model as training settings; the graph shows the stages built from them",
    "deeph": "DeepH: crystal-graph messages in local bond frames",
    "deeph_e3": "DeepH-E3: gated tensor products of node and edge features",
    "mace_h": "MACE-H: atomic-basis densities, symmetric products and degree expansion",
    "equivariant_nodes_bond_frame_readout": "Hybrid: equivariant node updates with a bond-frame readout",
    "symmetric_products_attention_pair_readout": "Hybrid: MACE products with attention and a pair readout",
    "tensor_messages_projected_attention": "Hybrid: tensor-product messages with projected attention",
    "tensor_products_degree_expansion_shell_readout": "Hybrid: tensor products, degree expansion and a shell readout",
    "graphene_mandala": "MANDALA pilot for graphene",
    "mos2_mandala": "MANDALA pilot for MoS2",
    "silicon": "First model: a small MANDALA for silicon",
    "cookbook": "Layer cookbook: custom layers from a plugin",
}

PRESET_SYSTEMS = {
    "mandala_paper": "SiO2 s2p2d1",
    "mandala_paper_blocks": "SiO2 s2p2d1",
    "graphene_mandala": "Graphene C s2p2d1",
    "mos2_mandala": "MoS2",
    "silicon": "Si s2p2d1",
}


def relative_path(path):
    """Repository-relative path when inside the repository, absolute otherwise."""
    try:
        return str(Path(path).resolve().relative_to(ROOT))
    except ValueError:
        return str(Path(path).resolve())


def presets():
    """Every recipe shipped with the repository, grouped for the preset menu."""
    entries = []
    for group, directory in PRESET_DIRECTORIES:
        for path in sorted(directory.glob("*.yaml")):
            entries.append(_preset(group, path, None))
    for group, path, plugin in EXTRA_PRESETS:
        if path.exists():
            entries.append(_preset(group, path, plugin))
    return entries


def _preset(group, path, plugin):
    try:
        document = yaml.safe_load(path.read_text()) or {}
    except yaml.YAMLError:
        document = {}
    summary = ""
    first = path.read_text().splitlines()[:3]
    comments = [line.lstrip("# ").strip() for line in first if line.startswith("#")]
    if comments:
        summary = comments[0]
    summary = PRESET_SUMMARIES.get(path.stem, summary)
    return {
        "group": group,
        "name": path.stem,
        "path": relative_path(path),
        "kind": "pipeline" if document.get("pipeline") else "flat",
        "summary": summary,
        "plugin": plugin,
        "system": PRESET_SYSTEMS.get(path.stem),
    }


def load_preset(path):
    """Read a preset by repository-relative path (never outside the repository)."""
    resolved = (ROOT / path).resolve()
    if ROOT not in resolved.parents or resolved.suffix not in {".yaml", ".yml"}:
        raise ValueError("Presets must be YAML files inside the repository")
    document = yaml.safe_load(resolved.read_text()) or {}
    plugin = next(
        (p["plugin"] for p in presets() if p["path"] == relative_path(resolved)), None
    )
    if plugin and plugin not in document.get("plugins", []):
        document["plugins"] = [*document.get("plugins", []), plugin]
    return document


def parse_yaml(text):
    document = yaml.safe_load(text)
    if document is None:
        return {}
    if not isinstance(document, dict):
        raise ValueError("A recipe is a YAML mapping")
    return document


PLUGIN_DIRECTORIES = (ROOT / "plugins", ROOT / "examples")


def check_plugin_paths(document):
    """Plugins are Python files under the repository's plugins/ or examples/."""
    plugins = document.get("plugins") or []
    if not isinstance(plugins, list):
        raise ValueError("plugins must be a list of paths")
    for entry in plugins:
        resolved = (ROOT / str(entry)).resolve()
        if resolved.suffix != ".py" or not any(
            directory in resolved.parents for directory in PLUGIN_DIRECTORIES
        ):
            raise ValueError(
                f"Plugin {entry!r} must be a .py file under plugins/ or examples/"
            )


def inline_block_libraries(document):
    """Copy imported library blocks into ``pipeline.blocks`` for editing.

    The editor edits block definitions in place; an exported recipe is then
    self-contained. Each copied block keeps its ``library`` name.
    """
    pipeline = document.get("pipeline")
    if not isinstance(pipeline, dict) or not pipeline.get("block_libraries"):
        return document
    from net.block_library import merged_blocks

    pipeline = {k: v for k, v in pipeline.items() if k != "block_libraries"}
    pipeline["blocks"] = merged_blocks(document["pipeline"])
    return {**document, "pipeline": pipeline}


def framework_document(document):
    """The document as the framework sees it (Studio-only keys removed)."""
    return {key: value for key, value in document.items() if key not in STUDIO_KEYS}


def system_for(document, requested=None):
    """Species shells: an explicit choice, the recipe's species, or silicon."""
    if requested:
        return {str(k): [int(l) for l in v] for k, v in requested.items()}
    species = document.get("species") or {}
    if isinstance(species.get("shells"), dict):
        return {str(k): [int(l) for l in v] for k, v in species["shells"].items()}
    order = species.get("order")
    if (
        isinstance(order, list)
        and order
        and all(isinstance(name, str) for name in order)
    ):
        return {name: ELEMENT_SHELLS.get(name, DEFAULT_SHELLS) for name in order}
    return dict(SYSTEMS["Si s2p2d1"])


class _Dumper(yaml.SafeDumper):
    pass


def _represent_list(dumper, data):
    flow = all(not isinstance(item, (dict, list)) for item in data)
    return dumper.represent_sequence("tag:yaml.org,2002:seq", data, flow_style=flow)


def _represent_dict(dumper, data):
    flow = (
        0 < len(data) <= 6
        and all(
            not isinstance(value, (dict, list))
            or (
                isinstance(value, list)
                and len(value) <= 6
                and all(not isinstance(v, (dict, list)) for v in value)
            )
            for value in data.values()
        )
        and sum(len(str(k)) + len(str(v)) for k, v in data.items()) < 90
    )
    return dumper.represent_mapping(
        "tag:yaml.org,2002:map", data.items(), flow_style=flow
    )


def _represent_str(dumper, data):
    style = (
        "'"
        if re.search(r"[:{}\[\],#&*!|>'\"%@`$]", data)
        or data in {"null", "true", "false", "yes", "no"}
        else None
    )
    if "\n" in data:
        style = "|"
    return dumper.represent_scalar("tag:yaml.org,2002:str", data, style=style)


_Dumper.add_representer(list, _represent_list)
_Dumper.add_representer(tuple, _represent_list)
_Dumper.add_representer(dict, _represent_dict)
_Dumper.add_representer(str, _represent_str)


def _stage_first(document):
    """Order recipe stages as id, transform, inputs, outputs, options, ..."""
    pipeline = document.get("pipeline")
    if not isinstance(pipeline, dict):
        return document
    order = (
        "id",
        "transform",
        "role",
        "inputs",
        "outputs",
        "options",
        "initialization",
    )

    def arrange(stage):
        return {
            **{k: stage[k] for k in order if k in stage},
            **{k: v for k, v in stage.items() if k not in order},
        }

    pipeline = dict(pipeline)
    if "stages" in pipeline:
        pipeline["stages"] = [arrange(stage) for stage in pipeline["stages"]]
    if "blocks" in pipeline:
        pipeline["blocks"] = {
            name: {
                **block,
                "stages": [arrange(stage) for stage in block.get("stages", [])],
            }
            for name, block in pipeline["blocks"].items()
        }
    return {**document, "pipeline": pipeline}


# Sections that define the model, written first and in this order; the
# remaining keys (training, data and run settings) follow under a comment.
MODEL_SECTIONS = (
    "species",
    "parameters",
    "block_libraries",
    "plugins",
    "pipeline",
    "objective",
)


def dump_yaml(document, *, header=None):
    framework = framework_document(document)
    model = {key: framework[key] for key in MODEL_SECTIONS if key in framework}
    settings = {key: value for key, value in framework.items() if key not in model}

    def dump(part):
        return yaml.dump(
            _stage_first(part),
            Dumper=_Dumper,
            sort_keys=False,
            allow_unicode=True,
            width=100,
        )

    text = ""
    if model:
        text += "# Model\n" + dump(model)
    if settings:
        text += (
            ("\n" if text else "")
            + "# Training, data and run settings\n"
            + dump(settings)
        )
    if document.get("studio"):
        text += "\n# Editor layout (ignored by training)\n" + dump(
            {"studio": document["studio"]}
        )
    if header:
        text = "".join(f"# {line}\n" for line in header.splitlines()) + text
    return text
