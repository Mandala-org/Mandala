"""Compile a recipe for a declared chemical system and describe every stage.

``analyze_recipe`` is what model-building tools call: it resolves the recipe
exactly like training does (layout variables, flat-config lowering), compiles
it while collecting every error, and reports per stage the completed options
(explicit, inferred or default), the output contracts, the parameter count and
an asymptotic cost estimate. Built stages are memoized, so re-analysing an
edited recipe only rebuilds the stages whose inputs or options changed.
"""

from __future__ import annotations

import copy
import json
from collections import OrderedDict

import torch

from net.cost_model import SystemSize, load_constants, stage_cost
from net.features import default_input_specs
from net.registry import build_transform

PARITY = {0: "e", 1: "o", 2: "e", 3: "o", 4: "e", 5: "o", 6: "e"}


class _StageCache:
    """LRU memo of ``build_transform`` keyed by transform, options and inputs."""

    def __init__(self, size=512):
        self.size, self.entries = size, OrderedDict()

    def __call__(self, name, options, inputs, *, mapper=None, detailed=True):
        key = json.dumps(
            [name, options, [spec.to_json() for spec in inputs], id(mapper)],
            sort_keys=True,
            default=str,
        )
        if key in self.entries:
            self.entries.move_to_end(key)
            return self.entries[key]
        built = build_transform(name, options, inputs, mapper=mapper, detailed=True)
        self.entries[key] = built
        if len(self.entries) > self.size:
            self.entries.popitem(last=False)
        return built


_CACHE = _StageCache()


def clear_cache():
    """Forget built stages, e.g. after a transformation was re-registered."""
    _CACHE.entries.clear()


_MAPPERS: dict = {}


def orbital_mapper(species_shells):
    """Mapper for ``{species: [l, ...]}`` shell lists (memoized)."""
    from core.block_irrep_mapper import BlockIrrepMapper
    from core.orbital_irrep_config import OrbitalIrrepConfig

    key = json.dumps(species_shells, sort_keys=True)
    if key not in _MAPPERS:
        orbital = OrbitalIrrepConfig.from_dict(
            {
                species: [f"1x{l}{PARITY[l]}" for l in shells]
                for species, shells in species_shells.items()
            }
        )
        _MAPPERS[key] = BlockIrrepMapper(orbital, dtype=torch.get_default_dtype())
    return _MAPPERS[key]


def resolve_recipe(document, species_shells):
    """``(Config, pipeline specification)`` for a recipe or flat configuration."""
    from net.common import Config
    from net.component_defaults import pipeline_specification
    from net.configuration import resolve_configuration

    resolved, _ = resolve_configuration(
        copy.deepcopy(dict(document)), species_shells=species_shells
    )
    cfg = Config.from_mapping(resolved)
    return cfg, pipeline_specification(cfg, orbital_mapper(species_shells))


def _ordered(options):
    return {key: options[key] for key in options}


def analyze_pipeline(pipeline, *, species_shells, size=None, constants=None):
    """Describe a pipeline specification compiled for ``species_shells``."""
    from net.pipeline import FeaturePipeline

    size = size or SystemSize()
    constants = constants or load_constants()
    mapper = orbital_mapper(species_shells)
    compiled = FeaturePipeline(
        pipeline,
        num_species=len(species_shells),
        mapper=mapper,
        errors="collect",
        build=_CACHE,
    )
    specs = compiled.feature_specs
    stages, total_parameters, total_seconds = [], 0, 0.0
    for record in compiled.records:
        entry = {
            "id": record.id,
            "transform": record.transform,
            "group": record.group,
            "role": record.role,
            "inputs": list(record.inputs),
            "outputs": list(record.outputs),
            "options": _jsonable(record.options),
            "explicit": sorted(record.explicit),
            "inferred": sorted(record.inferred),
            "static": record.static,
            "enabled": record.enabled,
            "error": record.error,
        }
        if record.error is None:
            module = compiled.transforms[record.id]
            parameters = sum(p.numel() for p in module.parameters())
            operations, seconds = stage_cost(
                record.transform,
                module,
                [specs[slot] for slot in record.inputs],
                [specs[slot] for slot in record.outputs],
                size,
                constants,
            )
            entry.update(parameters=parameters, operations=operations, seconds=seconds)
            total_parameters += parameters
            total_seconds += seconds
        stages.append(entry)
    builtin = default_input_specs(len(species_shells))
    declared = set((pipeline.get("inputs") or {}))
    read = compiled.required_inputs()
    return {
        "stages": stages,
        "slots": {slot: spec.to_json() for slot, spec in specs.items()},
        "inputs": {
            slot: {
                **specs[slot].to_json(),
                "builtin": slot in builtin,
                "used": slot in read,
            }
            for slot in specs
            if slot in builtin or slot in declared
        },
        "groups": _jsonable(compiled.groups),
        "blocks": _block_contracts(pipeline),
        "outputs": dict(compiled.outputs),
        "errors": [
            {"stage": stage, "message": message} for stage, message in compiled.errors
        ],
        "totals": {"parameters": total_parameters, "seconds": total_seconds},
        "system": {"species_shells": species_shells, **size.__dict__},
    }


def _block_contracts(pipeline):
    """Port contracts of the blocks a pipeline defines or imports."""
    from net.block_library import merged_blocks
    from net.blocks import block_ports

    try:
        return _jsonable(block_ports(merged_blocks(pipeline)))
    except (ValueError, KeyError, TypeError):
        return {}


def block_library_catalog():
    """Every library block with its definition and port contract."""
    from net.block_library import available_libraries, library_blocks
    from net.blocks import block_ports

    blocks = library_blocks(available_libraries())
    ports = block_ports(blocks)
    return {
        name: {**_jsonable(block), "ports": _jsonable(ports[name])}
        for name, block in blocks.items()
    }


def analyze_recipe(document, *, species_shells, size=None, constants=None):
    """Resolve and analyse a full recipe (sections, parameters, layout rules)."""
    try:
        cfg, pipeline = resolve_recipe(document, species_shells)
    except Exception as error:  # resolution failures are reported, not raised
        return {
            "stages": [],
            "slots": {},
            "inputs": {},
            "groups": {},
            "blocks": {},
            "outputs": {},
            "errors": [{"stage": "configuration", "message": str(error)}],
            "totals": {"parameters": 0, "seconds": 0.0},
            "pipeline": None,
        }
    result = analyze_pipeline(
        pipeline, species_shells=species_shells, size=size, constants=constants
    )
    result["pipeline"] = _jsonable(pipeline)
    return result


def _jsonable(value):
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_jsonable(item) for item in value]
    if isinstance(value, torch.dtype):
        return str(value).removeprefix("torch.")
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)
