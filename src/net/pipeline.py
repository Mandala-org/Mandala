"""Typed, ordered composition of registered tensor transformations.

A pipeline specification lists input contracts, stages and named outputs::

    pipeline:
      inputs: {extra_slot: {domain: edge, irreps: 8x0e}}
      blocks: {name: {inputs, outputs, options, stages}}   # optional, see net.blocks
      block_libraries: [deeph_e3]                          # optional, net.block_library
      stages:
        - {id: radial, transform: soft_one_hot_basis, inputs: [edge_lengths],
           outputs: [radial], options: {number: 16, end: 6.0}}
      outputs: {hamiltonian: matrix}

Stages read and write named slots; a slot is written once. Every stage is a
registered transformation (``net.registry``); composite blocks expand into
stages before compilation. Architecture names never enter dispatch.

A stage with ``enabled: false`` is bypassed: its single output is one of its
inputs (the first, or the update of a residual), which makes ablations a
one-line recipe change. A block may declare an
``initialization_order`` (see ``net.blocks``) so that seeded parameters are
drawn in the order of a reference implementation.
"""

from __future__ import annotations

import math
from contextlib import nullcontext
from dataclasses import dataclass, field, replace
from typing import Mapping

import torch
from torch import nn

from net.blocks import expand_blocks
from net.features import (  # noqa: F401  (public re-exports)
    STRUCTURE_SLOTS,
    FeatureSpec,
    default_input_specs,
    global_irreps,
    index_spec,
    same_angular_basis,
    same_support,
)
from net.registry import (  # noqa: F401  (public re-exports)
    BuiltStage,
    Option,
    Port,
    TransformResult,
    build_transform,
    get_transform,
    register_transform,
    transform_catalog,
)

SPECIFICATION_KEYS = {
    "inputs",
    "blocks",
    "block_libraries",
    "stages",
    "outputs",
    "operator_layout",
}
STAGE_KEYS = {
    "id",
    "transform",
    "inputs",
    "outputs",
    "options",
    "role",
    "initialization",
    "enabled",
}
INITIALIZATION_KEYS = {"seed", "kind", "gain", "std", "bound"}


@dataclass
class StageRecord:
    """A compiled stage: its wiring, completed options and provenance."""

    id: str
    transform: str
    inputs: tuple
    outputs: tuple
    options: dict
    explicit: frozenset
    inferred: frozenset
    role: str = "feature"
    group: str | None = None
    static: bool = False
    initialization: dict = field(default_factory=dict)
    error: str | None = None
    enabled: bool = True


class Bypass(nn.Module):
    """A disabled stage: returns one of its inputs unchanged."""

    def __init__(self, position=0):
        super().__init__()
        self.position = position

    def forward(self, *inputs):
        return inputs[self.position]


def _initialize(module, initialization):
    kind = initialization.get("kind", "component")
    if kind not in {"component", "normal", "uniform", "zeros"}:
        raise ValueError(f"Unknown initialization kind {kind!r}")
    with torch.no_grad():
        for parameter in module.parameters():
            if kind == "normal":
                parameter.normal_(std=initialization.get("std", 0.01))
            elif kind == "uniform":
                bound = initialization.get("bound", 0.01)
                parameter.uniform_(-bound, bound)
            elif kind == "zeros":
                parameter.zero_()
            parameter.mul_(initialization.get("gain", 1.0))


def _check_initialization(initialization):
    unknown = set(initialization) - INITIALIZATION_KEYS
    if unknown:
        raise ValueError(f"Unknown initialization options: {sorted(unknown)}")
    for option in ("gain", "std", "bound"):
        if option in initialization and not math.isfinite(
            float(initialization[option])
        ):
            raise ValueError(f"Initialization {option} must be finite")
    seed = initialization.get("seed")
    if seed is not None and (
        isinstance(seed, bool) or not isinstance(seed, int) or seed < 0
    ):
        raise ValueError("Initialization seed must be a nonnegative integer")
    return seed


def _refine(slot, old, new):
    """Accept a tighter contract for an existing slot, never a different one."""
    if (old.domain, old.representation, old.frame, old.support) != (
        new.domain,
        new.representation,
        new.frame,
        new.support,
    ) or (
        (old.dim is not None and old.dim != new.dim)
        or (old.irreps is not None and old.irreps != new.irreps)
    ):
        raise ValueError(f"Stages disagree about the layout of {slot}")
    return new


def static_fingerprint(records):
    """Identity of the static stages: their transforms, wiring and options."""
    import hashlib
    import json

    payload = [
        (
            r.transform,
            r.inputs,
            r.outputs,
            sorted((k, repr(v)) for k, v in r.options.items()),
        )
        for r in records
        if r.static
    ]
    return hashlib.sha256(json.dumps(payload, default=str).encode()).hexdigest()[:16]


class FeaturePipeline(nn.Module):
    """A validated sequence of stages; outputs never overwrite saved slots."""

    def __init__(
        self,
        specification: Mapping,
        *,
        num_species,
        mapper=None,
        errors="raise",
        build=build_transform,
        static_only=False,
    ):
        """Compile ``specification``.

        With ``errors="collect"`` a failing stage is recorded (``record.error``)
        and every stage depending on it is skipped, so tools can report all
        problems of a recipe at once; the pipeline is then not runnable.
        ``static_only`` compiles only the parameter-free stages that depend on
        the structure alone (see :meth:`evaluate_static`).
        """
        super().__init__()
        if errors not in {"raise", "collect"}:
            raise ValueError("errors must be 'raise' or 'collect'")
        from copy import deepcopy

        unknown = set(specification) - SPECIFICATION_KEYS
        if unknown:
            raise ValueError(f"Unknown pipeline keys: {sorted(unknown)}")
        self.specification = dict(specification)
        self.operator_layout = deepcopy(specification.get("operator_layout"))
        self.errors: list[tuple[str, str]] = []
        from net.block_library import merged_blocks

        blocks = merged_blocks(specification)
        if errors == "raise":
            stages, self.groups = expand_blocks(list(specification["stages"]), blocks)
        else:
            # Expand instance by instance so one faulty block hides nothing else.
            stages, self.groups, failed_instances = [], {}, set()
            for stage in specification["stages"]:
                try:
                    expanded, self.groups = expand_blocks(
                        [stage], blocks, groups=self.groups
                    )
                    stages.extend(expanded)
                except (ValueError, KeyError, TypeError) as error:
                    sid = stage.get("id", "?") if isinstance(stage, dict) else "?"
                    self.errors.append((sid, f"Stage {sid}: {error}"))
                    failed_instances.update(
                        stage.get("outputs", ()) if isinstance(stage, dict) else ()
                    )
        self.convention_adapters = nn.ModuleDict()
        specs = default_input_specs(num_species)
        for slot, value in (specification.get("inputs") or {}).items():
            declared = FeatureSpec(**value)
            if (
                declared.representation == "geometry"
                and declared.frame == "global"
                and declared.angular_basis != "xyz"
            ):
                raise ValueError("Normalize Cartesian geometry to xyz at import")
            if slot in specs and declared != specs[slot]:
                # A recipe may tighten a built-in contract (e.g. give a width).
                try:
                    declared = _refine(slot, specs[slot], declared)
                except ValueError:
                    raise ValueError(
                        f"Cannot redefine built-in input contract {slot}"
                    ) from None
            specs[slot] = declared
        self.input_layouts = {
            slot: {"entity": spec.support or spec.domain, **spec.batch}
            for slot, spec in specs.items()
            if spec.batch is not None
        }
        self.input_adapters = tuple(
            slot for slot, spec in specs.items() if self._needs_adapter(spec)
        )
        specs = {slot: self._canonicalize(slot, spec) for slot, spec in specs.items()}
        static_slots = set(STRUCTURE_SLOTS) & set(specs)
        group_of = {
            stage_id: path
            for path, group in self.groups.items()
            for stage_id in group["stages"]
        }
        self.transforms = nn.ModuleDict()
        self.records: list[StageRecord] = []
        self._build_inputs: dict[str, tuple] = {}
        initial_rng = torch.random.get_rng_state()
        failed: set[str] = set(failed_instances) if errors == "collect" else set()
        seen: set[str] = set()
        for stage in stages:
            if static_only and not (
                set(stage["inputs"]) <= static_slots
                and (
                    stage.get("enabled", True) is False
                    or get_transform(stage["transform"]).pure
                )
            ):
                continue
            try:
                if (
                    not isinstance(stage, dict)
                    or "id" not in stage
                    or "transform" not in stage
                ):
                    raise ValueError("Every stage needs an id and a transform")
                if stage["id"] in seen:
                    raise ValueError(
                        f"Stage IDs must be unique module names: {stage['id']}"
                    )
                seen.add(stage["id"])
                if not isinstance(stage.get("options") or {}, Mapping):
                    raise ValueError(f"Stage {stage['id']}: options must be a mapping")
                built, reads, writes, initialization = self._compile_stage(
                    stage, specs, failed, mapper, build
                )
                for position, refined in built.refined_inputs.items():
                    slot = reads[position]
                    specs[slot] = _refine(slot, specs[slot], refined)
            except (ValueError, KeyError, TypeError) as error:
                if errors == "raise":
                    raise
                stage = stage if isinstance(stage, dict) else {}
                given = stage.get("options")
                given = dict(given) if isinstance(given, Mapping) else {}
                sid = stage.get("id", "?")
                self.errors.append((sid, str(error)))
                failed.update(stage.get("outputs", ()))
                self.records.append(
                    StageRecord(
                        id=sid,
                        transform=stage.get("transform", "?"),
                        inputs=tuple(stage.get("inputs", ())),
                        outputs=tuple(stage.get("outputs", ())),
                        options=given,
                        explicit=frozenset(given),
                        inferred=frozenset(),
                        role=stage.get("role", "feature"),
                        group=group_of.get(sid),
                        error=str(error),
                    )
                )
                continue
            sid = stage["id"]
            enabled = stage.get("enabled", True)
            static = (
                (not enabled or get_transform(stage["transform"]).pure)
                and not any(True for _ in built.module.parameters())
                and set(reads) <= static_slots
            )
            if static:
                static_slots.update(writes)
            elif static_only:
                continue
            self.transforms[sid] = built.module
            self.records.append(
                StageRecord(
                    id=sid,
                    transform=stage["transform"],
                    inputs=reads,
                    outputs=writes,
                    options=built.options,
                    explicit=built.explicit,
                    inferred=built.inferred,
                    role=stage.get("role", "feature"),
                    group=group_of.get(sid),
                    static=static,
                    initialization=initialization,
                    enabled=enabled,
                )
            )
            specs.update(
                (slot, self._canonicalize(slot, spec))
                for slot, spec in zip(writes, built.outputs)
            )
        self.outputs = {} if static_only else dict(specification["outputs"])
        undefined = set(self.outputs.values()) - specs.keys() - failed
        if undefined:
            message = f"Pipeline outputs reference undefined slots {sorted(undefined)}"
            if errors == "raise":
                raise ValueError(message)
            self.errors.append(("outputs", message))
        self.feature_specs = specs
        if static_only:
            # Modules built only to be discarded must not shift model seeds.
            torch.random.set_rng_state(initial_rng)
        elif not self.errors:
            self._apply_initialization_order(initial_rng, mapper, build)
        del self._build_inputs
        self.static_slots = frozenset(
            slot for record in self.records if record.static for slot in record.outputs
        )
        self.static_fingerprint = static_fingerprint(self.records)

    def _compile_stage(self, stage, specs, failed, mapper, build):
        extra = set(stage) - STAGE_KEYS
        if extra:
            raise ValueError(f"Unknown stage keys: {sorted(extra)}")
        sid = stage["id"]
        if not sid or "." in sid or sid in self.transforms:
            raise ValueError(f"Stage IDs must be unique module names: {sid}")
        reads, writes = tuple(stage["inputs"]), tuple(stage["outputs"])
        unconnected = [position for position, slot in enumerate(reads) if not slot]
        if unconnected:
            ports = [port.name for port in get_transform(stage["transform"]).inputs]
            names = [
                ports[min(i, len(ports) - 1)] if ports else f"#{i}" for i in unconnected
            ]
            raise ValueError(f"Stage {sid}: input(s) {', '.join(names)} not connected")
        upstream = set(reads) & failed
        if upstream:
            raise ValueError(
                f"Stage {sid} reads {sorted(upstream)}, produced by a failing stage"
            )
        missing = set(reads) - specs.keys()
        if missing:
            raise ValueError(f"Stage {sid} reads undefined slots: {sorted(missing)}")
        if len(set(writes)) != len(writes) or set(writes) & specs.keys():
            raise ValueError(f"Stage {sid} overwrites a saved feature slot")
        initialization = dict(stage.get("initialization") or {})
        _check_initialization(initialization)
        enabled = stage.get("enabled", True)
        if not isinstance(enabled, bool):
            raise ValueError(f"Stage {sid}: enabled must be true or false")
        input_specs = [specs[s] for s in reads]
        if not enabled:
            position = self._bypass_position(stage["transform"])
            if len(writes) != 1 or position >= len(reads):
                raise ValueError(
                    f"Stage {sid}: a disabled stage needs one output and an input "
                    "to pass through"
                )
            given = dict(stage.get("options") or {})
            try:  # options are checked even while the stage is bypassed
                get_transform(stage["transform"]).resolve_options(
                    given, input_specs, mapper
                )
            except ValueError as error:
                if "Unknown transformation" not in str(error):
                    raise ValueError(f"Stage {sid}: {error}") from error
            built = BuiltStage(
                Bypass(position),
                [input_specs[position]],
                given,
                frozenset(given),
                frozenset(),
                {},
            )
            return built, reads, writes, initialization
        built = self._build(stage, input_specs, mapper, build)
        self._build_inputs[sid] = (stage, input_specs)
        if len(writes) != len(built.outputs):
            raise ValueError(
                f"Stage {sid} declares {len(writes)} outputs; "
                f"{stage['transform']} produces {len(built.outputs)}"
            )
        return built, reads, writes, initialization

    @staticmethod
    def _bypass_position(transform):
        try:
            return get_transform(transform).bypass
        except ValueError:
            return 0  # a disabled block instance passes its first input

    @staticmethod
    def _build(stage, input_specs, mapper, build):
        initialization = dict(stage.get("initialization") or {})
        seed = initialization.get("seed")
        with torch.random.fork_rng(devices=[]) if seed is not None else nullcontext():
            if seed is not None:
                torch.random.default_generator.manual_seed(seed)
            try:
                built = build(
                    stage["transform"],
                    stage.get("options") or {},
                    input_specs,
                    mapper=mapper,
                    detailed=True,
                )
            except (ValueError, TypeError, KeyError) as error:
                raise ValueError(
                    f"Stage {stage['id']} ({stage['transform']}): {error}"
                ) from error
            if initialization:
                _initialize(built.module, initialization)
        return built

    def _initialization_sequence(self):
        """Stage ids in parameter-drawing order, or ``None`` for list order.

        A block instance with an ``initialization_order`` contributes its
        stages in that order (repeats draw again) where its first stage
        appears; every other stage keeps its list position.
        """
        ordered = {
            path: group
            for path, group in self.groups.items()
            if group.get("initialization_order")
        }
        if not ordered:
            return None
        parameterized = {
            r.id
            for r in self.records
            if any(True for _ in self.transforms[r.id].parameters())
        }

        def expand(path):
            group = self.groups[path]
            if not group.get("initialization_order"):
                return [sid for sid in group["stages"] if sid in self._build_inputs]
            sequence = []
            for entry in group["initialization_order"]:
                if entry in self.groups:
                    sequence.extend(expand(entry))
                elif entry in self._build_inputs:
                    sequence.append(entry)
            missing = (parameterized & set(group["stages"])) - set(sequence)
            if missing:
                raise ValueError(
                    f"Block instance {path}: initialization_order omits the "
                    f"parameterized stages {sorted(missing)}"
                )
            return sequence

        def outermost(sid):
            paths = [p for p, g in ordered.items() if sid in g["stages"]]
            return min(paths, key=len) if paths else None

        sequence, emitted = [], set()
        for record in self.records:
            path = outermost(record.id)
            if path is None:
                if record.id in self._build_inputs:
                    sequence.append(record.id)
            elif path not in emitted:
                emitted.add(path)
                sequence.extend(expand(path))
        return sequence

    def _apply_initialization_order(self, initial_rng, mapper, build):
        sequence = self._initialization_sequence()
        if sequence is None:
            return
        torch.random.set_rng_state(initial_rng)
        for sid in sequence:
            stage, input_specs = self._build_inputs[sid]
            self.transforms[sid] = self._build(stage, input_specs, mapper, build).module

    @staticmethod
    def _needs_adapter(spec):
        return (
            spec.frame == "global"
            and spec.irreps
            and spec.angular_basis not in {None, "xyz"}
        )

    def _canonicalize(self, slot, spec):
        if self._needs_adapter(spec):
            from net.transforms.orbital import IrrepCartesianBasis

            self.convention_adapters[slot] = IrrepCartesianBasis(
                spec.irreps, spec.angular_basis, "xyz"
            )
            return replace(spec, angular_basis="xyz")
        return spec

    @property
    def bindings(self):
        return [(record.id, record.inputs, record.outputs) for record in self.records]

    @property
    def roles(self):
        return {record.id: record.role for record in self.records}

    def stages_with_role(self, role):
        """``{stage id: module}`` of the stages declared with ``role``."""
        return {r.id: self.transforms[r.id] for r in self.records if r.role == role}

    def required_inputs(self):
        """Input slots read by some stage (and not produced by an earlier one)."""
        produced, required = set(), set()
        for record in self.records:
            required.update(set(record.inputs) - produced)
            produced.update(record.outputs)
        return required

    def evaluate_static(self, inputs):
        """Outputs of the parameter-free stages that depend on the structure only."""
        slots, computed = dict(inputs), {}
        for record in self.records:
            if record.static and not set(record.outputs) <= slots.keys():
                values = self._run(record, slots)
                slots.update(values)
                computed.update(values)
        return computed

    def _run(self, record, slots, **kwargs):
        missing = set(record.inputs) - slots.keys()
        if missing:
            raise ValueError(
                f"Stage {record.id} is missing runtime inputs {sorted(missing)}"
            )
        module = self.transforms[record.id]
        extra = {}
        if getattr(module, "accepts_physical_derivatives", False):
            extra["physical_derivatives"] = kwargs.get("physical_derivatives", False)
        result = module(*(slots[key] for key in record.inputs), **extra)
        writes = record.outputs
        if len(writes) == 1:
            values = (result,)
        elif isinstance(result, dict):
            values = tuple(result.values())
        else:
            values = tuple(result)
        if len(values) != len(writes):
            raise RuntimeError(
                f"Stage {record.id} returned an unexpected number of tensors"
            )
        return {
            slot: (
                self.convention_adapters[slot](value)
                if slot in self.convention_adapters
                else value
            )
            for slot, value in zip(writes, values)
        }

    def forward(
        self,
        inputs,
        *,
        return_trace=False,
        physical_derivatives=False,
        activation_mags=None,
    ):
        """Run every stage. Static stages whose outputs are supplied are skipped."""
        slots = dict(inputs)
        for slot in self.input_adapters:
            if slot in slots:
                slots[slot] = self.convention_adapters[slot](slots[slot])
        for record in self.records:
            if record.static and set(record.outputs) <= slots.keys():
                continue
            values = self._run(record, slots, physical_derivatives=physical_derivatives)
            if activation_mags is not None:
                from net.features import irrep_magnitudes

                for slot, value in values.items():
                    spec = self.feature_specs[slot]
                    if spec.representation == "irreps":
                        for irrep, magnitude in irrep_magnitudes(
                            value, spec.irreps
                        ).items():
                            activation_mags[f"mag_{record.id}_{irrep}"] = magnitude
            slots.update(values)
        output = {name: slots[slot] for name, slot in self.outputs.items()}
        return (output, slots) if return_trace else output
