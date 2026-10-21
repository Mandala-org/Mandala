"""Resolve declarative model recipes against an explicit orbital vocabulary.

Resolution depends on feature contracts and tensor-product reachability, never
on an architecture name. Presets and custom hybrid recipes follow the same
path. No labels or fitted statistics enter configuration resolution.

The reachability-completion rule is adapted from DeepH-E3's ``refine_post_node``
(commit 036bf95b996052ef778c5c38e3c54c2ac0e2b591).
Copyright (c) 2023 Xiaoxun-Gong, MIT License:
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import torch
from ase.data import atomic_numbers
from e3nn.o3 import Irrep, Irreps
from omegaconf import OmegaConf

from net.operator_codecs import ShellTensorCodec


def validate_configuration_sections(config):
    """Reject misspelled policies instead of silently using another procedure."""
    allowed = {
        "schema_version",
        "reference",
        "parameters",
        "species",
        "layout",
        "layout_rules",
        "codec",
        "data",
        "training",
        "objective",
        "optimization",
        "pipeline",
    }
    unknown = set(config) - allowed
    if unknown:
        raise ValueError(f"Unknown configuration sections: {sorted(unknown)}")
    sections = {
        "data": {
            "root",
            "splits",
            "matrix_files",
            "support_file",
            "local_coordinates",
            "num_l",
            "frame_source",
            "target_frame",
            "units",
            "graph_source",
            "graph_radius",
        },
        "objective": {"kind", "epoch_weight", "weights", "complex_reduction"},
        "optimization": {"optimizer", "scheduler", "stop_lr"},
        "pipeline": {"inputs", "stages", "outputs", "operator_layout"},
    }
    for section, names in sections.items():
        values = config.get(section, {})
        if not OmegaConf.is_dict(values) and not isinstance(values, Mapping):
            raise ValueError(f"{section} must be a mapping")
        unknown = set(values) - names
        if unknown:
            raise ValueError(f"Unknown {section} options: {sorted(unknown)}")
    objective = config.get("objective", {})
    if objective.get("kind", "mse") not in {"mse", "mae"}:
        raise ValueError("objective.kind must be mse or mae")
    if objective.get("epoch_weight", "edges") not in {"edges", "elements"}:
        raise ValueError("objective.epoch_weight must be edges or elements")
    if objective.get("complex_reduction", "magnitude") not in {
        "magnitude",
        "components",
    }:
        raise ValueError("objective.complex_reduction must be magnitude or components")
    units = config.get("data", {}).get("units", {})
    expected = {
        "energy": "eV",
        "length": "Angstrom",
        "orbitals": "openmx",
        "spin": "spin_major",
    }
    for key, value in units.items():
        if key not in expected or value != expected[key]:
            raise ValueError(
                f"Native operator interchange requires {key}={expected.get(key)!r}"
            )


def _path_exists(left, right, output):
    return any(output in a * b for _, a in left for _, b in right)


def complete_tensor_product_intermediate(
    candidate, node_input, edge_input, angular, output
):
    """Add reachable irrep types needed by the following endpoint product.

    Candidate types keep their original order and multiplicities. New types
    follow the source's smallest-first greedy path choice; their multiplicity
    follows an existing type with the same degree, or the candidate's last
    field. This is an explicit configuration-time shape rule, not a dynamic
    modification of a forward pass.
    """
    candidate, node_input, edge_input, angular, output = map(
        Irreps, (candidate, node_input, edge_input, angular, output)
    )
    angular = angular.sort().irreps.simplify()
    if not candidate:
        raise ValueError("Intermediate completion needs a nonempty candidate layout.")
    alternatives = []
    for _, required in output.sort().irreps.simplify():
        if _path_exists(candidate + edge_input, angular, required):
            continue
        paths = []
        for _, harmonic in angular:
            parity = required.p * harmonic.p
            for ell in range(abs(required.l - harmonic.l), required.l + harmonic.l + 1):
                irrep = Irrep(ell, parity)
                if irrep not in paths:
                    paths.append(irrep)
        if not paths:
            raise ValueError(f"No angular path can generate required irrep {required}.")
        alternatives.append(sorted(paths, reverse=True))
    additions, excluded = [], set()
    while True:
        additions = [irrep for irrep in additions if irrep not in excluded]
        for paths in alternatives:
            while paths and paths[-1] in excluded:
                paths.pop()
            if not paths:
                raise ValueError(
                    "No reachable intermediate can complete the output layout."
                )
            if paths[-1] not in additions:
                additions.append(paths[-1])
        invalid = [
            irrep
            for irrep in additions
            if not _path_exists(node_input + edge_input, angular, irrep)
        ]
        if not invalid:
            break
        excluded.update(invalid)
    extra = Irreps()
    for irrep in additions:
        multiplicities = [mul for mul, old in candidate if old.l == irrep.l]
        if not multiplicities:
            multiplicities = [candidate[-1].mul]
        for multiplicity in multiplicities:
            extra += Irreps([(multiplicity, irrep)])
    return str(candidate + extra)


def _species_order(shells, requested):
    if requested is not None:
        result = tuple(map(str, requested))
        if len(set(result)) != len(result) or set(result) != set(shells):
            raise ValueError("species.order must contain every species exactly once.")
        return result
    # The reference HDF5 loaders index species by ascending atomic number.
    # Non-chemical user labels remain valid with their explicit insertion order.
    if all(name in atomic_numbers for name in shells):
        return tuple(sorted(shells, key=atomic_numbers.__getitem__))
    if all(name.isdecimal() for name in shells):
        return tuple(sorted(shells, key=int))
    return tuple(shells)


def resolve_configuration(
    source: str | Path | Mapping,
    *,
    species_shells: Mapping,
    spinful: bool | None = None,
    overrides: Mapping | None = None,
):
    """Return ``(plain_resolved_configuration, native_shell_codec)``.

    ``source`` is a YAML filename or mapping. ``overrides`` is a nested mapping
    merged before interpolation. Species order, shell channels, dimensions and
    optional declared irrep-layout rules are then resolved. The returned codec
    uses the caller's current default torch dtype; set it before resolution.

    ``layout_rules`` supports ``channel_harmonics`` and
    ``tensor_product_completion``. They can be used by arbitrary custom recipes.
    """
    if isinstance(source, (str, Path)):
        config = OmegaConf.load(Path(source))
    elif OmegaConf.is_config(source):
        config = OmegaConf.create(OmegaConf.to_container(source, resolve=False))
    elif isinstance(source, Mapping):
        config = OmegaConf.create(dict(source))
    else:
        raise TypeError("Configuration must be a YAML path or mapping.")
    if overrides:
        config = OmegaConf.merge(config, overrides)
    validate_configuration_sections(config)
    if config.get("schema_version", 1) != 1:
        raise ValueError("Unsupported modular configuration schema_version.")
    shells = {
        str(key): tuple(map(int, values)) for key, values in species_shells.items()
    }
    if not shells:
        raise ValueError("Provide the complete nonempty species-shell vocabulary.")
    requested_order = OmegaConf.select(config, "species.order")
    order = _species_order(shells, requested_order)
    shells = {key: shells[key] for key in order}
    OmegaConf.update(
        config,
        "species",
        {
            "order": list(order),
            "count": len(order),
            "pair_count": len(order) ** 2,
            "embedding_count": len(order) + 5,
            "shells": {key: list(value) for key, value in shells.items()},
        },
        merge=True,
    )
    if "codec" not in config:
        config.codec = {}
    if spinful is not None:
        config.codec.spinful = bool(spinful)
    codec_options = OmegaConf.to_container(config.codec, resolve=True)
    codec = ShellTensorCodec.from_species_shells(
        shells,
        dtype=torch.get_default_dtype(),
        **codec_options,
    )
    layout = {
        "shell_pairs": [list(pair) for pair in codec.shell_pairs],
        "coupled_irreps": str(codec.irreps_out),
        "post_irreps": str(codec.irreps_out.sort().irreps.simplify()),
        "coupled_dim": codec.irreps_out.dim,
        "packed_dim": codec.packed_dim,
        "real_packed_dim": codec.packed_dim * (8 if codec.spinful else 1),
        "spinful": codec.spinful,
        "ignore_parity": codec.ignore_parity,
        "sort_irreps": bool(codec_options.get("sort_irreps", False)),
    }
    OmegaConf.update(config, "layout", layout, merge=True)
    for name in config.get("layout_rules", {}):
        rule = OmegaConf.to_container(config.layout_rules[name], resolve=True)
        operation = rule.pop("kind")
        if operation == "channel_harmonics":
            hidden = Irreps(rule.pop("hidden_irreps"))
            channels = hidden.count(Irrep("0e"))
            lmax = int(rule.pop("lmax"))
            if not channels:
                raise ValueError("Channel harmonics require at least one even scalar.")
            value = str(
                (Irreps.spherical_harmonics(lmax) * channels).sort()[0].simplify()
            )
            if rule:
                raise ValueError(f"Unknown channel_harmonics options: {sorted(rule)}")
        elif operation == "tensor_product_completion":
            value = complete_tensor_product_intermediate(**rule)
        else:
            raise ValueError(f"Unknown layout rule {operation!r}.")
        OmegaConf.update(config, f"layout.{name}", value, merge=False)
    if "pipeline" in config:
        signature = codec.layout_signature()
        existing = OmegaConf.select(config, "pipeline.operator_layout")
        if (
            existing is not None
            and OmegaConf.to_container(existing, resolve=True) != signature
        ):
            raise ValueError(
                "Frozen pipeline operator layout differs from its declared codec"
            )
        OmegaConf.update(config, "pipeline.operator_layout", signature, merge=False)
    resolved = OmegaConf.to_container(config, resolve=True, throw_on_missing=True)
    if "pipeline" in resolved:
        weights = resolved.get("objective", {}).get("weights", {})
        unknown = set(weights) - set(resolved["pipeline"]["outputs"])
        if unknown:
            raise ValueError(
                f"Objective weights name unknown operators: {sorted(unknown)}"
            )
    return resolved, codec
