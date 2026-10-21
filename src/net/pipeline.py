"""Typed, ordered composition of reusable tensor transformations.

Architecture names are deliberately absent from dispatch. Presets use the same
stage vocabulary as custom models; saved feature slots support multiple streams
and long residual links without embedding Python expressions in configuration.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Mapping

import torch
from torch import nn
from e3nn import o3


@dataclass(frozen=True)
class FeatureSpec:
    domain: str
    representation: str = "scalar"
    irreps: str | None = None
    frame: str = "global"
    width: int | None = None
    support: str | None = None
    angular_basis: str | None = None

    def __post_init__(self):
        if self.domain not in {"node", "edge", "bond_neighbor", "graph", "relation"}:
            raise ValueError(f"Unknown feature domain {self.domain}")
        if self.representation not in {
            "scalar",
            "irreps",
            "channel_irreps",
            "index",
            "geometry",
            "matrix",
            "local_components",
        }:
            raise ValueError(f"Unknown representation {self.representation}")
        if self.support is None and self.domain in {
            "edge",
            "relation",
            "bond_neighbor",
        }:
            object.__setattr__(self, "support", "output")
        if self.support is not None and (
            not isinstance(self.support, str) or not self.support
        ):
            raise ValueError("Feature support must be a nonempty index-set name")
        if self.angular_basis is None and (
            self.representation == "geometry"
            or (
                self.irreps is not None
                and any(ir.l > 0 for _, ir in o3.Irreps(self.irreps))
            )
        ):
            object.__setattr__(self, "angular_basis", "xyz")
        if self.angular_basis is not None and sorted(self.angular_basis) != [
            "x",
            "y",
            "z",
        ]:
            raise ValueError("Angular basis must name each Cartesian axis exactly once")
        if self.irreps is not None:
            dimension = o3.Irreps(self.irreps).dim
            if self.representation == "irreps" and self.width not in (None, dimension):
                raise ValueError("Feature width disagrees with its irrep layout")

    @property
    def dim(self):
        return o3.Irreps(self.irreps).dim if self.irreps is not None else self.width


def default_input_specs(num_species):
    return {
        "node_type_idx": FeatureSpec("node", "index", frame="none"),
        "node_one_hot": FeatureSpec(
            "node", irreps=f"{num_species}x0e", width=num_species
        ),
        "edge_type_idx": FeatureSpec("edge", "index", frame="none"),
        "edge_one_hot": FeatureSpec(
            "edge", irreps=f"{num_species**2}x0e", width=num_species**2
        ),
        "edge_lengths": FeatureSpec("edge", width=1),
        "edge_vectors": FeatureSpec("edge", "geometry", width=3),
        "edge_index": FeatureSpec("relation", "index", frame="none"),
        "message_edge_index": FeatureSpec(
            "relation", "index", frame="none", support="message"
        ),
        "message_vectors": FeatureSpec("edge", "geometry", width=3, support="message"),
        "message_lengths": FeatureSpec("edge", width=1, support="message"),
        "message_to_output": FeatureSpec(
            "relation", "index", frame="none", support="message"
        ),
        "node_batch": FeatureSpec("node", "index", frame="none"),
        "edge_batch": FeatureSpec("edge", "index", frame="none"),
        "selfloop_edge": FeatureSpec("edge", "index", frame="none"),
        "sub_atom_idx": FeatureSpec("bond_neighbor", "index", frame="none"),
        "sub_edge_idx": FeatureSpec("bond_neighbor", "index", frame="none"),
        "sub_edge_ang": FeatureSpec("bond_neighbor", "local_components", frame="bond"),
        "sub_index": FeatureSpec("bond_neighbor", "index", frame="none"),
        "bond_frames": FeatureSpec("edge", "geometry", frame="bond", width=9),
    }


class _OneHot(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes

    def forward(self, indices):
        return torch.nn.functional.one_hot(indices, self.num_classes).to(
            torch.get_default_dtype()
        )


class _Add(nn.Module):
    def forward(self, *args):
        result = args[0]
        for x in args[1:]:
            result = result + x
        return result


class _Concat(nn.Module):
    def forward(self, *args):
        return torch.cat(args, dim=-1)


class _Harmonics(nn.Module):
    def __init__(self, irreps, permutation=(0, 1, 2), normalization="component"):
        super().__init__()
        if sorted(permutation) != [0, 1, 2]:
            raise ValueError(
                "permutation must contain each Cartesian axis exactly once"
            )
        self.permutation = tuple(permutation)
        self.sh = o3.SphericalHarmonics(
            irreps, normalize=True, normalization=normalization
        )

    def forward(self, vectors):
        return self.sh(vectors[:, self.permutation])


def _global_irreps(spec, domain=None):
    if (
        spec.frame != "global"
        or spec.irreps is None
        or spec.representation not in {"scalar", "irreps"}
    ):
        raise ValueError(
            "This transformation requires global irrep features; insert an explicit frame adapter"
        )
    if domain and spec.domain != domain:
        raise ValueError(f"Expected {domain} features, received {spec.domain}")
    return o3.Irreps(spec.irreps)


def _same_support(*specs):
    """Require the same entities in the same order, not merely equal shapes."""
    supports = {spec.support for spec in specs}
    if len(supports) > 1:
        raise ValueError(
            f"Feature/index support mismatch: {sorted(map(str, supports))}"
        )
    return specs[0].support if specs else None


def _index_spec(spec, domain):
    if spec.domain != domain or spec.representation != "index":
        raise ValueError(f"Expected {domain}-scoped integer index features")


def _same_angular_basis(*specs):
    """Check non-scalar representations; invariant scalar channels are neutral."""
    bases = {
        spec.angular_basis
        for spec in specs
        if spec.irreps is not None and any(ir.l > 0 for _, ir in o3.Irreps(spec.irreps))
    }
    if len(bases) > 1:
        raise ValueError(
            f"Angular basis mismatch: {sorted(map(str, bases))}; insert irrep_cartesian_basis"
        )
    return next(iter(bases)) if bases else None


class _LocalOrbitalOutput(nn.Module):
    """Decode scalar orbital entries and explicitly ordered SOC components.

    The original local-coordinate reader interleaves real/imaginary values per
    orbital in order uu, dd, ud, du. The common codec uses uu, ud, du, dd.
    """

    def __init__(
        self, packed_dim, spinful=False, component_order=("uu", "dd", "ud", "du")
    ):
        super().__init__()
        self.packed_dim = int(packed_dim)
        self.spinful = bool(spinful)
        if self.packed_dim <= 0:
            raise ValueError("packed_dim must be positive")
        if len(component_order) != 4 or set(component_order) != {
            "uu",
            "ud",
            "du",
            "dd",
        }:
            raise ValueError(
                "SOC component_order must be a permutation of uu, ud, du, dd"
            )
        self.component_indices = tuple(
            component_order.index(c) for c in ("uu", "ud", "du", "dd")
        )

    def forward(self, value):
        expected = self.packed_dim * (8 if self.spinful else 1)
        if value.ndim != 2 or value.shape[1] != expected:
            raise ValueError(f"Local orbital output expects shape (edges,{expected})")
        if not self.spinful:
            return value
        pairs = value.reshape(value.shape[0], self.packed_dim, 4, 2)
        complex_values = torch.complex(pairs[..., 0], pairs[..., 1])
        return complex_values[:, :, self.component_indices].transpose(1, 2)


def build_transform(name, options, inputs, *, mapper=None):
    """Return (module, output specs). Inputs are positional, never inferred by name."""
    p = dict(options)
    if name == "embedding":
        if inputs[0].representation != "index":
            raise ValueError("embedding requires integer species indices")
        n, width = p.pop("num_embeddings"), p.pop("width")
        n += p.pop("extra_embeddings", 0)
        return nn.Embedding(n, width, **p), [
            FeatureSpec(
                inputs[0].domain,
                irreps=f"{width}x0e",
                width=width,
                support=inputs[0].support,
            )
        ]
    if name == "one_hot":
        n = p["num_classes"]
        _index_spec(inputs[0], inputs[0].domain)
        return _OneHot(**p), [
            FeatureSpec(
                inputs[0].domain, irreps=f"{n}x0e", width=n, support=inputs[0].support
            )
        ]
    if name == "irrep_linear":
        irreps = _global_irreps(inputs[0])
        out = p.pop("irreps_out")
        return o3.Linear(irreps, out, **p), [
            FeatureSpec(
                inputs[0].domain,
                "irreps",
                str(o3.Irreps(out)),
                support=inputs[0].support,
                angular_basis=inputs[0].angular_basis,
            )
        ]
    if name == "bond_local_linear":
        if (
            len(inputs) != 1
            or inputs[0].representation != "local_components"
            or inputs[0].frame != "bond"
        ):
            raise ValueError(
                "bond_local_linear requires explicitly rotated bond-frame components"
            )
        width = p.pop("out_features")
        return nn.Linear(inputs[0].dim, width, **p), [
            FeatureSpec(
                inputs[0].domain,
                "local_components",
                frame="bond",
                width=width,
                support=inputs[0].support,
            )
        ]
    if name == "scalar_linear":
        if inputs[0].frame != "global" or (
            inputs[0].irreps
            and any(ir.l or ir.p != 1 for _, ir in o3.Irreps(inputs[0].irreps))
        ):
            raise ValueError("scalar_linear accepts invariant scalar channels")
        width = p.pop("out_features")
        return nn.Linear(inputs[0].dim, width, **p), [
            FeatureSpec(
                inputs[0].domain,
                irreps=f"{width}x0e",
                width=width,
                support=inputs[0].support,
            )
        ]
    if name in {"add", "concat"}:
        if not inputs or any(
            (s.domain, s.frame, s.representation)
            != (inputs[0].domain, inputs[0].frame, inputs[0].representation)
            for s in inputs
        ):
            raise ValueError(
                "Combined features must share entity, frame and representation"
            )
        _same_support(*inputs)
        _same_angular_basis(*inputs)
        if name == "add":
            if any(s != inputs[0] for s in inputs):
                raise ValueError("Residual addition requires identical feature layouts")
            return _Add(), [inputs[0]]
        ir = (
            str(sum((o3.Irreps(s.irreps) for s in inputs), o3.Irreps()))
            if all(s.irreps for s in inputs)
            else None
        )
        width = (
            sum(s.dim for s in inputs)
            if all(s.dim is not None for s in inputs)
            else None
        )
        return _Concat(), [
            FeatureSpec(
                inputs[0].domain,
                inputs[0].representation,
                ir,
                inputs[0].frame,
                width,
                inputs[0].support,
                _same_angular_basis(*inputs),
            )
        ]
    if name == "gaussian_basis":
        from net.transforms.equivariant import GaussianBasis

        width = p["n_gaussians"]
        if inputs[0].domain != "edge" or inputs[0].representation not in {
            "scalar",
            "geometry",
        }:
            raise ValueError("gaussian_basis requires edge-scoped distances")
        return GaussianBasis(**p), [
            FeatureSpec(
                inputs[0].domain,
                irreps=f"{width}x0e",
                width=width,
                support=inputs[0].support,
            )
        ]
    if name == "spherical_harmonics":
        if inputs[0].representation != "geometry" or inputs[0].frame != "global":
            raise ValueError("spherical_harmonics requires global Cartesian vectors")
        basis = inputs[0].angular_basis or "xyz"
        basis = "".join(basis[index] for index in p.get("permutation", (0, 1, 2)))
        return _Harmonics(**p), [
            FeatureSpec(
                inputs[0].domain,
                "irreps",
                str(o3.Irreps(p["irreps"])),
                support=inputs[0].support,
                angular_basis=basis,
            )
        ]
    if name in {"endpoint_tensor_product_node", "endpoint_tensor_product_edge"}:
        from net.transforms.equivariant import (
            EndpointTensorProductNodeUpdate,
            EndpointTensorProductEdgeUpdate,
        )

        if len(inputs) < 7:
            raise ValueError(
                "Endpoint tensor products require node, species, angular, edge, radial, edge index and node batch inputs"
            )
        node = _global_irreps(inputs[0], "node")
        _index_spec(inputs[5], "relation")
        _index_spec(inputs[6], "node")
        support = _same_support(inputs[2], inputs[3], inputs[4], inputs[5])
        basis = _same_angular_basis(inputs[0], inputs[2], inputs[3])
        if inputs[4].domain != "edge":
            raise ValueError("Radial features must be edge-scoped")
        if name.endswith("node"):
            _global_irreps(inputs[1], "node")
            _same_support(inputs[0], inputs[1], inputs[6])
            if len(inputs) > 7:
                _index_spec(inputs[7], "edge")
                _same_support(inputs[5], inputs[7])
            if len(inputs) > 8:
                _same_support(inputs[5], inputs[8])
            sh, edge = (
                _global_irreps(inputs[2], "edge"),
                _global_irreps(inputs[3], "edge"),
            )
            module = EndpointTensorProductNodeUpdate(
                irreps_in_node=node, irreps_in_edge=edge, irreps_sh=sh, **p
            )
            domain = "node"
        else:
            _global_irreps(inputs[1], "edge")
            _same_support(inputs[1], inputs[5])
            _same_support(inputs[0], inputs[6])
            sh, edge = (
                _global_irreps(inputs[2], "edge"),
                _global_irreps(inputs[3], "edge"),
            )
            module = EndpointTensorProductEdgeUpdate(
                irreps_in_node=node, irreps_in_edge=edge, irreps_sh=sh, **p
            )
            domain = "edge"
        return module, [
            FeatureSpec(
                domain,
                "irreps",
                str(module.irreps_out),
                support=inputs[0].support if domain == "node" else support,
                angular_basis=basis,
            )
        ]
    if name == "graph_irrep_norm":
        from net.transforms.equivariant import GraphIrrepNorm

        if len(inputs) != 2:
            raise ValueError(
                "Graph normalization requires features and entity batch indices"
            )
        _index_spec(inputs[1], inputs[0].domain)
        _same_support(*inputs)
        return GraphIrrepNorm(_global_irreps(inputs[0]), **p), [inputs[0]]
    if name in {
        "scalar_gated_node",
        "endpoint_scalar_edge",
        "local_coordinate_message",
        "channel_linear",
        "pair_distance_affine",
        "irrep_to_bond",
    }:
        from net.transforms.local_coordinates import build_local_transform

        if name == "scalar_gated_node":
            _index_spec(inputs[1], "relation")
            _same_support(inputs[1], inputs[2], inputs[4])
            _index_spec(inputs[3], "node")
            _same_support(inputs[0], inputs[3])
            support = inputs[0].support
        elif name == "endpoint_scalar_edge":
            _index_spec(inputs[1], "relation")
            support = _same_support(inputs[1], inputs[2])
        elif name == "local_coordinate_message":
            if len(inputs) != 7:
                raise ValueError(
                    "Local coordinate messages require all four bond-neighborhood relations"
                )
            for spec in (inputs[2], inputs[3], inputs[5]):
                _index_spec(spec, "bond_neighbor")
            if inputs[4].domain != "bond_neighbor":
                raise ValueError("Local angles must be bond-neighborhood-scoped")
            support = _same_support(*inputs[1:])
        elif name == "irrep_to_bond":
            _index_spec(inputs[1], "relation")
            if inputs[2].domain != "edge" or inputs[2].frame != "bond":
                raise ValueError("Bond-frame rotation requires edge-scoped bond frames")
            support = _same_support(inputs[1], inputs[2])
            basis = inputs[0].angular_basis or "xyz"
            permutation = tuple("xyz".index(axis) for axis in basis)
            if "axis_permutation" in p and tuple(p["axis_permutation"]) != permutation:
                raise ValueError(
                    "Bond-frame axis_permutation disagrees with input angular basis"
                )
            p["axis_permutation"] = permutation
        elif name == "pair_distance_affine":
            _index_spec(inputs[1], "edge")
            support = _same_support(*inputs)
        else:
            support = inputs[0].support
        module, produced = build_local_transform(name, p, inputs)
        return module, [
            replace(
                spec,
                support=support,
                angular_basis=(
                    inputs[0].angular_basis
                    if name in {"irrep_to_bond", "channel_linear"}
                    else spec.angular_basis
                ),
            )
            for spec in produced
        ]
    if name in {
        "neighbor_density",
        "symmetric_product",
        "angular_degree_expansion",
        "cutoff_radial_expansion",
    }:
        from net.transforms.cluster import build_cluster_transform

        return build_cluster_transform(name, p, inputs)
    if name == "shell_tensor_decode":
        from net.operator_codecs import ShellTensorCodec

        p.setdefault("dtype", torch.get_default_dtype())
        codec = ShellTensorCodec(**p)
        if _global_irreps(inputs[0], "edge") != codec.irreps_out:
            raise ValueError(
                "Readout irreps must exactly match the shell-channel codec"
            )
        basis = _same_angular_basis(inputs[0])
        if basis not in {None, "yzx"}:
            raise ValueError(
                "Native shell decoding requires cyclic yzx irreps; insert irrep_cartesian_basis"
            )
        return _CodecDecode(codec), [
            FeatureSpec(
                "edge",
                "matrix",
                frame="global",
                width=codec.packed_dim,
                support=inputs[0].support,
            )
        ]
    if name == "local_orbital_output":
        if (
            len(inputs) != 1
            or inputs[0].domain != "edge"
            or inputs[0].representation != "local_components"
            or inputs[0].frame != "bond"
        ):
            raise ValueError(
                "local_orbital_output requires bond-frame local edge components"
            )
        module = _LocalOrbitalOutput(**p)
        expected = module.packed_dim * (8 if module.spinful else 1)
        if inputs[0].dim is not None and inputs[0].dim != expected:
            raise ValueError("Local orbital channel width disagrees with packed output")
        return module, [
            FeatureSpec(
                "edge",
                "matrix",
                frame="bond",
                width=module.packed_dim,
                support=inputs[0].support,
            )
        ]
    if name == "species_irrep_affine":
        from net.training_policies import SpeciesIrrepAffine

        _index_spec(inputs[1], "edge")
        _same_support(*inputs)
        return SpeciesIrrepAffine(irreps=_global_irreps(inputs[0], "edge"), **p), [
            inputs[0]
        ]
    if name in {
        "endpoint_message_aggregation",
        "partitioned_operator_readout",
        "irrep_cartesian_basis",
    }:
        from net.transforms.mandala_components import build_mandala_transform

        return build_mandala_transform(name, p, inputs, mapper=mapper)
    raise ValueError(f"Unknown transformation {name!r}")


class _CodecDecode(nn.Module):
    def __init__(self, codec):
        super().__init__()
        self.codec = codec

    def forward(self, x):
        return self.codec.decode(x)


class FeaturePipeline(nn.Module):
    """A validated sequence; outputs never overwrite input or saved slots."""

    def __init__(self, specification: Mapping, *, num_species, mapper=None):
        super().__init__()
        self.specification = dict(specification)
        from copy import deepcopy

        self.operator_layout = deepcopy(specification.get("operator_layout"))
        specs = default_input_specs(num_species)
        for slot, value in specification.get("inputs", {}).items():
            declared = FeatureSpec(**value)
            if slot in specs and declared != specs[slot]:
                raise ValueError(f"Cannot redefine built-in input contract {slot}")
            specs[slot] = declared
        self.transforms = nn.ModuleDict()
        self.bindings = []
        for stage in specification["stages"]:
            extra = set(stage) - {"id", "transform", "inputs", "outputs", "options"}
            if extra:
                raise ValueError(f"Unknown stage keys: {sorted(extra)}")
            sid = stage["id"]
            if not sid or "." in sid or sid in self.transforms:
                raise ValueError(f"Stage IDs must be unique module names: {sid}")
            reads, writes = tuple(stage["inputs"]), tuple(stage["outputs"])
            missing = set(reads) - specs.keys()
            if missing:
                raise ValueError(
                    f"Stage {sid} reads undefined slots: {sorted(missing)}"
                )
            if len(set(writes)) != len(writes) or set(writes) & specs.keys():
                raise ValueError(f"Stage {sid} overwrites a saved feature slot")
            module, produced = build_transform(
                stage["transform"],
                stage.get("options", {}),
                [specs[s] for s in reads],
                mapper=mapper,
            )
            if len(writes) != len(produced):
                raise ValueError(f"Stage {sid} declares the wrong number of outputs")
            self.transforms[sid] = module
            self.bindings.append((sid, reads, writes))
            specs.update(zip(writes, produced))
        self.outputs = dict(specification["outputs"])
        if set(self.outputs.values()) - specs.keys():
            raise ValueError("Pipeline output references an undefined feature slot")
        self.feature_specs = specs

    def forward(self, inputs, *, return_trace=False, physical_derivatives=False):
        slots = dict(inputs)
        for sid, reads, writes in self.bindings:
            missing = set(reads) - slots.keys()
            if missing:
                raise ValueError(
                    f"Stage {sid} is missing runtime inputs {sorted(missing)}"
                )
            module = self.transforms[sid]
            kwargs = (
                {"physical_derivatives": physical_derivatives}
                if getattr(module, "accepts_physical_derivatives", False)
                else {}
            )
            result = module(*(slots[key] for key in reads), **kwargs)
            values = (
                (result,)
                if len(writes) == 1
                else (
                    tuple(result.values())
                    if isinstance(result, dict)
                    else tuple(result)
                )
            )
            if len(values) != len(writes):
                raise RuntimeError(
                    f"Stage {sid} returned an unexpected number of tensors"
                )
            slots.update(zip(writes, values))
        output = {name: slots[slot] for name, slot in self.outputs.items()}
        return (output, slots) if return_trace else output
