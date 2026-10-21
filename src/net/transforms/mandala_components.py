"""Typed bridges to existing MANDALA transformations and explicit basis changes.

These components reuse the existing attention update and partitioned readout;
there is no second model or architecture-name dispatch.
"""

from __future__ import annotations

import torch
from torch import nn
from e3nn import o3


class IrrepCartesianBasis(nn.Module):
    """Change Cartesian harmonic coordinates without changing irrep layout."""

    def __init__(self, irreps, source, target):
        super().__init__()
        self.irreps = o3.Irreps(irreps)
        permutation = torch.eye(3, dtype=torch.float64)[
            [source.index(axis) for axis in target]
        ]
        # Construct in double precision even when the model default is float32.
        previous = torch.get_default_dtype()
        try:
            torch.set_default_dtype(torch.float64)
            matrix = self.irreps.D_from_matrix(permutation)
        finally:
            torch.set_default_dtype(previous)
        self.register_buffer("matrix", matrix.to(previous))

    def forward(self, features):
        return features @ self.matrix.to(features).T


class EndpointMessageAggregation(nn.Module):
    def __init__(
        self,
        node_irreps,
        edge_irreps,
        angular_irreps,
        radial_width,
        species_count,
        *,
        irreps_out,
        config=None,
    ):
        super().__init__()
        from net.common import Config
        from net.layers import NodeUpdateBlock

        options = dict(config or {})
        options.setdefault("dtype", torch.get_default_dtype())
        angular = o3.Irreps(angular_irreps)
        if angular != o3.Irreps.spherical_harmonics(angular.lmax):
            raise ValueError(
                "Existing endpoint message aggregation requires full ordered spherical harmonics"
            )
        for key, value in (("l_max", angular.lmax), ("n_radial", radial_width)):
            if key in options and options[key] != value:
                raise ValueError(
                    f"{key} is determined by the typed angular/radial inputs"
                )
            options[key] = value
        self.block = NodeUpdateBlock(
            o3.Irreps(node_irreps),
            o3.Irreps(edge_irreps),
            species_count,
            Config(**options),
            node_irreps_out=o3.Irreps(irreps_out),
            n_edge_types=species_count**2,
        )
        self.irreps_out = self.block.irreps_out

    def forward(self, node, edge, edge_index, angular, radial, species, pair_types):
        return self.block(node, edge, edge_index, angular, radial, species, pair_types)


class PartitionedOperatorReadout(nn.Module):
    """Existing pair head, followed by explicit xyz → native orbital conversion."""

    def __init__(
        self,
        node_irreps,
        edge_irreps,
        *,
        mapper,
        species_shells,
        irreps_neck,
        species_order=None,
        config=None,
    ):
        super().__init__()
        from net.common import Config
        from net.heads import DeepHead
        from net.operator_codecs import ShellTensorCodec

        if mapper is None:
            raise ValueError(
                "partitioned_operator_readout requires the E3GNN orbital mapper"
            )
        supplied_shells = {str(k): tuple(v) for k, v in species_shells.items()}
        self.species_order = (
            tuple(map(str, species_order))
            if species_order is not None
            else tuple(supplied_shells)
        )
        if len(set(self.species_order)) != len(self.species_order) or set(
            self.species_order
        ) != set(supplied_shells):
            raise ValueError(
                "Readout species_order must contain every shell species exactly once"
            )
        self.species_shells = {
            species: supplied_shells[species] for species in self.species_order
        }
        elements = mapper.orbital_cfg.elements()
        if set(elements) != set(self.species_shells):
            raise ValueError("Readout shell vocabulary must match mapper elements")
        for species in elements:
            expected = o3.Irreps(
                [(1, (ell, (-1) ** ell)) for ell in self.species_shells[species]]
            )
            if (
                expected.simplify()
                != mapper.orbital_cfg.element_to_irreps[species].simplify()
            ):
                raise ValueError("Readout shell ordering/irreps disagree with mapper")
        self.codec = ShellTensorCodec.from_species_shells(
            self.species_shells, dtype=torch.get_default_dtype()
        )
        head_options = dict(config or {})
        head_options.setdefault("dtype", torch.get_default_dtype())
        self.head = DeepHead(
            o3.Irreps(node_irreps),
            o3.Irreps(edge_irreps),
            o3.Irreps(irreps_neck),
            list(mapper.edge_types),
            mapper,
            Config(**head_options),
        )
        # Dataset pair IDs follow the resolved vocabulary, independent of the
        # legacy mapper's alphabetically ordered projection dictionary.
        self.edge_types = tuple(
            f"{left}-{right}"
            for left in self.species_order
            for right in self.species_order
        )
        for species in elements:
            irreps = o3.Irreps(
                [(1, (ell, (-1) ** ell)) for ell in self.species_shells[species]]
            )
            basis = IrrepCartesianBasis(irreps, "xyz", "yzx").matrix
            permutation, _ = self.codec.rotation.basis_matrices(
                self.species_shells[species], self.species_shells[species]
            )
            self.register_buffer(f"native_basis_{species}", permutation.T @ basis)

    def forward(self, node, edge, edge_index, pair_types, onsite):
        pair_edges, partitions = {}, {}
        for pair_index, pair in enumerate(self.edge_types):
            selected = torch.where(pair_types == pair_index)[0]
            ends = edge_index[:, selected]
            diagonal = onsite[selected]
            shifted = (
                (ends[0] == ends[1]) & ~diagonal
                if self.head.separate_shifted_self
                else torch.zeros_like(diagonal)
            )
            pair_edges[pair] = torch.cat(
                [ends.new_zeros((3, len(selected))), ends], dim=0
            )
            partitions[pair] = {
                "global_idx": selected,
                "diag_local_idx": torch.where(diagonal)[0],
                "shifted_self_local_idx": torch.where(shifted)[0],
                "offdiag_local_idx": torch.where(~diagonal & ~shifted)[0],
            }
        vectors = self.head(node, edge, pair_edges, partitions)
        output = edge.new_zeros((len(edge), self.codec.packed_dim))
        for pair, values in vectors.items():
            left, right = pair.split("-")
            blocks = self.head.mapper.vectors_to_blocks(pair, values)
            a, b = (
                getattr(self, f"native_basis_{left}").to(blocks),
                getattr(self, f"native_basis_{right}").to(blocks),
            )
            packed = self.codec.pack_blocks((left, right), a @ blocks @ b.T)
            output = output.index_copy(0, partitions[pair]["global_idx"], packed)
        return output


def build_mandala_transform(name, parameters, inputs, *, mapper=None):
    from dataclasses import replace
    from net.pipeline import (
        FeatureSpec,
        _global_irreps,
        _same_support,
        _same_angular_basis,
        _index_spec,
    )

    p = dict(parameters)
    if name == "irrep_cartesian_basis":
        if len(inputs) != 1:
            raise ValueError("Cartesian irrep conversion accepts one input")
        irreps = _global_irreps(inputs[0])
        target = p.pop("target")
        if sorted(target) != ["x", "y", "z"]:
            raise ValueError("target must permute xyz")
        if p:
            raise ValueError(f"Unknown Cartesian basis options {sorted(p)}")
        source = inputs[0].angular_basis or "xyz"
        return IrrepCartesianBasis(irreps, source, target), [
            replace(inputs[0], angular_basis=target)
        ]
    if name == "endpoint_message_aggregation":
        if len(inputs) != 7:
            raise ValueError(
                "Endpoint aggregation expects node, edge, index, angular, radial, species, pair types"
            )
        node, edge = (
            _global_irreps(inputs[0], "node"),
            _global_irreps(inputs[1], "edge"),
        )
        angular, radial, attrs = (
            _global_irreps(inputs[3], "edge"),
            _global_irreps(inputs[4], "edge"),
            _global_irreps(inputs[5], "node"),
        )
        _index_spec(inputs[2], "relation")
        _index_spec(inputs[6], "edge")
        _same_support(inputs[1], inputs[2], inputs[3], inputs[4], inputs[6])
        _same_support(inputs[0], inputs[5])
        basis = _same_angular_basis(inputs[0], inputs[1], inputs[3])
        if basis not in {None, "xyz"}:
            raise ValueError(
                "Existing MANDALA aggregation requires xyz; insert irrep_cartesian_basis"
            )
        if any(ir.l or ir.p != 1 for _, ir in radial + attrs):
            raise ValueError("Radial and species inputs must be invariant scalars")
        module = EndpointMessageAggregation(
            node, edge, angular, radial.dim, attrs.dim, **p
        )
        return module, [
            FeatureSpec(
                "node",
                "irreps",
                str(module.irreps_out),
                support=inputs[0].support,
                angular_basis="xyz",
            )
        ]
    if name == "partitioned_operator_readout":
        if len(inputs) != 5:
            raise ValueError(
                "Partitioned readout expects node, edge, edge index, pair types and onsite mask"
            )
        node, edge = (
            _global_irreps(inputs[0], "node"),
            _global_irreps(inputs[1], "edge"),
        )
        _index_spec(inputs[2], "relation")
        _index_spec(inputs[3], "edge")
        _index_spec(inputs[4], "edge")
        support = _same_support(*inputs[1:])
        basis = _same_angular_basis(inputs[0], inputs[1])
        if basis not in {None, "xyz"}:
            raise ValueError(
                "Partitioned orbital readout requires xyz; insert irrep_cartesian_basis"
            )
        module = PartitionedOperatorReadout(node, edge, mapper=mapper, **p)
        return module, [
            FeatureSpec(
                "edge", "matrix", width=module.codec.packed_dim, support=support
            )
        ]
    raise ValueError(f"Unknown MANDALA transformation {name!r}")
