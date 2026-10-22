"""Orbital packing and explicit Cartesian basis transformations.

Learned readouts remain independent graph stages.
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
            # Axis permutations have exact zero/unit entries. Remove angular
            # exponential roundoff there, which otherwise creates spurious
            # near-zero gradients amplified by Adam's epsilon normalization.
            for value in (0.0, 1.0, -1.0):
                matrix = torch.where(
                    (matrix - value).abs() < 1e-14, matrix.new_tensor(value), matrix
                )
        finally:
            torch.set_default_dtype(previous)
        self.register_buffer("matrix", matrix.to(previous))

    def forward(self, features):
        return features @ self.matrix.to(features).T


class PartitionedBlockPack(nn.Module):
    """Pack partitioned orbital vectors with an explicit Cartesian basis change."""

    def __init__(
        self,
        *,
        mapper,
        species_shells,
        species_order=None,
    ):
        super().__init__()
        from core.operator_layout import ShellTensorCodec

        if mapper is None:
            raise ValueError(
                "partitioned_blocks_to_shells requires the E3GNN orbital mapper"
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
        self.mapper = mapper
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

    def forward(self, vectors, partitions):
        count = sum(len(part["global_idx"]) for part in partitions.values())
        exemplar = next(iter(vectors.values()))
        output = exemplar.new_zeros((count, self.codec.packed_dim))
        for pair, values in vectors.items():
            left, right = pair.split("-")
            blocks = self.mapper.vectors_to_blocks(pair, values)
            a, b = (
                getattr(self, f"native_basis_{left}").to(blocks),
                getattr(self, f"native_basis_{right}").to(blocks),
            )
            packed = self.codec.pack_blocks((left, right), a @ blocks @ b.T)
            output = output.index_copy(0, partitions[pair]["global_idx"], packed)
        return output


def build_orbital_transform(name, parameters, inputs, *, mapper=None):
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
        from net.layers import NodeUpdateBlock
        from net.common import Config

        options = dict(p.pop("config", {}))
        options.setdefault("dtype", torch.get_default_dtype())
        for key, value in (("l_max", angular.lmax), ("n_radial", radial.dim)):
            if key in options and options[key] != value:
                raise ValueError(
                    f"{key} is determined by the typed angular/radial inputs"
                )
            options[key] = value
        if angular != o3.Irreps.spherical_harmonics(angular.lmax):
            raise ValueError(
                "Endpoint aggregation requires full ordered spherical harmonics"
            )
        module = NodeUpdateBlock(
            node,
            edge,
            attrs.dim,
            Config(**options),
            node_irreps_out=o3.Irreps(p.pop("irreps_out")),
            n_edge_types=attrs.dim**2,
        )
        if p:
            raise ValueError(f"Unknown endpoint aggregation options: {sorted(p)}")
        return module, [
            FeatureSpec(
                "node",
                "irreps",
                str(module.irreps_out),
                support=inputs[0].support,
                angular_basis="xyz",
            )
        ]
    if name == "partitioned_blocks_to_shells":
        if (
            len(inputs) != 2
            or inputs[0].representation != "block_vectors"
            or inputs[1].representation != "object"
        ):
            raise ValueError(
                "Block packing expects partitioned vectors and edge partitions"
            )
        module = PartitionedBlockPack(mapper=mapper, **p)
        return module, [
            FeatureSpec(
                "edge",
                "matrix",
                width=module.codec.packed_dim,
                support=inputs[0].support,
            )
        ]
    raise ValueError(f"Unknown orbital transformation {name!r}")
