"""Orbital packing and explicit Cartesian basis transformations.

Learned readouts remain independent graph stages.
"""

from __future__ import annotations

import torch
from torch import nn
from e3nn import o3

from dataclasses import replace
from net.features import FeatureSpec, global_irreps
from net.registry import Option, Port, register_transform


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


@register_transform(
    "irrep_cartesian_basis",
    category="frame",
    summary="Relabel the Cartesian axis order of irrep features (e.g. xyz -> yzx)",
    inputs=[Port("features", None, "irreps")],
    options=[
        Option(
            "target",
            "choice",
            key=True,
            choices=("xyz", "yzx", "zxy", "xzy", "yxz", "zyx"),
        )
    ],
)
def _irrep_cartesian_basis(p, inputs, mapper):
    irreps = global_irreps(inputs[0])
    source = inputs[0].angular_basis or "xyz"
    return IrrepCartesianBasis(irreps, source, p["target"]), [
        replace(inputs[0], angular_basis=p["target"])
    ]


@register_transform(
    "partitioned_blocks_to_shells",
    category="output",
    summary="Pack per species-pair orbital vectors into the shared shell layout",
    inputs=[
        Port("vectors", "edge", "block_vectors"),
        Port("partitions", None, "object"),
    ],
    options=[
        Option("species_shells", "mapping", key=True, doc="Usually ${species.shells}"),
        Option("species_order", "list", None),
    ],
)
def _partitioned_blocks_to_shells(p, inputs, mapper):
    if (
        inputs[0].representation != "block_vectors"
        or inputs[1].representation != "object"
    ):
        raise ValueError(
            "Block packing expects partitioned vectors and edge partitions"
        )
    module = PartitionedBlockPack(mapper=mapper, **p)
    return module, [
        FeatureSpec(
            "edge", "matrix", width=module.codec.packed_dim, support=inputs[0].support
        )
    ]
