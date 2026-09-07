"""Pair-local O(3)-equivariant mapper references for Stage 4.

The implementations in this module prioritize an auditable float64 reference
over production throughput. Every public mapper consumes exactly the two
endpoint descriptors and their bond and returns one complete raw full-block
irrep vector. Hermiticity is projected only after independent directed forward
calls. No graph aggregation or mutable node state is present.
"""

from __future__ import annotations

from itertools import product
import math
from typing import Literal

from e3nn import o3
from e3nn.o3 import Irrep, Irreps
import torch
from torch import nn

from pair_hamiltonian.output_schema import (
    FullBlockIrrepTransform,
    o3_representation_matrix,
)

Architecture = Literal["m0", "m2", "m3", "m5", "m7"]
TrainingScope = Literal["onsite", "offsite", "joint"]


def _module_key(pair: tuple[str, str]) -> str:
    return f"{pair[0]}__{pair[1]}"


def _copy_slices(irreps: Irreps) -> list[tuple[Irrep, slice]]:
    result: list[tuple[Irrep, slice]] = []
    offset = 0
    for multiplicity, irrep in irreps:
        for _ in range(multiplicity):
            result.append((irrep, slice(offset, offset + irrep.dim)))
            offset += irrep.dim
    return result


def _multiplicity_capped_irreps(irreps: Irreps, cap: int) -> Irreps:
    if cap < 1:
        raise ValueError("descriptor multiplicity cap must be positive")
    counts: dict[Irrep, int] = {}
    for multiplicity, irrep in irreps:
        counts[irrep] = counts.get(irrep, 0) + multiplicity
    return Irreps(
        (min(multiplicity, cap), irrep)
        for irrep, multiplicity in sorted(
            counts.items(), key=lambda item: (item[0].l, item[0].p)
        )
    )


class BondExpansion(nn.Module):
    """Fixed radial-spherical bond basis with full natural O(3) parity."""

    def __init__(self, *, n_radial: int, l_max: int, cutoff: float) -> None:
        super().__init__()
        if n_radial < 1 or l_max < 0 or cutoff <= 0:
            raise ValueError("invalid bond expansion settings")
        self.n_radial = int(n_radial)
        self.l_max = int(l_max)
        self.cutoff = float(cutoff)
        self.irreps_out = Irreps(
            [
                (1, Irrep(l_value, (-1) ** l_value))
                for _ in range(n_radial)
                for l_value in range(l_max + 1)
            ]
        )

    def forward(self, displacement: torch.Tensor) -> torch.Tensor:
        if displacement.shape[-1] != 3:
            raise ValueError("displacement must end in three Cartesian components")
        distance = torch.linalg.vector_norm(displacement, dim=-1)
        if torch.any(distance <= torch.finfo(displacement.dtype).eps):
            raise ValueError("offsite bond displacement must be nonzero")
        unit = displacement / distance[..., None]
        scaled = (distance / self.cutoff).clamp(0.0, 1.0)
        envelope = torch.where(
            distance < self.cutoff,
            1.0 - 3.0 * scaled.square() + 2.0 * scaled**3,
            torch.zeros_like(scaled),
        )
        parts: list[torch.Tensor] = []
        for radial_index in range(self.n_radial):
            radial = envelope * torch.cos((radial_index + 0.5) * math.pi * scaled)
            for l_value in range(self.l_max + 1):
                harmonics = o3.spherical_harmonics(
                    l_value, unit, normalize=True, normalization="component"
                )
                parts.append(radial[..., None] * harmonics)
        return torch.cat(parts, dim=-1)


class InvariantSummary(nn.Module):
    """True 0e coordinates and copy norms; raw 0o values never enter an MLP."""

    def __init__(self, irreps: Irreps | str) -> None:
        super().__init__()
        self.irreps = Irreps(irreps)
        self.copy_slices = _copy_slices(self.irreps)
        self.dimension = len(self.copy_slices)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        if values.shape[-1] != self.irreps.dim:
            raise ValueError("values do not match invariant-summary irreps")
        invariants: list[torch.Tensor] = []
        for irrep, component_slice in self.copy_slices:
            block = values[..., component_slice]
            if irrep.l == 0 and irrep.p == 1:
                invariants.append(block)
            else:
                invariants.append(block.square().sum(dim=-1, keepdim=True))
        return torch.cat(invariants, dim=-1)


class LinearPairKernel(nn.Module):
    """M0 block-diagonal equal-irrep readout of endpoints and bond."""

    def __init__(
        self,
        descriptor_irreps: Irreps,
        bond_irreps: Irreps,
        target_irreps: Irreps,
    ) -> None:
        super().__init__()
        self.input_irreps = descriptor_irreps + descriptor_irreps + bond_irreps
        self.target_irreps = target_irreps
        self.linear = o3.Linear(self.input_irreps, target_irreps)

    def forward(
        self,
        descriptor_i: torch.Tensor,
        bond: torch.Tensor,
        descriptor_j: torch.Tensor,
    ) -> torch.Tensor:
        return self.linear(torch.cat((descriptor_i, descriptor_j, bond), dim=-1))


def _reachable_hidden(input_irreps: Irreps, *, multiplicity: int, l_max: int) -> Irreps:
    present = {irrep for _, irrep in input_irreps}
    for _, left in input_irreps:
        for _, right in input_irreps:
            present.update(irrep for irrep in left * right if irrep.l <= l_max)
    return Irreps(
        [(multiplicity, irrep) for irrep in sorted(present, key=lambda x: (x.l, x.p))]
    )


class DenseCGPairKernel(nn.Module):
    """M2 dense global-O(3) bilinear residual E3MLP reference."""

    def __init__(
        self,
        descriptor_irreps: Irreps,
        bond_irreps: Irreps,
        target_irreps: Irreps,
        *,
        hidden_multiplicity: int,
        hidden_l_max: int,
    ) -> None:
        super().__init__()
        self.input_irreps = descriptor_irreps + descriptor_irreps + bond_irreps
        self.target_irreps = target_irreps
        hidden = _reachable_hidden(
            self.input_irreps,
            multiplicity=hidden_multiplicity,
            l_max=hidden_l_max,
        )
        self.hidden_irreps = hidden
        self.pre = o3.Linear(self.input_irreps, hidden)
        self.left = o3.Linear(hidden, hidden)
        self.right = o3.Linear(hidden, hidden)
        self.tensor_product = o3.FullyConnectedTensorProduct(hidden, hidden, hidden)
        self.activation = o3.Norm(hidden, squared=True)
        self.gain = nn.Sequential(
            nn.Linear(hidden.num_irreps, max(16, hidden.num_irreps)),
            nn.SiLU(),
            nn.Linear(max(16, hidden.num_irreps), hidden.num_irreps),
            nn.Sigmoid(),
        )
        component_copy: list[int] = []
        for copy_index, (irrep, _component_slice) in enumerate(_copy_slices(hidden)):
            component_copy.extend([copy_index] * irrep.dim)
        self.register_buffer(
            "component_copy",
            torch.tensor(component_copy, dtype=torch.long),
            persistent=False,
        )
        self.output = o3.Linear(hidden, target_irreps)

    def forward(
        self,
        descriptor_i: torch.Tensor,
        bond: torch.Tensor,
        descriptor_j: torch.Tensor,
    ) -> torch.Tensor:
        values = torch.cat((descriptor_i, descriptor_j, bond), dim=-1)
        hidden = self.pre(values)
        interaction = self.tensor_product(self.left(hidden), self.right(hidden))
        gains = 2.0 * self.gain(self.activation(interaction))
        component_gains = gains.index_select(-1, self.component_copy)
        return self.output(hidden + interaction * component_gains)


class ReducedCoefficientPairKernel(nn.Module):
    """M3 fixed CG generators with invariant neural reduced coefficients."""

    def __init__(
        self,
        descriptor_irreps: Irreps,
        bond_irreps: Irreps,
        target_irreps: Irreps,
        *,
        invariant_hidden: int,
        factorization_rank: int | None = None,
        generator_multiplicity: int | None = None,
    ) -> None:
        super().__init__()
        self.endpoint_irreps = descriptor_irreps + descriptor_irreps
        self.bond_irreps = bond_irreps
        self.target_irreps = target_irreps
        if generator_multiplicity is None:
            self.generator_irreps = target_irreps
            self.output: nn.Module = nn.Identity()
        else:
            if generator_multiplicity < 1:
                raise ValueError("generator multiplicity must be positive")
            target_types = sorted(
                {irrep for _, irrep in target_irreps},
                key=lambda irrep: (irrep.l, irrep.p),
            )
            self.generator_irreps = Irreps(
                (generator_multiplicity, irrep) for irrep in target_types
            )
            self.output = o3.Linear(self.generator_irreps, target_irreps)
        self.tensor_product = o3.FullyConnectedTensorProduct(
            self.endpoint_irreps,
            bond_irreps,
            self.generator_irreps,
            internal_weights=False,
            shared_weights=False,
        )
        self.invariants = InvariantSummary(self.endpoint_irreps)
        invariant_dimension = self.invariants.dimension + 1
        self.factorization_rank = factorization_rank
        coefficient_dimension = (
            self.tensor_product.weight_numel
            if factorization_rank is None
            else factorization_rank
        )
        self.coefficients = nn.Sequential(
            nn.Linear(invariant_dimension, invariant_hidden),
            nn.SiLU(),
            nn.Linear(invariant_hidden, coefficient_dimension),
        )
        if factorization_rank is not None:
            if factorization_rank < 1:
                raise ValueError("factorization_rank must be positive")
            self.weight_basis = nn.Parameter(
                torch.empty(factorization_rank, self.tensor_product.weight_numel)
            )
            nn.init.orthogonal_(self.weight_basis)
        else:
            self.register_parameter("weight_basis", None)

    def forward(
        self,
        descriptor_i: torch.Tensor,
        bond: torch.Tensor,
        descriptor_j: torch.Tensor,
    ) -> torch.Tensor:
        endpoints = torch.cat((descriptor_i, descriptor_j), dim=-1)
        distance_proxy = bond[..., :1].square()
        invariants = torch.cat((self.invariants(endpoints), distance_proxy), dim=-1)
        weights = self.coefficients(invariants)
        if self.weight_basis is not None:
            weights = weights @ self.weight_basis
        return self.output(self.tensor_product(endpoints, bond, weights))


def _bond_frame(displacement: torch.Tensor, roll: torch.Tensor | None) -> torch.Tensor:
    """Return Q with ``r_local = r_global @ Q.T`` and local z along the bond."""
    z_axis = displacement / torch.linalg.vector_norm(displacement, dim=-1)[..., None]
    coordinate_axes = torch.eye(3, dtype=displacement.dtype, device=displacement.device)
    alignment = torch.abs(z_axis[..., None, :] * coordinate_axes).sum(dim=-1)
    reference = coordinate_axes[alignment.argmin(dim=-1)]
    x_axis = torch.linalg.cross(reference, z_axis, dim=-1)
    x_axis = x_axis / torch.linalg.vector_norm(x_axis, dim=-1)[..., None]
    y_axis = torch.linalg.cross(z_axis, x_axis, dim=-1)
    frame = torch.stack((x_axis, y_axis, z_axis), dim=-2)
    if roll is None:
        return frame
    roll = torch.as_tensor(roll, dtype=displacement.dtype, device=displacement.device)
    roll = torch.broadcast_to(roll, displacement.shape[:-1])
    cosine, sine = torch.cos(roll), torch.sin(roll)
    zero, one = torch.zeros_like(cosine), torch.ones_like(cosine)
    row0 = torch.stack((cosine, -sine, zero), dim=-1)
    row1 = torch.stack((sine, cosine, zero), dim=-1)
    row2 = torch.stack((zero, zero, one), dim=-1)
    return torch.stack((row0, row1, row2), dim=-2) @ frame


class BondFramePairKernel(nn.Module):
    """M7 auditable bond-frame reference with exact embedded O(2) covariance.

    The local kernel deliberately retains a full O(3)-CG generator bank.  This
    is a complete, conservative reference for the O(2) restriction and makes
    roll/reflection correctness transparent; a later optimized kernel may drop
    the redundant local paths only after matching this implementation.
    """

    def __init__(
        self,
        descriptor_irreps: Irreps,
        bond_irreps: Irreps,
        target_irreps: Irreps,
        *,
        invariant_hidden: int,
        generator_multiplicity: int | None = None,
    ) -> None:
        super().__init__()
        self.descriptor_irreps = descriptor_irreps
        self.target_irreps = target_irreps
        self.local_kernel = ReducedCoefficientPairKernel(
            descriptor_irreps,
            bond_irreps,
            target_irreps,
            invariant_hidden=invariant_hidden,
            generator_multiplicity=generator_multiplicity,
        )

    def local_coordinates(
        self,
        descriptor_i: torch.Tensor,
        displacement: torch.Tensor,
        descriptor_j: torch.Tensor,
        bond_expansion: BondExpansion,
        *,
        roll: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        frame = _bond_frame(displacement, roll)
        descriptor_action = o3_representation_matrix(self.descriptor_irreps, frame)
        target_action = o3_representation_matrix(self.target_irreps, frame)
        local_i = torch.einsum("...i,...ji->...j", descriptor_i, descriptor_action)
        local_j = torch.einsum("...i,...ji->...j", descriptor_j, descriptor_action)
        local_displacement = torch.einsum("...i,...ji->...j", displacement, frame)
        local_bond = bond_expansion(local_displacement)
        return local_i, local_bond, local_j, target_action

    def forward(
        self,
        descriptor_i: torch.Tensor,
        displacement: torch.Tensor,
        descriptor_j: torch.Tensor,
        bond_expansion: BondExpansion,
        *,
        roll: torch.Tensor | None = None,
    ) -> torch.Tensor:
        local_i, local_bond, local_j, target_action = self.local_coordinates(
            descriptor_i,
            displacement,
            descriptor_j,
            bond_expansion,
            roll=roll,
        )
        local_output = self.local_kernel(local_i, local_bond, local_j)
        return torch.einsum("...i,...ij->...j", local_output, target_action)


class OnsiteCGKernel(nn.Module):
    """Separate global-O(3) onsite path used by all nonlinear architectures."""

    def __init__(
        self,
        descriptor_irreps: Irreps,
        target_irreps: Irreps,
        *,
        hidden_multiplicity: int,
        hidden_l_max: int,
    ) -> None:
        super().__init__()
        hidden = _reachable_hidden(
            descriptor_irreps,
            multiplicity=hidden_multiplicity,
            l_max=hidden_l_max,
        )
        self.pre = o3.Linear(descriptor_irreps, hidden)
        self.tensor_product = o3.FullyConnectedTensorProduct(hidden, hidden, hidden)
        self.output = o3.Linear(hidden, target_irreps)

    def forward(self, descriptor: torch.Tensor) -> torch.Tensor:
        hidden = self.pre(descriptor)
        return self.output(hidden + self.tensor_product(hidden, hidden))


class FullBlockNeuralPairMapper(nn.Module):
    """Common full-block API for M0, M2, M3, M5, and reference M7."""

    def __init__(
        self,
        architecture: Architecture,
        target_transform: FullBlockIrrepTransform,
        descriptor_irreps: Irreps | str,
        *,
        bond_n_radial: int = 2,
        bond_l_max: int = 2,
        bond_cutoff: float = 6.0,
        hidden_multiplicity: int = 2,
        hidden_l_max: int = 3,
        invariant_hidden: int = 32,
        factorization_rank: int = 4,
        descriptor_multiplicity_cap: int | None = None,
        generator_multiplicity: int | None = None,
        separate_descriptor_projections: bool = False,
        enabled_scope: TrainingScope = "joint",
        dtype: torch.dtype = torch.float64,
    ) -> None:
        super().__init__()
        if architecture not in ("m0", "m2", "m3", "m5", "m7"):
            raise ValueError(f"unsupported architecture {architecture!r}")
        self.architecture = architecture
        if enabled_scope not in ("onsite", "offsite", "joint"):
            raise ValueError(f"unsupported enabled scope {enabled_scope!r}")
        self.enabled_scope = enabled_scope
        self.target_transform = target_transform
        self.input_descriptor_irreps = Irreps(descriptor_irreps)
        self.separate_descriptor_projections = bool(separate_descriptor_projections)
        if descriptor_multiplicity_cap is None:
            self.descriptor_irreps = self.input_descriptor_irreps
            if self.separate_descriptor_projections:
                if enabled_scope in ("joint", "onsite"):
                    self.onsite_descriptor_projection: nn.Module = nn.Identity()
                if enabled_scope in ("joint", "offsite"):
                    self.offsite_descriptor_projection: nn.Module = nn.Identity()
            else:
                self.descriptor_projection: nn.Module = nn.Identity()
        else:
            self.descriptor_irreps = _multiplicity_capped_irreps(
                self.input_descriptor_irreps, descriptor_multiplicity_cap
            )
            if self.separate_descriptor_projections:
                if enabled_scope in ("joint", "onsite"):
                    self.onsite_descriptor_projection = o3.Linear(
                        self.input_descriptor_irreps, self.descriptor_irreps
                    )
                if enabled_scope in ("joint", "offsite"):
                    self.offsite_descriptor_projection = o3.Linear(
                        self.input_descriptor_irreps, self.descriptor_irreps
                    )
            else:
                self.descriptor_projection = o3.Linear(
                    self.input_descriptor_irreps, self.descriptor_irreps
                )
        self.bond_expansion = BondExpansion(
            n_radial=bond_n_radial, l_max=bond_l_max, cutoff=bond_cutoff
        )
        elements = target_transform.orbital_config.elements()
        self.onsite_kernels = nn.ModuleDict()
        self.offsite_kernels = nn.ModuleDict()
        if enabled_scope in ("joint", "onsite"):
            for element in elements:
                target = target_transform.irreps((element, element))
                self.onsite_kernels[element] = (
                    o3.Linear(self.descriptor_irreps, target)
                    if architecture == "m0"
                    else OnsiteCGKernel(
                        self.descriptor_irreps,
                        target,
                        hidden_multiplicity=hidden_multiplicity,
                        hidden_l_max=hidden_l_max,
                    )
                )
        if enabled_scope in ("joint", "offsite"):
            for pair in product(elements, repeat=2):
                target = target_transform.irreps(pair)
                if architecture == "m0":
                    kernel: nn.Module = LinearPairKernel(
                        self.descriptor_irreps,
                        self.bond_expansion.irreps_out,
                        target,
                    )
                elif architecture == "m2":
                    kernel = DenseCGPairKernel(
                        self.descriptor_irreps,
                        self.bond_expansion.irreps_out,
                        target,
                        hidden_multiplicity=hidden_multiplicity,
                        hidden_l_max=hidden_l_max,
                    )
                elif architecture in ("m3", "m5"):
                    kernel = ReducedCoefficientPairKernel(
                        self.descriptor_irreps,
                        self.bond_expansion.irreps_out,
                        target,
                        invariant_hidden=invariant_hidden,
                        factorization_rank=(
                            factorization_rank if architecture == "m5" else None
                        ),
                        generator_multiplicity=generator_multiplicity,
                    )
                else:
                    kernel = BondFramePairKernel(
                        self.descriptor_irreps,
                        self.bond_expansion.irreps_out,
                        target,
                        invariant_hidden=invariant_hidden,
                        generator_multiplicity=generator_multiplicity,
                    )
                self.offsite_kernels[_module_key(pair)] = kernel
        self.to(dtype=dtype)

    def _raw_offsite(
        self,
        pair: tuple[str, str],
        descriptor_i: torch.Tensor,
        displacement_ij: torch.Tensor,
        descriptor_j: torch.Tensor,
        *,
        roll: torch.Tensor | None = None,
    ) -> torch.Tensor:
        kernel = self.offsite_kernels[_module_key(pair)]
        if self.architecture == "m7":
            assert isinstance(kernel, BondFramePairKernel)
            return kernel(
                descriptor_i,
                displacement_ij,
                descriptor_j,
                self.bond_expansion,
                roll=roll,
            )
        bond = self.bond_expansion(displacement_ij)
        return kernel(descriptor_i, bond, descriptor_j)

    def predict_onsite(self, species: str, descriptor: torch.Tensor) -> torch.Tensor:
        if self.enabled_scope not in ("joint", "onsite"):
            raise RuntimeError("onsite path is not present in this scoped model")
        projection = (
            self.onsite_descriptor_projection
            if self.separate_descriptor_projections
            else self.descriptor_projection
        )
        descriptor = projection(descriptor)
        return self.onsite_kernels[species](descriptor)

    def predict_offsite(
        self,
        pair: tuple[str, str],
        descriptor_i: torch.Tensor,
        displacement_ij: torch.Tensor,
        descriptor_j: torch.Tensor,
        *,
        roll: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.enabled_scope not in ("joint", "offsite"):
            raise RuntimeError("offsite path is not present in this scoped model")
        projection = (
            self.offsite_descriptor_projection
            if self.separate_descriptor_projections
            else self.descriptor_projection
        )
        descriptor_i = projection(descriptor_i)
        descriptor_j = projection(descriptor_j)
        return self._raw_offsite(
            pair,
            descriptor_i,
            displacement_ij,
            descriptor_j,
            roll=roll,
        )

    def parameters_for_scope(self, scope: TrainingScope):
        """Yield exactly the parameters owned by an independent fit scope."""
        if scope not in ("onsite", "offsite", "joint"):
            raise ValueError(f"unsupported training scope {scope!r}")
        if scope == "joint":
            yield from self.parameters()
            return
        if not self.separate_descriptor_projections:
            raise ValueError(
                "independent optimization requires separate descriptor projections"
            )
        projection = (
            self.onsite_descriptor_projection
            if scope == "onsite"
            else self.offsite_descriptor_projection
        )
        yield from projection.parameters()
        kernels = self.onsite_kernels if scope == "onsite" else self.offsite_kernels
        yield from kernels.parameters()

    def forward(
        self,
        descriptor_i: torch.Tensor,
        descriptor_j: torch.Tensor,
        displacement_ij: torch.Tensor,
        species_i: str,
        species_j: str,
        pair_type_metadata: dict[str, object],
    ) -> torch.Tensor:
        onsite = pair_type_metadata.get("onsite")
        if not isinstance(onsite, bool):
            raise ValueError("pair_type_metadata must contain boolean 'onsite'")
        if onsite:
            if species_i != species_j:
                raise ValueError("onsite blocks require identical endpoint species")
            if descriptor_i.shape != descriptor_j.shape or not torch.equal(
                descriptor_i, descriptor_j
            ):
                raise ValueError("onsite endpoint descriptors must be identical")
            return self.predict_onsite(species_i, descriptor_i)
        return self.predict_offsite(
            (species_i, species_j), descriptor_i, displacement_ij, descriptor_j
        )
