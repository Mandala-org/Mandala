"""Native real-basis ACE covariants for non-message-passing pair models.

The implementation deliberately starts from a transparent float64 reference:
species-resolved atom-density coefficients are coupled by fixed Clebsch--Gordan
maps.  No coefficient in this module is learned.  The resulting feature vector
is a single direct sum of O(3) irreps, with every multiplicity carrying explicit
path provenance.

The onsite basis contains density correlation orders zero through two.  The
offsite basis follows ACEhamiltonians' low-order construction: every quadratic
path contains exactly one tagged bond factor and one endpoint-environment
factor.  It remains a pair-local map of ``(D_i, r_ij, D_j)``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from typing import Literal

from e3nn import o3
from e3nn.o3 import Irrep, Irreps
import torch
from torch import nn

from pair_hamiltonian.output_schema import _accurate_irrep_intertwiner


@dataclass(frozen=True, slots=True)
class CovariantChannel:
    """One multiplicity copy in a deterministic equivariant feature vector."""

    index: int
    start: int
    stop: int
    l: int
    parity: int
    degree: int
    correlation_order: int
    family: str
    species: int | None = None
    radial_index: int | None = None
    left_channel: int | None = None
    right_channel: int | None = None
    endpoint: Literal["i", "j"] | None = None

    @property
    def irrep(self) -> Irrep:
        return Irrep(self.l, self.parity)


@dataclass(frozen=True, slots=True)
class EquivariantFeatureLayout:
    """Immutable ordered metadata for an ACE feature vector."""

    irreps: str
    channels: tuple[CovariantChannel, ...]
    convention: str = "mandala-native-real-ace-v1"

    @property
    def dimension(self) -> int:
        return sum(channel.stop - channel.start for channel in self.channels)

    @property
    def content_hash(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()


def _smooth_cutoff(distance: torch.Tensor, cutoff: float) -> torch.Tensor:
    """Compact C1 smoothstep, equal to one at zero and zero at the cutoff."""
    x = (distance / cutoff).clamp(0.0, 1.0)
    return torch.where(distance < cutoff, 1.0 - 3.0 * x.square() + 2.0 * x**3, 0.0)


def _sine_radial_basis(
    distance: torch.Tensor, *, count: int, cutoff: float
) -> torch.Tensor:
    """Stable compact radial functions ``sin(n*pi*r/R)/r`` with a soft wall."""
    orders = torch.arange(1, count + 1, dtype=distance.dtype, device=distance.device)
    x = distance[:, None] / cutoff
    # torch.sinc(x) = sin(pi*x)/(pi*x), including its analytic value at zero.
    radial = (orders * math.pi / cutoff) * torch.sinc(x * orders)
    return radial * _smooth_cutoff(distance, cutoff)[:, None]


class AtomicNeighborDensity(nn.Module):
    """Species-resolved one-particle ``R_n(r) Y_lm(rhat)`` coefficients.

    The input is a ragged edge list. ``center_index[e]`` identifies the atom
    whose environment contains displacement ``e``.  This is deterministic
    descriptor precomputation, not a learned graph/message-passing operation.
    """

    def __init__(
        self,
        species: tuple[int, ...],
        *,
        n_radial: int,
        l_max: int,
        cutoff: float,
    ) -> None:
        super().__init__()
        if not species or len(set(species)) != len(species):
            raise ValueError(
                "species must be a non-empty tuple of unique atomic numbers"
            )
        if n_radial <= 0 or l_max < 0 or cutoff <= 0:
            raise ValueError(
                "n_radial and cutoff must be positive and l_max non-negative"
            )
        self.species = tuple(int(value) for value in species)
        self.n_radial = int(n_radial)
        self.l_max = int(l_max)
        self.cutoff = float(cutoff)
        channels: list[CovariantChannel] = []
        parts: list[tuple[int, Irrep]] = []
        offset = 0
        for atomic_number in self.species:
            for radial_index in range(self.n_radial):
                for l_value in range(self.l_max + 1):
                    irrep = Irrep(l_value, (-1) ** l_value)
                    channels.append(
                        CovariantChannel(
                            index=len(channels),
                            start=offset,
                            stop=offset + irrep.dim,
                            l=l_value,
                            parity=irrep.p,
                            degree=radial_index + l_value + 1,
                            correlation_order=1,
                            family="density",
                            species=atomic_number,
                            radial_index=radial_index,
                        )
                    )
                    parts.append((1, irrep))
                    offset += irrep.dim
        self.layout = EquivariantFeatureLayout(str(Irreps(parts)), tuple(channels))

    @property
    def irreps_out(self) -> Irreps:
        return Irreps(self.layout.irreps)

    def forward(
        self,
        displacements: torch.Tensor,
        neighbor_species: torch.Tensor,
        center_index: torch.Tensor | None = None,
        *,
        num_centers: int | None = None,
    ) -> torch.Tensor:
        if displacements.ndim != 2 or displacements.shape[1] != 3:
            raise ValueError("displacements must have shape (neighbors, 3)")
        if neighbor_species.shape != (displacements.shape[0],):
            raise ValueError("neighbor_species must have one entry per displacement")
        if center_index is None:
            center_index = torch.zeros(
                displacements.shape[0], dtype=torch.long, device=displacements.device
            )
            num_centers = 1 if num_centers is None else num_centers
        if center_index.shape != (displacements.shape[0],):
            raise ValueError("center_index must have one entry per displacement")
        center_index = center_index.to(device=displacements.device, dtype=torch.long)
        if num_centers is None:
            num_centers = (
                int(center_index.max().item()) + 1 if center_index.numel() else 0
            )
        if num_centers < 0 or (
            center_index.numel() and int(center_index.max().item()) >= num_centers
        ):
            raise ValueError("num_centers is inconsistent with center_index")
        unknown = set(neighbor_species.detach().cpu().tolist()) - set(self.species)
        if unknown:
            raise ValueError(f"Unknown neighbor atomic numbers: {sorted(unknown)}")
        result = torch.zeros(
            num_centers,
            self.layout.dimension,
            dtype=displacements.dtype,
            device=displacements.device,
        )
        if displacements.shape[0] == 0:
            return result
        distance = torch.linalg.vector_norm(displacements, dim=-1)
        if torch.any(distance <= torch.finfo(displacements.dtype).eps):
            raise ValueError("Neighbor density excludes zero-displacement center atoms")
        radial = _sine_radial_basis(distance, count=self.n_radial, cutoff=self.cutoff)
        unit = displacements / distance[:, None]
        harmonics = {
            l_value: o3.spherical_harmonics(
                l_value, unit, normalize=True, normalization="component"
            )
            for l_value in range(self.l_max + 1)
        }
        species_tensor = neighbor_species.to(device=displacements.device)
        for channel in self.layout.channels:
            mask = species_tensor == channel.species
            values = (
                radial[:, channel.radial_index, None]
                * harmonics[channel.l]
                * mask[:, None]
            )
            result[:, channel.start : channel.stop].index_add_(0, center_index, values)
        return result


@dataclass(frozen=True, slots=True)
class _ProductPath:
    left: int
    right: int
    output: int


class ACECovariantBasis(nn.Module):
    """Fixed onsite ACE covariants through correlation order two."""

    def __init__(
        self,
        density_layout: EquivariantFeatureLayout,
        *,
        correlation_order: int,
        max_degree: int,
        dtype: torch.dtype = torch.float64,
    ) -> None:
        super().__init__()
        if correlation_order not in (1, 2):
            raise ValueError("Reference ACE basis currently supports order one or two")
        if max_degree <= 0:
            raise ValueError("max_degree must be positive")
        self.density_layout = density_layout
        self.correlation_order = correlation_order
        self.max_degree = max_degree
        output: list[CovariantChannel] = []
        paths: list[_ProductPath] = []
        parts: list[tuple[int, Irrep]] = []
        offset = 0

        def append(**values) -> int:
            nonlocal offset
            irrep = Irrep(values["l"], values["parity"])
            index = len(output)
            output.append(
                CovariantChannel(
                    index=index, start=offset, stop=offset + irrep.dim, **values
                )
            )
            parts.append((1, irrep))
            offset += irrep.dim
            return index

        append(
            l=0,
            parity=1,
            degree=0,
            correlation_order=0,
            family="constant",
        )
        eligible = [
            channel
            for channel in density_layout.channels
            if channel.degree <= max_degree
        ]
        for channel in eligible:
            append(
                l=channel.l,
                parity=channel.parity,
                degree=channel.degree,
                correlation_order=1,
                family="density",
                species=channel.species,
                radial_index=channel.radial_index,
                left_channel=channel.index,
            )
        if correlation_order == 2:
            for left_position, left in enumerate(eligible):
                for right in eligible[left_position:]:
                    degree = left.degree + right.degree
                    if degree > max_degree:
                        continue
                    for l_out in range(abs(left.l - right.l), left.l + right.l + 1):
                        if left.index == right.index and l_out % 2 == 1:
                            continue
                        output_index = append(
                            l=l_out,
                            parity=left.parity * right.parity,
                            degree=degree,
                            correlation_order=2,
                            family="density_product",
                            left_channel=left.index,
                            right_channel=right.index,
                        )
                        paths.append(
                            _ProductPath(left.index, right.index, output_index)
                        )
        self.layout = EquivariantFeatureLayout(str(Irreps(parts)), tuple(output))
        self.paths = tuple(paths)
        for path_index, path in enumerate(self.paths):
            left = density_layout.channels[path.left]
            right = density_layout.channels[path.right]
            target = self.layout.channels[path.output]
            cg = _accurate_irrep_intertwiner(
                left.l, left.parity, right.l, right.parity, target.l
            )
            self.register_buffer(f"cg_{path_index}", cg.to(dtype=dtype))

    @property
    def irreps_out(self) -> Irreps:
        return Irreps(self.layout.irreps)

    def forward(self, density: torch.Tensor) -> torch.Tensor:
        if density.shape[-1] != self.density_layout.dimension:
            raise ValueError("density feature dimension does not match basis layout")
        result = torch.zeros(
            *density.shape[:-1],
            self.layout.dimension,
            dtype=density.dtype,
            device=density.device,
        )
        result[..., 0] = 1.0
        for channel in self.layout.channels[1:]:
            if channel.correlation_order == 1:
                source = self.density_layout.channels[channel.left_channel]
                result[..., channel.start : channel.stop] = density[
                    ..., source.start : source.stop
                ]
        for path_index, path in enumerate(self.paths):
            left = self.density_layout.channels[path.left]
            right = self.density_layout.channels[path.right]
            target = self.layout.channels[path.output]
            cg = getattr(self, f"cg_{path_index}").to(density)
            result[..., target.start : target.stop] = torch.einsum(
                "oij,...i,...j->...o",
                cg.reshape(target.irrep.dim, left.irrep.dim, right.irrep.dim),
                density[..., left.start : left.stop],
                density[..., right.start : right.stop],
            )
        return result


class TaggedBondACEBasis(nn.Module):
    """Low-order offsite ACE basis with one tagged bond in every product."""

    def __init__(
        self,
        density_layout: EquivariantFeatureLayout,
        *,
        bond_n_radial: int,
        bond_l_max: int,
        bond_cutoff: float,
        max_degree: int,
        dtype: torch.dtype = torch.float64,
    ) -> None:
        super().__init__()
        if bond_n_radial <= 0 or bond_l_max < 0 or bond_cutoff <= 0:
            raise ValueError("Invalid tagged-bond basis dimensions or cutoff")
        self.density_layout = density_layout
        self.bond_n_radial = bond_n_radial
        self.bond_l_max = bond_l_max
        self.bond_cutoff = bond_cutoff
        self.max_degree = max_degree
        bond_channels: list[CovariantChannel] = []
        offset = 0
        for radial_index in range(bond_n_radial):
            for l_value in range(bond_l_max + 1):
                irrep = Irrep(l_value, (-1) ** l_value)
                bond_channels.append(
                    CovariantChannel(
                        index=len(bond_channels),
                        start=offset,
                        stop=offset + irrep.dim,
                        l=l_value,
                        parity=irrep.p,
                        degree=radial_index + l_value + 1,
                        correlation_order=1,
                        family="bond",
                        radial_index=radial_index,
                    )
                )
                offset += irrep.dim
        self.bond_layout = EquivariantFeatureLayout(
            str(Irreps([(1, channel.irrep) for channel in bond_channels])),
            tuple(bond_channels),
        )
        output: list[CovariantChannel] = []
        parts: list[tuple[int, Irrep]] = []
        paths: list[tuple[int, int, str, int]] = []
        offset = 0

        def append(**values) -> int:
            nonlocal offset
            irrep = Irrep(values["l"], values["parity"])
            index = len(output)
            output.append(
                CovariantChannel(
                    index=index, start=offset, stop=offset + irrep.dim, **values
                )
            )
            parts.append((1, irrep))
            offset += irrep.dim
            return index

        append(l=0, parity=1, degree=0, correlation_order=0, family="constant")
        for bond in bond_channels:
            if bond.degree <= max_degree:
                append(
                    l=bond.l,
                    parity=bond.parity,
                    degree=bond.degree,
                    correlation_order=1,
                    family="bond",
                    radial_index=bond.radial_index,
                    left_channel=bond.index,
                )
        for endpoint in ("i", "j"):
            for bond in bond_channels:
                for environment in density_layout.channels:
                    degree = bond.degree + environment.degree
                    if degree > max_degree:
                        continue
                    for l_out in range(
                        abs(bond.l - environment.l), bond.l + environment.l + 1
                    ):
                        output_index = append(
                            l=l_out,
                            parity=bond.parity * environment.parity,
                            degree=degree,
                            correlation_order=2,
                            family="bond_environment",
                            left_channel=bond.index,
                            right_channel=environment.index,
                            endpoint=endpoint,
                        )
                        paths.append(
                            (bond.index, environment.index, endpoint, output_index)
                        )
        self.layout = EquivariantFeatureLayout(str(Irreps(parts)), tuple(output))
        self.paths = tuple(paths)
        for path_index, (
            bond_index,
            environment_index,
            _endpoint,
            output_index,
        ) in enumerate(self.paths):
            bond = self.bond_layout.channels[bond_index]
            environment = density_layout.channels[environment_index]
            target = self.layout.channels[output_index]
            cg = _accurate_irrep_intertwiner(
                bond.l,
                bond.parity,
                environment.l,
                environment.parity,
                target.l,
            )
            self.register_buffer(f"cg_{path_index}", cg.to(dtype=dtype))

    @property
    def irreps_out(self) -> Irreps:
        return Irreps(self.layout.irreps)

    def _bond_features(self, displacement: torch.Tensor) -> torch.Tensor:
        distance = torch.linalg.vector_norm(displacement, dim=-1)
        if torch.any(distance <= torch.finfo(displacement.dtype).eps):
            raise ValueError("Offsite tagged bonds must have nonzero displacement")
        radial = _sine_radial_basis(
            distance, count=self.bond_n_radial, cutoff=self.bond_cutoff
        )
        unit = displacement / distance[:, None]
        harmonics = {
            l_value: o3.spherical_harmonics(
                l_value, unit, normalize=True, normalization="component"
            )
            for l_value in range(self.bond_l_max + 1)
        }
        result = displacement.new_zeros(
            displacement.shape[0], self.bond_layout.dimension
        )
        for channel in self.bond_layout.channels:
            result[:, channel.start : channel.stop] = (
                radial[:, channel.radial_index, None] * harmonics[channel.l]
            )
        return result

    def forward(
        self,
        descriptor_i: torch.Tensor,
        displacement_ij: torch.Tensor,
        descriptor_j: torch.Tensor,
    ) -> torch.Tensor:
        if descriptor_i.shape != descriptor_j.shape:
            raise ValueError("Endpoint descriptor shapes must match")
        if (
            descriptor_i.ndim != 2
            or descriptor_i.shape[1] != self.density_layout.dimension
        ):
            raise ValueError("Endpoint descriptors do not match density layout")
        if displacement_ij.shape != (descriptor_i.shape[0], 3):
            raise ValueError("displacement_ij must have shape (pairs, 3)")
        bond_features = self._bond_features(displacement_ij)
        result = descriptor_i.new_zeros(descriptor_i.shape[0], self.layout.dimension)
        result[:, 0] = 1.0
        for channel in self.layout.channels[1:]:
            if channel.family == "bond":
                source = self.bond_layout.channels[channel.left_channel]
                result[:, channel.start : channel.stop] = bond_features[
                    :, source.start : source.stop
                ]
        endpoint_values = {"i": descriptor_i, "j": descriptor_j}
        for path_index, (
            bond_index,
            environment_index,
            endpoint,
            output_index,
        ) in enumerate(self.paths):
            bond = self.bond_layout.channels[bond_index]
            environment = self.density_layout.channels[environment_index]
            target = self.layout.channels[output_index]
            cg = getattr(self, f"cg_{path_index}").to(descriptor_i)
            result[:, target.start : target.stop] = torch.einsum(
                "oij,bi,bj->bo",
                cg.reshape(target.irrep.dim, bond.irrep.dim, environment.irrep.dim),
                bond_features[:, bond.start : bond.stop],
                endpoint_values[endpoint][:, environment.start : environment.stop],
            )
        return result
