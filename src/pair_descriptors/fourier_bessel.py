"""D3 Fourier-Bessel characteristic-function descriptors."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math

from e3nn import o3
from e3nn.o3 import Irrep, Irreps
import torch
from torch import nn

from pair_descriptors.density import spherical_bessel_j


@dataclass(frozen=True, slots=True)
class FourierBesselChannel:
    """One species/frequency/angular characteristic-function copy."""

    species: int
    frequency_index: int
    frequency_inverse_angstrom: float
    l: int
    start: int
    stop: int


class FourierBesselDescriptor(nn.Module):
    """Real spherical expansion on shared frequency shells ``q=k*pi/RD``."""

    convention = "mandala-d3-fourier-bessel-v1"

    def __init__(
        self,
        species: tuple[int, ...],
        *,
        frequency_count: int,
        l_max: int,
        cutoff: float,
    ) -> None:
        super().__init__()
        if frequency_count <= 0 or l_max < 0 or cutoff <= 0:
            raise ValueError("Invalid D3 dimensions or cutoff")
        self.species = tuple(int(value) for value in species)
        self.frequency_count = int(frequency_count)
        self.l_max = int(l_max)
        self.cutoff = float(cutoff)
        frequencies = (
            torch.arange(1, frequency_count + 1, dtype=torch.float64) * math.pi / cutoff
        )
        self.register_buffer("frequencies", frequencies)
        channels: list[FourierBesselChannel] = []
        irreps = []
        offset = 0
        for atomic_number in self.species:
            for frequency_index, frequency in enumerate(frequencies.tolist()):
                for l_value in range(l_max + 1):
                    irrep = Irrep(l_value, (-1) ** l_value)
                    channels.append(
                        FourierBesselChannel(
                            atomic_number,
                            frequency_index,
                            frequency,
                            l_value,
                            offset,
                            offset + irrep.dim,
                        )
                    )
                    irreps.append((1, irrep))
                    offset += irrep.dim
        self.channels = tuple(channels)
        self.irreps_out = Irreps(irreps)

    @property
    def metadata(self) -> dict[str, object]:
        value = {
            "convention": self.convention,
            "species": self.species,
            "frequency_count": self.frequency_count,
            "l_max": self.l_max,
            "cutoff_angstrom": self.cutoff,
            "irreps_out": str(self.irreps_out),
            "channels": [asdict(channel) for channel in self.channels],
        }
        encoded = json.dumps(value, sort_keys=True, separators=(",", ":"))
        return {**value, "content_hash": hashlib.sha256(encoded.encode()).hexdigest()}

    def contribution(
        self, neighbor_species: torch.Tensor, displacement: torch.Tensor
    ) -> torch.Tensor:
        if displacement.ndim != 2 or displacement.shape[1] != 3:
            raise ValueError("displacement must have shape (neighbors, 3)")
        if neighbor_species.shape != (len(displacement),):
            raise ValueError("neighbor_species must have one entry per neighbor")
        unknown = set(neighbor_species.detach().cpu().tolist()) - set(self.species)
        if unknown:
            raise ValueError(f"Unknown species: {sorted(unknown)}")
        distance = torch.linalg.vector_norm(displacement, dim=-1)
        if torch.any(distance <= torch.finfo(displacement.dtype).eps):
            raise ValueError("D3 excludes zero-displacement center atoms")
        unit = displacement / distance[:, None]
        relative = distance / self.cutoff
        window = torch.where(
            relative < 1.0,
            1.0 - 3.0 * relative.square() + 2.0 * relative**3,
            0.0,
        )
        harmonics = {
            l_value: o3.spherical_harmonics(
                l_value, unit, normalize=True, normalization="component"
            )
            for l_value in range(self.l_max + 1)
        }
        argument = distance[:, None] * self.frequencies.to(displacement)[None, :]
        radial = {
            l_value: spherical_bessel_j(l_value, argument)
            for l_value in range(self.l_max + 1)
        }
        output = displacement.new_zeros((len(displacement), self.irreps_out.dim))
        species = neighbor_species.to(displacement.device)
        for channel in self.channels:
            output[:, channel.start : channel.stop] = (
                ((species == channel.species) * window)[:, None]
                * radial[channel.l][:, channel.frequency_index, None]
                * harmonics[channel.l]
            )
        return output

    def forward(
        self,
        displacement: torch.Tensor,
        neighbor_species: torch.Tensor,
        center_index: torch.Tensor | None = None,
        *,
        num_centers: int | None = None,
    ) -> torch.Tensor:
        values = self.contribution(neighbor_species, displacement)
        if center_index is None:
            center_index = torch.zeros(
                len(values), dtype=torch.long, device=values.device
            )
            num_centers = 1 if num_centers is None else num_centers
        center_index = center_index.to(device=values.device, dtype=torch.long)
        if center_index.shape != (len(values),):
            raise ValueError("center_index must have one entry per neighbor")
        if num_centers is None:
            num_centers = int(center_index.max()) + 1 if len(values) else 0
        output = values.new_zeros((num_centers, self.irreps_out.dim))
        output.index_add_(0, center_index, values)
        return output

    def puncture(
        self,
        descriptor: torch.Tensor,
        partner_species: torch.Tensor,
        partner_displacement: torch.Tensor,
    ) -> torch.Tensor:
        removed = self.contribution(partner_species, partner_displacement)
        if descriptor.shape != removed.shape:
            raise ValueError("descriptor and partner batch shapes must agree")
        return descriptor - removed

    def jacobian(
        self, displacement: torch.Tensor, neighbor_species: torch.Tensor
    ) -> torch.Tensor:
        """Return ``dD/dr`` for one environment with shape ``(K,N,3)``."""
        coordinates = displacement.detach().requires_grad_(True)
        return torch.autograd.functional.jacobian(
            lambda value: self(value, neighbor_species)[0],
            coordinates,
            vectorize=True,
        )
