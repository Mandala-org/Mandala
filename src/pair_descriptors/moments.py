"""D2 irreducible Zernike/STF moment descriptors."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json

from e3nn.o3 import Irrep, Irreps
import torch
from torch import nn

from pair_descriptors.density import RawNeighborDensityDescriptor


@dataclass(frozen=True, slots=True)
class MomentChannel:
    """One species/radial/angular irreducible moment copy."""

    species: int
    radial_index: int
    l: int
    total_degree: int
    source_start: int
    source_stop: int
    start: int
    stop: int


class IrreducibleMomentDescriptor(nn.Module):
    """3D-Zernike/STF moments through total polynomial degree ``P``.

    A radial index ``p`` and angular order ``l`` has total degree ``2p+l``.
    The fixed real spherical channel transforms as ``l`` with parity
    ``(-1)**l``. The implementation uses an analytic Jacobi recurrence; its
    Jacobian is evaluated exactly through PyTorch automatic differentiation.
    """

    convention = "mandala-d2-zernike-stf-v1"

    def __init__(self, species: tuple[int, ...], *, max_degree: int, cutoff: float):
        super().__init__()
        if max_degree < 0:
            raise ValueError("max_degree must be non-negative")
        self.species = tuple(int(value) for value in species)
        self.max_degree = int(max_degree)
        self.base = RawNeighborDensityDescriptor(
            self.species,
            radial_basis="zernike",
            n_radial=max_degree // 2 + 1,
            l_max=max_degree,
            cutoff=cutoff,
        )
        channels: list[MomentChannel] = []
        irreps = []
        offset = 0
        for source in self.base.channels:
            degree = 2 * source.radial_index + source.l
            if degree > max_degree:
                continue
            dimension = source.stop - source.start
            channels.append(
                MomentChannel(
                    species=source.species,
                    radial_index=source.radial_index,
                    l=source.l,
                    total_degree=degree,
                    source_start=source.start,
                    source_stop=source.stop,
                    start=offset,
                    stop=offset + dimension,
                )
            )
            irreps.append((1, Irrep(source.l, (-1) ** source.l)))
            offset += dimension
        self.channels = tuple(channels)
        self.irreps_out = Irreps(irreps)

    @property
    def cutoff(self) -> float:
        return self.base.cutoff

    @property
    def metadata(self) -> dict[str, object]:
        value = {
            "convention": self.convention,
            "species": self.species,
            "max_degree": self.max_degree,
            "cutoff_angstrom": self.cutoff,
            "irreps_out": str(self.irreps_out),
            "channels": [asdict(channel) for channel in self.channels],
        }
        encoded = json.dumps(value, sort_keys=True, separators=(",", ":"))
        return {**value, "content_hash": hashlib.sha256(encoded.encode()).hexdigest()}

    def _select(self, values: torch.Tensor) -> torch.Tensor:
        return torch.cat(
            [
                values[..., channel.source_start : channel.source_stop]
                for channel in self.channels
            ],
            dim=-1,
        )

    def contribution(
        self, neighbor_species: torch.Tensor, displacement: torch.Tensor
    ) -> torch.Tensor:
        return self._select(self.base.contribution(neighbor_species, displacement))

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
        if displacement.ndim != 2:
            raise ValueError("jacobian expects one environment")
        coordinates = displacement.detach().requires_grad_(True)
        return torch.autograd.functional.jacobian(
            lambda value: self(value, neighbor_species)[0],
            coordinates,
            vectorize=True,
        )
