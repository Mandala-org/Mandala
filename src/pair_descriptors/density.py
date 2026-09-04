"""Species-resolved raw neighbor-density (D1) descriptors.

The stored channels are real spherical tensors ``(s, n, l)`` with parity
``(-1)**l``.  Both radial families are fixed, orthogonal on the unit ball
before application of the common smooth cutoff window, and contain no learned
parameters.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from typing import Literal

from e3nn import o3
from e3nn.o3 import Irrep, Irreps
import numpy as np
from scipy.optimize import brentq
from scipy.special import spherical_jn
import torch
from torch import nn


RadialKind = Literal["spherical_bessel", "zernike"]


@dataclass(frozen=True, slots=True)
class DensityChannel:
    species: int
    radial_index: int
    l: int
    start: int
    stop: int

    @property
    def parity(self) -> int:
        return (-1) ** self.l


def _spherical_bessel_zeros(l_value: int, count: int) -> np.ndarray:
    """Return the first positive zeros of ``j_l`` by bracketed scanning."""
    roots: list[float] = []
    step = math.pi / 16.0
    left = 1.0e-8
    f_left = float(spherical_jn(l_value, left))
    right = left + step
    while len(roots) < count:
        f_right = float(spherical_jn(l_value, right))
        if f_left * f_right < 0.0:
            root = brentq(lambda x: spherical_jn(l_value, x), left, right)
            if not roots or root - roots[-1] > 1.0e-8:
                roots.append(root)
        left, f_left = right, f_right
        right += step
        if right > (count + l_value + 4) * math.pi:
            raise RuntimeError("Failed to bracket spherical-Bessel zeros")
    return np.asarray(roots)


def _jacobi(n: int, alpha: float, beta: float, x: torch.Tensor) -> torch.Tensor:
    """Differentiable three-term Jacobi recurrence."""
    if n == 0:
        return torch.ones_like(x)
    p0 = torch.ones_like(x)
    p1 = 0.5 * ((alpha - beta) + (alpha + beta + 2.0) * x)
    if n == 1:
        return p1
    for k in range(1, n):
        kf = float(k)
        a1 = 2.0 * (kf + 1.0) * (kf + alpha + beta + 1.0) * (2.0 * kf + alpha + beta)
        a2 = (2.0 * kf + alpha + beta + 1.0) * (
            (2.0 * kf + alpha + beta) * (2.0 * kf + alpha + beta + 2.0) * x
            + alpha**2
            - beta**2
        )
        a3 = 2.0 * (kf + alpha) * (kf + beta) * (2.0 * kf + alpha + beta + 2.0)
        p0, p1 = p1, (a2 * p1 - a3 * p0) / a1
    return p1


def spherical_bessel_j(l_value: int, z: torch.Tensor) -> torch.Tensor:
    """Differentiable ``j_l(z)`` with a stable small-argument series."""
    tiny = torch.finfo(z.dtype).eps ** 0.5
    safe = torch.where(z.abs() < tiny, torch.full_like(z, tiny), z)
    j0 = torch.sin(safe) / safe
    if l_value == 0:
        return torch.where(z.abs() < tiny, torch.ones_like(z), j0)
    current = torch.sin(safe) / safe.square() - torch.cos(safe) / safe
    previous = j0
    for ell in range(1, l_value):
        following = (2 * ell + 1) / safe * current - previous
        previous, current = current, following
    double_factorial = math.prod(range(1, 2 * l_value + 2, 2))
    series = z.pow(l_value) / double_factorial
    series *= (
        1.0
        - z.square() / (2.0 * (2 * l_value + 3))
        + z.pow(4) / (8.0 * (2 * l_value + 3) * (2 * l_value + 5))
    )
    return torch.where(z.abs() < 0.2, series, current)


class OrthogonalBallRadialBasis(nn.Module):
    """Fixed radial functions indexed by ``(n,l)`` on ``[0, cutoff]``."""

    def __init__(self, kind: RadialKind, n_radial: int, l_max: int, cutoff: float):
        super().__init__()
        if kind not in ("spherical_bessel", "zernike"):
            raise ValueError(f"Unknown radial basis: {kind}")
        if n_radial <= 0 or l_max < 0 or cutoff <= 0:
            raise ValueError("n_radial/cutoff must be positive and l_max non-negative")
        self.kind = kind
        self.n_radial = int(n_radial)
        self.l_max = int(l_max)
        self.cutoff = float(cutoff)
        if kind == "spherical_bessel":
            roots = np.stack(
                [_spherical_bessel_zeros(lv, n_radial) for lv in range(l_max + 1)]
            )
            norms = np.sqrt(2.0) / np.abs(
                np.stack([spherical_jn(lv + 1, roots[lv]) for lv in range(l_max + 1)])
            )
            self.register_buffer("roots", torch.from_numpy(roots))
            self.register_buffer("norms", torch.from_numpy(norms))
        else:
            self.register_buffer("roots", torch.empty(0, dtype=torch.float64))
            self.register_buffer("norms", torch.empty(0, dtype=torch.float64))

    def forward(self, distance: torch.Tensor, l_value: int) -> torch.Tensor:
        if not 0 <= l_value <= self.l_max:
            raise ValueError("l_value outside configured range")
        x = distance / self.cutoff
        inside = x < 1.0
        if self.kind == "spherical_bessel":
            roots = self.roots[l_value].to(distance)
            norms = self.norms[l_value].to(distance)
            # torch.special.spherical_bessel_j0 only covers l=0; use the stable
            # upward recurrence, with analytic small-x limits supplied below.
            z = x[:, None] * roots[None, :]
            values = spherical_bessel_j(l_value, z)
            values = values * norms[None, :] / self.cutoff**1.5
        else:
            values = []
            argument = 2.0 * x.square() - 1.0
            for radial_index in range(self.n_radial):
                normalization = math.sqrt(4 * radial_index + 2 * l_value + 3)
                values.append(
                    normalization
                    * x.pow(l_value)
                    * _jacobi(radial_index, 0.0, l_value + 0.5, argument)
                    / self.cutoff**1.5
                )
            values = torch.stack(values, dim=-1)
        # Known C1 window. It is positive in the open ball and therefore does
        # not alter the complete-basis injectivity argument.
        window = torch.where(inside, 1.0 - 3.0 * x.square() + 2.0 * x**3, 0.0)
        return values * window[:, None]


class RawNeighborDensityDescriptor(nn.Module):
    """Additive D1 coefficients for a ragged periodic neighbor list."""

    convention = "mandala-d1-real-density-v1"

    def __init__(
        self,
        species: tuple[int, ...],
        *,
        radial_basis: RadialKind,
        n_radial: int,
        l_max: int,
        cutoff: float,
    ) -> None:
        super().__init__()
        if not species or len(species) != len(set(species)):
            raise ValueError("species must be unique and non-empty")
        self.species = tuple(int(z) for z in species)
        self.radial = OrthogonalBallRadialBasis(radial_basis, n_radial, l_max, cutoff)
        channels: list[DensityChannel] = []
        irreps: list[tuple[int, Irrep]] = []
        offset = 0
        for z in self.species:
            for n in range(n_radial):
                for lv in range(l_max + 1):
                    irrep = Irrep(lv, (-1) ** lv)
                    channels.append(
                        DensityChannel(z, n, lv, offset, offset + irrep.dim)
                    )
                    irreps.append((1, irrep))
                    offset += irrep.dim
        self.channels = tuple(channels)
        self.irreps_out = Irreps(irreps)

    @property
    def cutoff(self) -> float:
        return self.radial.cutoff

    @property
    def metadata(self) -> dict[str, object]:
        value = {
            "convention": self.convention,
            "species": self.species,
            "radial_basis": self.radial.kind,
            "n_radial": self.radial.n_radial,
            "l_max": self.radial.l_max,
            "cutoff_angstrom": self.cutoff,
            "irreps_out": str(self.irreps_out),
            "channels": [asdict(channel) for channel in self.channels],
        }
        payload = json.dumps(value, sort_keys=True, separators=(",", ":"))
        return {**value, "content_hash": hashlib.sha256(payload.encode()).hexdigest()}

    def contribution(
        self, neighbor_species: torch.Tensor, displacement: torch.Tensor
    ) -> torch.Tensor:
        """Return the exact additive term for each labelled neighbor/image."""
        if displacement.ndim != 2 or displacement.shape[-1] != 3:
            raise ValueError("displacement must have shape (neighbors, 3)")
        if neighbor_species.shape != (displacement.shape[0],):
            raise ValueError("neighbor_species must have one entry per neighbor")
        unknown = set(neighbor_species.detach().cpu().tolist()) - set(self.species)
        if unknown:
            raise ValueError(f"Unknown species: {sorted(unknown)}")
        distance = torch.linalg.vector_norm(displacement, dim=-1)
        if torch.any(distance <= torch.finfo(displacement.dtype).eps):
            raise ValueError("D1 excludes zero-displacement center atoms")
        unit = displacement / distance[:, None]
        harmonics = {
            lv: o3.spherical_harmonics(
                lv, unit, normalize=True, normalization="component"
            )
            for lv in range(self.radial.l_max + 1)
        }
        radial = {lv: self.radial(distance, lv) for lv in range(self.radial.l_max + 1)}
        output = displacement.new_zeros((displacement.shape[0], self.irreps_out.dim))
        z_values = neighbor_species.to(displacement.device)
        for channel in self.channels:
            output[:, channel.start : channel.stop] = (
                (z_values == channel.species)[:, None]
                * radial[channel.l][:, channel.radial_index, None]
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
                len(values), dtype=torch.long, device=displacement.device
            )
            num_centers = 1 if num_centers is None else num_centers
        center_index = center_index.to(device=displacement.device, dtype=torch.long)
        if center_index.shape != (len(values),):
            raise ValueError("center_index must have one entry per neighbor")
        if num_centers is None:
            num_centers = int(center_index.max().item()) + 1 if len(values) else 0
        if num_centers < 0 or (len(values) and center_index.max() >= num_centers):
            raise ValueError("num_centers is inconsistent with center_index")
        output = displacement.new_zeros((num_centers, self.irreps_out.dim))
        output.index_add_(0, center_index, values)
        return output

    def puncture(
        self,
        descriptor: torch.Tensor,
        partner_species: torch.Tensor,
        partner_displacement: torch.Tensor,
    ) -> torch.Tensor:
        """Remove one explicitly identified periodic partner contribution."""
        removed = self.contribution(partner_species, partner_displacement)
        if descriptor.shape != removed.shape:
            raise ValueError("descriptor and partner batch shapes must agree")
        return descriptor - removed
