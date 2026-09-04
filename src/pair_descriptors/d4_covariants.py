"""D4 deterministic ACE/NICE covariant lifts of a retained D1 descriptor."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json

from e3nn.o3 import Irrep, Irreps
import torch
from torch import nn

from pair_descriptors.density import RawNeighborDensityDescriptor
from pair_hamiltonian.output_schema import _accurate_irrep_intertwiner


@dataclass(frozen=True, slots=True)
class D4LiftPath:
    """One label-independent Clebsch--Gordan coupling path."""

    body_degree: int
    input_channels: tuple[int, ...]
    intermediate_l: int | None
    l: int
    parity: int
    radial_degree: int
    start: int
    stop: int


class DeterministicCovariantLiftDescriptor(nn.Module):
    """Retain D1 and append a capped deterministic bank of degree-2/3 lifts.

    Candidate paths are ordered and capped solely by radial/angular indices and
    species-channel indices.  Hamiltonian labels, validation loss, PCA, and
    learned contractions never enter the manifest.
    """

    convention = "mandala-d4-d1-cg-lifts-v1"

    def __init__(
        self,
        species: tuple[int, ...],
        *,
        radial_basis: str,
        n_radial: int,
        l_max: int,
        cutoff: float,
        body_degree: int,
        lift_max_input_l: int,
        lift_max_radial_index: int,
        lift_radial_degree_budget: int,
        lift_max_intermediate_l: int,
        target_irreps: tuple[tuple[int, int], ...],
        max_paths_per_degree_irrep: int,
        dtype: torch.dtype = torch.float64,
    ) -> None:
        super().__init__()
        if body_degree not in (2, 3):
            raise ValueError("D4 body_degree must be 2 or 3")
        if (
            min(
                lift_max_input_l,
                lift_max_radial_index,
                lift_radial_degree_budget,
                lift_max_intermediate_l,
            )
            < 0
        ):
            raise ValueError("D4 lift budgets must be non-negative")
        if max_paths_per_degree_irrep <= 0:
            raise ValueError("D4 path cap must be positive")
        self.base = RawNeighborDensityDescriptor(
            species,
            radial_basis=radial_basis,
            n_radial=n_radial,
            l_max=l_max,
            cutoff=cutoff,
        )
        self.body_degree = int(body_degree)
        self.lift_max_input_l = int(lift_max_input_l)
        self.lift_max_radial_index = int(lift_max_radial_index)
        self.lift_radial_degree_budget = int(lift_radial_degree_budget)
        self.lift_max_intermediate_l = int(lift_max_intermediate_l)
        self.target_irreps = tuple(sorted(set(target_irreps)))
        self.max_paths_per_degree_irrep = int(max_paths_per_degree_irrep)
        if not self.target_irreps:
            raise ValueError("D4 requires at least one target-reachable irrep")
        for l_value, parity in self.target_irreps:
            Irrep(l_value, parity)

        eligible = tuple(
            index
            for index, channel in enumerate(self.base.channels)
            if channel.l <= self.lift_max_input_l
            and channel.radial_index <= self.lift_max_radial_index
        )
        candidates: list[tuple[int, tuple[int, ...], int | None, int, int, int]] = []
        for left_pos, left_index in enumerate(eligible):
            left = self.base.channels[left_index]
            for right_index in eligible[left_pos:]:
                right = self.base.channels[right_index]
                radial_degree = left.radial_index + right.radial_index
                if radial_degree > self.lift_radial_degree_budget:
                    continue
                parity = left.parity * right.parity
                for l_out in range(abs(left.l - right.l), left.l + right.l + 1):
                    if (l_out, parity) not in self.target_irreps:
                        continue
                    # The antisymmetric square of an identical irrep vanishes.
                    if left_index == right_index and l_out % 2:
                        continue
                    candidates.append(
                        (
                            2,
                            (left_index, right_index),
                            None,
                            l_out,
                            parity,
                            radial_degree,
                        )
                    )

        if self.body_degree == 3:
            for a_pos, a_index in enumerate(eligible):
                a = self.base.channels[a_index]
                for b_pos in range(a_pos, len(eligible)):
                    b_index = eligible[b_pos]
                    b = self.base.channels[b_index]
                    for c_index in eligible[b_pos:]:
                        c = self.base.channels[c_index]
                        radial_degree = a.radial_index + b.radial_index + c.radial_index
                        if radial_degree > self.lift_radial_degree_budget:
                            continue
                        parity = a.parity * b.parity * c.parity
                        for intermediate_l in range(abs(a.l - b.l), a.l + b.l + 1):
                            if intermediate_l > self.lift_max_intermediate_l:
                                continue
                            if a_index == b_index and intermediate_l % 2:
                                continue
                            for l_out in range(
                                abs(intermediate_l - c.l), intermediate_l + c.l + 1
                            ):
                                if (l_out, parity) in self.target_irreps:
                                    candidates.append(
                                        (
                                            3,
                                            (a_index, b_index, c_index),
                                            intermediate_l,
                                            l_out,
                                            parity,
                                            radial_degree,
                                        )
                                    )

        # Low radial degree and low angular complexity win ties.  The final
        # channel tuple makes the selection fully deterministic.
        candidates.sort(
            key=lambda item: (
                item[0],
                item[3],
                -item[4],
                item[5],
                sum(self.base.channels[index].l for index in item[1]),
                item[2] if item[2] is not None else -1,
                item[1],
            )
        )
        counts: dict[tuple[int, int, int], int] = {}
        selected = []
        for candidate in candidates:
            group = (candidate[0], candidate[3], candidate[4])
            if counts.get(group, 0) >= self.max_paths_per_degree_irrep:
                continue
            counts[group] = counts.get(group, 0) + 1
            selected.append(candidate)

        paths: list[D4LiftPath] = []
        irreps = list(self.base.irreps_out)
        offset = self.base.irreps_out.dim
        coupling_keys: set[tuple[int, int, int, int, int]] = set()
        for degree, inputs, intermediate_l, l_out, parity, radial_degree in selected:
            irrep = Irrep(l_out, parity)
            paths.append(
                D4LiftPath(
                    body_degree=degree,
                    input_channels=inputs,
                    intermediate_l=intermediate_l,
                    l=l_out,
                    parity=parity,
                    radial_degree=radial_degree,
                    start=offset,
                    stop=offset + irrep.dim,
                )
            )
            irreps.append((1, irrep))
            offset += irrep.dim
            first, second = (self.base.channels[index] for index in inputs[:2])
            first_out = l_out if degree == 2 else int(intermediate_l)
            coupling_keys.add(
                (first.l, first.parity, second.l, second.parity, first_out)
            )
            if degree == 3:
                third = self.base.channels[inputs[2]]
                coupling_keys.add(
                    (
                        first_out,
                        first.parity * second.parity,
                        third.l,
                        third.parity,
                        l_out,
                    )
                )

        self.paths = tuple(paths)
        self.irreps_out = Irreps(irreps)
        self._coupling_names: dict[tuple[int, int, int, int, int], str] = {}
        for index, key in enumerate(sorted(coupling_keys)):
            name = f"cg_{index}"
            self.register_buffer(
                name, _accurate_irrep_intertwiner(*key).to(dtype=dtype)
            )
            self._coupling_names[key] = name

    @property
    def cutoff(self) -> float:
        return self.base.cutoff

    @property
    def metadata(self) -> dict[str, object]:
        value = {
            "convention": self.convention,
            "base": self.base.metadata,
            "body_degree": self.body_degree,
            "lift_max_input_l": self.lift_max_input_l,
            "lift_max_radial_index": self.lift_max_radial_index,
            "lift_radial_degree_budget": self.lift_radial_degree_budget,
            "lift_max_intermediate_l": self.lift_max_intermediate_l,
            "target_irreps": self.target_irreps,
            "max_paths_per_degree_irrep": self.max_paths_per_degree_irrep,
            "irreps_out": str(self.irreps_out),
            "raw_dimension": self.base.irreps_out.dim,
            "lift_dimension": self.irreps_out.dim - self.base.irreps_out.dim,
            "paths": [asdict(path) for path in self.paths],
        }
        encoded = json.dumps(value, sort_keys=True, separators=(",", ":"))
        return {**value, "content_hash": hashlib.sha256(encoded.encode()).hexdigest()}

    def _couple(
        self,
        left: torch.Tensor,
        right: torch.Tensor,
        key: tuple[int, int, int, int, int],
    ) -> torch.Tensor:
        l_left, _p_left, l_right, _p_right, l_out = key
        cg = getattr(self, self._coupling_names[key]).to(left)
        return torch.einsum(
            "oij,...i,...j->...o",
            cg.reshape(2 * l_out + 1, 2 * l_left + 1, 2 * l_right + 1),
            left,
            right,
        )

    def lift(self, density: torch.Tensor) -> torch.Tensor:
        """Append the frozen covariant lifts to precomputed D1 values."""
        if density.shape[-1] != self.base.irreps_out.dim:
            raise ValueError("D4 input does not match its retained D1 layout")
        output = density.new_zeros(*density.shape[:-1], self.irreps_out.dim)
        output[..., : self.base.irreps_out.dim] = density
        for path in self.paths:
            source = [self.base.channels[index] for index in path.input_channels]
            left = density[..., source[0].start : source[0].stop]
            right = density[..., source[1].start : source[1].stop]
            first_l = path.l if path.body_degree == 2 else int(path.intermediate_l)
            value = self._couple(
                left,
                right,
                (source[0].l, source[0].parity, source[1].l, source[1].parity, first_l),
            )
            if path.body_degree == 3:
                third = source[2]
                value = self._couple(
                    value,
                    density[..., third.start : third.stop],
                    (
                        first_l,
                        source[0].parity * source[1].parity,
                        third.l,
                        third.parity,
                        path.l,
                    ),
                )
            output[..., path.start : path.stop] = value
        return output

    def forward(
        self,
        displacement: torch.Tensor,
        neighbor_species: torch.Tensor,
        center_index: torch.Tensor | None = None,
        *,
        num_centers: int | None = None,
    ) -> torch.Tensor:
        return self.lift(
            self.base(
                displacement,
                neighbor_species,
                center_index,
                num_centers=num_centers,
            )
        )

    def puncture(
        self,
        descriptor: torch.Tensor,
        partner_species: torch.Tensor,
        partner_displacement: torch.Tensor,
    ) -> torch.Tensor:
        """Exactly rebuild D4 after subtracting one partner from retained D1."""
        raw_dimension = self.base.irreps_out.dim
        removed = self.base.contribution(partner_species, partner_displacement)
        if descriptor.shape[:-1] != removed.shape[:-1]:
            raise ValueError("descriptor and partner batch shapes must agree")
        return self.lift(descriptor[..., :raw_dimension] - removed)
