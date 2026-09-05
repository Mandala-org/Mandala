"""Closed-form M0 full-block mapper for production screening."""

from __future__ import annotations

from itertools import combinations_with_replacement
from typing import Mapping

from e3nn.o3 import Irreps
import torch
from torch import nn

from pair_hamiltonian.output_schema import FullBlockIrrepTransform
from pair_mappers.linear import (
    EquivariantRidgeAccumulator,
    EquivariantRidgeRegressor,
    RidgeFitDiagnostics,
)
from pair_mappers.neural import BondExpansion


def _module_key(pair: tuple[str, str]) -> str:
    return f"{pair[0]}__{pair[1]}"


class ClosedFormM0PairMapper(nn.Module):
    """Equal-irrep ridge readout of ``(D_i, D_j, b_ij)``.

    M0 creates no tensor-product channels. Missing target irrep types are
    returned as exact zeros and recorded through infinite-condition entries in
    the fit diagnostics. Canonical species ordering and an exact homonuclear
    reversal projection enforce the full-block Hermiticity convention.
    """

    def __init__(
        self,
        target_transform: FullBlockIrrepTransform,
        descriptor_irreps: Irreps | str,
        *,
        bond_n_radial: int,
        bond_l_max: int,
        bond_cutoff: float,
        ridge: float,
        dtype: torch.dtype = torch.float64,
    ) -> None:
        super().__init__()
        self.target_transform = target_transform
        self.descriptor_irreps = Irreps(descriptor_irreps)
        self.bond_expansion = BondExpansion(
            n_radial=bond_n_radial,
            l_max=bond_l_max,
            cutoff=bond_cutoff,
        )
        self.offsite_irreps = (
            self.descriptor_irreps
            + self.descriptor_irreps
            + self.bond_expansion.irreps_out
        )
        elements = target_transform.orbital_config.elements()
        self._element_rank = {element: index for index, element in enumerate(elements)}
        self.onsite_regressors = nn.ModuleDict(
            {
                element: EquivariantRidgeRegressor(
                    self.descriptor_irreps,
                    target_transform.irreps((element, element)),
                    ridge=ridge,
                    dtype=dtype,
                    allow_missing_target_irreps=True,
                )
                for element in elements
            }
        )
        self.offsite_regressors = nn.ModuleDict(
            {
                _module_key(pair): EquivariantRidgeRegressor(
                    self.offsite_irreps,
                    target_transform.irreps(pair),
                    ridge=ridge,
                    dtype=dtype,
                    allow_missing_target_irreps=True,
                )
                for pair in combinations_with_replacement(elements, 2)
            }
        )

    def _canonical_pair(self, pair: tuple[str, str]) -> tuple[tuple[str, str], bool]:
        try:
            reverse = self._element_rank[pair[0]] > self._element_rank[pair[1]]
        except KeyError as exc:
            raise KeyError(f"unknown species pair {pair}") from exc
        return ((pair[1], pair[0]) if reverse else pair), reverse

    def onsite_accumulator(
        self, species: str, *, device: torch.device | str
    ) -> EquivariantRidgeAccumulator:
        return EquivariantRidgeAccumulator(
            self.descriptor_irreps,
            self.target_transform.irreps((species, species)),
            dtype=torch.float64,
            device=device,
            allow_missing_target_irreps=True,
        )

    def offsite_accumulator(
        self, pair: tuple[str, str], *, device: torch.device | str
    ) -> EquivariantRidgeAccumulator:
        canonical, _ = self._canonical_pair(pair)
        return EquivariantRidgeAccumulator(
            self.offsite_irreps,
            self.target_transform.irreps(canonical),
            dtype=torch.float64,
            device=device,
            allow_missing_target_irreps=True,
        )

    def onsite_features(self, descriptor: torch.Tensor) -> torch.Tensor:
        return descriptor

    def offsite_features(
        self,
        descriptor_i: torch.Tensor,
        displacement_ij: torch.Tensor,
        descriptor_j: torch.Tensor,
    ) -> torch.Tensor:
        bond = self.bond_expansion(displacement_ij)
        return torch.cat((descriptor_i, descriptor_j, bond), dim=-1)

    def fit_onsite_from_accumulator(
        self, species: str, accumulator: EquivariantRidgeAccumulator
    ) -> RidgeFitDiagnostics:
        return self.onsite_regressors[species].fit_from_accumulator(accumulator)

    def fit_offsite_from_accumulator(
        self,
        pair: tuple[str, str],
        accumulator: EquivariantRidgeAccumulator,
    ) -> RidgeFitDiagnostics:
        canonical, was_reversed = self._canonical_pair(pair)
        if was_reversed:
            raise ValueError("streaming accumulators must use canonical pairs")
        return self.offsite_regressors[_module_key(canonical)].fit_from_accumulator(
            accumulator
        )

    def predict_onsite(self, species: str, descriptor: torch.Tensor) -> torch.Tensor:
        prediction = self.onsite_regressors[species](descriptor)
        return 0.5 * (
            prediction + self.target_transform.reverse((species, species), prediction)
        )

    def predict_offsite(
        self,
        pair: tuple[str, str],
        descriptor_i: torch.Tensor,
        displacement_ij: torch.Tensor,
        descriptor_j: torch.Tensor,
    ) -> torch.Tensor:
        canonical, was_reversed = self._canonical_pair(pair)
        if was_reversed:
            descriptor_i, descriptor_j = descriptor_j, descriptor_i
            displacement_ij = -displacement_ij
        regressor = self.offsite_regressors[_module_key(canonical)]
        prediction = regressor(
            self.offsite_features(descriptor_i, displacement_ij, descriptor_j)
        )
        if canonical[0] == canonical[1]:
            opposite = regressor(
                self.offsite_features(descriptor_j, -displacement_ij, descriptor_i)
            )
            prediction = 0.5 * (
                prediction + self.target_transform.reverse(canonical, opposite)
            )
        return (
            self.target_transform.reverse(canonical, prediction)
            if was_reversed
            else prediction
        )

    def forward(
        self,
        descriptor_i: torch.Tensor,
        descriptor_j: torch.Tensor,
        displacement_ij: torch.Tensor,
        species_i: str,
        species_j: str,
        pair_type_metadata: Mapping[str, object],
    ) -> torch.Tensor:
        onsite = pair_type_metadata.get("onsite")
        if not isinstance(onsite, bool):
            raise ValueError("pair_type_metadata must contain boolean 'onsite'")
        if onsite:
            if species_i != species_j or not torch.equal(descriptor_i, descriptor_j):
                raise ValueError("onsite endpoints must be identical and same-species")
            return self.predict_onsite(species_i, descriptor_i)
        return self.predict_offsite(
            (species_i, species_j), descriptor_i, displacement_ij, descriptor_j
        )
