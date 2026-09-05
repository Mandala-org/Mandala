"""Full-block native ACE-style Hamiltonian mapper."""

from __future__ import annotations

from itertools import product
from typing import Mapping

import torch
from torch import nn

from pair_descriptors.ace_covariants import (
    ACECovariantBasis,
    EquivariantFeatureLayout,
    TaggedBondACEBasis,
)
from pair_hamiltonian.output_schema import FullBlockIrrepTransform
from pair_mappers.linear import (
    EquivariantRidgeAccumulator,
    EquivariantRidgeRegressor,
    RidgeFitDiagnostics,
)


def _module_key(pair: tuple[str, str]) -> str:
    return f"{pair[0]}__{pair[1]}"


class NativeACEPairMapper(nn.Module):
    """Pair-local ACE basis plus equivariant linear full-block regression.

    There is one fit per directed species-pair type, and every fit predicts the
    complete raw block-irrep vector in one call. Hermitian projection is an
    evaluation/export operation and is deliberately absent from this mapper.
    """

    def __init__(
        self,
        target_transform: FullBlockIrrepTransform,
        density_layout: EquivariantFeatureLayout,
        *,
        onsite_correlation_order: int,
        onsite_max_degree: int,
        bond_n_radial: int,
        bond_l_max: int,
        bond_cutoff: float,
        offsite_max_degree: int,
        ridge: float,
        dtype: torch.dtype = torch.float64,
    ) -> None:
        super().__init__()
        self.target_transform = target_transform
        self.density_layout = density_layout
        self.onsite_basis = ACECovariantBasis(
            density_layout,
            correlation_order=onsite_correlation_order,
            max_degree=onsite_max_degree,
            dtype=dtype,
        )
        self.offsite_basis = TaggedBondACEBasis(
            density_layout,
            bond_n_radial=bond_n_radial,
            bond_l_max=bond_l_max,
            bond_cutoff=bond_cutoff,
            max_degree=offsite_max_degree,
            dtype=dtype,
        )
        elements = target_transform.orbital_config.elements()
        self.onsite_regressors = nn.ModuleDict(
            {
                element: EquivariantRidgeRegressor(
                    self.onsite_basis.irreps_out,
                    target_transform.irreps((element, element)),
                    ridge=ridge,
                    dtype=dtype,
                )
                for element in elements
            }
        )
        self.offsite_regressors = nn.ModuleDict(
            {
                _module_key(pair): EquivariantRidgeRegressor(
                    self.offsite_basis.irreps_out,
                    target_transform.irreps(pair),
                    ridge=ridge,
                    dtype=dtype,
                )
                for pair in product(elements, repeat=2)
            }
        )

    def onsite_features(self, descriptor: torch.Tensor) -> torch.Tensor:
        return self.onsite_basis(descriptor)

    def offsite_features(
        self,
        descriptor_i: torch.Tensor,
        displacement_ij: torch.Tensor,
        descriptor_j: torch.Tensor,
    ) -> torch.Tensor:
        return self.offsite_basis(descriptor_i, displacement_ij, descriptor_j)

    @torch.no_grad()
    def fit_onsite(
        self, species: str, descriptor: torch.Tensor, target: torch.Tensor
    ) -> RidgeFitDiagnostics:
        return self.onsite_regressors[species].fit(
            self.onsite_features(descriptor), target
        )

    @torch.no_grad()
    def fit_offsite(
        self,
        pair: tuple[str, str],
        descriptor_i: torch.Tensor,
        displacement_ij: torch.Tensor,
        descriptor_j: torch.Tensor,
        target: torch.Tensor,
    ) -> RidgeFitDiagnostics:
        features = self.offsite_features(descriptor_i, displacement_ij, descriptor_j)
        return self.offsite_regressors[_module_key(pair)].fit(features, target)

    @torch.no_grad()
    def fit_onsite_from_accumulator(
        self, species: str, accumulator: EquivariantRidgeAccumulator
    ) -> RidgeFitDiagnostics:
        """Finalize one bounded-memory onsite sufficient-statistics fit."""
        return self.onsite_regressors[species].fit_from_accumulator(accumulator)

    @torch.no_grad()
    def fit_offsite_from_accumulator(
        self,
        pair: tuple[str, str],
        accumulator: EquivariantRidgeAccumulator,
    ) -> RidgeFitDiagnostics:
        """Finalize one directed bounded-memory offsite fit."""
        return self.offsite_regressors[_module_key(pair)].fit_from_accumulator(
            accumulator
        )

    def predict_onsite(self, species: str, descriptor: torch.Tensor) -> torch.Tensor:
        return self.onsite_regressors[species](self.onsite_features(descriptor))

    def predict_offsite(
        self,
        pair: tuple[str, str],
        descriptor_i: torch.Tensor,
        displacement_ij: torch.Tensor,
        descriptor_j: torch.Tensor,
    ) -> torch.Tensor:
        regressor = self.offsite_regressors[_module_key(pair)]
        return regressor(
            self.offsite_features(descriptor_i, displacement_ij, descriptor_j)
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
        """Implement the frozen public full-block pair-mapper interface."""
        onsite = pair_type_metadata.get("onsite")
        if not isinstance(onsite, bool):
            raise ValueError("pair_type_metadata must contain boolean 'onsite'")
        if onsite:
            if species_i != species_j:
                raise ValueError("Onsite blocks require identical endpoint species")
            if descriptor_i.shape != descriptor_j.shape or not torch.equal(
                descriptor_i, descriptor_j
            ):
                raise ValueError("Onsite endpoint descriptors must be identical")
            return self.predict_onsite(species_i, descriptor_i)
        return self.predict_offsite(
            (species_i, species_j), descriptor_i, displacement_ij, descriptor_j
        )
