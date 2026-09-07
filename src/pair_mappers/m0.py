"""Closed-form M0 full-block mapper for production screening."""

from __future__ import annotations

from itertools import product
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
    the fit diagnostics. Every ordered species pair is fitted separately and
    predictions remain raw until post-training Hermitian projection.
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
        onsite_affine: bool = False,
        enabled_scope: str = "joint",
        dtype: torch.dtype = torch.float64,
    ) -> None:
        super().__init__()
        self.target_transform = target_transform
        if enabled_scope not in ("onsite", "offsite", "joint"):
            raise ValueError(f"unsupported enabled scope {enabled_scope!r}")
        self.enabled_scope = enabled_scope
        self.descriptor_irreps = Irreps(descriptor_irreps)
        self.onsite_affine = bool(onsite_affine)
        self.onsite_irreps = (
            self.descriptor_irreps + Irreps("1x0e")
            if self.onsite_affine
            else self.descriptor_irreps
        )
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
        self.onsite_regressors = nn.ModuleDict(
            {
                element: EquivariantRidgeRegressor(
                    self.onsite_irreps,
                    target_transform.irreps((element, element)),
                    ridge=ridge,
                    dtype=dtype,
                    allow_missing_target_irreps=True,
                )
                for element in elements
                if enabled_scope in ("joint", "onsite")
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
                for pair in product(elements, repeat=2)
                if enabled_scope in ("joint", "offsite")
            }
        )

    def onsite_accumulator(
        self, species: str, *, device: torch.device | str
    ) -> EquivariantRidgeAccumulator:
        if self.enabled_scope not in ("joint", "onsite"):
            raise RuntimeError("onsite path is not present in this scoped model")
        return EquivariantRidgeAccumulator(
            self.onsite_irreps,
            self.target_transform.irreps((species, species)),
            dtype=torch.float64,
            device=device,
            allow_missing_target_irreps=True,
        )

    def offsite_accumulator(
        self, pair: tuple[str, str], *, device: torch.device | str
    ) -> EquivariantRidgeAccumulator:
        if self.enabled_scope not in ("joint", "offsite"):
            raise RuntimeError("offsite path is not present in this scoped model")
        return EquivariantRidgeAccumulator(
            self.offsite_irreps,
            self.target_transform.irreps(pair),
            dtype=torch.float64,
            device=device,
            allow_missing_target_irreps=True,
        )

    def onsite_features(self, descriptor: torch.Tensor) -> torch.Tensor:
        if not self.onsite_affine:
            return descriptor
        return torch.cat(
            (descriptor, descriptor.new_ones(*descriptor.shape[:-1], 1)), dim=-1
        )

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
        return self.offsite_regressors[_module_key(pair)].fit_from_accumulator(
            accumulator
        )

    def predict_onsite(self, species: str, descriptor: torch.Tensor) -> torch.Tensor:
        if self.enabled_scope not in ("joint", "onsite"):
            raise RuntimeError("onsite path is not present in this scoped model")
        return self.onsite_regressors[species](self.onsite_features(descriptor))

    def predict_offsite(
        self,
        pair: tuple[str, str],
        descriptor_i: torch.Tensor,
        displacement_ij: torch.Tensor,
        descriptor_j: torch.Tensor,
    ) -> torch.Tensor:
        if self.enabled_scope not in ("joint", "offsite"):
            raise RuntimeError("offsite path is not present in this scoped model")
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
