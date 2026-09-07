"""Composition of independently optimized onsite and directed offsite maps."""

from __future__ import annotations

from typing import Mapping

import torch
from torch import nn


class IndependentOnsiteOffsiteMapper(nn.Module):
    """Dispatch raw blocks to two parameter-disjoint fitted mappers.

    This class deliberately performs no Hermitian projection. Global onsite or
    reverse-pair projection belongs to evaluation/export after both raw paths
    have run.
    """

    def __init__(self, onsite_mapper: nn.Module, offsite_mapper: nn.Module) -> None:
        super().__init__()
        if onsite_mapper is offsite_mapper:
            raise ValueError("onsite and offsite mappers must be distinct objects")
        onsite_parameters = {id(parameter) for parameter in onsite_mapper.parameters()}
        offsite_parameters = {
            id(parameter) for parameter in offsite_mapper.parameters()
        }
        if not onsite_parameters.isdisjoint(offsite_parameters):
            raise ValueError("onsite and offsite mappers must not share parameters")
        self.onsite_mapper = onsite_mapper
        self.offsite_mapper = offsite_mapper

    def predict_onsite(self, species: str, descriptor: torch.Tensor) -> torch.Tensor:
        return self.onsite_mapper.predict_onsite(species, descriptor)

    def predict_offsite(
        self,
        pair: tuple[str, str],
        descriptor_i: torch.Tensor,
        displacement_ij: torch.Tensor,
        descriptor_j: torch.Tensor,
    ) -> torch.Tensor:
        return self.offsite_mapper.predict_offsite(
            pair, descriptor_i, displacement_ij, descriptor_j
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
