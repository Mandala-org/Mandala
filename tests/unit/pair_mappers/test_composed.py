import pytest
import torch
from torch import nn

from pair_mappers.composed import IndependentOnsiteOffsiteMapper


class _Onsite(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(2.0))

    def predict_onsite(self, species, descriptor):
        del species
        return descriptor * self.weight


class _Offsite(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(3.0))

    def predict_offsite(self, pair, descriptor_i, displacement, descriptor_j):
        del pair, displacement
        return (descriptor_i + descriptor_j) * self.weight


@pytest.mark.unit
def test_composition_dispatches_to_independent_raw_paths():
    onsite, offsite = _Onsite(), _Offsite()
    model = IndependentOnsiteOffsiteMapper(onsite, offsite)
    descriptor_i = torch.ones(2, 1)
    descriptor_j = torch.full((2, 1), 2.0)
    displacement = torch.ones(2, 3)
    assert torch.equal(model.predict_onsite("O", descriptor_i), torch.full((2, 1), 2.0))
    assert torch.equal(
        model.predict_offsite(("O", "Si"), descriptor_i, displacement, descriptor_j),
        torch.full((2, 1), 9.0),
    )


@pytest.mark.unit
def test_composition_rejects_shared_parameter_objects():
    onsite = _Onsite()
    with pytest.raises(ValueError, match="distinct objects"):
        IndependentOnsiteOffsiteMapper(onsite, onsite)
