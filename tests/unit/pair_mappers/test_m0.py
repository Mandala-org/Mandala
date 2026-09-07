import pytest
import torch
from e3nn import o3

from core.orbital_irrep_config import OrbitalIrrepConfig
from pair_hamiltonian.output_schema import FullBlockIrrepTransform
from pair_mappers import ClosedFormM0PairMapper
from pair_mappers.linear import EquivariantRidgeAccumulator, EquivariantRidgeRegressor


@pytest.mark.unit
def test_ridge_can_return_missing_target_irrep_as_zero():
    model = EquivariantRidgeRegressor(
        "2x0e", "1x0e + 1x1e", allow_missing_target_irreps=True
    )
    accumulator = EquivariantRidgeAccumulator(
        "2x0e", "1x0e + 1x1e", allow_missing_target_irreps=True
    )
    features = torch.randn(12, 2, dtype=torch.float64)
    targets = torch.randn(12, 4, dtype=torch.float64)
    accumulator.update(features, targets)
    diagnostics = model.fit_from_accumulator(accumulator)
    prediction = model(features)
    assert torch.count_nonzero(prediction[:, 1:]) == 0
    assert diagnostics.condition_numbers["1e"] == float("inf")


@pytest.mark.unit
def test_m0_uses_uncoupled_bond_channels_without_training_time_projection():
    config = OrbitalIrrepConfig.from_dict({"A": "1s1p", "B": "1s1p"})
    transform = FullBlockIrrepTransform(config, dtype=torch.float64)
    descriptor_irreps = o3.Irreps("2x0e + 1x1o")
    model = ClosedFormM0PairMapper(
        transform,
        descriptor_irreps,
        bond_n_radial=2,
        bond_l_max=2,
        bond_cutoff=6.0,
        ridge=1e-10,
    )
    assert model.offsite_irreps.dim > 2 * descriptor_irreps.dim
    assert set(model.offsite_regressors) == {"A__A", "A__B", "B__A", "B__B"}
    descriptor_i = torch.randn(5, descriptor_irreps.dim, dtype=torch.float64)
    descriptor_j = torch.randn_like(descriptor_i)
    displacement = torch.randn(5, 3, dtype=torch.float64)
    for name, value in model.named_buffers():
        if "weight_" in name and value.numel() > 0:
            value.copy_(torch.randn_like(value))
    forward = model.predict_offsite(
        ("A", "A"), descriptor_i, displacement, descriptor_j
    )
    reverse = model.predict_offsite(
        ("A", "A"), descriptor_j, -displacement, descriptor_i
    )
    assert not torch.allclose(
        reverse, transform.reverse(("A", "A"), forward), atol=1e-12, rtol=1e-12
    )
    onsite = model.predict_onsite("A", descriptor_i)
    assert not torch.allclose(
        onsite,
        transform.reverse(("A", "A"), onsite),
        atol=1e-12,
        rtol=1e-12,
    )


@pytest.mark.unit
def test_onsite_affine_ridge_adds_only_an_invariant_scalar_feature():
    config = OrbitalIrrepConfig.from_dict({"A": "1s1p"})
    transform = FullBlockIrrepTransform(config, dtype=torch.float64)
    model = ClosedFormM0PairMapper(
        transform,
        "1x0e + 1x1o",
        bond_n_radial=1,
        bond_l_max=0,
        bond_cutoff=6.0,
        ridge=1e-10,
        onsite_affine=True,
        enabled_scope="onsite",
    )
    descriptor = torch.randn(7, 4, dtype=torch.float64)
    features = model.onsite_features(descriptor)
    assert model.onsite_irreps == o3.Irreps("1x0e + 1x1o + 1x0e")
    assert torch.equal(features[:, :-1], descriptor)
    assert torch.equal(features[:, -1], torch.ones(7, dtype=torch.float64))
    assert not model.offsite_regressors
