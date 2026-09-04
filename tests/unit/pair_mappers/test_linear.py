import pytest
import torch
from e3nn import o3

from pair_mappers.linear import EquivariantRidgeAccumulator, EquivariantRidgeRegressor


@pytest.mark.unit
def test_equivariant_ridge_recovers_shared_magnetic_coefficients():
    torch.manual_seed(56)
    features_irreps = o3.Irreps("3x0e + 2x1o + 1x1e")
    targets_irreps = o3.Irreps("2x0e + 3x1o + 1x1e")
    model = EquivariantRidgeRegressor(
        features_irreps, targets_irreps, ridge=1e-12, dtype=torch.float64
    )
    features = torch.randn(80, features_irreps.dim, dtype=torch.float64)
    truth = EquivariantRidgeRegressor(
        features_irreps, targets_irreps, ridge=0.0, dtype=torch.float64
    )
    for name, buffer in truth.named_buffers():
        if name.startswith("weight_"):
            buffer.copy_(torch.randn_like(buffer))
    targets = truth(features)
    diagnostics = model.fit(features, targets)
    prediction = model(features)
    assert diagnostics.sample_count == 80
    assert torch.allclose(prediction, targets, atol=1e-10, rtol=1e-10)


@pytest.mark.unit
@pytest.mark.equivariance
def test_fitted_linear_map_commutes_with_improper_rotation():
    torch.manual_seed(78)
    irreps = o3.Irreps("2x0e + 2x1o + 2x1e + 1x2e")
    model = EquivariantRidgeRegressor(irreps, irreps, ridge=1e-10, dtype=torch.float64)
    features = torch.randn(60, irreps.dim, dtype=torch.float64)
    model.fit(features, torch.randn(60, irreps.dim, dtype=torch.float64))
    rotation = -o3.rand_matrix(dtype=torch.float64)
    previous = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float64)
        action = irreps.D_from_matrix(rotation)
    finally:
        torch.set_default_dtype(previous)
    expected = model(features) @ action.T
    actual = model(features @ action.T)
    assert torch.allclose(actual, expected, atol=1e-10, rtol=1e-10)


@pytest.mark.unit
def test_streaming_sufficient_statistics_match_in_memory_fit():
    torch.manual_seed(91)
    feature_irreps = o3.Irreps("3x0e + 2x1o + 2x1e")
    target_irreps = o3.Irreps("2x0e + 1x1o + 1x1e")
    features = torch.randn(51, feature_irreps.dim, dtype=torch.float64)
    targets = torch.randn(51, target_irreps.dim, dtype=torch.float64)
    direct = EquivariantRidgeRegressor(feature_irreps, target_irreps, ridge=1e-6)
    streamed = EquivariantRidgeRegressor(feature_irreps, target_irreps, ridge=1e-6)
    direct.fit(features, targets)
    accumulator = EquivariantRidgeAccumulator(feature_irreps, target_irreps)
    accumulator.update(features[:17], targets[:17])
    accumulator.update(features[17:39], targets[17:39])
    accumulator.update(features[39:], targets[39:])
    streamed.fit_from_accumulator(accumulator)
    assert torch.allclose(streamed(features), direct(features), atol=1e-12, rtol=1e-12)
