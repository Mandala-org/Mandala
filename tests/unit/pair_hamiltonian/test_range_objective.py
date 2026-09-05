import pytest
import torch

from pair_hamiltonian.range_objective import (
    closed_form_range_batch,
    physical_offsite_prediction,
    range_factored_mse,
)
from pair_mappers.linear import EquivariantRidgeAccumulator


@pytest.mark.unit
def test_physical_and_weighted_normalized_losses_have_identical_residual_weighting():
    torch.manual_seed(3)
    output = torch.randn(7, 4, dtype=torch.float64, requires_grad=True)
    target = torch.randn_like(output)
    envelope = torch.logspace(-5, -1, 7, dtype=torch.float64)
    physical = range_factored_mse(output, target, envelope, mode="physical_mse")
    normalized = range_factored_mse(
        output, target, envelope, mode="weighted_normalized_mse"
    )
    assert torch.allclose(normalized, physical, atol=1e-12, rtol=1e-12)
    physical_gradient = torch.autograd.grad(physical, output, retain_graph=True)[0]
    normalized_gradient = torch.autograd.grad(normalized, output)[0]
    assert torch.allclose(
        normalized_gradient, physical_gradient, atol=1e-12, rtol=1e-12
    )
    assert torch.allclose(
        physical_offsite_prediction(output, envelope), output * envelope[:, None]
    )


@pytest.mark.unit
def test_weighted_normalized_closed_form_statistics_equal_physical_design():
    torch.manual_seed(5)
    features = torch.randn(13, 3, dtype=torch.float64)
    target = torch.randn(13, 2, dtype=torch.float64)
    envelope = torch.logspace(-4, -1, 13, dtype=torch.float64)
    physical_batch = closed_form_range_batch(
        features, target, envelope, mode="physical_design"
    )
    normalized_batch = closed_form_range_batch(
        features, target, envelope, mode="weighted_normalized_target"
    )
    physical = EquivariantRidgeAccumulator("3x0e", "2x0e")
    normalized = EquivariantRidgeAccumulator("3x0e", "2x0e")
    physical.update(physical_batch[0], physical_batch[1])
    normalized.update(
        normalized_batch[0],
        normalized_batch[1],
        sample_weight=normalized_batch[2],
    )
    assert torch.allclose(
        physical.xtx[next(iter(physical.xtx))],
        normalized.xtx[next(iter(normalized.xtx))],
    )
    assert torch.allclose(
        physical.xty[next(iter(physical.xty))],
        normalized.xty[next(iter(normalized.xty))],
    )
    assert torch.allclose(
        physical.sum_square[next(iter(physical.sum_square))],
        normalized.sum_square[next(iter(normalized.sum_square))],
    )


@pytest.mark.unit
def test_weighted_h_over_g_loss_stays_finite_at_envelope_floor_in_float32():
    output = torch.randn(8, 169, dtype=torch.float32, requires_grad=True)
    target = torch.randn_like(output) * 1.0e-4
    envelope = torch.full((8,), 1.0e-8, dtype=torch.float32)
    loss = range_factored_mse(output, target, envelope, mode="weighted_normalized_mse")
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(output.grad).all()
