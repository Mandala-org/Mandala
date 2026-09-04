import pytest
import torch

from data.envelope import evaluate_pair_envelope


@pytest.mark.unit
def test_pair_envelope_gradient_matches_finite_difference():
    params = torch.tensor([[0.1, -0.4, 0.3, 3.0, -0.7]], dtype=torch.float64)
    distance = torch.tensor([2.2], dtype=torch.float64, requires_grad=True)
    value = evaluate_pair_envelope(
        distance,
        family="slater_soft_cutoff",
        pair_params=params,
    )
    derivative = torch.autograd.grad(value.sum(), distance)[0]

    step = 1.0e-6
    plus = evaluate_pair_envelope(
        distance.detach() + step,
        family="slater_soft_cutoff",
        pair_params=params,
    )
    minus = evaluate_pair_envelope(
        distance.detach() - step,
        family="slater_soft_cutoff",
        pair_params=params,
    )
    finite_difference = (plus - minus) / (2.0 * step)

    assert torch.isfinite(value).all()
    assert torch.isfinite(derivative).all()
    assert torch.allclose(derivative, finite_difference, atol=1e-8, rtol=1e-6)
