import pytest
import torch

from core.norms import zero_safe_norm


@pytest.mark.parametrize("keepdim", [False, True])
def test_zero_safe_norm_values_first_and_second_derivatives(keepdim):
    values = torch.tensor(
        [[0.0, 0.0, 0.0], [3.0, 4.0, 0.0]], dtype=torch.float64, requires_grad=True
    )
    norm = zero_safe_norm(values, keepdim=keepdim)
    torch.testing.assert_close(
        norm.flatten(), torch.tensor([0.0, 5.0], dtype=values.dtype)
    )
    gradient = torch.autograd.grad(norm.sum(), values, create_graph=True)[0]
    torch.testing.assert_close(
        gradient, torch.tensor([[0.0, 0.0, 0.0], [0.6, 0.8, 0.0]], dtype=values.dtype)
    )
    hessian = torch.stack(
        [
            torch.autograd.grad(gradient[:, axis].sum(), values, retain_graph=True)[0]
            for axis in range(3)
        ],
        dim=1,
    )
    torch.testing.assert_close(hessian[0], torch.zeros(3, 3, dtype=values.dtype))
    # Hessian of |v| away from zero: I/r - v v^T/r^3.
    vector = values[1].detach()
    expected = (
        torch.eye(3, dtype=values.dtype) / 5.0 - torch.outer(vector, vector) / 125.0
    )
    torch.testing.assert_close(hessian[1], expected)
