from __future__ import annotations

import copy

import pytest
import torch

from net.e3gnn import E3GNN

pytestmark = pytest.mark.gpu


def _require_cuda() -> None:
    if not torch.cuda.is_available():
        pytest.fail("MANDALA_RUN_GPU_TESTS=1 was set, but CUDA is unavailable.")


def test_torch_scatter_cuda_kernel_forward_backward():
    """Exercise the compiled extension, including wheels on an older driver."""
    from torch_scatter import scatter_max

    _require_cuda()
    values = torch.tensor([1.0, 3.0, 2.0], device="cuda", requires_grad=True)
    indices = torch.tensor([0, 0, 1], device="cuda")
    maximum, _ = scatter_max(values, indices, dim=0)
    torch.testing.assert_close(maximum, torch.tensor([3.0, 2.0], device="cuda"))
    maximum.sum().backward()
    torch.testing.assert_close(
        values.grad, torch.tensor([0.0, 1.0, 1.0], device="cuda")
    )


def test_tiny_model_forward_backward_and_strict_reload(small_angular_dataset_e3nn):
    _require_cuda()
    dataset, mapper, base_cfg = small_angular_dataset_e3nn
    # This GPU-only session owns the fixture. Preprocessed energy targets can
    # have grad_fn, so PyTorch cannot deepcopy the complete dataset.
    cfg = copy.deepcopy(base_cfg)
    cfg.device = torch.device("cuda")
    cfg.matrix_targets = ["hamiltonian"]
    dataset.to("cuda")
    model = E3GNN(mapper, cfg).cuda()
    x, y = dataset[0]
    model.log_dict = lambda *args, **kwargs: None

    loss = model.training_step((x, y), batch_idx=0)
    assert loss is not None and torch.isfinite(loss)
    loss.backward()
    gradients = [
        parameter.grad for parameter in model.parameters() if parameter.grad is not None
    ]
    assert gradients and all(torch.isfinite(gradient).all() for gradient in gradients)
    assert any(torch.count_nonzero(gradient) > 0 for gradient in gradients)

    reloaded = E3GNN(mapper, cfg).cuda()
    reloaded.load_state_dict(model.state_dict(), strict=True)
    with torch.no_grad():
        expected = model(x)["hamiltonian"].pair_vectors
        actual = reloaded(x)["hamiltonian"].pair_vectors
    for key in expected:
        assert torch.equal(expected[key], actual[key])
