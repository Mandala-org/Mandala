from __future__ import annotations

import copy

import pytest
import torch

from net.e3gnn import E3GNN


pytestmark = pytest.mark.gpu


def _require_cuda() -> None:
    if not torch.cuda.is_available():
        pytest.fail("MANDALA_RUN_GPU_TESTS=1 was set, but CUDA is unavailable.")


def test_cuda_tensor_and_e3nn_runtime():
    _require_cuda()
    x = torch.randn(1024, 3, device="cuda")
    loss = x.square().mean()
    assert torch.isfinite(loss)
    assert torch.cuda.get_device_capability()[0] >= 7


def test_tiny_model_forward_backward_and_strict_reload(small_angular_dataset_e3nn):
    _require_cuda()
    dataset, mapper, base_cfg = small_angular_dataset_e3nn
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
    assert any(parameter.grad is not None for parameter in model.parameters())

    reloaded = E3GNN(mapper, cfg).cuda()
    reloaded.load_state_dict(model.state_dict(), strict=True)
    with torch.no_grad():
        expected = model(x)["hamiltonian"].pair_vectors
        actual = reloaded(x)["hamiltonian"].pair_vectors
    for key in expected:
        assert torch.equal(expected[key], actual[key])
