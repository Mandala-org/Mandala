"""Checkpoint readability must not silently change a trained nonlinearity."""

import math

import pytest
import torch

from net.checkpoint_compat import set_checkpoint_film_semantics
from net.e3mlp_variants import InvariantFiLMActivation


def activation():
    layer = InvariantFiLMActivation(
        "1x0e+1x0o+1x1o", hidden_dim=1, scalar_activation="tanh"
    ).double()
    # A fixed asymmetric trained MLP makes both historical semantic changes
    # observable. Conditioning reads the odd scalar, with a unit odd bias.
    with torch.no_grad():
        for p in layer.parameters():
            p.zero_()
        layer.mlp[0].weight[0, 1] = 1
        layer.mlp[2].weight[2, 0] = 1
        layer.mlp[2].bias[4] = 1
    return torch.nn.Sequential(layer)


def test_historical_replay_matches_july_signed_conditioning_and_odd_bias():
    model = activation().eval()
    x = torch.tensor([[0.0, -2.0, 1.0, 0.0, 0.0]], dtype=torch.float64)
    corrected = model(x)
    initial_weights = {k: v.clone() for k, v in model.state_dict().items()}
    report = set_checkpoint_film_semantics(model, "legacy_2026_07")
    expected = torch.tensor(
        [[0.0, math.tanh(-1.0), 2 / (1 + math.exp(-math.tanh(-2.0))), 0.0, 0.0]],
        dtype=x.dtype,
    )
    torch.testing.assert_close(model(x), expected, atol=1e-14, rtol=1e-14)
    assert report["modules"] == ["0"] and not report["learned_parameters_changed"]
    assert all(
        torch.equal(v, initial_weights[k]) for k, v in model.state_dict().items()
    )
    set_checkpoint_film_semantics(model, "o3")
    torch.testing.assert_close(model(x), corrected, atol=0, rtol=0)


def test_default_remains_inversion_equivariant_and_replay_requires_eval():
    model = activation()
    with pytest.raises(ValueError, match="model.eval"):
        set_checkpoint_film_semantics(model, "legacy_2026_07")
    model.eval()
    x = torch.tensor([[0.4, -2.0, 1.0, 0.2, -0.3]], dtype=torch.float64)
    parity = torch.tensor([1.0, -1.0, -1.0, -1.0, -1.0])
    torch.testing.assert_close(model(x * parity), model(x) * parity)
    with pytest.raises(ValueError, match="Unknown"):
        set_checkpoint_film_semantics(model, "guess")
