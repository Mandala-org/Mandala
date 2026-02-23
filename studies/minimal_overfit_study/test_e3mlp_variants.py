import sys
from pathlib import Path

import pytest
import torch
from e3nn.o3 import Irreps, rand_matrix

PROJECT_ROOT = Path(__file__).resolve().parents[2]
STUDY_DIR = Path(__file__).resolve().parent
for p in (PROJECT_ROOT, STUDY_DIR):
    p_str = str(p)
    if p_str not in sys.path:
        sys.path.append(p_str)

from e3mlp_variants import (  # noqa: E402
    BilinearSelfTPE3MLP,
    GateE3MLP,
    GateMagnitudesE3MLP,
    GateScalarsMLPE3MLP,
    InvariantMoEE3MLP,
    InvariantSelfAttentionE3MLP,
    NormActE3MLP,
    ResidualE3MLP,
    ScalarMagnitudeSelfTPGatedE3MLP,
)

IRREPS_IN = Irreps("2x0e + 2x1o + 2x2e + 2x3o + 2x4e")
IRREPS_HIDDEN = Irreps("3x0e + 2x1o + 2x1e + 2x2o + 1x2e + 1x3o + 2x3e + 1x4e")
IRREPS_OUT = Irreps("1x0e + 1x1o + 1x2e + 1x3o + 1x4e")
BATCH = 3


def _make_variants(num_layers: int):
    return [
        (
            "NormActE3MLP",
            NormActE3MLP(
                IRREPS_IN,
                IRREPS_OUT,
                irreps_hidden=IRREPS_HIDDEN,
                num_layers=num_layers,
            ),
        ),
        (
            "GateE3MLP",
            GateE3MLP(
                IRREPS_IN,
                IRREPS_OUT,
                irreps_hidden=IRREPS_HIDDEN,
                num_layers=num_layers,
            ),
        ),
        (
            "GateScalarsMLPE3MLP",
            GateScalarsMLPE3MLP(
                IRREPS_IN,
                IRREPS_OUT,
                irreps_hidden=IRREPS_HIDDEN,
                num_layers=num_layers,
            ),
        ),
        (
            "GateMagnitudesE3MLP",
            GateMagnitudesE3MLP(
                IRREPS_IN,
                IRREPS_OUT,
                irreps_hidden=IRREPS_HIDDEN,
                num_layers=num_layers,
            ),
        ),
        (
            "ResidualE3MLP",
            ResidualE3MLP(
                IRREPS_IN,
                IRREPS_OUT,
                irreps_hidden=IRREPS_HIDDEN,
                num_layers=num_layers,
            ),
        ),
        (
            "BilinearSelfTPE3MLP",
            BilinearSelfTPE3MLP(
                IRREPS_IN,
                IRREPS_OUT,
                irreps_hidden=IRREPS_HIDDEN,
                num_layers=num_layers,
                tp_mul_per_irrep=1,
            ),
        ),
        (
            "InvariantSelfAttentionE3MLP",
            InvariantSelfAttentionE3MLP(
                IRREPS_IN,
                IRREPS_OUT,
                irreps_hidden=IRREPS_HIDDEN,
                num_layers=num_layers,
                attention_dim=8,
                context_dim=4,
                context_hidden_dims=(16,),
                token_hidden_dims=(16,),
                scalar_update_hidden_dims=(16,),
            ),
        ),
        (
            "InvariantMoEE3MLP",
            InvariantMoEE3MLP(
                IRREPS_IN,
                IRREPS_OUT,
                irreps_hidden=IRREPS_HIDDEN,
                num_layers=num_layers,
                num_experts=3,
                expert_num_layers=2,
                router_hidden_dims=(16,),
            ),
        ),
        (
            "ScalarMagnitudeSelfTPGatedE3MLP",
            ScalarMagnitudeSelfTPGatedE3MLP(
                IRREPS_IN,
                IRREPS_OUT,
                irreps_hidden=IRREPS_HIDDEN,
                num_layers=num_layers,
                mlp_hidden_dims=(16,),
                tp_mul_per_irrep=1,
            ),
        ),
    ]


def _assert_equivariant(module: torch.nn.Module, atol: float = 5e-4):
    module.eval()

    x = IRREPS_IN.randn(BATCH, -1)
    rot = rand_matrix()

    d_in = IRREPS_IN.D_from_matrix(rot)
    d_out = IRREPS_OUT.D_from_matrix(rot)

    y_from_rot_input = module(x @ d_in.T)
    y_rot_from_output = module(x) @ d_out.T

    assert torch.allclose(y_from_rot_input, y_rot_from_output, atol=atol, rtol=atol)


@pytest.mark.unit
@pytest.mark.parametrize(
    "num_layers", [1, 2], ids=["linear_depth1", "nonlinear_depth2"]
)
def test_e3mlp_variants_initialization_and_forward(num_layers):
    torch.manual_seed(0)
    x = IRREPS_IN.randn(BATCH, -1)

    for name, model in _make_variants(num_layers):
        y = model(x)
        assert y.shape == (BATCH, IRREPS_OUT.dim), name


@pytest.mark.unit
def test_e3mlp_variants_equivariance():
    torch.manual_seed(0)
    for name, model in _make_variants(num_layers=2):
        _assert_equivariant(model)
