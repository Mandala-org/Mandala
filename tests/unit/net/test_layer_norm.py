import torch
from e3nn.o3 import Irreps

from net.layer_norm import E3LayerNorm


def test_e3layernorm_forward_does_not_depend_on_irreps_runtime_object():
    irreps = Irreps("2x0e+1x1o")
    norm = E3LayerNorm(irreps, affine=True)
    x = torch.randn(5, irreps.dim)
    batch = torch.tensor([0, 0, 1, 1, 1], dtype=torch.long)

    # Forward should use the cached plain-Python field specs, not iterate the
    # runtime Irreps object. This keeps the runtime path independent from
    # e3nn's custom container semantics.
    norm.irreps_in = object()  # type: ignore[assignment]

    out = norm(x, batch)

    assert out.shape == x.shape
    assert torch.isfinite(out).all()
