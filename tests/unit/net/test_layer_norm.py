import torch
from e3nn.o3 import Irreps

from net.layer_norm import E3LayerNorm


def test_e3layernorm_forward_does_not_depend_on_irreps_runtime_object():
    irreps = Irreps("2x0e+1x1o")
    eps = 0.1
    norm = E3LayerNorm(irreps, eps=eps, affine=True)
    x = torch.tensor(
        [
            [1.0, 3.0, 1.0, 2.0, 3.0],
            [5.0, 7.0, 2.0, 0.0, 4.0],
            [10.0, 14.0, 2.0, 4.0, 6.0],
            [18.0, 22.0, 4.0, 2.0, 0.0],
            [14.0, 16.0, 0.0, 6.0, 2.0],
        ]
    )
    batch = torch.tensor([0, 0, 1, 1, 1], dtype=torch.long)
    with torch.no_grad():
        norm.weight.copy_(torch.tensor([2.0, 0.5, 3.0]))
        norm.bias.copy_(torch.tensor([0.25, -0.5]))

    # Forward should use the cached plain-Python field specs, not iterate the
    # runtime Irreps object. This keeps the runtime path independent from
    # e3nn's custom container semantics.
    norm.irreps_in = object()  # type: ignore[assignment]

    out = norm(x, batch)

    assert out.shape == x.shape
    assert torch.isfinite(out).all()
    expected = torch.empty_like(x)
    for graph_id in (0, 1):
        selected = batch == graph_id
        scalars = x[selected, :2]
        centered = scalars - scalars.mean()
        scalar_scale = (centered.square().mean() + eps).sqrt() + eps
        expected[selected, :2] = centered / scalar_scale * norm.weight[:2] + norm.bias
        vectors = x[selected, 2:].reshape(-1, 1, 3)
        vector_scale = (vectors.square().mean() + eps).sqrt() + eps
        expected[selected, 2:] = (vectors / vector_scale * norm.weight[2]).reshape(
            -1, 3
        )
    torch.testing.assert_close(out, expected)
