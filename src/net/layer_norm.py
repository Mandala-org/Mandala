from __future__ import annotations

import torch
from torch import nn
from e3nn.o3 import Irreps
from torch_scatter import scatter


class E3LayerNorm(nn.Module):
    """
    DeepH-E3 style equivariant layer normalization.

    This is intentionally close to the study implementation so the silicon
    sweeps can reproduce the same normalization path.
    """

    def __init__(
        self,
        irreps_in: Irreps,
        eps: float = 1e-5,
        affine: bool = True,
        normalization: str = "component",
    ) -> None:
        super().__init__()
        self.irreps_in = Irreps(irreps_in)
        self.eps = float(eps)
        self.normalization = normalization
        self.field_specs: list[tuple[int, int, bool, int]] = []

        if affine:
            ib, iw = 0, 0
            weight_slices: list[slice] = []
            bias_slices: list[slice | None] = []
            for mul, ir in self.irreps_in:
                self.field_specs.append((mul, ir.dim, ir.is_scalar(), ir.l))
                if ir.is_scalar():
                    bias_slices.append(slice(ib, ib + mul))
                    ib += mul
                else:
                    bias_slices.append(None)
                weight_slices.append(slice(iw, iw + mul))
                iw += mul
            self.weight = nn.Parameter(torch.ones([iw]))
            self.bias = nn.Parameter(torch.zeros([ib]))
            self.weight_slices = weight_slices
            self.bias_slices = bias_slices
        else:
            for mul, ir in self.irreps_in:
                self.field_specs.append((mul, ir.dim, ir.is_scalar(), ir.l))
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)
            self.weight_slices = []
            self.bias_slices = []

    def forward(
        self,
        x: torch.Tensor,
        batch: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if batch is None:
            batch = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)

        batch_size = int(batch.max().item()) + 1 if batch.numel() > 0 else 1
        batch_degree = (
            torch.bincount(batch, minlength=batch_size)
            .clamp_min(1)
            .to(device=x.device, dtype=x.dtype)
        )

        out = []
        ix = 0
        for index, (mul, ir_dim, is_scalar, l_value) in enumerate(self.field_specs):
            field = x[:, ix : ix + mul * ir_dim].reshape(-1, mul, ir_dim)

            if l_value == 0:
                mean = (
                    scatter(
                        field, batch, dim=0, dim_size=batch_size, reduce="add"
                    ).mean(dim=1, keepdim=True)
                    / batch_degree[:, None, None]
                )
                field = field - mean[batch]

            norm = scatter(
                field.abs().pow(2),
                batch,
                dim=0,
                dim_size=batch_size,
                reduce="mean",
            ).mean(dim=[1, 2], keepdim=True)
            if self.normalization == "norm":
                norm = norm * ir_dim

            safe_norm = torch.sqrt(norm + self.eps)
            field = field / (safe_norm[batch] + self.eps)

            if self.weight is not None:
                weight = self.weight[self.weight_slices[index]]
                field = field * weight[None, :, None]
            if self.bias is not None and is_scalar:
                bias_slice = self.bias_slices[index]
                assert bias_slice is not None
                bias = self.bias[bias_slice]
                field = field + bias[None, :, None]

            out.append(field.reshape(-1, mul * ir_dim))
            ix += mul * ir_dim

        return torch.cat(out, dim=-1)
