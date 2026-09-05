"""Range-factored objectives with physical-space sample weighting."""

from __future__ import annotations

from typing import Literal

import torch

RangeGradientLoss = Literal["physical_mse", "weighted_normalized_mse"]
RangeClosedFormFit = Literal["physical_design", "weighted_normalized_target"]


def physical_offsite_prediction(
    model_output: torch.Tensor, envelope: torch.Tensor
) -> torch.Tensor:
    if envelope.shape != model_output.shape[:-1]:
        raise ValueError("envelope must have one scalar per model output")
    return model_output * envelope[..., None]


def range_factored_mse(
    model_output: torch.Tensor,
    target: torch.Tensor,
    envelope: torch.Tensor,
    *,
    mode: RangeGradientLoss,
) -> torch.Tensor:
    """Evaluate a range-factored loss without overweighting weak long bonds.

    ``physical_mse`` is exactly ``mean((G*M-H)^2)``.  The normalized-target
    form applies the required ``G^2`` sample weights. It is algebraically equal
    to the physical-space loss, including its global scale and gradients.
    """
    if model_output.shape != target.shape:
        raise ValueError("model_output and target must have identical shapes")
    if envelope.shape != target.shape[:-1] or torch.any(envelope <= 0):
        raise ValueError("envelope must be positive with one value per sample")
    physical_residual = physical_offsite_prediction(model_output, envelope) - target
    if mode == "physical_mse":
        return physical_residual.square().mean()
    if mode == "weighted_normalized_mse":
        normalized_residual = model_output - target / envelope[..., None]
        weights = envelope.square()
        return (weights[..., None] * normalized_residual.square()).mean()
    raise ValueError(f"Unknown range-factored loss mode {mode!r}")


def closed_form_range_batch(
    features: torch.Tensor,
    target: torch.Tensor,
    envelope: torch.Tensor,
    *,
    mode: RangeClosedFormFit,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Return an equivalent physical or weighted-normalized ridge batch."""
    if features.ndim != 2 or target.ndim != 2 or features.shape[0] != target.shape[0]:
        raise ValueError("features and target must be two-dimensional sample arrays")
    if envelope.shape != (features.shape[0],) or torch.any(envelope <= 0):
        raise ValueError("envelope must be positive with one value per sample")
    if mode == "physical_design":
        return features * envelope[:, None], target, None
    if mode == "weighted_normalized_target":
        return features, target / envelope[:, None], envelope.square()
    raise ValueError(f"Unknown closed-form range mode {mode!r}")
