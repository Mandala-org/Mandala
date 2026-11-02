"""Unweighted errors over observed physical matrix elements.

Each directed stored matrix element contributes once, including both reverse
edges. There is no species-pair balancing, irrep weighting, or padding. Callers
choose physical basis/frame, projection, and units before using this function.
"""

from __future__ import annotations

import torch


@torch.no_grad()
def matrix_error_totals(prediction, target, *, edge_selector=None):
    """Return absolute/squared error sums and the number of observed entries.

    Targets must be an aligned prefix of the predicted support. Partial shell
    coverage and an optional edge selector restrict the evaluated entries. Sums
    use float64 to permit stable accumulation over an entire dataset.
    """
    if prediction.basis != target.basis:
        raise ValueError("Matrix metrics require the same physical basis")
    absolute, squared, count = None, None, 0
    coverage = getattr(target, "coverage_masks", {})
    for key, values in target.pair_blocks.items():
        n = values.shape[0]
        if key not in prediction.pair_blocks:
            raise ValueError(f"Missing predicted matrix pair: {key}")
        predicted = prediction.pair_blocks[key][:n]
        if predicted.shape != values.shape or not torch.equal(
            prediction.pair_edges[key][:, :n], target.pair_edges[key]
        ):
            raise ValueError(f"Matrix metric support/shape mismatch for {key}")
        mask = coverage.get(key)
        mask = (
            torch.ones_like(values, dtype=torch.bool) if mask is None else mask.clone()
        )
        if edge_selector is not None:
            selected = edge_selector(target.pair_edges[key])
            if selected is not None:
                mask &= selected.reshape((-1,) + (1,) * (values.ndim - 1))
        difference = (predicted - values)[mask].abs().double()
        if not torch.isfinite(difference).all():
            raise ValueError(f"Nonfinite observed matrix error for {key}")
        a, s = difference.sum(), difference.square().sum()
        absolute = a if absolute is None else absolute + a
        squared = s if squared is None else squared + s
        count += difference.numel()
    if not count:
        raise ValueError("No observed matrix elements to evaluate")
    return {"absolute_sum": absolute, "squared_sum": squared, "count": count}


def matrix_error_metrics(prediction, target, *, scale=1.0, edge_selector=None):
    """Element-weighted MAE/MSE/RMSE for one structure, in caller-selected units.

    The shared trainer averages these per-structure values over the epoch, matching
    the reference trainers. Dataset-wide entry weighting can instead sum totals.
    """
    totals = matrix_error_totals(prediction, target, edge_selector=edge_selector)
    mse = totals["squared_sum"] * scale**2 / totals["count"]
    return {
        "mae_elementwise": totals["absolute_sum"] * scale / totals["count"],
        "mse_elementwise": mse,
        "rmse_elementwise": mse.sqrt(),
    }
