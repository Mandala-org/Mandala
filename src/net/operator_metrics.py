"""Unweighted errors over observed physical matrix elements.

Each directed stored matrix element contributes once, including both reverse
edges. There is no species-pair balancing, irrep weighting, or padding. Callers
choose physical basis/frame, projection, and units before using this function.
"""

from __future__ import annotations

import torch


def _observed_matrix_errors(prediction, target, *, edge_selector=None):
    """Yield absolute errors and references on the aligned observed support."""
    if prediction.basis != target.basis:
        raise ValueError("Matrix metrics require the same physical basis")
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
        yield difference, values[mask]


@torch.no_grad()
def matrix_error_totals(prediction, target, *, edge_selector=None):
    """Return absolute/squared error sums and the number of observed entries.

    Targets must be an aligned prefix of the predicted support. Partial shell
    coverage and an optional edge selector restrict the evaluated entries. Sums
    use float64 to permit stable accumulation over an entire dataset.
    """
    absolute, squared, count = None, None, 0
    for difference, _ in _observed_matrix_errors(
        prediction, target, edge_selector=edge_selector
    ):
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


class OperatorErrorDistribution:
    """Pool physical element errors without retaining a validation tensor.

    Absolute errors are appended as float64 to a temporary file; NumPy computes
    exact linear percentiles in place on a memory map (selection, no histogram
    approximation). Reference moments use the parallel/Chan variance update.
    The caller supplies the historical paper MAE separately: its weighting may
    differ from the pooled element MAE reported here as a diagnostic.
    """

    def __init__(self, *, scratch_dir=None):
        import tempfile

        self._file = tempfile.TemporaryFile(dir=scratch_dir)
        self.count = 0
        self.structures = 0
        self.absolute_sum = 0.0
        self.squared_sum = 0.0
        self.maximum = 0.0
        self.reference_mean = 0.0
        self.reference_m2 = 0.0
        self._finalized = False

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def close(self):
        self._file.close()

    @torch.no_grad()
    def update(self, prediction, target, *, scale=1.0, edge_selector=None):
        """Add one structure and return its pooled physical-element MAE."""
        import math
        import numpy as np

        if self._finalized:
            raise RuntimeError("Cannot update a finalized error distribution")
        if not math.isfinite(scale) or scale <= 0:
            raise ValueError("Metric scale must be finite and positive")
        structure_count, structure_absolute = 0, 0.0
        for error, reference in _observed_matrix_errors(
            prediction, target, edge_selector=edge_selector
        ):
            n = error.numel()
            if not n:
                continue
            errors = error.cpu().numpy() * scale
            dtype = torch.complex128 if reference.is_complex() else torch.float64
            refs = reference.to(dtype=dtype).cpu().numpy() * scale
            if not np.isfinite(refs).all():
                raise ValueError("Nonfinite observed reference")
            chunk_mean = refs.mean()
            chunk_m2 = float(np.square(np.abs(refs - chunk_mean)).sum())
            combined = self.count + n
            delta = chunk_mean - self.reference_mean
            self.reference_m2 += (
                chunk_m2 + abs(delta) ** 2 * self.count * n / combined
            )
            self.reference_mean += delta * n / combined
            self.count = combined
            absolute = float(errors.sum())
            self.absolute_sum += absolute
            self.squared_sum += float(np.dot(errors, errors))
            self.maximum = max(self.maximum, float(errors.max()))
            self._file.write(errors.tobytes())
            structure_count += n
            structure_absolute += absolute
        if not structure_count:
            raise ValueError("No observed matrix elements to evaluate")
        self.structures += 1
        return structure_absolute / structure_count

    def compute(self):
        import math
        import numpy as np

        if not self.count:
            raise ValueError("No observed matrix elements to evaluate")
        self._file.flush()
        values = np.memmap(self._file, dtype=np.float64, mode="r+", shape=(self.count,))
        try:
            p95, p99 = np.quantile(
                values, [0.95, 0.99], method="linear", overwrite_input=True
            )
        finally:
            del values
        self._finalized = True
        return {
            "mae_elementwise_pooled": self.absolute_sum / self.count,
            "rmse": math.sqrt(self.squared_sum / self.count),
            "max_absolute_error": self.maximum,
            "p95_absolute_error": float(p95),
            "p99_absolute_error": float(p99),
            "r2": (
                1.0 - self.squared_sum / self.reference_m2
                if self.reference_m2 > 0.0
                else None
            ),
            "element_count": self.count,
            "structures": self.structures,
            "residual_sum_squares": self.squared_sum,
            "reference_centered_sum_squares": float(self.reference_m2),
        }
