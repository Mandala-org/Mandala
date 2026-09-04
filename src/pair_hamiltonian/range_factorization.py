"""Training-only range-magnitude fitting for pair Hamiltonian targets."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from os import PathLike
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares
import torch

from data.envelope import evaluate_slater_soft_cutoff

HARTREE_TO_MEV = 27_211.386245988


@dataclass(frozen=True, slots=True)
class SlaterEnvelopeFit:
    """Frozen scalar envelope for one unordered species-pair type."""

    pair: str
    theta: tuple[float, float, float, float, float]
    cutoff_angstrom: float
    bin_width_angstrom: float
    fitted_block_count: int
    nonempty_bin_count: int
    log_rmse: float

    def evaluate(self, distance_angstrom: torch.Tensor) -> torch.Tensor:
        parameters = distance_angstrom.new_tensor(self.theta).expand(
            distance_angstrom.numel(), -1
        )
        return evaluate_slater_soft_cutoff(
            distance_angstrom.reshape(-1), parameters
        ).reshape(distance_angstrom.shape)


class RangeEnvelopeAccumulator:
    """Accumulate train-only log-RMS statistics without retaining blocks."""

    def __init__(self, cutoff_angstrom: float, bin_width_angstrom: float) -> None:
        if cutoff_angstrom <= 0 or bin_width_angstrom <= 0:
            raise ValueError("cutoff and bin width must be positive")
        self.cutoff_angstrom = float(cutoff_angstrom)
        self.bin_width_angstrom = float(bin_width_angstrom)
        self.bin_count = int(np.ceil(cutoff_angstrom / bin_width_angstrom)) + 1
        self.count = np.zeros(self.bin_count, dtype=np.int64)
        self.log_sum = np.zeros(self.bin_count, dtype=np.float64)
        self.log_square_sum = np.zeros(self.bin_count, dtype=np.float64)

    def update(self, distances: np.ndarray, target_irreps: np.ndarray) -> None:
        distances = np.asarray(distances, dtype=np.float64)
        targets = np.asarray(target_irreps, dtype=np.float64)
        if targets.ndim != 2 or targets.shape[0] != distances.size:
            raise ValueError("targets must have shape (pairs, target_dimension)")
        if distances.size == 0:
            return
        # The AO<->irrep map is orthonormal, so this is physical element RMS.
        magnitude = np.sqrt(np.mean(np.square(targets), axis=1))
        valid = np.isfinite(magnitude) & (magnitude > np.finfo(np.float64).tiny)
        indices = np.minimum(
            (distances[valid] / self.bin_width_angstrom).astype(np.int64),
            self.bin_count - 1,
        )
        logs = np.log(magnitude[valid])
        self.count += np.bincount(indices, minlength=self.bin_count)
        self.log_sum += np.bincount(indices, weights=logs, minlength=self.bin_count)
        self.log_square_sum += np.bincount(
            indices, weights=np.square(logs), minlength=self.bin_count
        )

    def fit(self, pair: str) -> SlaterEnvelopeFit:
        selected = self.count > 0
        if np.count_nonzero(selected) < 5:
            raise ValueError(f"Too few populated distance bins to fit {pair}")
        radius = (
            np.arange(self.bin_count, dtype=np.float64) + 0.5
        ) * self.bin_width_angstrom
        x = radius[selected]
        y = self.log_sum[selected] / self.count[selected]
        weights = np.sqrt(np.minimum(self.count[selected], 10_000)).astype(np.float64)
        weights /= np.sqrt(np.mean(np.square(weights)))

        initial = np.asarray(
            [
                float(np.median(y[: min(5, y.size)])),
                np.log(0.75),
                0.0,
                self.cutoff_angstrom - 0.25,
                np.log(0.2),
            ],
            dtype=np.float64,
        )

        def residual(theta: np.ndarray) -> np.ndarray:
            distance = torch.from_numpy(x)
            params = torch.from_numpy(theta).expand(x.size, -1)
            prediction = evaluate_slater_soft_cutoff(distance, params).numpy()
            return weights * (np.log(np.maximum(prediction, 1.0e-30)) - y)

        result = least_squares(
            residual,
            initial,
            bounds=(
                np.asarray([-30.0, -6.0, -8.0, 0.5, -6.0]),
                np.asarray([10.0, 3.0, 8.0, self.cutoff_angstrom + 2.0, 1.0]),
            ),
            loss="soft_l1",
            f_scale=0.5,
            max_nfev=2_000,
        )
        if not result.success or not np.all(np.isfinite(result.x)):
            raise RuntimeError(f"Envelope fit failed for {pair}: {result.message}")
        unweighted = residual(result.x) / weights
        return SlaterEnvelopeFit(
            pair=pair,
            theta=tuple(float(value) for value in result.x),
            cutoff_angstrom=self.cutoff_angstrom,
            bin_width_angstrom=self.bin_width_angstrom,
            fitted_block_count=int(self.count.sum()),
            nonempty_bin_count=int(np.count_nonzero(selected)),
            log_rmse=float(np.sqrt(np.mean(np.square(unweighted)))),
        )


def envelope_manifest(
    fits: Iterable[SlaterEnvelopeFit], *, dataset_fingerprint: str, split_hash: str
) -> dict[str, object]:
    """Build the immutable JSON payload consumed by existing envelope code."""
    ordered = sorted(fits, key=lambda fit: fit.pair)
    payload: dict[str, object] = {
        "version": "mandala-pair-range-v1",
        "family": "slater_soft_cutoff",
        "fit_partition": "train",
        "magnitude": "per-block physical matrix-element RMS in hartree",
        "dataset_fingerprint": dataset_fingerprint,
        "split_hash": split_hash,
        "pairs": {
            fit.pair: {
                "theta": list(fit.theta),
                "cutoff_angstrom": fit.cutoff_angstrom,
                "bin_width_angstrom": fit.bin_width_angstrom,
                "fitted_block_count": fit.fitted_block_count,
                "nonempty_bin_count": fit.nonempty_bin_count,
                "log_rmse": fit.log_rmse,
            }
            for fit in ordered
        },
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    payload["content_hash"] = hashlib.sha256(canonical.encode()).hexdigest()
    return payload


def plot_range_envelope_fits(
    fits: Iterable[SlaterEnvelopeFit],
    samples: dict[str, tuple[np.ndarray, np.ndarray]],
    output_path: str | PathLike[str],
    *,
    max_scatter_points: int = 50_000,
) -> list[dict[str, float | int | str]]:
    """Plot training block RMS magnitudes and frozen fitted range curves.

    ``samples[pair]`` contains distances in angstrom and per-block physical
    matrix-element RMS magnitudes in hartree.  Scatter downsampling is
    deterministic and affects only rendering, never fitting or bin summaries.
    """
    ordered = sorted(fits, key=lambda fit: fit.pair)
    if max_scatter_points <= 0:
        raise ValueError("max_scatter_points must be positive")
    figure, axes = plt.subplots(
        1, len(ordered), figsize=(5.2 * len(ordered), 4.6), sharey=True
    )
    axes = np.atleast_1d(axes)
    rows: list[dict[str, float | int | str]] = []
    for axis, fit in zip(axes, ordered):
        distance, magnitude = samples[fit.pair]
        distance = np.asarray(distance, dtype=np.float64)
        magnitude = np.asarray(magnitude, dtype=np.float64)
        valid = (
            np.isfinite(distance)
            & np.isfinite(magnitude)
            & (magnitude > np.finfo(np.float64).tiny)
        )
        distance = distance[valid]
        magnitude = magnitude[valid]
        if distance.size == 0:
            raise ValueError(f"No finite positive samples for {fit.pair}")
        if distance.size > max_scatter_points:
            shown = np.linspace(
                0, distance.size - 1, max_scatter_points, dtype=np.int64
            )
        else:
            shown = np.arange(distance.size)
        axis.scatter(
            distance[shown],
            magnitude[shown] * HARTREE_TO_MEV,
            s=3,
            alpha=0.08,
            color="#4472c4",
            linewidths=0,
            rasterized=True,
            label="training blocks",
        )
        bin_index = np.floor(distance / fit.bin_width_angstrom).astype(np.int64)
        bin_centers: list[float] = []
        geometric_means: list[float] = []
        for index in np.unique(bin_index):
            selected = magnitude[bin_index == index]
            center = (index + 0.5) * fit.bin_width_angstrom
            geometric_mean = float(np.exp(np.mean(np.log(selected))))
            bin_centers.append(center)
            geometric_means.append(geometric_mean)
            rows.append(
                {
                    "species_pair": fit.pair,
                    "distance_center_angstrom": center,
                    "block_count": int(selected.size),
                    "geometric_mean_rms_mev": geometric_mean * HARTREE_TO_MEV,
                    "q10_rms_mev": float(np.quantile(selected, 0.1)) * HARTREE_TO_MEV,
                    "q90_rms_mev": float(np.quantile(selected, 0.9)) * HARTREE_TO_MEV,
                }
            )
        axis.plot(
            bin_centers,
            np.asarray(geometric_means) * HARTREE_TO_MEV,
            color="#111111",
            linewidth=1.4,
            marker="o",
            markersize=2.5,
            label="binned geometric mean",
        )
        fit_distance = torch.linspace(
            max(1.0e-4, float(distance.min())),
            fit.cutoff_angstrom,
            500,
            dtype=torch.float64,
        )
        fit_magnitude = fit.evaluate(fit_distance).numpy() * HARTREE_TO_MEV
        axis.plot(
            fit_distance.numpy(),
            fit_magnitude,
            color="#c43b35",
            linewidth=2.2,
            label="frozen Slater fit",
        )
        axis.set_yscale("log")
        axis.set_xlim(left=0.0, right=fit.cutoff_angstrom)
        axis.set_xlabel("Pair distance (angstrom)")
        axis.set_title(fit.pair)
        axis.grid(True, which="both", alpha=0.18, linewidth=0.6)
    axes[0].set_ylabel("Per-block matrix-element RMS magnitude (meV)")
    axes[0].legend(loc="lower left", frameon=False)
    figure.suptitle("Training-only SiO2 range-envelope fits")
    figure.tight_layout()
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)
    return rows
