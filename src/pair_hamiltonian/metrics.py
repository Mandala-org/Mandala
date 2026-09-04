"""Streaming physical metrics for full-block irrep predictions."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import torch

from pair_hamiltonian.hamgnn_sio2 import HARTREE_TO_MEV
from pair_hamiltonian.output_schema import FullBlockIrrepTransform


@dataclass
class _Sums:
    absolute: float = 0.0
    squared: float = 0.0
    count: int = 0

    def update(self, difference: torch.Tensor) -> None:
        self.absolute += float(difference.abs().sum().item())
        self.squared += float(difference.square().sum().item())
        self.count += difference.numel()

    def metrics(self, scale: float) -> dict[str, float | int]:
        if self.count == 0:
            return {"mae": 0.0, "rmse": 0.0, "scalar_count": 0}
        return {
            "mae": self.absolute / self.count * scale,
            "rmse": (self.squared / self.count) ** 0.5 * scale,
            "scalar_count": self.count,
        }


@dataclass
class _BlockNormSums:
    norm_sum: float = 0.0
    norm_square_sum: float = 0.0
    count: int = 0

    def update(self, difference: torch.Tensor) -> None:
        norms = torch.linalg.matrix_norm(difference, ord="fro")
        self.norm_sum += float(norms.sum().item())
        self.norm_square_sum += float(norms.square().sum().item())
        self.count += norms.numel()

    def metrics(self, scale: float) -> dict[str, float | int]:
        if self.count == 0:
            return {"mae": 0.0, "rmse": 0.0, "block_count": 0}
        return {
            "mae": self.norm_sum / self.count * scale,
            "rmse": (self.norm_square_sum / self.count) ** 0.5 * scale,
            "block_count": self.count,
        }


class FullBlockMetricAccumulator:
    """Accumulate headline reconstructed-element and diagnostic irrep metrics."""

    def __init__(
        self,
        transform: FullBlockIrrepTransform,
        *,
        distance_bin_width_angstrom: float = 0.5,
    ) -> None:
        if distance_bin_width_angstrom <= 0:
            raise ValueError("distance_bin_width_angstrom must be positive")
        self.transform = transform
        self.distance_bin_width_angstrom = distance_bin_width_angstrom
        self.global_sums = _Sums()
        self.block_norm_sums = _BlockNormSums()
        self.by_pair: dict[str, _Sums] = defaultdict(_Sums)
        self.by_site: dict[str, _Sums] = defaultdict(_Sums)
        self.by_distance: dict[int, _Sums] = defaultdict(_Sums)
        self.by_irrep: dict[str, _Sums] = defaultdict(_Sums)
        self.by_irrep_copy: dict[str, _Sums] = defaultdict(_Sums)

    def update(
        self,
        pair: tuple[str, str] | str,
        prediction: torch.Tensor,
        target: torch.Tensor,
        *,
        onsite: bool,
        distances_angstrom: torch.Tensor | None = None,
    ) -> None:
        schema = self.transform.schema(pair)
        if (
            prediction.shape != target.shape
            or prediction.shape[-1] != schema.vector_dimension
        ):
            raise ValueError(
                "Prediction and target must share the full schema dimension"
            )
        pair_name = f"{schema.species_i}-{schema.species_j}"
        predicted_blocks = self.transform.irreps_to_blocks(pair, prediction)
        target_blocks = self.transform.irreps_to_blocks(pair, target)
        block_difference = predicted_blocks - target_blocks
        self.global_sums.update(block_difference)
        self.block_norm_sums.update(block_difference)
        self.by_pair[pair_name].update(block_difference)
        self.by_site["onsite" if onsite else "offsite"].update(block_difference)
        if distances_angstrom is not None:
            if distances_angstrom.shape != prediction.shape[:-1]:
                raise ValueError("distances must have one value per block")
            indices = torch.floor(
                distances_angstrom / self.distance_bin_width_angstrom
            ).to(torch.long)
            for bin_index in torch.unique(indices).tolist():
                self.by_distance[int(bin_index)].update(
                    block_difference[indices == bin_index]
                )
        vector_difference = prediction - target
        for copy in schema.copies:
            difference = vector_difference[..., copy.vector_start : copy.vector_stop]
            self.by_irrep[copy.irrep_label].update(difference)
            self.by_irrep_copy[
                f"{pair_name}:q{copy.copy_index:03d}:{copy.irrep_label}"
            ].update(difference)

    def compute(self) -> dict[str, object]:
        matrix_scale = HARTREE_TO_MEV
        return {
            "units": {"matrix_mae_rmse": "meV", "irrep_mae_rmse": "meV"},
            "matrix_elements": self.global_sums.metrics(matrix_scale),
            "block_frobenius": self.block_norm_sums.metrics(matrix_scale),
            "by_species_pair": {
                key: value.metrics(matrix_scale)
                for key, value in sorted(self.by_pair.items())
            },
            "by_site": {
                key: value.metrics(matrix_scale)
                for key, value in sorted(self.by_site.items())
            },
            "by_distance": {
                f"{index * self.distance_bin_width_angstrom:.3f}-"
                f"{(index + 1) * self.distance_bin_width_angstrom:.3f}_angstrom": value.metrics(
                    matrix_scale
                )
                for index, value in sorted(self.by_distance.items())
            },
            "by_target_irrep": {
                key: value.metrics(matrix_scale)
                for key, value in sorted(self.by_irrep.items())
            },
            "by_target_irrep_copy": {
                key: value.metrics(matrix_scale)
                for key, value in sorted(self.by_irrep_copy.items())
            },
        }
