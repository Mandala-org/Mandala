"""Closed-form equivariant linear regression in multiplicity space."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from e3nn.o3 import Irrep, Irreps
import torch
from torch import nn


def _expanded_slices(irreps: Irreps) -> list[tuple[Irrep, slice]]:
    result: list[tuple[Irrep, slice]] = []
    offset = 0
    for multiplicity, irrep in irreps:
        for _ in range(multiplicity):
            result.append((irrep, slice(offset, offset + irrep.dim)))
            offset += irrep.dim
    return result


def _key(irrep: Irrep) -> str:
    return f"l{irrep.l}_{'e' if irrep.p == 1 else 'o'}"


@dataclass(frozen=True, slots=True)
class RidgeFitDiagnostics:
    sample_count: int
    scalar_equation_count: int
    feature_dimension: int
    target_dimension: int
    ridge: float
    condition_numbers: dict[str, float]


class EquivariantRidgeAccumulator:
    """Bounded-memory sufficient statistics for an equivariant ridge fit."""

    def __init__(
        self,
        feature_irreps: Irreps | str,
        target_irreps: Irreps | str,
        *,
        dtype: torch.dtype = torch.float64,
        device: torch.device | str = "cpu",
        allow_missing_target_irreps: bool = False,
    ) -> None:
        self.feature_irreps = Irreps(feature_irreps)
        self.target_irreps = Irreps(target_irreps)
        self.feature_groups: dict[Irrep, list[slice]] = defaultdict(list)
        self.target_groups: dict[Irrep, list[slice]] = defaultdict(list)
        for irrep, component_slice in _expanded_slices(self.feature_irreps):
            self.feature_groups[irrep].append(component_slice)
        for irrep, component_slice in _expanded_slices(self.target_irreps):
            self.target_groups[irrep].append(component_slice)
        self.sample_count = 0
        self.xtx: dict[Irrep, torch.Tensor] = {}
        self.xty: dict[Irrep, torch.Tensor] = {}
        self.sum_square: dict[Irrep, torch.Tensor] = {}
        self.equation_count: dict[Irrep, int] = defaultdict(int)
        for irrep, target_slices in self.target_groups.items():
            feature_count = len(self.feature_groups[irrep])
            if feature_count == 0 and not allow_missing_target_irreps:
                raise ValueError(f"Feature basis lacks required target irrep: {irrep}")
            self.xtx[irrep] = torch.zeros(
                feature_count, feature_count, dtype=dtype, device=device
            )
            self.xty[irrep] = torch.zeros(
                feature_count, len(target_slices), dtype=dtype, device=device
            )
            self.sum_square[irrep] = torch.zeros(
                feature_count, dtype=dtype, device=device
            )

    @torch.no_grad()
    def update(self, features: torch.Tensor, targets: torch.Tensor) -> None:
        if features.ndim != 2 or features.shape[1] != self.feature_irreps.dim:
            raise ValueError("features do not match accumulator feature irreps")
        if targets.shape != (features.shape[0], self.target_irreps.dim):
            raise ValueError("targets do not match accumulator target irreps")
        if features.shape[0] == 0:
            return
        self.sample_count += features.shape[0]
        for irrep, target_slices in self.target_groups.items():
            if not self.feature_groups[irrep]:
                continue
            x = EquivariantRidgeRegressor._gather(
                features, self.feature_groups[irrep]
            ).to(self.xtx[irrep])
            y = EquivariantRidgeRegressor._gather(targets, target_slices).to(
                self.xty[irrep]
            )
            x = x.permute(0, 2, 1).reshape(-1, x.shape[-2])
            y = y.permute(0, 2, 1).reshape(-1, y.shape[-2])
            self.xtx[irrep].add_(x.T @ x)
            self.xty[irrep].add_(x.T @ y)
            self.sum_square[irrep].add_(x.square().sum(dim=0))
            self.equation_count[irrep] += x.shape[0]


class EquivariantRidgeRegressor(nn.Module):
    """One scalar coefficient per equal-irrep feature/target copy pair.

    Magnetic components of an irrep copy share coefficients exactly.  Feature
    RMS normalization is fitted on training samples and is likewise shared
    across all ``m`` components.  No centering is applied to nonscalars.
    """

    def __init__(
        self,
        feature_irreps: Irreps | str,
        target_irreps: Irreps | str,
        *,
        ridge: float = 1.0e-8,
        dtype: torch.dtype = torch.float64,
        allow_missing_target_irreps: bool = False,
    ) -> None:
        super().__init__()
        if ridge < 0:
            raise ValueError("ridge must be non-negative")
        self.feature_irreps = Irreps(feature_irreps)
        self.target_irreps = Irreps(target_irreps)
        self.ridge = float(ridge)
        self.feature_groups: dict[Irrep, list[slice]] = defaultdict(list)
        self.target_groups: dict[Irrep, list[slice]] = defaultdict(list)
        for irrep, component_slice in _expanded_slices(self.feature_irreps):
            self.feature_groups[irrep].append(component_slice)
        for irrep, component_slice in _expanded_slices(self.target_irreps):
            self.target_groups[irrep].append(component_slice)
        missing = sorted(
            {
                str(irrep)
                for irrep in self.target_groups
                if irrep not in self.feature_groups
            }
        )
        if missing and not allow_missing_target_irreps:
            raise ValueError(f"Feature basis lacks required target irreps: {missing}")
        for irrep, feature_slices in self.feature_groups.items():
            self.register_buffer(
                f"scale_{_key(irrep)}",
                torch.ones(len(feature_slices), dtype=dtype),
            )
            target_count = len(self.target_groups.get(irrep, ()))
            self.register_buffer(
                f"weight_{_key(irrep)}",
                torch.zeros(target_count, len(feature_slices), dtype=dtype),
            )
        self.diagnostics: RidgeFitDiagnostics | None = None

    @staticmethod
    def _gather(values: torch.Tensor, slices: list[slice]) -> torch.Tensor:
        return torch.stack([values[..., item] for item in slices], dim=-2)

    @torch.no_grad()
    def fit(self, features: torch.Tensor, targets: torch.Tensor) -> RidgeFitDiagnostics:
        if features.ndim != 2 or features.shape[1] != self.feature_irreps.dim:
            raise ValueError("features must have shape (samples, feature_irreps.dim)")
        if targets.shape != (features.shape[0], self.target_irreps.dim):
            raise ValueError("targets must have shape (samples, target_irreps.dim)")
        if features.shape[0] == 0:
            raise ValueError("Cannot fit an empty dataset")
        dtype = next(self.buffers()).dtype
        features = features.to(dtype=dtype)
        targets = targets.to(dtype=dtype, device=features.device)
        conditions: dict[str, float] = {}
        equations = 0
        for irrep, target_slices in self.target_groups.items():
            feature_slices = self.feature_groups[irrep]
            if not feature_slices:
                conditions[str(irrep)] = float("inf")
                continue
            x = self._gather(features, feature_slices)  # sample, copy, m
            y = self._gather(targets, target_slices)
            scale = torch.sqrt(torch.mean(x.square(), dim=(0, 2))).clamp_min(1.0e-12)
            getattr(self, f"scale_{_key(irrep)}").copy_(scale)
            x = (
                (x / scale[None, :, None])
                .permute(0, 2, 1)
                .reshape(-1, len(feature_slices))
            )
            y = y.permute(0, 2, 1).reshape(-1, len(target_slices))
            gram = x.T @ x
            regularized = gram + self.ridge * torch.eye(
                gram.shape[0], dtype=gram.dtype, device=gram.device
            )
            solution = torch.linalg.solve(regularized, x.T @ y)
            getattr(self, f"weight_{_key(irrep)}").copy_(solution.T)
            conditions[str(irrep)] = float(torch.linalg.cond(regularized).item())
            equations += x.shape[0]
        self.diagnostics = RidgeFitDiagnostics(
            sample_count=features.shape[0],
            scalar_equation_count=equations,
            feature_dimension=self.feature_irreps.dim,
            target_dimension=self.target_irreps.dim,
            ridge=self.ridge,
            condition_numbers=conditions,
        )
        return self.diagnostics

    @torch.no_grad()
    def fit_from_accumulator(
        self, accumulator: EquivariantRidgeAccumulator
    ) -> RidgeFitDiagnostics:
        if (
            accumulator.feature_irreps != self.feature_irreps
            or accumulator.target_irreps != self.target_irreps
        ):
            raise ValueError("Accumulator irreps do not match this regressor")
        if accumulator.sample_count == 0:
            raise ValueError("Cannot fit empty sufficient statistics")
        conditions: dict[str, float] = {}
        equations = 0
        for irrep, target_slices in self.target_groups.items():
            if not self.feature_groups[irrep]:
                conditions[str(irrep)] = float("inf")
                continue
            count = accumulator.equation_count[irrep]
            scale = torch.sqrt(accumulator.sum_square[irrep] / count).clamp_min(1e-12)
            inverse_scale = scale.reciprocal()
            gram = (
                accumulator.xtx[irrep] * inverse_scale[:, None] * inverse_scale[None, :]
            )
            rhs = accumulator.xty[irrep] * inverse_scale[:, None]
            regularized = gram + self.ridge * torch.eye(
                gram.shape[0], dtype=gram.dtype, device=gram.device
            )
            solution = torch.linalg.solve(regularized, rhs)
            getattr(self, f"scale_{_key(irrep)}").copy_(
                scale.to(getattr(self, f"scale_{_key(irrep)}"))
            )
            getattr(self, f"weight_{_key(irrep)}").copy_(
                solution.T.to(getattr(self, f"weight_{_key(irrep)}"))
            )
            conditions[str(irrep)] = float(torch.linalg.cond(regularized).item())
            equations += count
        self.diagnostics = RidgeFitDiagnostics(
            sample_count=accumulator.sample_count,
            scalar_equation_count=equations,
            feature_dimension=self.feature_irreps.dim,
            target_dimension=self.target_irreps.dim,
            ridge=self.ridge,
            condition_numbers=conditions,
        )
        return self.diagnostics

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.shape[-1] != self.feature_irreps.dim:
            raise ValueError("features do not match feature_irreps")
        result = features.new_zeros(*features.shape[:-1], self.target_irreps.dim)
        for irrep, target_slices in self.target_groups.items():
            feature_slices = self.feature_groups[irrep]
            if not feature_slices:
                continue
            x = self._gather(features, feature_slices)
            scale = getattr(self, f"scale_{_key(irrep)}").to(features)
            weight = getattr(self, f"weight_{_key(irrep)}").to(features)
            values = torch.einsum("...fm,tf->...tm", x / scale[..., :, None], weight)
            for target_index, target_slice in enumerate(target_slices):
                result[..., target_slice] = values[..., target_index, :]
        return result
