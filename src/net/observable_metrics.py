from __future__ import annotations

from typing import Any

import torch

from core.sparse_math import trace_matmul_sparse_block_matrix_aligned
from data.block_matrix import BlockMatrix
from utils.units import HARTREE_TO_EV


def _normalize_scalar_target(
    target: torch.Tensor | None,
    *,
    name: str,
) -> torch.Tensor | None:
    if target is None:
        return None
    if not torch.is_tensor(target):
        return torch.as_tensor(target)
    if target.ndim == 0:
        return target
    if target.numel() == 1:
        return target.reshape(())
    raise ValueError(
        f"{name} target must be scalar or length-1 tensor, got shape={tuple(target.shape)}"
    )


def validate_observable_config(cfg) -> None:
    targets = set(cfg.matrix_targets)

    if cfg.train_on_energy:
        if cfg.train_observables_on_gt:
            if not ({"hamiltonian", "density"} & targets):
                raise ValueError(
                    "train_on_energy=True with train_observables_on_gt=True requires "
                    "predicting 'hamiltonian' and/or 'density'."
                )
        elif not {"hamiltonian", "density"}.issubset(targets):
            raise ValueError(
                "train_on_energy=True requires predicting both 'hamiltonian' and "
                "'density' unless train_observables_on_gt=True."
            )

    if cfg.train_on_num_electrons:
        if cfg.train_observables_on_gt:
            if not ({"overlap", "density"} & targets):
                raise ValueError(
                    "train_on_num_electrons=True with train_observables_on_gt=True "
                    "requires predicting 'overlap' and/or 'density'."
                )
        elif not {"overlap", "density"}.issubset(targets):
            raise ValueError(
                "train_on_num_electrons=True requires predicting both 'overlap' and "
                "'density' unless train_observables_on_gt=True."
            )


def build_observable_predictions(
    preds_matrix: dict[str, BlockMatrix],
    *,
    pred_trace_alignment: dict[str, Any],
    H_true: BlockMatrix | None,
    D_true: BlockMatrix | None,
    S_true: BlockMatrix | None,
) -> dict[str, torch.Tensor]:
    values: dict[str, torch.Tensor] = {}

    if {"hamiltonian", "density"}.issubset(preds_matrix):
        values["energy"] = trace_matmul_sparse_block_matrix_aligned(
            preds_matrix["hamiltonian"],
            preds_matrix["density"],
            pred_trace_alignment,
        )
    if D_true is not None and "hamiltonian" in preds_matrix:
        values["energy_gt_density"] = trace_matmul_sparse_block_matrix_aligned(
            preds_matrix["hamiltonian"],
            D_true,
            pred_trace_alignment,
        )
    if H_true is not None and "density" in preds_matrix:
        values["energy_gt_hamiltonian"] = trace_matmul_sparse_block_matrix_aligned(
            H_true,
            preds_matrix["density"],
            pred_trace_alignment,
        )

    if {"density", "overlap"}.issubset(preds_matrix):
        values["num_electrons"] = trace_matmul_sparse_block_matrix_aligned(
            preds_matrix["density"],
            preds_matrix["overlap"],
            pred_trace_alignment,
        )
    if S_true is not None and "density" in preds_matrix:
        values["num_electrons_gt_overlap"] = trace_matmul_sparse_block_matrix_aligned(
            preds_matrix["density"],
            S_true,
            pred_trace_alignment,
        )
    if D_true is not None and "overlap" in preds_matrix:
        values["num_electrons_gt_density"] = trace_matmul_sparse_block_matrix_aligned(
            D_true,
            preds_matrix["overlap"],
            pred_trace_alignment,
        )

    return values


def add_observable_metrics(
    metrics: dict[str, torch.Tensor],
    *,
    stage: str,
    cfg,
    observable_values: dict[str, torch.Tensor],
    energy_target: torch.Tensor | None,
    num_electrons_target: torch.Tensor | None,
) -> None:
    energy_target = _normalize_scalar_target(energy_target, name="energy")
    num_electrons_target = _normalize_scalar_target(
        num_electrons_target, name="num_electrons"
    )

    if cfg.enable_energy and energy_target is not None:
        if "energy" in observable_values:
            metrics[f"{stage}/energy_mae"] = (
                torch.mean(torch.abs(observable_values["energy"] - energy_target))
                * HARTREE_TO_EV
            )
        if cfg.log_partial_gt_observables or cfg.train_observables_on_gt:
            if "energy_gt_hamiltonian" in observable_values:
                metrics[f"{stage}/energy_mae_gt_hamiltonian"] = (
                    torch.mean(
                        torch.abs(
                            observable_values["energy_gt_hamiltonian"] - energy_target
                        )
                    )
                    * HARTREE_TO_EV
                )
            if "energy_gt_density" in observable_values:
                metrics[f"{stage}/energy_mae_gt_density"] = (
                    torch.mean(
                        torch.abs(
                            observable_values["energy_gt_density"] - energy_target
                        )
                    )
                    * HARTREE_TO_EV
                )

    if cfg.enable_num_electrons and num_electrons_target is not None:
        if "num_electrons" in observable_values:
            metrics[f"{stage}/num_electrons_mae"] = torch.mean(
                torch.abs(observable_values["num_electrons"] - num_electrons_target)
            )
        if cfg.log_partial_gt_observables or cfg.train_observables_on_gt:
            if "num_electrons_gt_overlap" in observable_values:
                metrics[f"{stage}/num_electrons_mae_gt_overlap"] = torch.mean(
                    torch.abs(
                        observable_values["num_electrons_gt_overlap"]
                        - num_electrons_target
                    )
                )
            if "num_electrons_gt_density" in observable_values:
                metrics[f"{stage}/num_electrons_mae_gt_density"] = torch.mean(
                    torch.abs(
                        observable_values["num_electrons_gt_density"]
                        - num_electrons_target
                    )
                )


def observable_loss(
    *,
    cfg,
    observable_values: dict[str, torch.Tensor],
    energy_target: torch.Tensor | None,
    num_electrons_target: torch.Tensor | None,
    mse_fn,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    loss_E_weighted = torch.tensor(0.0, device=device)
    loss_N_weighted = torch.tensor(0.0, device=device)
    energy_target = _normalize_scalar_target(energy_target, name="energy")
    num_electrons_target = _normalize_scalar_target(
        num_electrons_target, name="num_electrons"
    )

    if cfg.train_on_energy and energy_target is not None:
        if cfg.train_observables_on_gt:
            losses = []
            if "energy_gt_hamiltonian" in observable_values:
                losses.append(
                    mse_fn(observable_values["energy_gt_hamiltonian"], energy_target)
                )
            if "energy_gt_density" in observable_values:
                losses.append(
                    mse_fn(observable_values["energy_gt_density"], energy_target)
                )
            if losses:
                loss_E_weighted = cfg.loss_coef_observables * sum(losses) / len(losses)
        elif "energy" in observable_values:
            loss_E_weighted = cfg.loss_coef_observables * mse_fn(
                observable_values["energy"], energy_target
            )

    if cfg.train_on_num_electrons and num_electrons_target is not None:
        if cfg.train_observables_on_gt:
            losses = []
            if "num_electrons_gt_overlap" in observable_values:
                losses.append(
                    mse_fn(
                        observable_values["num_electrons_gt_overlap"],
                        num_electrons_target,
                    )
                )
            if "num_electrons_gt_density" in observable_values:
                losses.append(
                    mse_fn(
                        observable_values["num_electrons_gt_density"],
                        num_electrons_target,
                    )
                )
            if losses:
                loss_N_weighted = cfg.loss_coef_observables * sum(losses) / len(losses)
        elif "num_electrons" in observable_values:
            loss_N_weighted = cfg.loss_coef_observables * mse_fn(
                observable_values["num_electrons"], num_electrons_target
            )

    return loss_E_weighted, loss_N_weighted
