from __future__ import annotations


import torch

from core.sparse_math import (
    trace_matmul_sparse_block_matrix_aligned,
)
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


def truncate_pred_block_matrix_to_target_prefix(
    pred_matrix: BlockMatrix,
    target_matrix: BlockMatrix,
) -> BlockMatrix:
    pair_blocks: dict[str, torch.Tensor] = {}
    pair_edges: dict[str, torch.Tensor] = {}
    lookup: dict[tuple[int, int, int, int, int], tuple[str, int]] = {}

    for key, target_edges in target_matrix.pair_edges.items():
        if key not in pred_matrix.pair_blocks:
            raise ValueError(f"Prediction is missing key '{key}' required by target")
        pred_blocks = pred_matrix.pair_blocks[key]
        target_n = target_edges.shape[1]
        if target_n <= 0:
            raise ValueError(f"Target edge prefix for key '{key}' is empty.")
        if pred_blocks.shape[0] < target_n:
            raise ValueError(
                f"Prediction is missing the target prefix for key '{key}': "
                f"pred_len={pred_blocks.shape[0]} target_len={target_n}"
            )
        pair_blocks[key] = pred_blocks[:target_n]
        pair_edges[key] = target_edges
        for idx, (sx, sy, sz, i, j) in enumerate(target_edges.t().tolist()):
            lookup[(sx, sy, sz, i, j)] = (key, idx)

    return BlockMatrix(
        atoms=pred_matrix.atoms,
        atom_counts=pred_matrix.atom_counts,
        pair_blocks=pair_blocks,
        pair_edges=pair_edges,
        lookup=lookup,
        orbital_cfg=pred_matrix.orbital_cfg,
        basis=pred_matrix.basis,
    )


def build_observable_predictions(
    preds_matrix: dict[str, BlockMatrix],
    *,
    trace_alignment,
    H_true: BlockMatrix | None,
    D_true: BlockMatrix | None,
    S_true: BlockMatrix | None,
) -> dict[str, torch.Tensor]:
    values: dict[str, torch.Tensor] = {}

    if {"hamiltonian", "density"}.issubset(preds_matrix):
        values["energy"] = trace_matmul_sparse_block_matrix_aligned(
            preds_matrix["hamiltonian"],
            preds_matrix["density"],
            trace_alignment,
        )
    if D_true is not None and "hamiltonian" in preds_matrix:
        values["energy_gt_density"] = trace_matmul_sparse_block_matrix_aligned(
            preds_matrix["hamiltonian"],
            D_true,
            trace_alignment,
        )
    if H_true is not None and "density" in preds_matrix:
        values["energy_gt_hamiltonian"] = trace_matmul_sparse_block_matrix_aligned(
            H_true,
            preds_matrix["density"],
            trace_alignment,
        )

    if {"density", "overlap"}.issubset(preds_matrix):
        values["num_electrons"] = trace_matmul_sparse_block_matrix_aligned(
            preds_matrix["density"],
            preds_matrix["overlap"],
            trace_alignment,
        )
    if S_true is not None and "density" in preds_matrix:
        values["num_electrons_gt_overlap"] = trace_matmul_sparse_block_matrix_aligned(
            preds_matrix["density"],
            S_true,
            trace_alignment,
        )
    if D_true is not None and "overlap" in preds_matrix:
        values["num_electrons_gt_density"] = trace_matmul_sparse_block_matrix_aligned(
            D_true,
            preds_matrix["overlap"],
            trace_alignment,
        )

    return values


def rescale_density_prediction_to_num_electrons(
    preds_matrix: dict[str, BlockMatrix],
    *,
    trace_alignment,
    num_electrons_target: torch.Tensor | None,
    overlap_true: BlockMatrix | None,
    eps: float = 1e-12,
) -> tuple[dict[str, BlockMatrix], torch.Tensor | None]:
    """Rescale predicted density to satisfy the electron-count trace exactly.

    Predicted overlap is used when available, matching fully predicted H/D/S
    inference. Ground-truth overlap is the fallback for density-only models.
    This helper is for reported metrics only. The scale is deliberately detached
    so density normalization cannot alter observable-guided training gradients.
    """
    target = _normalize_scalar_target(num_electrons_target, name="num_electrons")
    density = preds_matrix.get("density")
    overlap = preds_matrix.get("overlap", overlap_true)
    if density is None or overlap is None or target is None:
        return preds_matrix, None

    predicted_count = trace_matmul_sparse_block_matrix_aligned(
        density,
        overlap,
        trace_alignment,
    )
    eps_tensor = torch.full_like(predicted_count, float(eps))
    safe_count = torch.where(
        torch.abs(predicted_count) > float(eps), predicted_count, eps_tensor
    )
    density_scale = float((target / safe_count).detach().cpu().item())
    scaled = dict(preds_matrix)
    scaled["density"] = density * density_scale
    return scaled, predicted_count


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
    loss_kind = str(getattr(cfg, "observable_loss_kind", "mse")).lower()
    if loss_kind not in {"mse", "mae"}:
        raise ValueError("observable_loss_kind must be one of 'mse' or 'mae'.")

    def scalar_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if loss_kind == "mae":
            return torch.mean(torch.abs(prediction - target))
        return mse_fn(prediction, target)

    if cfg.train_on_energy and energy_target is not None:
        if cfg.train_observables_on_gt:
            losses = []
            if "energy_gt_hamiltonian" in observable_values:
                losses.append(
                    scalar_loss(
                        observable_values["energy_gt_hamiltonian"], energy_target
                    )
                )
            if "energy_gt_density" in observable_values:
                losses.append(
                    scalar_loss(observable_values["energy_gt_density"], energy_target)
                )
            if losses:
                loss_E_weighted = cfg.loss_coef_observables * sum(losses) / len(losses)
        elif "energy" in observable_values:
            loss_E_weighted = cfg.loss_coef_observables * scalar_loss(
                observable_values["energy"], energy_target
            )

    if cfg.train_on_num_electrons and num_electrons_target is not None:
        if cfg.train_observables_on_gt:
            losses = []
            if "num_electrons_gt_overlap" in observable_values:
                losses.append(
                    scalar_loss(
                        observable_values["num_electrons_gt_overlap"],
                        num_electrons_target,
                    )
                )
            if "num_electrons_gt_density" in observable_values:
                losses.append(
                    scalar_loss(
                        observable_values["num_electrons_gt_density"],
                        num_electrons_target,
                    )
                )
            if losses:
                loss_N_weighted = cfg.loss_coef_observables * sum(losses) / len(losses)
        elif "num_electrons" in observable_values:
            loss_N_weighted = cfg.loss_coef_observables * scalar_loss(
                observable_values["num_electrons"], num_electrons_target
            )

    return loss_E_weighted, loss_N_weighted
