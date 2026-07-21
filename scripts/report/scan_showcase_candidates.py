"""Scan precomputed evaluation artifacts for preliminary showcase candidates.

This deliberately performs no model inference. It only reads saved tensors and
plots' numerical sidecars, so the output is suitable for triage rather than
publication-quality benchmarking.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from data.block_matrix import BlockMatrix  # noqa: E402
from utils.units import HARTREE_TO_EV  # noqa: E402

EVAL_ROOT = ROOT / "eval_outputs"
OUT_ROOT = ROOT / "analysis_outputs"


def _load(path: Path) -> Any:
    return torch.load(path, map_location="cpu", weights_only=False)


def _scalar(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            return None
        return float(value.item())
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _dataset(path: Path) -> str:
    text = str(path).lower()
    if "siox" in text:
        return "SiOx"
    if "zncusn" in text or "zncuse" in text:
        return "ZnCuSnSeS"
    if "si_perturbed" in text:
        return "Silicon-perturbed"
    if "silicon" in text or "small_si" in text or "/si_" in text or "/si/" in text:
        return "Silicon"
    return "Other"


def _model_label(path: Path) -> str:
    parts = path.relative_to(EVAL_ROOT).parts
    return "/".join(parts)


def _metric_row(path: Path) -> dict[str, Any]:
    row: dict[str, Any] = {
        "evaluation_dir": str(path.relative_to(ROOT)),
        "dataset": _dataset(path),
        "model_or_bundle": _model_label(path),
        "evidence": "artifact-only",
        "hamiltonian_mae": None,
        "hamiltonian_mae_unit": "eV",
        "hamiltonian_mse": None,
        "hamiltonian_mse_unit": "eV^2",
        "hamiltonian_rmse": None,
        "hamiltonian_rmse_unit": "eV",
        "hamiltonian_corr": None,
        "hamiltonian_r2": None,
        "block_abs_mae_mean": None,
        "block_rel_mae_mean": None,
        "density_mae": None,
        "density_mae_unit": "native",
        "density_mse": None,
        "density_mse_unit": "native^2",
        "density_corr": None,
        "overlap_mae": None,
        "overlap_mae_unit": "dimensionless",
        "overlap_mse": None,
        "overlap_mse_unit": "dimensionless^2",
        "dos_l1": None,
        "dos_fermi_abs_error_ev": None,
        "dos_electron_abs_error": None,
        "band_mae_ev": None,
        "band_near_fermi_mae_ev": None,
        "band_fermi_available": False,
        "has_hamiltonian_plot": (
            path / "hamiltonian_first_atoms_comparison.png"
        ).exists(),
        "has_dos_plot": (path / "dos_comparison.png").exists(),
        "has_band_plot": (path / "band_structure_comparison.png").exists(),
        "has_density_plot": (path / "density_first_atoms_comparison.png").exists(),
        "notes": [],
    }

    metric_path = path / "evaluation_matrix_metrics.json"
    if metric_path.exists():
        try:
            metric_payload = json.loads(metric_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            metric_payload = {}
            row["notes"].append("invalid matrix metric sidecar")
        if not isinstance(metric_payload, dict):
            metric_payload = {}
            row["notes"].append("invalid matrix metric sidecar")
        metrics = metric_payload.get("metrics", {})
        if not isinstance(metrics, dict):
            metrics = {}
            row["notes"].append("invalid matrix metric mapping")
        row["hamiltonian_mae"] = _scalar(metrics.get("val/hamiltonian_mae"))
        row["hamiltonian_mse"] = _scalar(metrics.get("val/hamiltonian_mse"))
        row["hamiltonian_rmse"] = (
            None if row["hamiltonian_mse"] is None else row["hamiltonian_mse"] ** 0.5
        )
        row["density_mae"] = _scalar(metrics.get("val/density_mae"))
        row["density_mse"] = _scalar(metrics.get("val/density_mse"))
        row["overlap_mae"] = _scalar(metrics.get("val/overlap_mae"))
        row["overlap_mse"] = _scalar(metrics.get("val/overlap_mse"))

    corr_path = path / "hamiltonian_correlation.pt"
    if corr_path.exists():
        data = _load(corr_path)
        row["hamiltonian_corr"] = _scalar(data.get("corr"))
        row["hamiltonian_r2"] = _scalar(data.get("r2"))

    block_path = path / "hamiltonian_block_error_metrics.pt"
    if block_path.exists():
        data = _load(block_path)
        for key, out in (
            ("abs_mae", "block_abs_mae_mean"),
            ("rel_mae", "block_rel_mae_mean"),
        ):
            values = data.get(key)
            if isinstance(values, list) and values:
                row[out] = float(torch.as_tensor(values, dtype=torch.float64).mean())
        # Backfill old bundles without the exact metric artifact.  Weight each
        # per-edge block MAE by its number of orbital matrix elements; never use
        # the dense correlation payload, which folds periodic images together.
        if row["hamiltonian_mae"] is None:
            pred_path = path / "pred_hamiltonian.pt"
            pair_keys = data.get("pair_key", [])
            edge_maes = data.get("abs_mae", [])
            if pred_path.exists() and len(pair_keys) == len(edge_maes):
                pred_matrix = BlockMatrix.load(pred_path)
                absolute_sum = 0.0
                scalar_count = 0
                for pair_key, edge_mae in zip(pair_keys, edge_maes):
                    rows, cols = pred_matrix.orbital_cfg.block_dims(str(pair_key))
                    count = int(rows * cols)
                    absolute_sum += float(edge_mae) * count
                    scalar_count += count
                if scalar_count:
                    row["hamiltonian_mae"] = (
                        absolute_sum / scalar_count
                    ) * HARTREE_TO_EV
                    row["notes"].append(
                        "Hamiltonian MAE reconstructed from block sidecar; rerun for exact MSE"
                    )

    density_corr_path = path / "density_correlation.pt"
    if density_corr_path.exists():
        data = _load(density_corr_path)
        row["density_corr"] = _scalar(data.get("corr"))

    dos_path = path / "dos_comparison.pt"
    if dos_path.exists():
        data = _load(dos_path)
        grid = data.get("grid_true")
        true, pred = data.get("dos_true"), data.get("dos_pred")
        if all(isinstance(x, torch.Tensor) for x in (grid, true, pred)):
            grid = grid.to(torch.float64)
            true = true.to(torch.float64)
            pred = pred.to(torch.float64)
            row["dos_l1"] = float(torch.trapezoid((true - pred).abs(), grid))
        fermi_true = _scalar(data.get("fermi_true_ev"))
        fermi_pred = _scalar(data.get("fermi_pred_ev"))
        if fermi_true is not None and fermi_pred is not None:
            row["dos_fermi_abs_error_ev"] = abs(fermi_true - fermi_pred)
        n_true = _scalar(data.get("num_electrons_true"))
        n_pred = _scalar(data.get("num_electrons_pred"))
        if n_true is not None and n_pred is not None:
            row["dos_electron_abs_error"] = abs(n_true - n_pred)
        if (
            _scalar(data.get("energy_min_ev")) == -10.0
            and _scalar(data.get("energy_max_ev")) == 15.0
        ):
            row["notes"].append("legacy DOS window; inspect Fermi level")

    pred_band = path / "band_structure_pred_gt_overlap.pt"
    gt_band = path / "band_structure_gt_gt_overlap.pt"
    if not pred_band.exists():
        pred_band = path / "band_structure_pred.pt"
    if pred_band.exists() and gt_band.exists():
        pred_data, gt_data = _load(pred_band), _load(gt_band)
        pred, gt = pred_data.get("eigenvalues"), gt_data.get("eigenvalues")
        if (
            isinstance(pred, torch.Tensor)
            and isinstance(gt, torch.Tensor)
            and pred.shape == gt.shape
        ):
            diff = (pred.to(torch.float64) - gt.to(torch.float64)).abs()
            row["band_mae_ev"] = float(diff.mean())
            fermi = _scalar(gt_data.get("fermi_level"))
            if fermi is None:
                fermi = _scalar(pred_data.get("fermi_level"))
            if fermi is not None:
                row["band_fermi_available"] = True
                mask = (gt.to(torch.float64) - fermi).abs() <= 2.0
                if bool(mask.any()):
                    row["band_near_fermi_mae_ev"] = float(diff[mask].mean())
        else:
            row["notes"].append("band tensors missing or shape mismatch")
    elif (path / "band_structure_pred.pt").exists():
        row["notes"].append("prediction-only band artifact; no GT comparison")

    if not corr_path.exists():
        row["notes"].append("no Hamiltonian correlation sidecar")
    if not metric_path.exists():
        row["notes"].append("no exact matrix metric sidecar; rerun evaluation")
    row["notes"] = "; ".join(row["notes"])
    return row


def main() -> None:
    candidates = []
    for path in sorted(EVAL_ROOT.glob("**/*")):
        if not path.is_dir():
            continue
        if any(
            (path / name).exists()
            for name in (
                "hamiltonian_correlation.pt",
                "dos_comparison.pt",
                "band_structure_pred.pt",
                "band_structure_pred_gt_overlap.pt",
            )
        ):
            candidates.append(_metric_row(path))

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    json_path = OUT_ROOT / "showcase_model_scan.json"
    csv_path = OUT_ROOT / "showcase_model_scan.csv"
    json_path.write_text(json.dumps(candidates, indent=2), encoding="utf-8")
    fields = list(candidates[0]) if candidates else []
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(candidates)
    print(f"Scanned {len(candidates)} evaluation bundles")
    print(json_path)
    print(csv_path)


if __name__ == "__main__":
    main()
