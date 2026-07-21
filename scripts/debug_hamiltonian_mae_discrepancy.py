#!/usr/bin/env python

"""Compare every Hamiltonian-MAE path used by training and evaluation.

This is intentionally a read-only diagnostic.  It runs one checkpoint on one
snapshot, then reports MAEs for raw, envelope-physicalized, symmetrized,
prefix-paired, and exact-edge-aligned block matrices.  The output makes it
possible to determine whether a discrepancy comes from units, physicalization,
symmetrization, edge ordering/support, or aggregation.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from analysis import evaluation as analysis_eval  # noqa: E402
from data.factory import DatasetFactory  # noqa: E402
from net.e3gnn import E3GNN  # noqa: E402
from scripts import evaluate_checkpoint_materials as eval_ckpt  # noqa: E402
from utils.units import HARTREE_TO_EV  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--snapshot-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--convention", default="e3nn")
    return parser.parse_args()


def _prefix_pair(
    pred: Any, target: Any
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    preds: dict[str, torch.Tensor] = {}
    targets: dict[str, torch.Tensor] = {}
    for key, target_blocks in target.pair_blocks.items():
        target_n = int(target_blocks.shape[0])
        preds[key] = pred.pair_blocks[key][:target_n]
        targets[key] = target_blocks
    return preds, targets


def _exact_pair(
    pred: Any, target: Any
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], dict[str, Any]]:
    aligned, debug = analysis_eval.align_prediction_to_target(pred, target)
    preds = {key: aligned.pair_blocks[key] for key in target.pair_blocks}
    targets = {
        key: target.pair_blocks[key][: preds[key].shape[0]]
        for key in target.pair_blocks
    }
    return preds, targets, debug


def _stats(
    preds: dict[str, torch.Tensor], targets: dict[str, torch.Tensor]
) -> dict[str, Any]:
    abs_sums: dict[str, float] = {}
    counts: dict[str, int] = {}
    pair_mean_sum_ha = 0.0
    for key, target in targets.items():
        pred = preds[key]
        if pred.shape != target.shape:
            raise ValueError(
                f"Shape mismatch for {key}: {pred.shape} != {target.shape}"
            )
        error = torch.abs(pred.to(torch.float64) - target.to(torch.float64))
        abs_sums[key] = float(error.sum().item())
        counts[key] = int(error.numel())
        pair_mean_sum_ha += float(error.mean().item())
    total_abs = sum(abs_sums.values())
    total_count = sum(counts.values())
    global_mae_ha = total_abs / total_count
    snapshot_mean_of_pair_means_ha = pair_mean_sum_ha / max(len(abs_sums), 1)
    return {
        "global_mae_ha": global_mae_ha,
        "global_mae_ev": global_mae_ha * HARTREE_TO_EV,
        "sum_of_pair_maes_ha": pair_mean_sum_ha,
        "sum_of_pair_maes_ev": pair_mean_sum_ha * HARTREE_TO_EV,
        "mean_of_pair_maes_ha": snapshot_mean_of_pair_means_ha,
        "mean_of_pair_maes_ev": snapshot_mean_of_pair_means_ha * HARTREE_TO_EV,
        "total_elements": total_count,
        "per_pair_mae_ev": {
            key: (abs_sums[key] / counts[key]) * HARTREE_TO_EV
            for key in sorted(abs_sums)
        },
    }


def _edge_prefix_debug(pred: Any, target: Any) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, target_edges in target.pair_edges.items():
        target_n = int(target_edges.shape[1])
        pred_edges = pred.pair_edges[key]
        prefix = pred_edges[:, :target_n]
        match = prefix.shape == target_edges.shape and torch.equal(prefix, target_edges)
        mismatch_columns = None
        if prefix.shape == target_edges.shape:
            mismatch_columns = int(
                torch.any(prefix != target_edges, dim=0).sum().item()
            )
        result[key] = {
            "pred_edges": int(pred_edges.shape[1]),
            "target_edges": target_n,
            "extra_pred_edges": int(pred_edges.shape[1] - target_n),
            "prefix_exact": bool(match),
            "prefix_mismatch_columns": mismatch_columns,
        }
    return result


def main() -> None:
    args = parse_args()
    checkpoint = eval_ckpt._load_checkpoint(args.checkpoint)
    cfg = eval_ckpt._restore_config(checkpoint)
    cfg.dataset_device = None
    cfg.snapshot_cache_dir = None
    cfg.spectral_loss_enabled = False
    cfg.spectral_fermi_cache_path = None

    matrix_path, info_path = eval_ckpt._discover_snapshot_paths(
        args.snapshot_path, None, None
    )
    factory = DatasetFactory(cfg, convention=args.convention)
    factory.add_snapshot(matrix_path, info_path, purpose="train")
    dataset, _, mapper = factory.create()
    x, y = dataset[0]
    device = eval_ckpt._resolve_device(args.device)
    x = eval_ckpt._move_to_device(x, device)
    y = eval_ckpt._move_to_device(y, device)

    model = E3GNN(mapper=mapper, cfg=cfg).to(device).eval()
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    with torch.no_grad():
        pred_irreps = model(x)
        raw = pred_irreps["hamiltonian"].to_blocks(model.mapper)
        physical = model._physicalize_predicted_matrix(
            name="hamiltonian", pred_matrix=raw, x=x
        )
        raw_sym = analysis_eval.symmetrize_block_matrix(raw)
        physical_sym = analysis_eval.symmetrize_block_matrix(physical)

    target = y["hamiltonian"]
    report: dict[str, Any] = {
        "checkpoint": str(args.checkpoint.resolve()),
        "snapshot_path": str(args.snapshot_path),
        "config": {
            "hamiltonian_envelope_mode": cfg.hamiltonian_envelope_mode,
            "apply_cutoff_to_targets": cfg.apply_cutoff_to_targets,
            "cutoff_radius": cfg.cutoff_radius,
            "symmetrize_output": cfg.symmetrize_output,
            "require_exact_edge_match": cfg.require_exact_edge_match,
            "log_hamiltonian_irrep_contrib_metrics": cfg.log_hamiltonian_irrep_contrib_metrics,
            "log_hamiltonian_pair_contrib_metrics": cfg.log_hamiltonian_pair_contrib_metrics,
        },
        "edge_prefix": _edge_prefix_debug(physical_sym, target),
        "metrics": {},
    }

    for label, matrix in (
        ("raw_prefix", raw),
        ("raw_sym_prefix", raw_sym),
        ("physical_prefix", physical),
        ("physical_sym_prefix_training_validation", physical_sym),
    ):
        p, t = _prefix_pair(matrix, target)
        report["metrics"][label] = _stats(p, t)

    p, t, align_debug = _exact_pair(physical_sym, target)
    report["metrics"]["physical_sym_exact_edge_evaluation"] = _stats(p, t)
    report["alignment"] = align_debug

    rendered = json.dumps(report, indent=2, sort_keys=True)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
