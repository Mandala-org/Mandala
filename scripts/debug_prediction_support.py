#!/usr/bin/env python

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from analysis import evaluation as analysis_eval  # noqa: E402
from data.factory import DatasetFactory  # noqa: E402
from data.openmx_info_parser import parse_info_out  # noqa: E402
from scripts import evaluate_checkpoint_materials as eval_ckpt  # noqa: E402
from net.e3gnn import E3GNN  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Debug raw-vs-aligned sparse support for a checkpoint evaluation snapshot."
        )
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--snapshot-path", type=Path, default=None)
    parser.add_argument("--matrix-path", type=Path, default=None)
    parser.add_argument("--info-path", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--matrix-name",
        type=str,
        default="hamiltonian",
        choices=["hamiltonian", "density", "overlap"],
    )
    parser.add_argument(
        "--analysis-cutoff-radius",
        type=float,
        default=None,
        help="Optional post-inference cutoff applied before support diagnostics.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
    )
    parser.add_argument("--convention", type=str, default="e3nn")
    parser.add_argument("--zero-tol", type=float, default=1.0e-12)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint = eval_ckpt._load_checkpoint(args.checkpoint)
    cfg = eval_ckpt._restore_config(checkpoint)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    matrix_path, info_path = eval_ckpt._discover_snapshot_paths(
        args.snapshot_path,
        args.matrix_path,
        args.info_path,
    )
    cfg.dataset_device = None
    cfg.snapshot_cache_dir = eval_ckpt._resolve_snapshot_cache_dir(cfg, output_dir)
    factory = DatasetFactory(cfg, convention=args.convention)
    factory.add_snapshot(matrix_path, info_path, purpose="train")
    dataset, _, mapper = factory.create()
    x, y = dataset[0]

    device = eval_ckpt._resolve_device(args.device)
    x = eval_ckpt._move_to_device(x, device)
    y = eval_ckpt._move_to_device(y, device)

    model = E3GNN(mapper=mapper, cfg=cfg)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.to(device)
    model.eval()

    with torch.no_grad():
        predictions_irreps = model(x)
        if "overlap" in predictions_irreps:
            predictions_irreps["overlap"] = (
                analysis_eval.clean_predicted_overlap_irreps(
                    predictions_irreps["overlap"],
                    model.mapper,
                )
            )
        pred_mats = {
            name: analysis_eval.symmetrize_block_matrix(pred.to_blocks(model.mapper))
            for name, pred in predictions_irreps.items()
        }

    info = parse_info_out(info_path)
    positions = x["positions"]
    box = x["box"]
    gt_snapshot = eval_ckpt._build_snapshot_from_matrices(
        {name: y[name] for name in ("hamiltonian", "density", "overlap")},
        positions=positions,
        box=box,
        info=info,
    )
    gt_snapshot = eval_ckpt._maybe_apply_analysis_cutoff(
        gt_snapshot,
        args.analysis_cutoff_radius,
        cfg,
    )

    pred_snapshot = eval_ckpt._build_snapshot_from_matrices(
        {
            "hamiltonian": pred_mats.get("hamiltonian", gt_snapshot["hamiltonian"]),
            "density": pred_mats.get("density", gt_snapshot["density"]),
            "overlap": pred_mats.get("overlap", gt_snapshot["overlap"]),
        },
        positions=positions,
        box=box,
        info=info,
    )
    pred_snapshot = eval_ckpt._maybe_apply_analysis_cutoff(
        pred_snapshot,
        args.analysis_cutoff_radius,
        cfg,
    )

    pred_mat = pred_snapshot[args.matrix_name]
    gt_mat = gt_snapshot[args.matrix_name]
    diagnostics = analysis_eval.save_prediction_support_debug_artifacts(
        pred_mat,
        gt_mat,
        positions=positions,
        box=box,
        output_dir=output_dir,
        prefix=args.matrix_name,
        title=f"{args.matrix_name.capitalize()} support debug",
        zero_tol=args.zero_tol,
    )

    print(
        f"[DEBUG] matrix={args.matrix_name} "
        f"target_edges={diagnostics['counts']['target_edges_total']} "
        f"pred_edges_raw={diagnostics['counts']['pred_edges_total_raw']} "
        f"pred_edges_aligned={diagnostics['counts']['pred_edges_total_aligned']} "
        f"pred_only={diagnostics['counts']['pred_only_edges_total']} "
        f"missing={diagnostics['counts']['missing_edges_total']}"
    )
    print(
        f"[DEBUG] dense_corr raw={diagnostics['correlation']['raw_dense']:.6f} "
        f"aligned={diagnostics['correlation']['aligned_dense']:.6f}"
    )
    print(
        "[DEBUG] gt~0 spike "
        f"raw={diagnostics['dense_zero_support_raw']['target_zero_pred_nonzero_entries']} "
        f"aligned={diagnostics['dense_zero_support_aligned']['target_zero_pred_nonzero_entries']}"
    )
    if diagnostics["alignment"]["prefix_mismatch_keys"]:
        print(
            "[DEBUG] prefix mismatch keys:",
            ", ".join(diagnostics["alignment"]["prefix_mismatch_keys"]),
        )
    if diagnostics["top_pred_only_edges"]:
        print("[DEBUG] top prediction-only edges:")
        for rec in diagnostics["top_pred_only_edges"][:10]:
            print(
                "  "
                f"{rec['pair_key']} shift={tuple(rec['shift'])} "
                f"{rec['src_atom']}->{rec['dst_atom']} "
                f"dist={rec['edge_length']:.3f} "
                f"mean_abs_pred={rec['mean_abs_pred']:.6e} "
                f"max_abs_pred={rec['max_abs_pred']:.6e}"
            )


if __name__ == "__main__":
    main()
