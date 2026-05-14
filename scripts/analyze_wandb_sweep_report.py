from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from analysis.wandb_sweep_core import (  # noqa: E402
    analyze_sweep,
    build_rank_run_styles,
    build_summary_payload,
    default_output_dir,
    json_default,
    print_analysis_summary,
)
from analysis.wandb_sweep_plotly import (  # noqa: E402
    build_histogram_figure,
    build_html_report,
    build_run_detail_page,
    build_scatter_figure,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze a Weights & Biases sweep and build an interactive Plotly HTML report."
    )
    parser.add_argument("sweep_url", type=str)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--rank-metric", type=str, default=None)
    parser.add_argument(
        "--rank-goal",
        type=str,
        default="auto",
        choices=["auto", "minimize", "maximize"],
    )
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--top-k-ci", type=int, default=None)
    parser.add_argument("--k-fold", type=int, default=5)
    parser.add_argument("--bins", type=int, default=18)
    parser.add_argument("--max-categorical-label-chars", type=int, default=28)
    parser.add_argument(
        "--evaluation-cache-root",
        type=Path,
        default=Path("eval_cache"),
        help="Root directory scanned for precomputed local evaluation manifests.",
    )
    parser.add_argument(
        "--include-plotlyjs",
        type=str,
        default="cdn",
        choices=["cdn", "inline"],
        help="'inline' makes a self-contained HTML file; 'cdn' keeps it smaller.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = analyze_sweep(
        args.sweep_url,
        rank_metric=args.rank_metric,
        rank_goal=args.rank_goal,
        top_k=args.top_k,
        top_k_ci=args.top_k_ci,
    )
    output_dir = (
        Path(args.output_dir)
        if args.output_dir is not None
        else default_output_dir(result.sweep_path)
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    run_pages_dir = output_dir / "runs"
    run_pages_dir.mkdir(parents=True, exist_ok=True)

    print_analysis_summary(result)
    run_styles = build_rank_run_styles(result.ranked_records, len(result.top_runs))
    histogram_fig = build_histogram_figure(
        result,
        run_styles=run_styles,
        bins=args.bins,
        max_categorical_label_chars=args.max_categorical_label_chars,
    )
    scatter_fig = build_scatter_figure(
        result,
        run_styles=run_styles,
        k_fold=args.k_fold,
        max_categorical_label_chars=args.max_categorical_label_chars,
    )
    run_page_index = _prepare_run_pages(
        output_dir,
        result.ranked_records,
        args.evaluation_cache_root,
    )
    html_report = build_html_report(
        result,
        histogram_fig=histogram_fig,
        scatter_fig=scatter_fig,
        include_plotlyjs=True if args.include_plotlyjs == "inline" else "cdn",
        run_page_index=run_page_index,
    )
    html_path = output_dir / "sweep_report.html"
    html_path.write_text(html_report, encoding="utf-8")
    print(f"Saved HTML report: {html_path}")

    for record in result.ranked_records:
        run_html = build_run_detail_page(
            result,
            record,
            include_plotlyjs=True if args.include_plotlyjs == "inline" else "cdn",
            run_page_index=run_page_index,
        )
        run_path = run_pages_dir / f"{record.run_id}.html"
        run_path.write_text(run_html, encoding="utf-8")
    print(f"Saved run pages: {run_pages_dir}")

    summary_path = output_dir / "sweep_summary.json"
    summary_path.write_text(
        json.dumps(build_summary_payload(result), indent=2, default=json_default)
    )
    print(f"Saved summary: {summary_path}")


def _prepare_run_pages(
    output_dir: Path,
    ranked_records: list[Any],
    evaluation_cache_root: Path,
) -> dict[str, dict[str, Any]]:
    manifests = _discover_evaluation_manifests(evaluation_cache_root)
    run_page_index: dict[str, dict[str, Any]] = {}
    assets_root = output_dir / "run_assets"
    assets_root.mkdir(parents=True, exist_ok=True)
    for record in ranked_records:
        page_info: dict[str, Any] = {"href": f"runs/{record.run_id}.html"}
        manifest = manifests.get(record.run_id)
        if manifest is not None:
            evaluation = _copy_bundle_assets(
                manifest,
                assets_root / record.run_id,
                record.run_id,
            )
            page_info["evaluation"] = evaluation
        run_page_index[record.run_id] = page_info
    return run_page_index


def _discover_evaluation_manifests(
    evaluation_cache_root: Path,
) -> dict[str, dict[str, Any]]:
    manifests: dict[str, dict[str, Any]] = {}
    if not evaluation_cache_root.exists():
        return manifests
    for manifest_path in evaluation_cache_root.rglob("evaluation_manifest.json"):
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        source = payload.get("source", {})
        run_id = source.get("run_id")
        if not run_id:
            continue
        manifests[str(run_id)] = {
            "manifest_path": manifest_path,
            "bundle_dir": manifest_path.parent,
            "payload": payload,
        }
    return manifests


def _copy_bundle_assets(
    manifest: dict[str, Any],
    destination_dir: Path,
    run_id: str,
) -> dict[str, Any]:
    source_dir = Path(manifest["bundle_dir"])
    destination_dir.mkdir(parents=True, exist_ok=True)
    preferred_images = [
        "band_structure_comparison.png",
        "dos_comparison.png",
        "dos_error.png",
        "hamiltonian_first_atoms_comparison.png",
        "hamiltonian_correlation.png",
        "density_first_atoms_comparison.png",
        "density_correlation.png",
        "overlap_correlation.png",
    ]
    image_assets = []
    file_assets = []
    for filename in preferred_images:
        source_path = source_dir / filename
        if not source_path.exists():
            continue
        target_path = destination_dir / filename
        shutil.copy2(source_path, target_path)
        image_assets.append(
            {
                "label": _label_for_asset(filename),
                "href": f"../run_assets/{run_id}/{filename}",
            }
        )
    extra_files = [
        "evaluation_manifest.json",
        "band_structure_gt_gt_overlap.pt",
        "band_structure_pred_gt_overlap.pt",
        "band_structure_gt.pt",
        "band_structure_pred.pt",
    ]
    for filename in extra_files:
        source_path = source_dir / filename
        if not source_path.exists():
            continue
        target_path = destination_dir / filename
        shutil.copy2(source_path, target_path)
        file_assets.append(
            {
                "label": filename,
                "href": f"../run_assets/{run_id}/{filename}",
            }
        )
    return {
        "source_dir": str(source_dir),
        "image_assets": image_assets,
        "file_assets": file_assets,
    }


def _label_for_asset(filename: str) -> str:
    mapping = {
        "band_structure_comparison.png": "Band structure comparison",
        "dos_comparison.png": "DOS comparison",
        "dos_error.png": "DOS error",
        "hamiltonian_first_atoms_comparison.png": "Hamiltonian heatmap",
        "hamiltonian_correlation.png": "Hamiltonian correlation",
        "density_first_atoms_comparison.png": "Density heatmap",
        "density_correlation.png": "Density correlation",
        "overlap_correlation.png": "Overlap correlation",
    }
    return mapping.get(filename, filename)


if __name__ == "__main__":
    main()
