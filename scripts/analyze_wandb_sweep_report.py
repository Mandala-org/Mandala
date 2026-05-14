from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

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
    html_report = build_html_report(
        result,
        histogram_fig=histogram_fig,
        scatter_fig=scatter_fig,
        include_plotlyjs=True if args.include_plotlyjs == "inline" else "cdn",
    )
    html_path = output_dir / "sweep_report.html"
    html_path.write_text(html_report, encoding="utf-8")
    print(f"Saved HTML report: {html_path}")

    summary_path = output_dir / "sweep_summary.json"
    summary_path.write_text(
        json.dumps(build_summary_payload(result), indent=2, default=json_default)
    )
    print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    main()
