from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-codex")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from analysis.wandb_sweep_core import (  # noqa: E402
    analyze_sweep,
    build_markdown_report,
    build_rank_run_styles,
    build_summary_payload,
    default_output_dir,
    json_default,
    print_analysis_summary,
)
from analysis.wandb_sweep_matplotlib import (  # noqa: E402
    plot_metric_scatterplots,
    plot_variable_histograms,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze a Weights & Biases sweep and plot swept config histograms."
    )
    parser.add_argument(
        "sweep_url",
        type=str,
        help=(
            "W&B sweep URL or path, e.g. "
            "'https://wandb.ai/<entity>/<project>/sweeps/<sweep_id>'"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory for the plots and summary files.",
    )
    parser.add_argument(
        "--rank-metric",
        type=str,
        default=None,
        help=(
            "Metric used to rank runs. Defaults to the sweep objective metric "
            "from the sweep metadata."
        ),
    )
    parser.add_argument(
        "--rank-goal",
        type=str,
        default="auto",
        choices=["auto", "minimize", "maximize"],
        help=(
            "Direction for ranking by --rank-metric. 'auto' uses the sweep "
            "objective goal from sweep metadata."
        ),
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="How many top runs to overlay on the histograms.",
    )
    parser.add_argument(
        "--top-k-ci",
        type=int,
        default=None,
        help="How many top runs to use for the scatterplot confidence band. Defaults to --top-k.",
    )
    parser.add_argument(
        "--k-fold",
        type=int,
        default=5,
        help="Number of folds used to build the scatterplot confidence band.",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=18,
        help="Base bin count for numeric histograms.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=180,
        help="PNG export DPI.",
    )
    parser.add_argument(
        "--max-figure-width",
        type=float,
        default=34.0,
        help="Maximum width of the combined figure in inches.",
    )
    parser.add_argument(
        "--max-figure-height",
        type=float,
        default=20.0,
        help="Maximum height of the combined figure in inches.",
    )
    parser.add_argument(
        "--max-categorical-label-chars",
        type=int,
        default=28,
        help="Truncate long categorical tick labels to this many characters.",
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

    if result.variable_specs:
        fig = plot_variable_histograms(
            result.variable_specs,
            result.top_runs,
            run_styles,
            args.bins,
            result.sweep_path,
            result.rank_metric,
            args.max_figure_width,
            args.max_figure_height,
            args.max_categorical_label_chars,
        )
        figure_path = output_dir / "sweep_config_histograms.png"
        fig.savefig(figure_path, bbox_inches="tight", dpi=args.dpi)
        plt.close(fig)
        print(f"\nSaved figure: {figure_path}")

        scatter_fig = plot_metric_scatterplots(
            result.variable_specs,
            result.records,
            result.top_runs,
            run_styles,
            result.ci_runs,
            args.k_fold,
            result.sweep_path,
            result.rank_metric,
            args.max_figure_width,
            args.max_figure_height,
            args.max_categorical_label_chars,
        )
        scatter_path = output_dir / "sweep_metric_scatterplots.png"
        scatter_fig.savefig(scatter_path, bbox_inches="tight", dpi=args.dpi)
        plt.close(scatter_fig)
        print(f"Saved figure: {scatter_path}")
    else:
        print(
            "\nNo non-constant swept variables were found, so no histograms were generated."
        )

    summary_path = output_dir / "sweep_summary.json"
    summary_path.write_text(
        json.dumps(build_summary_payload(result), indent=2, default=json_default)
    )
    print(f"Saved summary: {summary_path}")

    report_path = output_dir / "sweep_summary.md"
    report_path.write_text(build_markdown_report(result))
    print(f"Saved report: {report_path}")


if __name__ == "__main__":
    main()
