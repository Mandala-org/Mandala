from __future__ import annotations

import argparse
import json
import math
import os
import hashlib
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import matplotlib
import numpy as np
from matplotlib.lines import Line2D

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-codex")
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import wandb
except Exception as exc:  # pragma: no cover - informative failure
    raise SystemExit(
        "wandb is required for this script. Install the project dependencies first."
    ) from exc


MISSING = object()


@dataclass(frozen=True)
class RunRecord:
    run_id: str
    name: str
    state: str
    score: float | None
    config: dict[str, Any]


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
            "Direction for ranking by --rank-metric. 'auto' follows the sweep "
            "objective goal when possible and otherwise uses a simple metric-name heuristic."
        ),
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
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
    sweep = _load_sweep(args.sweep_url)
    sweep_path = _sweep_path_from_ref(args.sweep_url)

    runs = list(sweep.runs)
    if not runs:
        raise SystemExit(f"Sweep {sweep_path} has no runs.")

    sweep_config = _normalize_mapping(_safe_mapping(getattr(sweep, "config", {})))
    sweep_params = _normalize_mapping(sweep_config.get("parameters", {}))
    objective = _safe_mapping(sweep_config.get("metric", {}))
    default_rank_metric = objective.get("name")
    rank_metric = args.rank_metric or default_rank_metric
    if not rank_metric:
        raise SystemExit(
            "Could not infer a ranking metric from the sweep metadata. "
            "Pass --rank-metric explicitly."
        )

    rank_goal = _resolve_rank_goal(
        rank_metric, args.rank_goal, objective.get("goal", "minimize")
    )
    records = _collect_run_records(runs, rank_metric)
    ranked_records = _rank_records(records, rank_goal)
    highlight_count = min(args.top_k, len(ranked_records))
    top_runs = ranked_records[:highlight_count]
    run_styles = _build_rank_run_styles(ranked_records, highlight_count)
    ci_count = min(
        args.top_k_ci if args.top_k_ci is not None else args.top_k,
        len(ranked_records),
    )
    ci_runs = ranked_records[:ci_count]
    variable_specs = _find_swept_variables(records, sweep_params)

    output_dir = (
        Path(args.output_dir)
        if args.output_dir is not None
        else Path("wandb_sweep_analysis") / _slugify(sweep_path)
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Sweep: {sweep_path}")
    print(f"Runs loaded: {len(runs)}")
    print(f"Ranking metric: {rank_metric} ({rank_goal})")
    print(f"Top runs used for overlays: {len(top_runs)}")
    print(f"Non-constant swept variables: {len(variable_specs)}")
    print()

    _print_run_states(runs)
    _print_variable_summary(variable_specs)
    _print_top_runs(top_runs, rank_metric)

    if variable_specs:
        fig = _plot_variables(
            variable_specs,
            top_runs,
            run_styles,
            args.bins,
            sweep_path,
            rank_metric,
            args.max_figure_width,
            args.max_figure_height,
            args.max_categorical_label_chars,
        )
        figure_path = output_dir / "sweep_config_histograms.png"
        fig.savefig(figure_path, bbox_inches="tight", dpi=args.dpi)
        plt.close(fig)
        print(f"\nSaved figure: {figure_path}")

        scatter_fig = _plot_metric_scatterplots(
            variable_specs,
            records,
            top_runs,
            run_styles,
            ci_runs,
            args.k_fold,
            sweep_path,
            rank_metric,
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
    summary = {
        "sweep_path": sweep_path,
        "rank_metric": rank_metric,
        "rank_goal": rank_goal,
        "top_runs": [
            {
                "run_id": rec.run_id,
                "name": rec.name,
                "state": rec.state,
                "score": rec.score,
                "config": rec.config,
            }
            for rec in top_runs
        ],
        "non_constant_variables": [
            {
                "name": spec["display_name"],
                "normalized_name": spec["key"],
                "distribution": spec["distribution"],
                "kind": spec["kind"],
                "unique_values": spec["unique_values"],
            }
            for spec in variable_specs
        ],
    }
    summary_path.write_text(json.dumps(summary, indent=2, default=_json_default))
    print(f"Saved summary: {summary_path}")

    report_path = output_dir / "sweep_summary.md"
    report_lines = [
        f"# Sweep summary: {sweep_path}",
        "",
        f"- Ranking metric: `{rank_metric}`",
        f"- Ranking goal: `{rank_goal}`",
        f"- Runs loaded: `{len(runs)}`",
        f"- Non-constant swept variables: `{len(variable_specs)}`",
        "",
        "## Top runs",
        "",
    ]
    for idx, rec in enumerate(top_runs, start=1):
        report_lines.append(f"{idx}. `{rec.name}` (`{rec.run_id}`) score={rec.score}")
    report_lines.extend(["", "## Variables", ""])
    for spec in variable_specs:
        report_lines.append(
            f"- `{spec['display_name']}`: {spec['kind']}"
            + (
                f", distribution={spec['distribution']}"
                if spec["distribution"] is not None
                else ""
            )
        )
    report_path.write_text("\n".join(report_lines) + "\n")
    print(f"Saved report: {report_path}")


def _load_sweep(ref: str):
    entity, project, sweep_id = _parse_sweep_ref(ref)
    api = wandb.Api()
    return api.sweep(f"{entity}/{project}/{sweep_id}")


def _parse_sweep_ref(ref: str) -> tuple[str, str, str]:
    if "://" in ref:
        parsed = urlparse(ref)
        parts = [part for part in parsed.path.split("/") if part]
    else:
        parts = [part for part in ref.split("/") if part]

    if len(parts) >= 4 and parts[2] == "sweeps":
        entity, project, _, sweep_id = parts[:4]
    elif len(parts) == 3:
        entity, project, sweep_id = parts
    else:
        raise ValueError(
            "Expected a W&B sweep URL like "
            "'https://wandb.ai/<entity>/<project>/sweeps/<sweep_id>'"
        )
    if not entity or not project or not sweep_id:
        raise ValueError(f"Invalid W&B sweep reference: {ref!r}")
    return entity, project, sweep_id


def _sweep_path_from_ref(ref: str) -> str:
    entity, project, sweep_id = _parse_sweep_ref(ref)
    return f"{entity}/{project}/{sweep_id}"


def _collect_run_records(runs: list[Any], rank_metric: str) -> list[RunRecord]:
    records: list[RunRecord] = []
    for run in runs:
        config = _normalize_mapping(_safe_mapping(getattr(run, "config", {})))
        score = _safe_float(_mapping_get(getattr(run, "summary", {}), rank_metric))
        records.append(
            RunRecord(
                run_id=str(getattr(run, "id", "")),
                name=str(getattr(run, "name", "")),
                state=str(getattr(run, "state", "")),
                score=score,
                config=config,
            )
        )
    return records


def _rank_records(records: list[RunRecord], rank_goal: str) -> list[RunRecord]:
    ranked = [record for record in records if record.score is not None]
    if not ranked:
        return []
    ranked.sort(
        key=lambda record: record.score if record.score is not None else math.inf
    )
    if rank_goal == "maximize":
        ranked.reverse()
    return ranked


def _resolve_rank_goal(rank_metric: str, rank_goal: str, sweep_goal: str) -> str:
    if rank_goal != "auto":
        return rank_goal
    if sweep_goal in {"minimize", "maximize"}:
        return sweep_goal
    lowered = rank_metric.lower()
    if any(
        token in lowered
        for token in ("loss", "error", "mae", "mse", "rmse", "distance")
    ):
        return "minimize"
    return "maximize"


def _find_swept_variables(
    records: list[RunRecord], sweep_params: dict[str, Any]
) -> list[dict[str, Any]]:
    all_keys = set(sweep_params)
    if not all_keys:
        for record in records:
            all_keys.update(record.config)

    variable_specs: list[dict[str, Any]] = []
    for key in sorted(all_keys):
        values = [record.config.get(key, MISSING) for record in records]
        present = [value for value in values if value is not MISSING]
        if len(present) < 2:
            continue

        unique = {_canonical_value(value) for value in present}
        if len(unique) <= 1:
            continue

        numeric_values = [_coerce_numeric(value) for value in present]
        is_numeric = all(value is not None for value in numeric_values)
        display_name = _display_name(key)
        param_meta = _safe_mapping(sweep_params.get(key, {}))
        distribution = param_meta.get("distribution")
        variable_specs.append(
            {
                "key": key,
                "display_name": display_name,
                "kind": "numeric" if is_numeric else "categorical",
                "distribution": distribution,
                "values": present,
                "unique_values": [
                    _display_value(value) for value in _unique_in_order(present)
                ],
                "numeric_values": [
                    value for value in numeric_values if value is not None
                ],
            }
        )
    return variable_specs


def _plot_variables(
    variable_specs: list[dict[str, Any]],
    top_runs: list[RunRecord],
    run_styles: dict[str, dict[str, Any]],
    base_bins: int,
    sweep_path: str,
    rank_metric: str,
    max_figure_width: float,
    max_figure_height: float,
    max_categorical_label_chars: int,
):
    n_plots = len(variable_specs)
    ncols = max(1, math.ceil(math.sqrt(n_plots * 1.2)))
    nrows = math.ceil(n_plots / ncols)
    fig_width = min(max_figure_width, max(14.0, 3.8 * ncols))
    fig_height = min(max_figure_height, max(8.0, 2.75 * nrows))
    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_width, fig_height))
    axes_arr = np.atleast_1d(axes).ravel()

    for ax, spec in zip(axes_arr, variable_specs):
        values = spec["values"]
        key = spec["display_name"]
        kind = spec["kind"]
        distribution = spec["distribution"]
        if kind == "numeric":
            numeric_values = np.asarray(
                [
                    value
                    for value in (_coerce_numeric(v) for v in values)
                    if value is not None
                ],
                dtype=float,
            )
            use_log = distribution == "log_uniform_values"
            bins, use_log = _numeric_bins(numeric_values, base_bins, use_log)
            ax.hist(
                numeric_values,
                bins=bins,
                color="#2c7fb8",
                alpha=0.85,
                edgecolor="white",
                linewidth=0.6,
            )
            if use_log:
                ax.set_xscale("log")
            for idx, run in enumerate(top_runs):
                value = _coerce_numeric(run.config.get(spec["key"]))
                if value is None:
                    continue
                color = run_styles[run.run_id]["color"]
                if len(spec["unique_values"]) < 8:
                    x_pos = _top_run_marker_x(
                        value,
                        idx,
                        len(top_runs),
                        numeric_values,
                        use_log=use_log,
                    )
                else:
                    x_pos = value
                ax.axvline(x_pos, color=color, lw=2.0, alpha=0.95)
            ax.set_xlabel("")
            if use_log:
                ax.set_title(f"{key} (log scale)", fontsize=10, pad=6)
            else:
                ax.set_title(key, fontsize=10, pad=6)
        else:
            labels = [
                _format_categorical_value(
                    spec["key"], value, max_categorical_label_chars
                )
                for value in values
            ]
            counts = Counter(labels)
            ordered_labels = sorted(
                counts.keys(), key=lambda label: (-counts[label], label)
            )
            positions = np.arange(len(ordered_labels), dtype=float)
            heights = [counts[label] for label in ordered_labels]
            ax.bar(
                positions,
                heights,
                color="#2c7fb8",
                alpha=0.85,
                edgecolor="white",
                linewidth=0.6,
            )
            for idx, run in enumerate(top_runs):
                value = run.config.get(spec["key"], MISSING)
                if value is MISSING:
                    continue
                label = _format_categorical_value(
                    spec["key"], value, max_categorical_label_chars
                )
                if label not in counts:
                    continue
                color = run_styles[run.run_id]["color"]
                position = ordered_labels.index(label)
                x_pos = _top_run_marker_x(
                    float(position),
                    idx,
                    len(top_runs),
                    np.asarray(positions, dtype=float),
                    use_log=False,
                    categorical=True,
                )
                ax.axvline(x_pos, color=color, lw=2.0, alpha=0.95)
            ax.set_xticks(positions)
            ax.set_xticklabels(ordered_labels, rotation=15, ha="right", fontsize=7)
            ax.set_xlabel("")
            ax.set_title(spec["display_name"], fontsize=10, pad=6)
        ax.set_ylabel("Run count", fontsize=9)
        ax.tick_params(axis="both", labelsize=8)
        ax.grid(True, axis="y", alpha=0.25)

    for ax in axes_arr[len(variable_specs) :]:
        ax.set_visible(False)

    for idx, ax in enumerate(axes_arr[: len(variable_specs)]):
        col = idx % ncols
        if col > 0:
            ax.tick_params(axis="y", labelleft=False)
            ax.set_ylabel("")

    fig.suptitle(
        f"W&B sweep parameter distributions\n{sweep_path}\nranked by {rank_metric}",
        fontsize=12,
        y=0.98,
    )
    legend_runs = top_runs[:5]
    if legend_runs:
        legend_handles = [
            Line2D(
                [0],
                [0],
                color=run_styles[run.run_id]["color"],
                lw=2.5,
                label=f"{idx + 1}. {run.name or run.run_id}",
            )
            for idx, run in enumerate(legend_runs)
        ]
        fig.legend(
            handles=legend_handles,
            loc="lower center",
            ncol=min(5, len(legend_handles)),
            frameon=False,
            bbox_to_anchor=(0.5, 0.0),
        )
        fig.tight_layout(rect=(0.02, 0.06, 1.0, 0.94), w_pad=1.0, h_pad=1.0)
    else:
        fig.tight_layout(rect=(0.02, 0.02, 1.0, 0.94), w_pad=1.0, h_pad=1.0)
    return fig


def _plot_metric_scatterplots(
    variable_specs: list[dict[str, Any]],
    records: list[RunRecord],
    top_runs: list[RunRecord],
    run_styles: dict[str, dict[str, Any]],
    ci_runs: list[RunRecord],
    k_fold: int,
    sweep_path: str,
    rank_metric: str,
    max_figure_width: float,
    max_figure_height: float,
    max_categorical_label_chars: int,
):
    n_plots = len(variable_specs)
    ncols = max(1, math.ceil(math.sqrt(n_plots * 1.2)))
    nrows = math.ceil(n_plots / ncols)
    fig_width = min(max_figure_width, max(14.0, 3.8 * ncols))
    fig_height = min(max_figure_height, max(8.0, 2.75 * nrows))
    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_width, fig_height))
    axes_arr = np.atleast_1d(axes).ravel()

    top_run_ids = {run.run_id for run in top_runs}
    ci_run_ids = {run.run_id for run in ci_runs}

    for ax, spec in zip(axes_arr, variable_specs):
        key = spec["key"]
        title = spec["display_name"]
        kind = spec["kind"]
        distribution = spec["distribution"]

        if kind == "numeric":
            points = []
            for record in records:
                x = _coerce_numeric(record.config.get(key))
                y = record.score
                if x is None or y is None or x <= 0 or y <= 0:
                    continue
                points.append((record, x, y))
            if points:
                non_top_xs = [
                    x for record, x, _ in points if record.run_id not in top_run_ids
                ]
                non_top_ys = [
                    y for record, _, y in points if record.run_id not in top_run_ids
                ]
                non_top_colors = [
                    run_styles[record.run_id]["color"]
                    for record, _, _ in points
                    if record.run_id not in top_run_ids
                ]
                if non_top_xs:
                    ax.scatter(
                        non_top_xs,
                        non_top_ys,
                        s=12,
                        c=non_top_colors,
                        alpha=0.34,
                        linewidths=0,
                    )
                band = _compute_numeric_ci_band_from_points(
                    points,
                    ci_run_ids,
                    k_fold,
                    use_log_x=distribution == "log_uniform_values",
                )
                if band is not None:
                    x_grid, y_center, y_low, y_high = band
                    ax.fill_between(
                        x_grid,
                        y_low,
                        y_high,
                        color="black",
                        alpha=0.08,
                        zorder=0,
                    )
                    ax.plot(
                        x_grid,
                        y_center,
                        color="black",
                        alpha=0.28,
                        linewidth=1.05,
                        zorder=1,
                    )
                for idx, run in enumerate(top_runs):
                    point = next(
                        (
                            (x, y)
                            for record, x, y in points
                            if record.run_id == run.run_id
                        ),
                        None,
                    )
                    if point is None:
                        continue
                    ax.scatter(
                        [point[0]],
                        [point[1]],
                        s=48,
                        c=[run_styles[run.run_id]["color"]],
                        edgecolors="black",
                        linewidths=0.4,
                        alpha=0.99,
                        zorder=4,
                    )
                use_log_x = distribution == "log_uniform_values"
                if use_log_x:
                    ax.set_xscale("log")
            else:
                ax.text(0.5, 0.5, "No positive data", ha="center", va="center")
                use_log_x = distribution == "log_uniform_values"
        else:
            labels = [
                _format_categorical_value(
                    key, record.config.get(key), max_categorical_label_chars
                )
                for record in records
                if record.config.get(key, MISSING) is not MISSING
                and record.score is not None
                and record.score > 0
            ]
            ordered_labels = sorted(set(labels))
            positions = {label: idx for idx, label in enumerate(ordered_labels)}
            for record in records:
                value = record.config.get(key, MISSING)
                if value is MISSING or record.score is None or record.score <= 0:
                    continue
                label = _format_categorical_value(
                    key, value, max_categorical_label_chars
                )
                x = positions[label]
                jitter = _categorical_jitter(record.run_id, key)
                color = run_styles[record.run_id]["color"]
                size = run_styles[record.run_id]["size"]
                alpha = run_styles[record.run_id]["alpha"]
                is_top = record.run_id in top_run_ids
                ax.scatter(
                    [x + jitter],
                    [record.score],
                    s=size,
                    c=[color],
                    alpha=alpha,
                    edgecolors="black" if is_top else "none",
                    linewidths=0.4 if is_top else 0,
                    zorder=3 if is_top else 2,
                )
            ax.set_xticks(list(positions.values()))
            ax.set_xticklabels(ordered_labels, rotation=15, ha="right", fontsize=7)
            ax.set_xlim(-0.5, max(len(ordered_labels) - 0.5, 0.5))
            use_log_x = False

        ax.set_yscale("log")
        ax.set_title(title, fontsize=10, pad=6)
        ax.set_ylabel(rank_metric, fontsize=9)
        ax.tick_params(axis="both", labelsize=8)
        ax.grid(True, axis="both", alpha=0.22)
        if kind == "numeric" and use_log_x:
            ax.set_xlabel("")
            ax.set_title(f"{title} (log x scale)", fontsize=10, pad=6)
        else:
            ax.set_xlabel("")

    for ax in axes_arr[len(variable_specs) :]:
        ax.set_visible(False)

    for idx, ax in enumerate(axes_arr[: len(variable_specs)]):
        col = idx % ncols
        if col > 0:
            ax.set_ylabel("")

    fig.suptitle(
        f"W&B sweep metric scatterplots\n{sweep_path}\nmetric={rank_metric} (log scale)",
        fontsize=12,
        y=0.98,
    )
    legend_runs = top_runs[:5]
    if legend_runs:
        legend_handles = [
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="none",
                markerfacecolor=run_styles[run.run_id]["color"],
                markeredgecolor="black",
                markeredgewidth=0.4,
                markersize=8,
                label=f"{idx + 1}. {run.name or run.run_id}",
            )
            for idx, run in enumerate(legend_runs)
        ]
        fig.legend(
            handles=legend_handles,
            loc="lower center",
            ncol=min(5, len(legend_handles)),
            frameon=False,
            bbox_to_anchor=(0.5, 0.0),
        )
    fig.tight_layout(rect=(0.02, 0.06, 1.0, 0.94), w_pad=1.0, h_pad=1.0)
    return fig


def _numeric_bins(
    values: np.ndarray, base_bins: int, use_log: bool
) -> tuple[np.ndarray, bool]:
    if values.size == 0:
        return np.linspace(0.0, 1.0, 2), False

    if use_log and np.any(values <= 0):
        use_log = False

    if use_log:
        positive = values[values > 0]
        lo = float(np.min(positive))
        hi = float(np.max(positive))
        if lo == hi:
            lo /= 1.5
            hi *= 1.5
        bin_count = max(8, min(30, base_bins))
        bins = np.logspace(np.log10(lo), np.log10(hi), bin_count)
        return bins, True

    lo = float(np.min(values))
    hi = float(np.max(values))
    if lo == hi:
        pad = 0.5 if lo == 0 else max(abs(lo) * 0.1, 1e-6)
        return np.linspace(lo - pad, hi + pad, max(4, base_bins)), False
    bins = np.histogram_bin_edges(values, bins=min(base_bins, max(4, values.size)))
    if len(bins) < 2:
        bins = np.linspace(lo, hi, max(4, base_bins))
    return bins, False


def _print_run_states(runs: list[Any]) -> None:
    state_counts = Counter(str(getattr(run, "state", "")) for run in runs)
    print("Run states:")
    for state, count in sorted(state_counts.items()):
        print(f"  {state}: {count}")
    print()


def _print_variable_summary(variable_specs: list[dict[str, Any]]) -> None:
    print("Non-constant swept variables:")
    for spec in variable_specs:
        distribution = spec["distribution"]
        suffix = f", distribution={distribution}" if distribution is not None else ""
        print(f"  - {spec['display_name']} ({spec['kind']}{suffix})")
    print()


def _print_top_runs(top_runs: list[RunRecord], rank_metric: str) -> None:
    print(f"Top runs by {rank_metric}:")
    for idx, record in enumerate(top_runs, start=1):
        print(f"  {idx}. {record.name} ({record.run_id}) -> {record.score}")
    print()


def _normalize_mapping(mapping: Any) -> dict[str, Any]:
    data = _safe_mapping(mapping)
    normalized: dict[str, Any] = {}
    for key, value in data.items():
        if not isinstance(key, str):
            continue
        normalized[key.replace("-", "_")] = _unwrap_wandb_value(value)
    return normalized


def _safe_mapping(mapping: Any) -> dict[str, Any]:
    if mapping is None:
        return {}
    if isinstance(mapping, dict):
        return mapping
    try:
        return dict(mapping)
    except Exception:
        return {}


def _unwrap_wandb_value(value: Any) -> Any:
    if isinstance(value, dict) and set(value) == {"value"}:
        return value["value"]
    return value


def _mapping_get(mapping: Any, key: str) -> Any:
    if hasattr(mapping, "get"):
        return mapping.get(key)
    try:
        return dict(mapping).get(key)
    except Exception:
        return None


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def _coerce_numeric(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float, np.number)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except Exception:
            return None
    return None


def _canonical_value(value: Any) -> Any:
    numeric = _coerce_numeric(value)
    if numeric is not None:
        return ("num", numeric)
    if isinstance(value, bool):
        return ("bool", bool(value))
    if isinstance(value, (list, tuple)):
        return ("seq", tuple(_canonical_value(item) for item in value))
    if isinstance(value, dict):
        return (
            "dict",
            tuple(sorted((str(k), _canonical_value(v)) for k, v in value.items())),
        )
    return ("str", str(value))


def _display_value(value: Any) -> str:
    numeric = _coerce_numeric(value)
    if numeric is not None:
        return f"{numeric:g}"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (list, tuple, dict)):
        return json.dumps(value, sort_keys=True, default=_json_default)
    return str(value)


def _shorten_label(text: str, max_chars: int) -> str:
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    if max_chars <= 4:
        return text[:max_chars]
    head = max(1, max_chars - 1)
    return text[: head - 1].rstrip() + "…"


def _display_name(name: str) -> str:
    return name.replace("_", "-")


def _format_categorical_value(key: str, value: Any, max_chars: int) -> str:
    text = _display_value(value)
    if key == "hidden_irreps":
        text = _truncate_hidden_irreps_value(text)
    return _shorten_label(text, max_chars)


def _truncate_hidden_irreps_value(value: str) -> str:
    parts = value.split("+")
    if len(parts) < 3:
        return value
    return "+".join(parts[:2]) + "+"


def _build_rank_run_styles(
    ranked_records: list[RunRecord], highlight_count: int
) -> dict[str, dict[str, Any]]:
    if not ranked_records:
        return {}

    highlight_count = max(0, min(highlight_count, len(ranked_records)))
    top_records = ranked_records[:highlight_count]
    rest_records = ranked_records[highlight_count:]

    top_colors = _gradient_colors("#d62728", "#f1c40f", len(top_records))
    rest_colors = _gradient_colors("#f1c40f", "#1f77b4", len(rest_records))

    styles: dict[str, dict[str, Any]] = {}
    for record, color in zip(top_records, top_colors):
        styles[record.run_id] = {
            "color": color,
            "size": 52,
            "alpha": 0.98,
        }
    for record, color in zip(rest_records, rest_colors):
        styles[record.run_id] = {
            "color": color,
            "size": 14,
            "alpha": 0.38,
        }
    return styles


def _gradient_colors(start_hex: str, end_hex: str, count: int) -> list[str]:
    if count <= 0:
        return []
    start = np.asarray(matplotlib.colors.to_rgb(start_hex), dtype=float)
    end = np.asarray(matplotlib.colors.to_rgb(end_hex), dtype=float)
    if count == 1:
        return [matplotlib.colors.to_hex(start)]
    colors = []
    for t in np.linspace(0.0, 1.0, count):
        rgb = start * (1.0 - t) + end * t
        colors.append(matplotlib.colors.to_hex(rgb))
    return colors


def _categorical_jitter(run_id: str, key: str, width: float = 0.12) -> float:
    seed_bytes = hashlib.sha1(f"{run_id}:{key}".encode("utf-8")).digest()[:8]
    seed = int.from_bytes(seed_bytes, "big", signed=False)
    rng = np.random.default_rng(seed)
    return float(rng.uniform(-width, width))


def _top_run_marker_x(
    base_x: float,
    idx: int,
    n_top_runs: int,
    values: np.ndarray,
    *,
    use_log: bool,
    categorical: bool = False,
) -> float:
    if n_top_runs <= 1:
        return base_x

    centered = idx - (n_top_runs - 1) / 2.0

    if categorical:
        return base_x + centered * 0.08

    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return base_x

    if use_log:
        # Keep the lines visually separated while preserving positive values.
        factor = 1.0 + centered * 0.025
        return max(base_x * factor, np.finfo(float).tiny)

    lo = float(np.min(arr))
    hi = float(np.max(arr))
    span = hi - lo
    if span <= 0:
        span = max(abs(lo), 1.0)
    step = max(span * 0.01, 1e-6)
    return base_x + centered * step


def _compute_numeric_ci_band_from_points(
    points: list[tuple[RunRecord, float, float]],
    ci_run_ids: set[str],
    k_fold: int,
    use_log_x: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    ci_points = [(x, y) for record, x, y in points if record.run_id in ci_run_ids]
    return _compute_numeric_ci_band_from_xy(ci_points, k_fold, use_log_x)


def _compute_numeric_ci_band_from_xy(
    points: list[tuple[float, float]],
    k_fold: int,
    use_log_x: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    if len(points) < 2:
        return None

    xs = np.asarray([x for x, _ in points], dtype=float)
    ys = np.asarray([y for _, y in points], dtype=float)
    valid = np.isfinite(xs) & np.isfinite(ys) & (ys > 0)
    if use_log_x:
        valid &= xs > 0
    xs = xs[valid]
    ys = ys[valid]
    if xs.size < 2:
        return None

    x_min = float(np.min(xs))
    x_max = float(np.max(xs))
    if x_min == x_max:
        return None

    if use_log_x:
        x_grid = np.logspace(np.log10(x_min), np.log10(x_max), 200)
    else:
        x_grid = np.linspace(x_min, x_max, 200)

    n = xs.size
    fold_count = max(2, min(int(k_fold), n))
    indices = np.arange(n)
    rng = np.random.default_rng(0)
    rng.shuffle(indices)
    folds = np.array_split(indices, fold_count)

    fold_preds: list[np.ndarray] = []
    for fold in folds:
        train_idx = np.setdiff1d(indices, fold, assume_unique=False)
        if train_idx.size < 1:
            continue
        fit = _fit_log_linear_regression(xs[train_idx], ys[train_idx])
        if fit is None:
            continue
        slope, intercept = fit
        fold_preds.append(_predict_log_linear_regression(x_grid, slope, intercept))

    if not fold_preds:
        return None

    center_fit = _fit_log_linear_regression(xs, ys)
    if center_fit is None:
        return None
    slope, intercept = center_fit
    y_center = _predict_log_linear_regression(x_grid, slope, intercept)
    y_stack = np.vstack(fold_preds)
    y_low = np.min(y_stack, axis=0)
    y_high = np.max(y_stack, axis=0)
    return x_grid, y_center, y_low, y_high


def _fit_log_linear_regression(
    xs: np.ndarray, ys: np.ndarray
) -> tuple[float, float] | None:
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    mask = np.isfinite(xs) & np.isfinite(ys) & (ys > 0)
    xs = xs[mask]
    ys = ys[mask]
    if xs.size == 0:
        return None
    if xs.size == 1 or np.allclose(xs, xs[0]):
        return 0.0, float(np.log(ys).mean())
    slope, intercept = np.polyfit(xs, np.log(ys), 1)
    return float(slope), float(intercept)


def _predict_log_linear_regression(
    xs: np.ndarray, slope: float, intercept: float
) -> np.ndarray:
    xs = np.asarray(xs, dtype=float)
    return np.exp(intercept + slope * xs)


def _unique_in_order(values: list[Any]) -> list[Any]:
    seen = set()
    ordered: list[Any] = []
    for value in values:
        marker = _canonical_value(value)
        if marker in seen:
            continue
        seen.add(marker)
        ordered.append(value)
    return ordered


def _slugify(text: str) -> str:
    cleaned = []
    for char in text.lower():
        if char.isalnum():
            cleaned.append(char)
        else:
            cleaned.append("-")
    slug = "".join(cleaned).strip("-")
    while "--" in slug:
        slug = slug.replace("--", "-")
    return slug or "wandb-sweep"


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    return str(value)


if __name__ == "__main__":
    main()
