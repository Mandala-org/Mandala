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
        default=3,
        help="How many top runs to overlay on the histograms.",
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
    top_runs = _select_top_runs(records, args.top_k, rank_metric, rank_goal)
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


def _select_top_runs(
    records: list[RunRecord], top_k: int, rank_metric: str, rank_goal: str
) -> list[RunRecord]:
    ranked = [record for record in records if record.score is not None]
    if not ranked:
        raise SystemExit(f"No run exposed a numeric value for metric {rank_metric!r}.")
    ranked.sort(
        key=lambda record: record.score if record.score is not None else math.inf
    )
    if rank_goal == "maximize":
        ranked.reverse()
    return ranked[:top_k]


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

    colors = ("#d62728", "#ff7f0e", "#f1c40f")
    line_handles = []
    for idx, run in enumerate(top_runs):
        color = colors[idx % len(colors)]
        line_handles.append(
            Line2D(
                [0],
                [0],
                color=color,
                lw=2.5,
                label=f"{idx + 1}. {run.name or run.run_id}",
            )
        )

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
                color = colors[idx % len(colors)]
                x_pos = _top_run_marker_x(
                    value,
                    idx,
                    len(top_runs),
                    numeric_values,
                    use_log=use_log,
                )
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
                color = colors[idx % len(colors)]
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
    if line_handles:
        fig.legend(
            handles=line_handles,
            loc="lower center",
            ncol=min(3, len(line_handles)),
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
    color_map = _build_scatter_color_map(records, top_run_ids)
    top_colors = ("#d62728", "#ff7f0e", "#f1c40f")
    top_color_by_id = {
        run.run_id: top_colors[idx % len(top_colors)]
        for idx, run in enumerate(top_runs)
    }

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
                    color_map[record.run_id]
                    for record, _, _ in points
                    if record.run_id not in top_run_ids
                ]
                if non_top_xs:
                    ax.scatter(
                        non_top_xs,
                        non_top_ys,
                        s=18,
                        c=non_top_colors,
                        alpha=0.45,
                        linewidths=0,
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
                        s=75,
                        c=[top_colors[idx % len(top_colors)]],
                        edgecolors="black",
                        linewidths=0.4,
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
                if record.run_id in top_color_by_id:
                    color = top_color_by_id[record.run_id]
                else:
                    color = color_map[record.run_id]
                ax.scatter(
                    [x + jitter],
                    [record.score],
                    s=18,
                    c=[color],
                    alpha=0.45 if record.run_id not in top_run_ids else 1.0,
                    linewidths=0,
                    zorder=3 if record.run_id in top_run_ids else 2,
                )
            for idx, run in enumerate(top_runs):
                value = run.config.get(key, MISSING)
                if value is MISSING or run.score is None or run.score <= 0:
                    continue
                label = _format_categorical_value(
                    key, value, max_categorical_label_chars
                )
                if label not in positions:
                    continue
                ax.scatter(
                    [positions[label]],
                    [run.score],
                    s=90,
                    c=[top_colors[idx % len(top_colors)]],
                    edgecolors="black",
                    linewidths=0.4,
                    zorder=5,
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

    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor=color,
            markeredgecolor="black",
            markeredgewidth=0.4,
            markersize=8,
            label=f"{idx + 1}. {run.name or run.run_id}",
        )
        for idx, (run, color) in enumerate(zip(top_runs, top_colors))
    ]

    fig.suptitle(
        f"W&B sweep metric scatterplots\n{sweep_path}\nmetric={rank_metric} (log scale)",
        fontsize=12,
        y=0.98,
    )
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=min(3, len(handles)),
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


def _build_scatter_color_map(
    records: list[RunRecord], top_run_ids: set[str]
) -> dict[str, str]:
    palette = list(plt.get_cmap("tab20").colors)
    rng = np.random.default_rng(0)
    rng.shuffle(palette)
    palette_hex = [matplotlib.colors.to_hex(color) for color in palette]
    color_map: dict[str, str] = {}
    idx = 0
    for record in records:
        if record.run_id in top_run_ids:
            continue
        color_map[record.run_id] = palette_hex[idx % len(palette_hex)]
        idx += 1
    return color_map


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
