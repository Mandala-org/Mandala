from __future__ import annotations

import math
from collections import Counter
from typing import Any

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D

from analysis.wandb_sweep_core import (
    RunRecord,
    categorical_jitter,
    coerce_numeric,
    compute_numeric_ci_band_from_points,
    format_categorical_value,
    numeric_bins,
    numeric_jitter,
    top_run_marker_x,
)


def plot_variable_histograms(
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
                    for value in (coerce_numeric(v) for v in values)
                    if value is not None
                ],
                dtype=float,
            )
            use_log = distribution == "log_uniform_values"
            bins, use_log = numeric_bins(numeric_values, base_bins, use_log)
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
                value = coerce_numeric(run.config.get(spec["key"]))
                if value is None:
                    continue
                color = run_styles[run.run_id]["color"]
                if len(spec["unique_values"]) < 8:
                    x_pos = top_run_marker_x(
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
            ax.set_title(
                f"{key} (log scale)" if use_log else key,
                fontsize=10,
                pad=6,
            )
        else:
            labels = [
                format_categorical_value(
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
                value = run.config.get(spec["key"])
                label = format_categorical_value(
                    spec["key"], value, max_categorical_label_chars
                )
                if label not in counts:
                    continue
                color = run_styles[run.run_id]["color"]
                position = ordered_labels.index(label)
                x_pos = top_run_marker_x(
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


def plot_metric_scatterplots(
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
                x = coerce_numeric(record.config.get(key))
                y = record.score
                if x is None or y is None or x <= 0 or y <= 0:
                    continue
                points.append((record, x, y))
            if points:
                numeric_x_values = np.asarray([x for _, x, _ in points], dtype=float)
                discrete_numeric = len(spec["unique_values"]) < 8
                non_top_points = [
                    (record, x, y)
                    for record, x, y in points
                    if record.run_id not in top_run_ids
                ]
                if non_top_points:
                    non_top_xs = [
                        (
                            numeric_jitter(
                                x,
                                record.run_id,
                                key,
                                numeric_x_values,
                                use_log=distribution == "log_uniform_values",
                            )
                            if discrete_numeric
                            else x
                        )
                        for record, x, _ in non_top_points
                    ]
                    non_top_ys = [y for _, _, y in non_top_points]
                    non_top_colors = [
                        run_styles[record.run_id]["color"]
                        for record, _, _ in non_top_points
                    ]
                    ax.scatter(
                        non_top_xs,
                        non_top_ys,
                        s=12,
                        c=non_top_colors,
                        alpha=0.34,
                        linewidths=0,
                    )
                band = compute_numeric_ci_band_from_points(
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
                for run in top_runs:
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
                    x_pos = point[0]
                    if discrete_numeric:
                        x_pos = numeric_jitter(
                            x_pos,
                            run.run_id,
                            key,
                            numeric_x_values,
                            use_log=distribution == "log_uniform_values",
                        )
                    ax.scatter(
                        [x_pos],
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
                format_categorical_value(
                    key, record.config.get(key), max_categorical_label_chars
                )
                for record in records
                if record.config.get(key) is not None
                and record.score is not None
                and record.score > 0
            ]
            ordered_labels = sorted(set(labels))
            positions = {label: idx for idx, label in enumerate(ordered_labels)}
            for record in records:
                value = record.config.get(key)
                if value is None or record.score is None or record.score <= 0:
                    continue
                label = format_categorical_value(
                    key, value, max_categorical_label_chars
                )
                x = positions[label]
                jitter = categorical_jitter(record.run_id, key)
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
        ax.set_title(
            f"{title} (log x scale)" if kind == "numeric" and use_log_x else title,
            fontsize=10,
            pad=6,
        )
        ax.set_ylabel(rank_metric, fontsize=9)
        ax.tick_params(axis="both", labelsize=8)
        ax.grid(True, axis="both", alpha=0.22)
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
