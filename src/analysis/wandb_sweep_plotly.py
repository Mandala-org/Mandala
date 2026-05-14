from __future__ import annotations

import html
import math
from collections import Counter
from typing import Any

import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

from analysis.wandb_sweep_core import (
    RunRecord,
    SweepAnalysisResult,
    categorical_jitter,
    coerce_numeric,
    compute_numeric_ci_band_from_points,
    format_categorical_value,
    numeric_jitter,
    run_url_for_record,
    top_run_marker_x,
)


def build_histogram_figure(
    result: SweepAnalysisResult,
    *,
    run_styles: dict[str, dict[str, Any]],
    bins: int,
    max_categorical_label_chars: int,
) -> go.Figure:
    variable_specs = result.variable_specs
    n_plots = len(variable_specs)
    ncols = max(1, math.ceil(math.sqrt(max(n_plots, 1) * 1.2)))
    nrows = max(1, math.ceil(max(n_plots, 1) / ncols))
    subplot_titles = [
        (
            f"{spec['display_name']} (log scale)"
            if spec["kind"] == "numeric"
            and spec["distribution"] == "log_uniform_values"
            else spec["display_name"]
        )
        for spec in variable_specs
    ]
    fig = make_subplots(rows=nrows, cols=ncols, subplot_titles=subplot_titles)

    for idx, spec in enumerate(variable_specs, start=1):
        row, col = _row_col(idx, ncols)
        values = spec["values"]
        if spec["kind"] == "numeric":
            numeric_values = np.asarray(
                [
                    value
                    for value in (coerce_numeric(v) for v in values)
                    if value is not None
                ],
                dtype=float,
            )
            fig.add_trace(
                go.Histogram(
                    x=numeric_values,
                    nbinsx=min(max(bins, 4), max(10, numeric_values.size)),
                    marker=dict(color="#2c7fb8"),
                    opacity=0.85,
                    showlegend=False,
                    hovertemplate="x=%{x}<br>count=%{y}<extra></extra>",
                ),
                row=row,
                col=col,
            )
            use_log = spec["distribution"] == "log_uniform_values"
            if use_log:
                fig.update_xaxes(type="log", row=row, col=col)
            for top_idx, run in enumerate(result.top_runs):
                value = coerce_numeric(run.config.get(spec["key"]))
                if value is None:
                    continue
                x_pos = value
                if len(spec["unique_values"]) < 8:
                    x_pos = top_run_marker_x(
                        value,
                        top_idx,
                        len(result.top_runs),
                        numeric_values,
                        use_log=use_log,
                    )
                fig.add_vline(
                    x=x_pos,
                    line_width=2,
                    line_color=run_styles[run.run_id]["color"],
                    opacity=0.95,
                    row=row,
                    col=col,
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
            fig.add_trace(
                go.Bar(
                    x=positions,
                    y=heights,
                    marker=dict(color="#2c7fb8"),
                    opacity=0.85,
                    showlegend=False,
                    hovertemplate="%{customdata}<br>count=%{y}<extra></extra>",
                    customdata=ordered_labels,
                ),
                row=row,
                col=col,
            )
            fig.update_xaxes(
                tickmode="array",
                tickvals=positions.tolist(),
                ticktext=ordered_labels,
                row=row,
                col=col,
            )
            for top_idx, run in enumerate(result.top_runs):
                label = format_categorical_value(
                    spec["key"],
                    run.config.get(spec["key"]),
                    max_categorical_label_chars,
                )
                if label not in counts:
                    continue
                position = ordered_labels.index(label)
                x_pos = top_run_marker_x(
                    float(position),
                    top_idx,
                    len(result.top_runs),
                    positions,
                    use_log=False,
                    categorical=True,
                )
                fig.add_vline(
                    x=x_pos,
                    line_width=2,
                    line_color=run_styles[run.run_id]["color"],
                    opacity=0.95,
                    row=row,
                    col=col,
                )
        fig.update_yaxes(title_text="Run count", row=row, col=col)

    fig.update_layout(
        barmode="overlay",
        title=(
            f"W&B sweep parameter distributions<br>{html.escape(result.sweep_path)}"
            f"<br>ranked by {html.escape(result.rank_metric)}"
        ),
        height=max(550, 280 * nrows),
        autosize=True,
        margin=dict(l=50, r=30, t=90, b=60),
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e8edf7"),
        legend=dict(font=dict(color="#e8edf7")),
    )
    return fig


def build_scatter_figure(
    result: SweepAnalysisResult,
    *,
    run_styles: dict[str, dict[str, Any]],
    k_fold: int,
    max_categorical_label_chars: int,
) -> go.Figure:
    variable_specs = result.variable_specs
    n_plots = len(variable_specs)
    ncols = max(1, math.ceil(math.sqrt(max(n_plots, 1) * 1.2)))
    nrows = max(1, math.ceil(max(n_plots, 1) / ncols))
    subplot_titles = [
        (
            f"{spec['display_name']} (log x scale)"
            if spec["kind"] == "numeric"
            and spec["distribution"] == "log_uniform_values"
            else spec["display_name"]
        )
        for spec in variable_specs
    ]
    fig = make_subplots(rows=nrows, cols=ncols, subplot_titles=subplot_titles)
    top_run_ids = {run.run_id for run in result.top_runs}
    ci_run_ids = {run.run_id for run in result.ci_runs}

    for idx, spec in enumerate(variable_specs, start=1):
        row, col = _row_col(idx, ncols)
        key = spec["key"]
        if spec["kind"] == "numeric":
            points = []
            for record in result.records:
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
                    xs = [
                        (
                            numeric_jitter(
                                x,
                                record.run_id,
                                key,
                                numeric_x_values,
                                use_log=spec["distribution"] == "log_uniform_values",
                            )
                            if discrete_numeric
                            else x
                        )
                        for record, x, _ in non_top_points
                    ]
                    ys = [y for _, _, y in non_top_points]
                    fig.add_trace(
                        go.Scattergl(
                            x=xs,
                            y=ys,
                            mode="markers",
                            marker=dict(
                                size=9,
                                color=[
                                    run_styles[record.run_id]["color"]
                                    for record, _, _ in non_top_points
                                ],
                                opacity=0.34,
                                line=dict(width=0),
                            ),
                            customdata=[
                                [record.name, record.run_id, record.state, x]
                                for record, x, _ in non_top_points
                            ],
                            hovertemplate=(
                                "run=%{customdata[0]}<br>"
                                "id=%{customdata[1]}<br>"
                                "state=%{customdata[2]}<br>"
                                f"{html.escape(spec['display_name'])}=%{{customdata[3]}}<br>"
                                f"{html.escape(result.rank_metric)}=%{{y}}<extra></extra>"
                            ),
                            showlegend=False,
                        ),
                        row=row,
                        col=col,
                    )
                band = compute_numeric_ci_band_from_points(
                    points,
                    ci_run_ids,
                    k_fold,
                    use_log_x=spec["distribution"] == "log_uniform_values",
                )
                if band is not None:
                    x_grid, y_center, y_low, y_high = band
                    fig.add_trace(
                        go.Scatter(
                            x=x_grid,
                            y=y_low,
                            mode="lines",
                            line=dict(width=0),
                            showlegend=False,
                            hoverinfo="skip",
                        ),
                        row=row,
                        col=col,
                    )
                    fig.add_trace(
                        go.Scatter(
                            x=x_grid,
                            y=y_high,
                            mode="lines",
                            line=dict(width=0),
                            fill="tonexty",
                            fillcolor="rgba(0,0,0,0.08)",
                            showlegend=False,
                            hoverinfo="skip",
                        ),
                        row=row,
                        col=col,
                    )
                    fig.add_trace(
                        go.Scatter(
                            x=x_grid,
                            y=y_center,
                            mode="lines",
                            line=dict(color="rgba(0,0,0,0.28)", width=1.2),
                            showlegend=False,
                            hoverinfo="skip",
                        ),
                        row=row,
                        col=col,
                    )
                highlighted = []
                for run in result.top_runs:
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
                            use_log=spec["distribution"] == "log_uniform_values",
                        )
                    highlighted.append((run, x_pos, point[1]))
                if highlighted:
                    fig.add_trace(
                        go.Scattergl(
                            x=[x for _run, x, _y in highlighted],
                            y=[y for _run, _x, y in highlighted],
                            mode="markers",
                            marker=dict(
                                size=13,
                                color=[
                                    run_styles[run.run_id]["color"]
                                    for run, _x, _y in highlighted
                                ],
                                opacity=0.99,
                                line=dict(color="black", width=1),
                            ),
                            customdata=[
                                [run.name, run.run_id, run.state, x]
                                for run, x, _y in highlighted
                            ],
                            hovertemplate=(
                                "run=%{customdata[0]}<br>"
                                "id=%{customdata[1]}<br>"
                                "state=%{customdata[2]}<br>"
                                f"{html.escape(spec['display_name'])}=%{{customdata[3]}}<br>"
                                f"{html.escape(result.rank_metric)}=%{{y}}<extra></extra>"
                            ),
                            showlegend=False,
                        ),
                        row=row,
                        col=col,
                    )
                if spec["distribution"] == "log_uniform_values":
                    fig.update_xaxes(type="log", row=row, col=col)
        else:
            labels = [
                format_categorical_value(
                    key, record.config.get(key), max_categorical_label_chars
                )
                for record in result.records
                if record.config.get(key) is not None
                and record.score is not None
                and record.score > 0
            ]
            ordered_labels = sorted(set(labels))
            positions = {label: idx for idx, label in enumerate(ordered_labels)}
            non_top_points: list[tuple[RunRecord, float, float, str]] = []
            top_points: list[tuple[RunRecord, float, float, str]] = []
            for record in result.records:
                value = record.config.get(key)
                if value is None or record.score is None or record.score <= 0:
                    continue
                label = format_categorical_value(
                    key, value, max_categorical_label_chars
                )
                x_pos = positions[label] + categorical_jitter(record.run_id, key)
                point = (record, x_pos, record.score, label)
                if record.run_id in top_run_ids:
                    top_points.append(point)
                else:
                    non_top_points.append(point)
            if non_top_points:
                fig.add_trace(
                    go.Scattergl(
                        x=[x for _record, x, _y, _label in non_top_points],
                        y=[y for _record, _x, y, _label in non_top_points],
                        mode="markers",
                        marker=dict(
                            size=[
                                max(7, int(run_styles[record.run_id]["size"] * 0.28))
                                for record, _x, _y, _label in non_top_points
                            ],
                            color=[
                                run_styles[record.run_id]["color"]
                                for record, _x, _y, _label in non_top_points
                            ],
                            opacity=[
                                run_styles[record.run_id]["alpha"]
                                for record, _x, _y, _label in non_top_points
                            ],
                            line=dict(width=0),
                        ),
                        customdata=[
                            [record.name, record.run_id, record.state, label]
                            for record, _x, _y, label in non_top_points
                        ],
                        hovertemplate=(
                            "run=%{customdata[0]}<br>"
                            "id=%{customdata[1]}<br>"
                            "state=%{customdata[2]}<br>"
                            f"{html.escape(spec['display_name'])}=%{{customdata[3]}}<br>"
                            f"{html.escape(result.rank_metric)}=%{{y}}<extra></extra>"
                        ),
                        showlegend=False,
                    ),
                    row=row,
                    col=col,
                )
            if top_points:
                fig.add_trace(
                    go.Scattergl(
                        x=[x for _record, x, _y, _label in top_points],
                        y=[y for _record, _x, y, _label in top_points],
                        mode="markers",
                        marker=dict(
                            size=[
                                max(8, int(run_styles[record.run_id]["size"] * 0.32))
                                for record, _x, _y, _label in top_points
                            ],
                            color=[
                                run_styles[record.run_id]["color"]
                                for record, _x, _y, _label in top_points
                            ],
                            opacity=0.98,
                            line=dict(color="black", width=1),
                        ),
                        customdata=[
                            [record.name, record.run_id, record.state, label]
                            for record, _x, _y, label in top_points
                        ],
                        hovertemplate=(
                            "run=%{customdata[0]}<br>"
                            "id=%{customdata[1]}<br>"
                            "state=%{customdata[2]}<br>"
                            f"{html.escape(spec['display_name'])}=%{{customdata[3]}}<br>"
                            f"{html.escape(result.rank_metric)}=%{{y}}<extra></extra>"
                        ),
                        showlegend=False,
                    ),
                    row=row,
                    col=col,
                )
            fig.update_xaxes(
                tickmode="array",
                tickvals=list(positions.values()),
                ticktext=ordered_labels,
                row=row,
                col=col,
            )

        fig.update_yaxes(type="log", title_text=result.rank_metric, row=row, col=col)

    fig.update_layout(
        title=(
            f"W&B sweep metric scatterplots<br>{html.escape(result.sweep_path)}"
            f"<br>metric={html.escape(result.rank_metric)} (log scale)"
        ),
        height=max(550, 300 * nrows),
        autosize=True,
        margin=dict(l=50, r=30, t=90, b=60),
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e8edf7"),
        legend=dict(font=dict(color="#e8edf7")),
    )
    return fig


def build_html_report(
    result: SweepAnalysisResult,
    *,
    histogram_fig: go.Figure,
    scatter_fig: go.Figure,
    include_plotlyjs: str | bool,
) -> str:
    summary_cards = f"""
<div class="summary-grid">
  <div class="card"><div class="k">Sweep</div><div class="v">{html.escape(result.sweep_path)}</div></div>
  <div class="card"><div class="k">Rank Metric</div><div class="v">{html.escape(result.rank_metric)}</div></div>
  <div class="card"><div class="k">Goal</div><div class="v">{html.escape(result.rank_goal)}</div></div>
  <div class="card"><div class="k">Runs Loaded</div><div class="v">{result.runs_loaded}</div></div>
  <div class="card"><div class="k">Skipped For Ranking</div><div class="v">{result.skipped_runs}</div></div>
  <div class="card"><div class="k">Swept Variables</div><div class="v">{len(result.variable_specs)}</div></div>
</div>
"""
    top_rows = "\n".join(
        f'<tr><td>{idx}</td><td><a href="{html.escape(run_url_for_record(result.sweep_path, rec))}">{html.escape(rec.name or rec.run_id)}</a></td>'
        f"<td>{html.escape(rec.run_id)}</td><td>{html.escape(rec.state)}</td><td>{rec.score if rec.score is not None else ''}</td></tr>"
        for idx, rec in enumerate(result.top_runs, start=1)
    )
    variable_rows = "\n".join(
        f"<tr><td>{html.escape(spec['display_name'])}</td><td>{html.escape(spec['kind'])}</td>"
        f"<td>{html.escape(str(spec['distribution'])) if spec['distribution'] is not None else ''}</td>"
        f"<td>{len(spec['unique_values'])}</td><td>{html.escape(', '.join(str(value) for value in spec['unique_values'][:4]))}</td></tr>"
        for spec in result.variable_specs
    )
    state_rows = "\n".join(
        f"<tr><td>{html.escape(state)}</td><td>{count}</td></tr>"
        for state, count in sorted(result.state_counts.items())
    )

    hist_html = pio.to_html(
        histogram_fig,
        include_plotlyjs=include_plotlyjs,
        full_html=False,
        default_width="100%",
        default_height="100%",
        div_id="histograms",
        config={"responsive": True},
    )
    scatter_html = pio.to_html(
        scatter_fig,
        include_plotlyjs=False,
        full_html=False,
        default_width="100%",
        default_height="100%",
        div_id="scatterplots",
        config={"responsive": True},
    )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>W&B Sweep Report</title>
  <style>
    :root {{
      color-scheme: dark;
    }}
    body {{
      font-family: ui-sans-serif, system-ui, sans-serif;
      margin: 0;
      background:
        radial-gradient(circle at top left, rgba(59, 130, 246, 0.14), transparent 30%),
        radial-gradient(circle at top right, rgba(16, 185, 129, 0.10), transparent 26%),
        linear-gradient(180deg, #0b1020 0%, #0f172a 42%, #111827 100%);
      color: #e5e7eb;
    }}
    .page {{ max-width: 1500px; margin: 0 auto; padding: 24px; }}
    h1, h2 {{ margin: 0 0 12px; }}
    .lead {{ color: #a8b3c7; margin-bottom: 24px; }}
    .summary-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 12px; margin-bottom: 24px; }}
    .card {{
      background: rgba(15, 23, 42, 0.82);
      border: 1px solid rgba(148, 163, 184, 0.15);
      border-radius: 14px;
      padding: 14px 16px;
      box-shadow: 0 20px 45px rgba(0, 0, 0, 0.24);
      backdrop-filter: blur(10px);
    }}
    .k {{ font-size: 12px; text-transform: uppercase; letter-spacing: 0.08em; color: #94a3b8; margin-bottom: 6px; }}
    .v {{ font-size: 16px; font-weight: 600; word-break: break-word; color: #f8fafc; }}
    .panel {{
      background: rgba(15, 23, 42, 0.82);
      border: 1px solid rgba(148, 163, 184, 0.15);
      border-radius: 18px;
      padding: 18px;
      box-shadow: 0 20px 45px rgba(0, 0, 0, 0.24);
      margin-bottom: 20px;
      backdrop-filter: blur(10px);
      overflow-x: hidden;
    }}
    .panel .plotly, .panel .plotly-graph-div, .panel .js-plotly-plot {{
      width: 100% !important;
      max-width: 100% !important;
    }}
    table {{ width: 100%; border-collapse: collapse; }}
    th, td {{ text-align: left; padding: 10px 8px; border-bottom: 1px solid rgba(148, 163, 184, 0.18); vertical-align: top; }}
    th {{ font-size: 12px; text-transform: uppercase; letter-spacing: 0.06em; color: #94a3b8; }}
    td {{ font-size: 14px; color: #e5e7eb; }}
    a {{ color: #7dd3fc; text-decoration: none; }}
    a:hover {{ text-decoration: underline; }}
    code {{
      background: rgba(148, 163, 184, 0.12);
      color: #f8fafc;
      padding: 2px 6px;
      border-radius: 6px;
    }}
  </style>
</head>
<body>
  <div class="page">
    <h1>W&amp;B Sweep Report</h1>
    <div class="lead">Interactive report for <code>{html.escape(result.sweep_path)}</code></div>
    {summary_cards}
    <div class="panel">
      <h2>Run States</h2>
      <table>
        <thead><tr><th>State</th><th>Count</th></tr></thead>
        <tbody>{state_rows}</tbody>
      </table>
    </div>
    <div class="panel">
      <h2>Top Runs</h2>
      <table>
        <thead><tr><th>Rank</th><th>Run</th><th>ID</th><th>State</th><th>{html.escape(result.rank_metric)}</th></tr></thead>
        <tbody>{top_rows}</tbody>
      </table>
    </div>
    <div class="panel">
      <h2>Variables</h2>
      <table>
        <thead><tr><th>Name</th><th>Kind</th><th>Distribution</th><th>Unique Values</th><th>Examples</th></tr></thead>
        <tbody>{variable_rows}</tbody>
      </table>
    </div>
    <div class="panel">
      <h2>Parameter Distributions</h2>
      {hist_html}
    </div>
    <div class="panel">
      <h2>Metric vs Parameter</h2>
      {scatter_html}
    </div>
  </div>
</body>
</html>"""


def _row_col(index: int, ncols: int) -> tuple[int, int]:
    zero = index - 1
    return zero // ncols + 1, zero % ncols + 1
