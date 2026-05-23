from __future__ import annotations

import html
import math
import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
import torch
from plotly.subplots import make_subplots

from analysis.wandb_sweep_core import (
    RunRecord,
    SweepAnalysisResult,
    categorical_jitter,
    coerce_numeric,
    compute_numeric_ci_band_from_points,
    filter_records_within_orders_of_magnitude,
    display_value,
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
    top_runs = filter_records_within_orders_of_magnitude(
        result.top_runs,
        rank_goal=result.rank_goal,
    )
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
            for top_idx, run in enumerate(top_runs):
                value = coerce_numeric(run.config.get(spec["key"]))
                if value is None:
                    continue
                x_pos = value
                if len(spec["unique_values"]) < 8:
                    x_pos = top_run_marker_x(
                        value,
                        top_idx,
                        len(top_runs),
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
            for top_idx, run in enumerate(top_runs):
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
                    len(top_runs),
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
    records = filter_records_within_orders_of_magnitude(
        result.records,
        rank_goal=result.rank_goal,
    )
    top_runs = filter_records_within_orders_of_magnitude(
        result.top_runs,
        rank_goal=result.rank_goal,
    )
    ci_runs = filter_records_within_orders_of_magnitude(
        result.ci_runs,
        rank_goal=result.rank_goal,
    )
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
    top_run_ids = {run.run_id for run in top_runs}
    ci_run_ids = {run.run_id for run in ci_runs}

    for idx, spec in enumerate(variable_specs, start=1):
        row, col = _row_col(idx, ncols)
        key = spec["key"]
        if spec["kind"] == "numeric":
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
                for record in records
                if record.config.get(key) is not None
                and record.score is not None
                and record.score > 0
            ]
            ordered_labels = sorted(set(labels))
            positions = {label: idx for idx, label in enumerate(ordered_labels)}
            non_top_points: list[tuple[RunRecord, float, float, str]] = []
            top_points: list[tuple[RunRecord, float, float, str]] = []
            for record in records:
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


def discover_evaluation_manifests(
    evaluation_cache_root: Path,
) -> list[dict[str, Any]]:
    manifests: list[dict[str, Any]] = []
    if not evaluation_cache_root.exists():
        return manifests
    for manifest_path in evaluation_cache_root.rglob("evaluation_manifest.json"):
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        manifests.append(
            {
                "manifest_path": manifest_path,
                "bundle_dir": manifest_path.parent,
                "payload": payload,
            }
        )
    return manifests


def prepare_run_page_index(
    output_dir: Path,
    ranked_records: list[RunRecord],
    evaluation_cache_root: Path,
) -> dict[str, dict[str, Any]]:
    manifests = discover_evaluation_manifests(evaluation_cache_root)
    run_page_index: dict[str, dict[str, Any]] = {}
    assets_root = output_dir / "run_assets"
    assets_root.mkdir(parents=True, exist_ok=True)
    for record in ranked_records:
        page_info: dict[str, Any] = {"href": f"runs/{record.run_id}.html"}
        manifest = find_manifest_for_run(record, manifests)
        if manifest is not None:
            evaluation = copy_evaluation_bundle_assets(
                manifest,
                assets_root / record.run_id,
                record.run_id,
            )
            page_info["evaluation"] = evaluation
        run_page_index[record.run_id] = page_info
    return run_page_index


def find_manifest_for_run(
    record: RunRecord,
    manifests: list[dict[str, Any]],
) -> dict[str, Any] | None:
    for manifest in manifests:
        source = manifest.get("payload", {}).get("source", {})
        if str(source.get("run_id") or "") == record.run_id:
            return manifest
    return None


def find_manifest_for_checkpoint(
    checkpoint_path: Path,
    manifests: list[dict[str, Any]],
) -> dict[str, Any] | None:
    resolved = str(checkpoint_path.expanduser().resolve())
    for manifest in manifests:
        source = manifest.get("payload", {}).get("source", {})
        manifest_checkpoint = source.get("checkpoint_path") or manifest.get(
            "payload", {}
        ).get("checkpoint")
        if not manifest_checkpoint:
            continue
        try:
            manifest_resolved = str(
                Path(str(manifest_checkpoint)).expanduser().resolve()
            )
        except Exception:
            manifest_resolved = str(manifest_checkpoint)
        if manifest_resolved == resolved:
            return manifest
    return None


def find_manifest_for_run_id(
    run_id: str,
    manifests: list[dict[str, Any]],
) -> dict[str, Any] | None:
    for manifest in manifests:
        source = manifest.get("payload", {}).get("source", {})
        if str(source.get("run_id") or "") == run_id:
            return manifest
    return None


def copy_evaluation_bundle_assets(
    manifest: dict[str, Any],
    destination_dir: Path,
    asset_namespace: str,
    *,
    href_prefix: str = "../run_assets",
) -> dict[str, Any]:
    source_dir = Path(manifest["bundle_dir"])
    destination_dir.mkdir(parents=True, exist_ok=True)
    preferred_images = [
        "band_structure_comparison.png",
        "band_structure_prediction.png",
        "dos_comparison.png",
        "dos_prediction.png",
        "dos_error.png",
        "hamiltonian_first_atoms_comparison.png",
        "hamiltonian_first_atoms_prediction.png",
        "hamiltonian_correlation.png",
        "density_first_atoms_comparison.png",
        "density_first_atoms_prediction.png",
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
                "label": label_for_asset(filename),
                "href": f"{href_prefix}/{asset_namespace}/{filename}",
            }
        )
    extra_files = [
        "evaluation_manifest.json",
        "band_structure_gt_gt_overlap.pt",
        "band_structure_pred_gt_overlap.pt",
        "band_structure_gt.pt",
        "band_structure_pred.pt",
        "pred_hamiltonian.pt",
        "pred_density.pt",
        "pred_overlap.pt",
        "structure_metadata.pt",
        "model_input.pt",
    ]
    copied_files: dict[str, Path] = {}
    for filename in extra_files:
        source_path = source_dir / filename
        if not source_path.exists():
            continue
        target_path = destination_dir / filename
        shutil.copy2(source_path, target_path)
        copied_files[filename] = target_path
        file_assets.append(
            {
                "label": filename,
                "href": f"{href_prefix}/{asset_namespace}/{filename}",
            }
        )
    return {
        "source_dir": str(source_dir),
        "asset_dir": str(destination_dir),
        "manifest": manifest.get("payload", {}),
        "image_assets": image_assets,
        "file_assets": file_assets,
        "copied_files": {name: str(path) for name, path in copied_files.items()},
    }


def label_for_asset(filename: str) -> str:
    mapping = {
        "band_structure_comparison.png": "Band structure comparison",
        "band_structure_prediction.png": "Band structure prediction",
        "dos_comparison.png": "DOS comparison",
        "dos_prediction.png": "DOS prediction",
        "dos_error.png": "DOS error",
        "hamiltonian_first_atoms_comparison.png": "Hamiltonian heatmap",
        "hamiltonian_first_atoms_prediction.png": "Hamiltonian prediction heatmap",
        "hamiltonian_correlation.png": "Hamiltonian correlation",
        "density_first_atoms_comparison.png": "Density heatmap",
        "density_first_atoms_prediction.png": "Density prediction heatmap",
        "density_correlation.png": "Density correlation",
        "overlap_correlation.png": "Overlap correlation",
    }
    return mapping.get(filename, filename)


def build_html_report(
    result: SweepAnalysisResult,
    *,
    histogram_fig: go.Figure,
    scatter_fig: go.Figure,
    include_plotlyjs: str | bool,
    run_page_index: dict[str, dict[str, Any]] | None = None,
) -> str:
    run_page_index = run_page_index or {}
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
        f"<tr><td>{idx}</td>"
        f"<td>{_run_page_link(rec, run_page_index)}</td>"
        f'<td><a href="{html.escape(run_url_for_record(result.sweep_path, rec))}">wandb</a></td>'
        f"<td>{html.escape(rec.run_id)}</td><td>{html.escape(rec.state)}</td><td>{rec.score if rec.score is not None else ''}</td>"
        f"<td>{html.escape(_evaluation_status_label(rec, run_page_index))}</td></tr>"
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
        <thead><tr><th>Rank</th><th>Run Page</th><th>W&amp;B</th><th>ID</th><th>State</th><th>{html.escape(result.rank_metric)}</th><th>Evaluation</th></tr></thead>
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


def build_run_detail_page(
    result: SweepAnalysisResult,
    record: RunRecord,
    *,
    include_plotlyjs: str | bool,
    run_page_index: dict[str, dict[str, Any]] | None = None,
) -> str:
    external_url = run_url_for_record(result.sweep_path, record)
    return build_model_detail_page(
        record,
        include_plotlyjs=include_plotlyjs,
        run_page_index=run_page_index,
        rank_metric=result.rank_metric,
        external_url=external_url,
        backlink_href="../sweep_report.html",
        backlink_label="Back to sweep report",
    )


def build_model_detail_page(
    record: RunRecord,
    *,
    include_plotlyjs: str | bool,
    run_page_index: dict[str, dict[str, Any]] | None = None,
    rank_metric: str | None = None,
    external_url: str | None = None,
    backlink_href: str | None = None,
    backlink_label: str | None = None,
) -> str:
    run_page_index = run_page_index or {}
    page_info = run_page_index.get(record.run_id, {})
    evaluation = page_info.get("evaluation", {})
    image_assets = evaluation.get("image_assets", [])
    file_assets = evaluation.get("file_assets", [])
    manifest = evaluation.get("manifest", {})
    settings = manifest.get("settings", {})
    config_rows = "\n".join(
        f"<tr><td>{html.escape(str(key))}</td><td>{html.escape(str(display_value(value)))}</td></tr>"
        for key, value in sorted(record.config.items())
    )
    band_html = _build_interactive_band_section(
        evaluation,
        include_plotlyjs=include_plotlyjs,
        div_id_prefix=f"band-{record.run_id}",
    )
    if image_assets:
        evaluation_html = "\n".join(
            f"""
<div class="asset-card">
  <div class="asset-title">{html.escape(asset['label'])}</div>
  <a href="{html.escape(asset['href'])}" target="_blank" rel="noopener noreferrer">
    <img src="{html.escape(asset['href'])}" alt="{html.escape(asset['label'])}">
  </a>
</div>
"""
            for asset in image_assets
        )
        downloads_html = "\n".join(
            f'<li><a href="{html.escape(asset["href"])}">{html.escape(asset["label"])}</a></li>'
            for asset in file_assets
        )
        settings_rows = "\n".join(
            f"<tr><td>{html.escape(str(key))}</td><td>{html.escape(str(display_value(value)))}</td></tr>"
            for key, value in sorted(settings.items())
        )
        settings_section = (
            f"""
  <h2>Evaluation Settings</h2>
  <table>
    <thead><tr><th>Key</th><th>Value</th></tr></thead>
    <tbody>{settings_rows}</tbody>
  </table>
"""
            if settings_rows
            else ""
        )
        evaluation_section = f"""
<div class="panel">
  <h2>Precomputed Evaluation</h2>
  <div class="meta-line">Source bundle: <code>{html.escape(str(evaluation.get("source_dir", "")))}</code></div>
  {band_html}
  <div class="asset-grid">
    {evaluation_html}
  </div>
  <h2>Downloads</h2>
  <ul class="downloads">
    {downloads_html}
  </ul>
  {settings_section}
</div>
"""
    else:
        evaluation_section = """
<div class="panel">
  <h2>Precomputed Evaluation</h2>
  <p class="empty">No precomputed local evaluation bundle was found for this run yet.</p>
</div>
"""
    score_label = rank_metric or "Score"
    backlink_html = (
        f'<a class="crumb" href="{html.escape(backlink_href)}">{html.escape(backlink_label or "Back")}</a>'
        if backlink_href is not None
        else ""
    )
    external_html = (
        f'<a class="cta" href="{html.escape(external_url)}" target="_blank" rel="noopener noreferrer">Open in W&amp;B</a>'
        if external_url
        else ""
    )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(record.name or record.run_id)} - W&amp;B Run Report</title>
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
    .topbar {{ display: flex; gap: 12px; flex-wrap: wrap; align-items: center; margin-bottom: 20px; }}
    .crumb, .cta {{
      display: inline-flex; align-items: center; gap: 8px; padding: 10px 14px;
      border-radius: 999px; border: 1px solid rgba(148, 163, 184, 0.18);
      background: rgba(15, 23, 42, 0.82); color: #c8d4ea; text-decoration: none;
    }}
    .page h1, h2 {{ margin: 0 0 12px; }}
    .lead {{ color: #a8b3c7; margin-bottom: 24px; }}
    .summary-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 12px; margin-bottom: 24px; }}
    .card, .panel {{
      background: rgba(15, 23, 42, 0.82);
      border: 1px solid rgba(148, 163, 184, 0.15);
      border-radius: 18px;
      padding: 18px;
      box-shadow: 0 20px 45px rgba(0, 0, 0, 0.24);
      backdrop-filter: blur(10px);
      margin-bottom: 20px;
    }}
    .card {{ border-radius: 14px; padding: 14px 16px; margin-bottom: 0; }}
    .k {{ font-size: 12px; text-transform: uppercase; letter-spacing: 0.08em; color: #94a3b8; margin-bottom: 6px; }}
    .v {{ font-size: 16px; font-weight: 600; word-break: break-word; color: #f8fafc; }}
    table {{ width: 100%; border-collapse: collapse; }}
    th, td {{ text-align: left; padding: 10px 8px; border-bottom: 1px solid rgba(148, 163, 184, 0.18); vertical-align: top; }}
    th {{ font-size: 12px; text-transform: uppercase; letter-spacing: 0.06em; color: #94a3b8; }}
    td {{ font-size: 14px; color: #e5e7eb; }}
    a {{ color: #7dd3fc; text-decoration: none; }}
    a:hover {{ text-decoration: underline; }}
    code {{ background: rgba(148, 163, 184, 0.12); color: #f8fafc; padding: 2px 6px; border-radius: 6px; }}
    .asset-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 16px; }}
    .plotly-panel {{
      margin-bottom: 18px;
      overflow-x: hidden;
    }}
    .plotly-panel .plotly, .plotly-panel .plotly-graph-div, .plotly-panel .js-plotly-plot {{
      width: 100% !important;
      max-width: 100% !important;
    }}
    .asset-card {{
      background: rgba(9, 14, 28, 0.72);
      border: 1px solid rgba(148, 163, 184, 0.14);
      border-radius: 16px;
      padding: 14px;
    }}
    .asset-title {{ font-weight: 600; margin-bottom: 10px; color: #f8fafc; }}
    .asset-card img {{ display: block; width: 100%; height: auto; border-radius: 10px; }}
    .downloads {{ margin: 0; padding-left: 18px; }}
    .meta-line {{ color: #a8b3c7; margin-bottom: 14px; }}
    .empty {{ color: #a8b3c7; }}
  </style>
</head>
<body>
  <div class="page">
    <div class="topbar">
      {backlink_html}
      {external_html}
    </div>
    <h1>{html.escape(record.name or record.run_id)}</h1>
    <div class="lead">Run detail page for <code>{html.escape(record.run_id)}</code></div>
    <div class="summary-grid">
      <div class="card"><div class="k">Metric</div><div class="v">{html.escape(score_label)}</div></div>
      <div class="card"><div class="k">Score</div><div class="v">{record.score if record.score is not None else "n/a"}</div></div>
      <div class="card"><div class="k">State</div><div class="v">{html.escape(record.state)}</div></div>
      <div class="card"><div class="k">Evaluation</div><div class="v">{html.escape(_evaluation_status_label(record, run_page_index))}</div></div>
    </div>
    {evaluation_section}
    <div class="panel">
      <h2>Config</h2>
      <table>
        <thead><tr><th>Key</th><th>Value</th></tr></thead>
        <tbody>{config_rows}</tbody>
      </table>
    </div>
  </div>
</body>
</html>"""


def _run_page_link(record: RunRecord, run_page_index: dict[str, dict[str, Any]]) -> str:
    page = run_page_index.get(record.run_id, {})
    href = page.get("href", f"runs/{record.run_id}.html")
    label = html.escape(record.name or record.run_id)
    return f'<a href="{html.escape(href)}">{label}</a>'


def _evaluation_status_label(
    record: RunRecord, run_page_index: dict[str, dict[str, Any]]
) -> str:
    page = run_page_index.get(record.run_id, {})
    evaluation = page.get("evaluation", {})
    if evaluation.get("image_assets"):
        return "precomputed"
    return "missing"


def _row_col(index: int, ncols: int) -> tuple[int, int]:
    zero = index - 1
    return zero // ncols + 1, zero % ncols + 1


def _build_interactive_band_section(
    evaluation: dict[str, Any],
    *,
    include_plotlyjs: str | bool,
    div_id_prefix: str,
) -> str:
    figure = _build_band_structure_figure(evaluation)
    if figure is None:
        return ""
    band_html = pio.to_html(
        figure,
        include_plotlyjs=include_plotlyjs,
        full_html=False,
        default_width="100%",
        default_height="100%",
        div_id=f"{div_id_prefix}-band",
        config={"responsive": True},
    )
    return f"""
  <h2>Interactive Band Structure</h2>
  <div class="plotly-panel">
    {band_html}
  </div>
"""


def _build_band_structure_figure(evaluation: dict[str, Any]) -> go.Figure | None:
    copied_files = evaluation.get("copied_files", {})
    gt_path = copied_files.get("band_structure_gt_gt_overlap.pt") or copied_files.get(
        "band_structure_gt.pt"
    )
    pred_path = copied_files.get(
        "band_structure_pred_gt_overlap.pt"
    ) or copied_files.get("band_structure_pred.pt")
    if gt_path is None and pred_path is None:
        return None

    payloads: list[tuple[str, dict[str, Any], str]] = []
    if gt_path is not None:
        payloads.append(("Ground truth", _load_band_payload(Path(gt_path)), "#e5e7eb"))
    if pred_path is not None:
        payloads.append(("Prediction", _load_band_payload(Path(pred_path)), "#60a5fa"))

    fig = make_subplots(
        rows=1,
        cols=len(payloads),
        shared_yaxes=True,
        subplot_titles=[title for title, _payload, _color in payloads],
    )

    for idx, (title, payload, color) in enumerate(payloads, start=1):
        linear_k = np.asarray(payload["linear_k"], dtype=float)
        energies = _band_energies_ev(payload)
        for band_idx in range(energies.shape[1]):
            fig.add_trace(
                go.Scattergl(
                    x=linear_k,
                    y=energies[:, band_idx],
                    mode="lines",
                    line=dict(color=color, width=1.1),
                    opacity=0.28 if title != "Ground truth" else 0.42,
                    hoverinfo="skip",
                    showlegend=False,
                ),
                row=1,
                col=idx,
            )
        tick_positions = np.asarray(payload["tick_positions"], dtype=float).tolist()
        tick_labels = [str(label) for label in payload["tick_labels"]]
        fig.update_xaxes(
            tickmode="array",
            tickvals=tick_positions,
            ticktext=tick_labels,
            row=1,
            col=idx,
        )
        for xpos in tick_positions:
            fig.add_vline(
                x=xpos,
                line_color="rgba(148,163,184,0.25)",
                line_width=1,
                row=1,
                col=idx,
            )

    fig.update_yaxes(title_text="Energy relative to Fermi (eV)", row=1, col=1)
    fig.update_layout(
        height=420,
        autosize=True,
        margin=dict(l=50, r=25, t=55, b=40),
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e8edf7"),
    )
    return fig


def _load_band_payload(path: Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise TypeError(f"Unexpected band payload type in {path}: {type(payload)!r}")
    return payload


def _band_energies_ev(payload: dict[str, Any]) -> np.ndarray:
    energies = payload["eigenvalues"]
    if torch.is_tensor(energies):
        energies = energies.detach().cpu().numpy()
    energies = np.asarray(energies, dtype=float) * 27.211386245988
    fermi = payload.get("fermi_level")
    if fermi is not None:
        if torch.is_tensor(fermi):
            fermi = float(fermi.detach().cpu().item())
        energies = energies - float(fermi) * 27.211386245988
    return energies
