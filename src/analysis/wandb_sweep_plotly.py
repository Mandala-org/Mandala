from __future__ import annotations

import html
import math
import json
import re
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

from matplotlib import cm as mpl_cm
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
import torch
from plotly.offline.offline import get_plotlyjs
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
                    opacity=1.0,
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
                    opacity=1.0,
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
                    opacity=1.0,
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
                    opacity=1.0,
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
                                opacity=1.0,
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
                            fillcolor="#e5e7eb",
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
                            line=dict(color="#6b7280", width=1.2),
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
                                opacity=1.0,
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
                            opacity=1.0,
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
                            opacity=1.0,
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
        "band_structure_and_dos_comparison.png",
        "band_structure_prediction.png",
        "dos_comparison_and_error.png",
        "eigenvalue_correlation.png",
        "dos_comparison.png",
        "dos_prediction.png",
        "dos_error.png",
        "hamiltonian_first_atoms_comparison.png",
        "hamiltonian_first_atoms_prediction.png",
        "density_first_atoms_comparison.png",
        "density_first_atoms_prediction.png",
    ]
    preferred_images.extend(
        path.name
        for path in sorted(source_dir.glob("*.png"))
        if path.name not in preferred_images
        and "sum_pbc" not in path.name
        and "shift_resolved" not in path.name
    )
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
        "evaluation_matrix_metrics.json",
        "band_structure_gt_gt_overlap.pt",
        "band_structure_pred_gt_overlap.pt",
        "band_structure_gt.pt",
        "band_structure_pred.pt",
        "dos_comparison.pt",
        "dos_prediction.pt",
        "tetrahedron_dos_cache_gt.pt",
        "tetrahedron_dos_cache_pred.pt",
        "dos_comparison.png",
        "dos_prediction.png",
        "dos_error.png",
        "hamiltonian_correlation.pt",
        "hamiltonian_correlation.png",
        "hamiltonian_block_error_metrics.pt",
        "hamiltonian_interactive_heatmaps.pt",
        "density_interactive_heatmaps.pt",
        "snapshot_3d_error_payload.pt",
        "snapshot_3d_error_payload_hamiltonian.pt",
        "snapshot_3d_error_payload_density.pt",
        "snapshot_3d_error_payload_overlap.pt",
        "overlap_interactive_heatmaps.pt",
        "density_block_error_metrics.pt",
        "density_correlation.pt",
        "density_correlation.png",
        "overlap_correlation.pt",
        "overlap_correlation.png",
        "overlap_block_error_metrics.pt",
        "eigenvalue_correlation.png",
        "pred_hamiltonian.pt",
        "pred_raw_hamiltonian.pt",
        "pred_density.pt",
        "pred_raw_density.pt",
        "pred_overlap.pt",
        "pred_raw_overlap.pt",
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
        "band_structure_and_dos_comparison.png": "Band structure and DOS comparison",
        "band_structure_prediction.png": "Band structure prediction",
        "dos_comparison.png": "DOS comparison",
        "dos_prediction.png": "DOS prediction",
        "dos_error.png": "DOS error",
        "dos_comparison.pt": "DOS comparison payload",
        "dos_prediction.pt": "DOS prediction payload",
        "tetrahedron_dos_cache_gt.pt": "Ground-truth DOS eigenvalue cache",
        "tetrahedron_dos_cache_pred.pt": "Prediction DOS eigenvalue cache",
        "hamiltonian_block_error_metrics.pt": "Hamiltonian block error metrics",
        "hamiltonian_interactive_heatmaps.pt": "Hamiltonian interactive heatmaps",
        "density_interactive_heatmaps.pt": "Density interactive heatmaps",
        "snapshot_3d_error_payload.pt": "Snapshot 3D error payload",
        "density_block_error_metrics.pt": "Density block error metrics",
        "hamiltonian_first_atoms_comparison.png": "Hamiltonian heatmap",
        "hamiltonian_first_atoms_prediction.png": "Hamiltonian prediction heatmap",
        "hamiltonian_correlation.png": "Hamiltonian correlation",
        "hamiltonian_correlation.pt": "Hamiltonian correlation payload",
        "pred_raw_hamiltonian.pt": "Raw predicted Hamiltonian blocks",
        "pred_raw_density.pt": "Raw predicted density blocks",
        "pred_raw_overlap.pt": "Raw predicted overlap blocks",
        "density_first_atoms_comparison.png": "Density heatmap",
        "density_first_atoms_prediction.png": "Density prediction heatmap",
        "density_correlation.png": "Density correlation",
        "density_correlation.pt": "Density correlation payload",
        "overlap_correlation.png": "Overlap correlation",
        "overlap_correlation.pt": "Overlap correlation payload",
    }
    return mapping.get(filename, filename)


def _load_evaluation_matrix_metrics(evaluation: dict[str, Any]) -> dict[str, Any]:
    path = evaluation.get("copied_files", {}).get("evaluation_matrix_metrics.json")
    if path is None:
        return {}
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _available_evaluation_matrices(
    evaluation: dict[str, Any], matrix_metrics: dict[str, Any]
) -> list[str]:
    copied_files = evaluation.get("copied_files", {})
    metric_matrices = matrix_metrics.get("matrices", {})
    available: list[str] = []
    for matrix_name in ("hamiltonian", "overlap", "density"):
        has_snapshot = bool(
            copied_files.get(f"snapshot_3d_error_payload_{matrix_name}.pt")
            or (
                matrix_name == "hamiltonian"
                and copied_files.get("snapshot_3d_error_payload.pt")
            )
        )
        has_heatmap = bool(copied_files.get(f"{matrix_name}_interactive_heatmaps.pt"))
        if matrix_name in metric_matrices or has_snapshot or has_heatmap:
            available.append(matrix_name)
    return available


def _format_report_metric(value: Any) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{float(value):.6g}"
    except (TypeError, ValueError):
        return html.escape(str(value))


def _build_matrix_metrics_section(matrix_metrics: dict[str, Any]) -> str:
    matrices = matrix_metrics.get("matrices", {})
    rows = []
    for matrix_name in ("hamiltonian", "overlap", "density"):
        values = matrices.get(matrix_name)
        if not isinstance(values, dict):
            continue
        units = values.get("units", {})
        rows.append(
            "<tr>"
            f"<td>{html.escape(matrix_name.title())}</td>"
            f"<td>{_format_report_metric(values.get('mae'))} {html.escape(str(units.get('mae', '')))}</td>"
            f"<td>{_format_report_metric(values.get('mse'))} {html.escape(str(units.get('mse', '')))}</td>"
            f"<td>{int(values.get('scalar_count', 0)):,}</td>"
            "</tr>"
        )
    if not rows:
        return ""
    definition = matrix_metrics.get("definition", "")
    definition_html = (
        f'<div class="meta-line">Definition: {html.escape(str(definition))}</div>'
        if definition
        else ""
    )
    return f"""
  <h2>Matrix Error Metrics</h2>
  {definition_html}
  <table>
    <thead><tr><th>Matrix</th><th>MAE</th><th>MSE</th><th>Scalar elements</th></tr></thead>
    <tbody>{''.join(rows)}</tbody>
  </table>
"""


def _build_matrix_selector_sections(
    run_id: str,
    matrix_names: list[str],
    snapshot_sections: dict[str, str],
    heatmap_sections: dict[str, str],
) -> tuple[str, str]:
    active_names = [
        name
        for name in matrix_names
        if snapshot_sections.get(name) or heatmap_sections.get(name)
    ]
    if not active_names:
        return "", ""
    safe_run_id = re.sub(r"[^A-Za-z0-9_-]+", "-", run_id)
    selector_id = f"evaluation-matrix-{safe_run_id}"
    options = "".join(
        f'<option value="{html.escape(name)}">{html.escape(name.title())}</option>'
        for name in active_names
    )
    selector = f"""
  <div class="matrix-selector">
    <label for="{html.escape(selector_id)}">Analyzed matrix</label>
    <select id="{html.escape(selector_id)}">{options}</select>
  </div>
"""
    panels = "".join(
        f'<div class="evaluation-matrix-panel" data-evaluation-matrix="{html.escape(name)}"'
        f'{"" if idx == 0 else " hidden"}>'
        f'{snapshot_sections.get(name, "")}{heatmap_sections.get(name, "")}</div>'
        for idx, name in enumerate(active_names)
    )
    script = f"""
  <script>
  (function() {{
    const selector = document.getElementById({json.dumps(selector_id)});
    if (!selector) return;
    const panels = Array.from(document.querySelectorAll(".evaluation-matrix-panel"));
    function updateMatrix() {{
      panels.forEach((panel) => {{
        panel.hidden = panel.dataset.evaluationMatrix !== selector.value;
      }});
      window.setTimeout(function() {{
        window.dispatchEvent(new Event("resize"));
      }}, 0);
    }}
    selector.addEventListener("change", updateMatrix);
    updateMatrix();
  }})();
  </script>
"""
    return selector, panels + script


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
    matrix_metrics = _load_evaluation_matrix_metrics(evaluation)
    config_rows = "\n".join(
        f"<tr><td>{html.escape(str(key))}</td><td>{html.escape(str(display_value(value)))}</td></tr>"
        for key, value in sorted(record.config.items())
    )
    matrix_names = _available_evaluation_matrices(evaluation, matrix_metrics)
    if include_plotlyjs in (True, "inline"):
        plotly_loader_html = f"<script>{get_plotlyjs()}</script>"
    elif include_plotlyjs == "cdn":
        plotly_loader_html = (
            '<script src="https://cdn.plot.ly/plotly-3.0.1.min.js"></script>'
        )
    else:
        plotly_loader_html = ""
    snapshot_sections: dict[str, str] = {}
    for matrix_name in matrix_names:
        section = _build_snapshot_3d_section(
            evaluation,
            include_plotlyjs=False,
            div_id_prefix=f"snapshot-3d-{matrix_name}-{record.run_id}",
            matrix_name=matrix_name,
        )
        snapshot_sections[matrix_name] = section
    heatmap_sections: dict[str, str] = {}
    for matrix_name in matrix_names:
        section = _build_hamiltonian_heatmap_section(
            evaluation,
            include_plotlyjs=False,
            div_id_prefix=f"matrix-heatmap-{matrix_name}-{record.run_id}",
            matrix_name=matrix_name,
        )
        heatmap_sections[matrix_name] = section
    matrix_selector_html, matrix_analysis_html = _build_matrix_selector_sections(
        record.run_id,
        matrix_names,
        snapshot_sections,
        heatmap_sections,
    )
    dos_html = _build_dos_section(
        evaluation,
        include_plotlyjs=False,
        div_id_prefix=f"dos-{record.run_id}",
    )
    band_html = _build_interactive_band_section(
        evaluation,
        include_plotlyjs=False,
        div_id_prefix=f"band-{record.run_id}",
    )
    eigenvalue_html = _build_eigenvalue_correlation_section(
        evaluation,
        include_plotlyjs=False,
        div_id_prefix=f"eigenvalue-{record.run_id}",
    )
    correlation_html = _build_correlation_section(
        evaluation,
        include_plotlyjs=False,
        div_id_prefix=f"corr-{record.run_id}",
    )
    block_error_html = _build_block_error_section(
        evaluation,
        include_plotlyjs=False,
        div_id_prefix=f"block-{record.run_id}",
    )
    matrix_metrics_html = _build_matrix_metrics_section(matrix_metrics)
    if image_assets or file_assets or matrix_analysis_html or matrix_metrics_html:
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
  {matrix_selector_html}
  {matrix_metrics_html}
  {matrix_analysis_html}
  {dos_html}
  {band_html}
  {eigenvalue_html}
  {correlation_html}
  {block_error_html}
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
    if block_error_html and not image_assets:
        evaluation_section += block_error_html
    report_hmae = matrix_metrics.get("metrics", {}).get("val/hamiltonian_mae")
    score_label = (
        "val/hamiltonian_mae" if report_hmae is not None else (rank_metric or "Score")
    )
    score_value = report_hmae if report_hmae is not None else record.score
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
  {plotly_loader_html}
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
    .metric-grid {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 14px;
    }}
    .correlation-grid {{
      grid-template-columns: repeat(auto-fit, minmax(620px, 1fr));
    }}
    .metric-card {{
      background: rgba(9, 14, 28, 0.72);
      border: 1px solid rgba(148, 163, 184, 0.14);
      border-radius: 16px;
      padding: 14px;
    }}
    .metric-title {{ font-weight: 600; margin-bottom: 8px; color: #f8fafc; }}
    .axis-controls {{
      display: flex;
      gap: 14px;
      flex-wrap: wrap;
      margin-bottom: 10px;
      color: #cbd5e1;
      font-size: 13px;
    }}
    .axis-toggle {{ display: inline-flex; align-items: center; gap: 6px; }}
    .metric-plot .plotly, .metric-plot .plotly-graph-div, .metric-plot .js-plotly-plot {{
      width: 100% !important;
      max-width: 100% !important;
    }}
    .heatmap-layout {{
      display: grid;
      grid-template-columns: minmax(0, 3.2fr) minmax(300px, 1.25fr);
      gap: 16px;
      align-items: start;
    }}
    .heatmap-main {{
      min-width: 0;
    }}
    .heatmap-side {{
      min-width: 0;
      position: sticky;
      top: 18px;
    }}
    .worst-edge-table {{
      width: 100%;
      border-collapse: collapse;
    }}
    .worst-edge-table th,
    .worst-edge-table td {{
      padding: 8px 6px;
      font-size: 12px;
      border-bottom: 1px solid rgba(148, 163, 184, 0.15);
    }}
    .worst-edge-table th {{
      color: #94a3b8;
      text-transform: uppercase;
      letter-spacing: 0.05em;
    }}
    .worst-edge-table td {{
      color: #e5e7eb;
      word-break: break-word;
    }}
    .heatmap-controls {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(170px, 1fr));
      gap: 12px;
      margin-bottom: 14px;
    }}
    .heatmap-controls label {{
      display: flex;
      flex-direction: column;
      gap: 6px;
      color: #cbd5e1;
      font-size: 13px;
    }}
    .heatmap-controls select,
    .heatmap-controls input,
    .heatmap-controls button {{
      border-radius: 10px;
      border: 1px solid rgba(148, 163, 184, 0.18);
      background: rgba(9, 14, 28, 0.82);
      color: #f8fafc;
      padding: 10px 12px;
      font-size: 14px;
    }}
    .snapshot3d-controls {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(170px, 1fr));
      gap: 12px;
      margin-bottom: 14px;
    }}
    .snapshot3d-controls label {{
      display: flex;
      flex-direction: column;
      gap: 6px;
      color: #cbd5e1;
      font-size: 13px;
    }}
    .snapshot3d-controls select,
    .snapshot3d-controls input {{
      border-radius: 10px;
      border: 1px solid rgba(148, 163, 184, 0.18);
      background: rgba(9, 14, 28, 0.82);
      color: #f8fafc;
      padding: 10px 12px;
      font-size: 14px;
    }}
    .snapshot3d-inline {{
      display: flex;
      align-items: center;
      gap: 8px;
      padding-top: 22px;
      color: #cbd5e1;
      font-size: 13px;
    }}
    .snapshot3d-inline input[type="checkbox"] {{
      width: 16px;
      height: 16px;
    }}
    .matrix-selector {{
      display: flex;
      align-items: center;
      gap: 12px;
      margin: 4px 0 18px;
      color: #cbd5e1;
    }}
    .matrix-selector select {{
      min-width: 220px;
      border-radius: 10px;
      border: 1px solid rgba(148, 163, 184, 0.18);
      background: rgba(9, 14, 28, 0.82);
      color: #f8fafc;
      padding: 10px 12px;
      font-size: 14px;
    }}
    .evaluation-matrix-panel[hidden] {{ display: none; }}
    .heatmap-controls button {{
      cursor: pointer;
      align-self: end;
    }}
    .heatmap-caption {{
      color: #a8b3c7;
      margin-bottom: 10px;
      min-height: 20px;
    }}
    @media (max-width: 980px) {{
      .heatmap-layout {{
        grid-template-columns: 1fr;
      }}
      .heatmap-side {{
        position: static;
      }}
      .metric-grid {{
        grid-template-columns: 1fr;
      }}
    }}
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
      <div class="card"><div class="k">Score</div><div class="v">{_format_report_metric(score_value)}</div></div>
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


def _bwr_plotly_colorscale(samples: int = 33) -> list[list[Any]]:
    cmap = mpl_cm.get_cmap("bwr")
    return [
        [
            float(idx / max(samples - 1, 1)),
            f"rgb({int(r*255)},{int(g*255)},{int(b*255)})",
        ]
        for idx, (r, g, b, _a) in enumerate(cmap(np.linspace(0.0, 1.0, samples)))
    ]


def _sequential_plotly_colorscale(
    name: str = "turbo", samples: int = 33
) -> list[list[Any]]:
    cmap = mpl_cm.get_cmap(name)
    return [
        [
            float(idx / max(samples - 1, 1)),
            f"rgb({int(r*255)},{int(g*255)},{int(b*255)})",
        ]
        for idx, (r, g, b, _a) in enumerate(cmap(np.linspace(0.0, 1.0, samples)))
    ]


def _load_snapshot_3d_payload(path: Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise TypeError(
            f"Unexpected snapshot 3D payload type in {path}: {type(payload)!r}"
        )
    return payload


def _build_snapshot_3d_section(
    evaluation: dict[str, Any],
    *,
    include_plotlyjs: str | bool,
    div_id_prefix: str,
    matrix_name: str = "hamiltonian",
) -> str:
    copied_files = evaluation.get("copied_files", {})
    payload_path = copied_files.get(f"snapshot_3d_error_payload_{matrix_name}.pt")
    if payload_path is None and matrix_name == "hamiltonian":
        payload_path = copied_files.get("snapshot_3d_error_payload.pt")
    if payload_path is None:
        return ""
    payload = _load_snapshot_3d_payload(Path(payload_path))
    colorscale = _sequential_plotly_colorscale("turbo")
    plot_div_id = f"{div_id_prefix}-plot"
    metric_id = f"{div_id_prefix}-metric"
    scale_id = f"{div_id_prefix}-scale"
    threshold_id = f"{div_id_prefix}-threshold"
    threshold_value_id = f"{div_id_prefix}-threshold-value"
    ghosts_id = f"{div_id_prefix}-ghosts"
    nodes_id = f"{div_id_prefix}-nodes"
    edges_id = f"{div_id_prefix}-edges"
    figure_html = pio.to_html(
        go.Figure(),
        include_plotlyjs=include_plotlyjs,
        full_html=False,
        default_width="100%",
        default_height="720px",
        div_id=plot_div_id,
        config={"responsive": True},
    )
    payload_json = json.dumps(payload)
    colorscale_json = json.dumps(colorscale)
    return f"""
  <h2>Snapshot 3D Error View</h2>
  <div class="snapshot3d-controls">
    <label>Metric
      <select id="{html.escape(metric_id)}">
        <option value="abs_mae">Absolute MAE</option>
        <option value="rel_mae">Relative error</option>
      </select>
    </label>
    <label>Scale
      <select id="{html.escape(scale_id)}">
        <option value="linear">Linear</option>
        <option value="log">Log10</option>
      </select>
    </label>
    <label>Threshold
      <input id="{html.escape(threshold_id)}" type="range" min="0" max="1" step="0.001" value="0">
    </label>
    <label>Threshold value
      <input id="{html.escape(threshold_value_id)}" type="text" value="0" readonly>
    </label>
    <label class="snapshot3d-inline">
      <input id="{html.escape(ghosts_id)}" type="checkbox">
      Show ghost nodes
    </label>
    <label class="snapshot3d-inline">
      <input id="{html.escape(nodes_id)}" type="checkbox">
      Show nodes
    </label>
    <label class="snapshot3d-inline">
      <input id="{html.escape(edges_id)}" type="checkbox" checked>
      Show edges
    </label>
  </div>
  <div class="plotly-panel">
    {figure_html}
  </div>
  <script>
  (function() {{
    const payload = {payload_json};
    const colorscale = {colorscale_json};
    const metricEl = document.getElementById({json.dumps(metric_id)});
    const scaleEl = document.getElementById({json.dumps(scale_id)});
    const thresholdEl = document.getElementById({json.dumps(threshold_id)});
    const thresholdValueEl = document.getElementById({json.dumps(threshold_value_id)});
    const ghostsEl = document.getElementById({json.dumps(ghosts_id)});
    const nodesEl = document.getElementById({json.dumps(nodes_id)});
    const edgesEl = document.getElementById({json.dumps(edges_id)});
    const plotDiv = document.getElementById({json.dumps(plot_div_id)});

    function lerp(a, b, t) {{
      return a + (b - a) * t;
    }}

    function parseRgb(text) {{
      const match = String(text).match(/rgb\\((\\d+),(\\d+),(\\d+)\\)/);
      if (!match) return [255, 255, 255];
      return [Number(match[1]), Number(match[2]), Number(match[3])];
    }}

    function colorForValue(t) {{
      const clamped = Math.max(0, Math.min(1, t));
      for (let idx = 1; idx < colorscale.length; idx += 1) {{
        const left = colorscale[idx - 1];
        const right = colorscale[idx];
        if (clamped <= right[0]) {{
          const span = Math.max(right[0] - left[0], 1e-12);
          const localT = (clamped - left[0]) / span;
          const a = parseRgb(left[1]);
          const b = parseRgb(right[1]);
          const rgb = [
            Math.round(lerp(a[0], b[0], localT)),
            Math.round(lerp(a[1], b[1], localT)),
            Math.round(lerp(a[2], b[2], localT)),
          ];
          return `rgb(${{rgb[0]}},${{rgb[1]}},${{rgb[2]}})`;
        }}
      }}
      return colorscale[colorscale.length - 1][1];
    }}

    function boxSegments(box) {{
      const a = box[0];
      const b = box[1];
      const c = box[2];
      const corners = [
        [0, 0, 0],
        a,
        b,
        c,
        [a[0] + b[0], a[1] + b[1], a[2] + b[2]],
        [a[0] + c[0], a[1] + c[1], a[2] + c[2]],
        [b[0] + c[0], b[1] + c[1], b[2] + c[2]],
        [a[0] + b[0] + c[0], a[1] + b[1] + c[1], a[2] + b[2] + c[2]],
      ];
      const edgePairs = [
        [0, 1], [0, 2], [0, 3], [1, 4], [1, 5], [2, 4],
        [2, 6], [3, 5], [3, 6], [4, 7], [5, 7], [6, 7],
      ];
      const x = [];
      const y = [];
      const z = [];
      edgePairs.forEach(([u, v]) => {{
        x.push(corners[u][0], corners[v][0], null);
        y.push(corners[u][1], corners[v][1], null);
        z.push(corners[u][2], corners[v][2], null);
      }});
      return {{
        type: "scatter3d",
        mode: "lines",
        x: x,
        y: y,
        z: z,
        line: {{color: "rgb(148,163,184)", width: 4}},
        hoverinfo: "skip",
        showlegend: false,
      }};
    }}

    function metricConfig() {{
      const metric = metricEl.value;
      const stats = payload.stats || {{}};
      const edgeMax = Number(stats[metric === "abs_mae" ? "edge_abs_max" : "edge_rel_max"] || 0);
      const nodeMax = Number(stats[metric === "abs_mae" ? "node_abs_max" : "node_rel_max"] || 0);
      const minPosEdge = Number(stats[metric === "abs_mae" ? "edge_abs_min_positive" : "edge_rel_min_positive"] || 1e-12);
      const minPosNode = Number(stats[metric === "abs_mae" ? "node_abs_min_positive" : "node_rel_min_positive"] || 1e-12);
      const maxValue = Math.max(edgeMax, nodeMax, minPosEdge, minPosNode);
      const minPositive = Math.max(Math.min(minPosEdge, minPosNode, maxValue), 1e-12);
      const logMin = Math.log10(minPositive) - 1.0;
      const logMax = Math.log10(Math.max(maxValue, minPositive * 10.0));
      return {{
        metric: metric,
        useLog: scaleEl.value === "log",
        maxValue: maxValue,
        minPositive: minPositive,
        logMin: logMin,
        logMax: Math.max(logMax, logMin + 1e-6),
      }};
    }}

    function sharedMetricValues(cfg, threshold) {{
      const values = [];
      if (edgesEl && edgesEl.checked) {{
        (payload.edges || []).forEach((edge) => {{
          const raw = Number(edge[cfg.metric] || 0);
          if (raw >= threshold) values.push(transformedValue(raw, cfg));
        }});
      }}
      if (nodesEl && nodesEl.checked) {{
        (payload.node_diagonal || []).forEach((node) => {{
          values.push(transformedValue(node[cfg.metric], cfg));
        }});
      }}
      if (ghostsEl && ghostsEl.checked) {{
        const nodeMap = new Map((payload.node_diagonal || []).map((node) => [Number(node.atom), node]));
        (payload.ghosts || []).forEach((ghost) => {{
          const metricSource = nodeMap.get(Number(ghost.atom)) || {{abs_mae: 0, rel_mae: 0}};
          values.push(transformedValue(metricSource[cfg.metric], cfg));
        }});
      }}
      return values;
    }}

    function thresholdValue(cfg) {{
      const t = Number(thresholdEl.value || 0);
      const exponent = cfg.logMin + t * (cfg.logMax - cfg.logMin);
      return Math.pow(10, exponent);
    }}

    function transformedValue(raw, cfg) {{
      const clipped = Math.max(Number(raw || 0), cfg.minPositive);
      return cfg.useLog ? Math.log10(clipped) : Number(raw || 0);
    }}

    function transformedRange(cfg, values) {{
      if (values && values.length) {{
        let minVal = Math.min(...values);
        let maxVal = Math.max(...values);
        if (maxVal <= minVal) maxVal = minVal + 1e-6;
        return [minVal, maxVal];
      }}
      if (cfg.useLog) {{
        return [Math.log10(cfg.minPositive), Math.log10(Math.max(cfg.maxValue, cfg.minPositive))];
      }}
      return [0, Math.max(cfg.maxValue, cfg.minPositive)];
    }}

    function edgeHover(edge) {{
      return [
        `edge=${{edge.src_atom}} -> ${{edge.dst_atom}}`,
        `shift=(${{edge.shift[0]}}, ${{edge.shift[1]}}, ${{edge.shift[2]}})`,
        `length=${{Number(edge.edge_length).toFixed(3)}}`,
        `abs_mae=${{Number(edge.abs_mae).toExponential(3)}}`,
        `rel_mae=${{Number(edge.rel_mae).toExponential(3)}}`,
        edge.missing_pred ? "prediction missing -> zero-filled for error" : "",
      ].filter(Boolean).join("<br>");
    }}

    function nodeHover(node) {{
      return [
        `atom=${{node.atom}}`,
        `abs_mae=${{Number(node.abs_mae).toExponential(3)}}`,
        `rel_mae=${{Number(node.rel_mae).toExponential(3)}}`,
        node.missing_pred ? "prediction missing -> zero-filled for error" : "",
      ].filter(Boolean).join("<br>");
    }}

    function ghostHover(ghost) {{
      return [
        `ghost atom=${{ghost.atom}}`,
        `shift=(${{ghost.shift[0]}}, ${{ghost.shift[1]}}, ${{ghost.shift[2]}})`,
        "value=0",
      ].join("<br>");
    }}

    function buildColorbarTrace(cfg, threshold) {{
      const visibleValues = sharedMetricValues(cfg, threshold);
      if (!visibleValues.length) return null;
      const [cmin, cmax] = transformedRange(cfg, visibleValues);
      return {{
        type: "scatter3d",
        mode: "markers",
        x: [0],
        y: [0],
        z: [0],
        hoverinfo: "skip",
        showlegend: false,
        marker: {{
          size: 0.1,
          opacity: 1.0,
          color: [cmin],
          colorscale: colorscale,
          cmin: cmin,
          cmax: cmax,
          showscale: true,
          colorbar: {{
            title: `${{cfg.useLog ? "log10 " : ""}}${{cfg.metric}}`,
            len: 0.8,
            y: 0.5,
          }},
        }},
      }};
    }}

    function buildEdgeTraces(cfg, threshold) {{
      const visibleEdges = (payload.edges || []).filter((edge) => Number(edge[cfg.metric] || 0) >= threshold);
      if (!visibleEdges.length) return [];
      const [cmin, cmax] = transformedRange(cfg, sharedMetricValues(cfg, threshold));
      const binCount = 18;
      const bins = Array.from({{length: binCount}}, () => ({{x: [], y: [], z: [], hover: [], values: []}}));
      visibleEdges.forEach((edge) => {{
        const value = transformedValue(edge[cfg.metric], cfg);
        const t = cmax <= cmin ? 0.5 : (value - cmin) / (cmax - cmin);
        const binIdx = Math.max(0, Math.min(binCount - 1, Math.floor(t * (binCount - 1))));
        bins[binIdx].x.push(edge.start[0], edge.end[0], null);
        bins[binIdx].y.push(edge.start[1], edge.end[1], null);
        bins[binIdx].z.push(edge.start[2], edge.end[2], null);
        bins[binIdx].hover.push(edgeHover(edge), edgeHover(edge), null);
        bins[binIdx].values.push(value);
      }});
      return bins
        .map((bin, idx) => {{
          if (!bin.x.length) return null;
          const binValue = Math.max(...bin.values);
          const color = colorForValue(cmax <= cmin ? 0.5 : (binValue - cmin) / (cmax - cmin));
          return {{
            type: "scatter3d",
            mode: "lines",
            x: bin.x,
            y: bin.y,
            z: bin.z,
            text: bin.hover,
            hovertemplate: "%{{text}}<extra></extra>",
            line: {{color: color, width: 4}},
            opacity: 1.0,
            showlegend: false,
          }};
        }})
        .filter(Boolean);
    }}

    function buildNodeTrace(cfg, showScale) {{
      if (!(nodesEl && nodesEl.checked)) return null;
      const sharedValues = sharedMetricValues(cfg, Number(thresholdEl.value || 0));
      const [cmin, cmax] = transformedRange(
        cfg,
        sharedValues.length ? sharedValues : [transformedValue(0, cfg)]
      );
      const nodes = payload.node_diagonal || [];
      return {{
        type: "scatter3d",
        mode: "markers",
        x: nodes.map((node) => node.position[0]),
        y: nodes.map((node) => node.position[1]),
        z: nodes.map((node) => node.position[2]),
        text: nodes.map((node) => nodeHover(node)),
        hovertemplate: "%{{text}}<extra></extra>",
        marker: {{
          size: 6,
          color: nodes.map((node) => transformedValue(node[cfg.metric], cfg)),
          colorscale: colorscale,
          cmin: cmin,
          cmax: cmax,
          opacity: 1.0,
          showscale: showScale,
          colorbar: showScale ? {{
            title: `${{cfg.useLog ? "log10 " : ""}}${{cfg.metric}}`,
            len: 0.8,
            y: 0.5,
          }} : undefined,
        }},
        name: "Atoms",
      }};
    }}

    function buildGhostTrace(cfg, showScale) {{
      if (!ghostsEl.checked) return null;
      const ghosts = payload.ghosts || [];
      if (!ghosts.length) return null;
      return {{
        type: "scatter3d",
        mode: "markers",
        x: ghosts.map((ghost) => ghost.position[0]),
        y: ghosts.map((ghost) => ghost.position[1]),
        z: ghosts.map((ghost) => ghost.position[2]),
        text: ghosts.map((ghost) => ghostHover(ghost)),
        hovertemplate: "%{{text}}<extra></extra>",
        marker: {{
          size: 4,
          symbol: "diamond",
          color: "rgb(148,163,184)",
          opacity: 1.0,
          line: {{color: "rgb(255,255,255)", width: 1}},
        }},
        name: "Ghosts",
      }};
    }}

    function render() {{
      if (!window.Plotly || !plotDiv) return;
      const cfg = metricConfig();
      const threshold = thresholdValue(cfg);
      thresholdValueEl.value = threshold.toExponential(3);
      const edgesVisible = !!(edgesEl && edgesEl.checked);
      const nodesVisible = !!(nodesEl && nodesEl.checked);
      const ghostsVisible = !!(ghostsEl && ghostsEl.checked);
      const traces = [
        boxSegments(payload.box || [[0,0,0],[1,0,0],[0,1,0]]),
        ...(edgesVisible ? buildEdgeTraces(cfg, threshold) : []),
      ];
      const colorbarTrace = edgesVisible
        ? buildColorbarTrace(cfg, threshold)
        : (nodesVisible
            ? null
            : (ghostsVisible ? null : null));
      if (colorbarTrace) traces.push(colorbarTrace);
      const nodeTrace = buildNodeTrace(cfg, !edgesVisible);
      if (nodeTrace) traces.push(nodeTrace);
      const ghostTrace = buildGhostTrace(cfg, !edgesVisible && !nodesVisible);
      if (ghostTrace) traces.push(ghostTrace);
      const layout = {{
        height: 720,
        margin: {{l: 0, r: 0, t: 10, b: 0}},
        template: "plotly_dark",
        paper_bgcolor: "rgba(0,0,0,0)",
        plot_bgcolor: "rgba(0,0,0,0)",
        font: {{color: "#e8edf7"}},
        scene: {{
          aspectmode: "data",
          xaxis: {{title: "x", backgroundcolor: "rgba(0,0,0,0)"}},
          yaxis: {{title: "y", backgroundcolor: "rgba(0,0,0,0)"}},
          zaxis: {{title: "z", backgroundcolor: "rgba(0,0,0,0)"}},
          camera: {{eye: {{x: 1.55, y: 1.55, z: 1.15}}}},
        }},
        showlegend: false,
      }};
      Plotly.react(plotDiv, traces, layout, {{responsive: true}});
    }}

    metricEl.addEventListener("change", render);
    scaleEl.addEventListener("change", render);
    thresholdEl.addEventListener("input", render);
    ghostsEl.addEventListener("change", render);
    if (nodesEl) nodesEl.addEventListener("change", render);
    if (edgesEl) edgesEl.addEventListener("change", render);
    render();
  }})();
  </script>
"""


def _load_hamiltonian_heatmap_payload(path: Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise TypeError(
            f"Unexpected Hamiltonian heatmap payload type in {path}: {type(payload)!r}"
        )
    return payload


def _fallback_heatmap_cutout(payload: dict[str, Any]) -> dict[str, Any] | None:
    for family in ("shift_resolved",):
        family_payload = payload.get(family, {})
        if family_payload.get("all") is not None:
            return family_payload["all"]
        for key in ("worst_abs", "worst_rel"):
            cutout = family_payload.get(key)
            if cutout is not None:
                return cutout
        closest = family_payload.get("closest_neighbors", {})
        if closest:
            first_key = sorted(closest.keys(), key=lambda value: int(value))[0]
            return closest[first_key]
        random_items = family_payload.get("random", [])
        if random_items:
            return random_items[0]
    return None


def _build_hamiltonian_heatmap_figure(
    cutout: dict[str, Any],
    *,
    colorscale: list[list[Any]],
    clim: float,
) -> go.Figure:
    axes = cutout.get("axes", {})
    tick_positions = axes.get("tick_positions", [])
    tick_labels = axes.get("labels", [])
    subplot_titles = ["Ground truth", "Prediction", "Difference"]
    fig = make_subplots(rows=1, cols=3, subplot_titles=subplot_titles)
    for idx, key in enumerate(("gt", "pred", "diff"), start=1):
        axis_suffix = "" if idx == 1 else str(idx)
        fig.add_trace(
            go.Heatmap(
                z=cutout.get(key, []),
                zmin=-float(clim),
                zmax=float(clim),
                colorscale=colorscale,
                colorbar=dict(len=0.78, y=0.5) if idx == 3 else None,
                showscale=idx == 3,
                hovertemplate="row=%{y}<br>col=%{x}<br>value=%{z:.6f}<extra></extra>",
            ),
            row=1,
            col=idx,
        )
        fig.update_xaxes(
            tickmode="array",
            tickvals=tick_positions,
            ticktext=tick_labels,
            tickangle=-35,
            constrain="domain",
            row=1,
            col=idx,
        )
        fig.update_yaxes(
            tickmode="array",
            tickvals=tick_positions,
            ticktext=tick_labels,
            autorange="reversed",
            scaleanchor=f"x{axis_suffix}",
            scaleratio=1,
            constrain="domain",
            row=1,
            col=idx,
        )
    fig.update_layout(
        height=460,
        margin=dict(l=60, r=30, t=60, b=90),
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e8edf7"),
    )
    return fig


def _worst_edge_rows(cutout: dict[str, Any]) -> str:
    rows = []
    for idx, edge in enumerate(cutout.get("worst_edges", []), start=1):
        shift = edge.get("shift")
        shift_text = ""
        if shift is not None:
            shift_text = f" [{shift[0]},{shift[1]},{shift[2]}]"
        edge_label = f"{edge.get('src_atom')} -> {edge.get('dst_atom')}{shift_text}"
        rows.append(
            f"<tr><td>{idx}</td><td>{html.escape(edge_label)}</td>"
            f"<td>{float(edge.get('abs_mae', 0.0)):.4e}</td>"
            f"<td>{float(edge.get('rel_mae', 0.0)):.4e}</td></tr>"
        )
    if not rows:
        return '<tr><td colspan="4" class="empty">No edges available.</td></tr>'
    return "\n".join(rows)


def _build_hamiltonian_heatmap_section(
    evaluation: dict[str, Any],
    *,
    include_plotlyjs: str | bool,
    div_id_prefix: str,
    matrix_name: str = "hamiltonian",
) -> str:
    copied_files = evaluation.get("copied_files", {})
    payload_path = copied_files.get(f"{matrix_name}_interactive_heatmaps.pt")
    if payload_path is None:
        return ""
    payload = _load_hamiltonian_heatmap_payload(Path(payload_path))
    initial_cutout = _fallback_heatmap_cutout(payload)
    if initial_cutout is None:
        return ""

    colorscale = _bwr_plotly_colorscale()
    initial_clim = float(payload.get("default_clim", 0.05))
    max_clim = max(float(payload.get("max_clim", initial_clim)), initial_clim)
    min_clim = max(min(initial_clim / 10.0, max_clim), 1.0e-4)
    heatmap_div_id = f"{div_id_prefix}-plot"
    caption_id = f"{div_id_prefix}-caption"
    family_id = f"{div_id_prefix}-family"
    strategy_id = f"{div_id_prefix}-strategy"
    atom_id = f"{div_id_prefix}-atom"
    clim_id = f"{div_id_prefix}-clim"
    clim_value_id = f"{div_id_prefix}-clim-value"
    random_button_id = f"{div_id_prefix}-randomize"
    edges_body_id = f"{div_id_prefix}-edges-body"
    figure_html = pio.to_html(
        _build_hamiltonian_heatmap_figure(
            initial_cutout,
            colorscale=colorscale,
            clim=initial_clim,
        ),
        include_plotlyjs=include_plotlyjs,
        full_html=False,
        default_width="100%",
        default_height="460px",
        div_id=heatmap_div_id,
        config={"responsive": True},
    )
    payload_json = json.dumps(payload)
    colorscale_json = json.dumps(colorscale)
    initial_edges_html = _worst_edge_rows(initial_cutout)
    log_min = math.log10(min_clim)
    log_max = math.log10(max_clim)
    log_initial = math.log10(initial_clim)
    log_step = max((log_max - log_min) / 200.0, 1.0e-3)
    has_all_atoms = payload.get("shift_resolved", {}).get("all") is not None
    if has_all_atoms:
        selection_options = '<option value="all" selected>All atoms</option>'
        side_panel_html = ""
    else:
        selection_options = """
        <option value="worst_abs">Worst matrix error</option>
        <option value="worst_rel">Worst relative error</option>
        <option value="closest_neighbors">Closest neighbors</option>
        <option value="random">Random</option>
        """
        side_panel_html = f"""
    <div class="heatmap-side">
      <div class="metric-card">
        <div class="metric-title">Worst Edges</div>
        <table class="worst-edge-table">
          <thead><tr><th>#</th><th>Edge</th><th>Abs</th><th>Rel</th></tr></thead>
          <tbody id="{html.escape(edges_body_id)}">
            {initial_edges_html}
          </tbody>
        </table>
      </div>
    </div>
        """
    return f"""
  <h2>Interactive matrix heatmaps</h2>
  <div class="heatmap-controls">
    <label>View
      <select id="{html.escape(family_id)}">
        <option value="shift_resolved">Shift resolved</option>
      </select>
    </label>
    <label>Selection
      <select id="{html.escape(strategy_id)}">
        {selection_options}
      </select>
    </label>
    <label>Anchor atom
      <input id="{html.escape(atom_id)}" type="number" min="0" max="{int(payload.get('atom_count', 0)) - 1}" value="0">
    </label>
    <label>Clim
      <input id="{html.escape(clim_id)}" type="range" min="{log_min:.6f}" max="{log_max:.6f}" step="{log_step:.6f}" value="{log_initial:.6f}">
    </label>
    <label>Clim value
      <input id="{html.escape(clim_value_id)}" type="text" value="{initial_clim:.3e}" readonly>
    </label>
    <button id="{html.escape(random_button_id)}" type="button">Randomize</button>
  </div>
  <div class="heatmap-caption" id="{html.escape(caption_id)}"></div>
  <div class="heatmap-layout">
    <div class="plotly-panel heatmap-main">
      {figure_html}
    </div>
    {side_panel_html}
  </div>
  <script>
  (function() {{
    const payload = {payload_json};
    const colorscale = {colorscale_json};
    const familyEl = document.getElementById({json.dumps(family_id)});
    const strategyEl = document.getElementById({json.dumps(strategy_id)});
    const atomEl = document.getElementById({json.dumps(atom_id)});
    const climEl = document.getElementById({json.dumps(clim_id)});
    const climValueEl = document.getElementById({json.dumps(clim_value_id)});
    const randomButton = document.getElementById({json.dumps(random_button_id)});
    const captionEl = document.getElementById({json.dumps(caption_id)});
    const edgesBody = document.getElementById({json.dumps(edges_body_id)});
    const plotDiv = document.getElementById({json.dumps(heatmap_div_id)});
    let randomIndex = 0;

    function firstClosestKey(familyPayload) {{
      const keys = Object.keys((familyPayload && familyPayload.closest_neighbors) || {{}});
      if (!keys.length) return null;
      return keys.sort((a, b) => Number(a) - Number(b))[0];
    }}

    function fallbackCutout(familyPayload) {{
      if (!familyPayload) return null;
      if (familyPayload.worst_abs) return familyPayload.worst_abs;
      if (familyPayload.worst_rel) return familyPayload.worst_rel;
      const closestKey = firstClosestKey(familyPayload);
      if (closestKey !== null) return familyPayload.closest_neighbors[closestKey];
      if (familyPayload.random && familyPayload.random.length) return familyPayload.random[0];
      return null;
    }}

    function currentCutout() {{
      const familyPayload = payload[familyEl.value] || payload.shift_resolved || {{}};
      if (strategyEl.value === "closest_neighbors") {{
        const key = String(Math.max(0, Math.min(Number(atomEl.value || 0), Number(payload.atom_count || 1) - 1)));
        return (familyPayload.closest_neighbors || {{}})[key] || fallbackCutout(familyPayload);
      }}
      if (strategyEl.value === "random") {{
        const items = familyPayload.random || [];
        if (!items.length) return fallbackCutout(familyPayload);
        return items[randomIndex % items.length];
      }}
      return familyPayload[strategyEl.value] || fallbackCutout(familyPayload);
    }}

    function heatmapTrace(z, clim, showscale, xaxisName, yaxisName) {{
      return {{
        type: "heatmap",
        z: z,
        zmin: -clim,
        zmax: clim,
        colorscale: colorscale,
        xaxis: xaxisName,
        yaxis: yaxisName,
        showscale: showscale,
        colorbar: showscale ? {{len: 0.78, y: 0.5}} : undefined,
        hovertemplate: "row=%{{y}}<br>col=%{{x}}<br>value=%{{z:.6f}}<extra></extra>",
      }};
    }}

    function buildLayout(cutout) {{
      const axes = cutout.axes || {{}};
      const tickvals = axes.tick_positions || [];
      const ticktext = axes.labels || [];
      return {{
        height: 460,
        margin: {{l: 60, r: 30, t: 60, b: 90}},
        template: "plotly_dark",
        paper_bgcolor: "rgba(0,0,0,0)",
        plot_bgcolor: "rgba(0,0,0,0)",
        font: {{color: "#e8edf7"}},
        grid: {{rows: 1, columns: 3, pattern: "independent"}},
        xaxis: {{tickmode: "array", tickvals: tickvals, ticktext: ticktext, tickangle: -35, constrain: "domain"}},
        xaxis2: {{tickmode: "array", tickvals: tickvals, ticktext: ticktext, tickangle: -35, constrain: "domain"}},
        xaxis3: {{tickmode: "array", tickvals: tickvals, ticktext: ticktext, tickangle: -35, constrain: "domain"}},
        yaxis: {{tickmode: "array", tickvals: tickvals, ticktext: ticktext, autorange: "reversed", scaleanchor: "x", scaleratio: 1, constrain: "domain"}},
        yaxis2: {{tickmode: "array", tickvals: tickvals, ticktext: ticktext, autorange: "reversed", scaleanchor: "x2", scaleratio: 1, constrain: "domain"}},
        yaxis3: {{tickmode: "array", tickvals: tickvals, ticktext: ticktext, autorange: "reversed", scaleanchor: "x3", scaleratio: 1, constrain: "domain"}},
        annotations: [
          {{text: "Ground truth", x: 0.145, y: 1.08, xref: "paper", yref: "paper", showarrow: false, font: {{size: 14}}}},
          {{text: "Prediction", x: 0.5, y: 1.08, xref: "paper", yref: "paper", showarrow: false, font: {{size: 14}}}},
          {{text: "Difference", x: 0.855, y: 1.08, xref: "paper", yref: "paper", showarrow: false, font: {{size: 14}}}},
        ],
      }};
    }}

    function render() {{
      const cutout = currentCutout();
      if (!cutout || !window.Plotly || !plotDiv) return;
      const clim = Math.pow(10, Number(climEl.value || Math.log10(payload.default_clim || 0.05)));
      climValueEl.value = clim.toExponential(3);
      const data = [
        heatmapTrace(cutout.gt, clim, false, "x", "y"),
        heatmapTrace(cutout.pred, clim, false, "x2", "y2"),
        heatmapTrace(cutout.diff, clim, true, "x3", "y3"),
      ];
      Plotly.react(plotDiv, data, buildLayout(cutout), {{responsive: true}});
      if (captionEl) {{
        captionEl.textContent = cutout.selection_label || "";
      }}
      if (edgesBody) {{
        const rows = (cutout.worst_edges || []).map((edge, idx) => {{
          const shift = edge.shift ? " [" + edge.shift[0] + "," + edge.shift[1] + "," + edge.shift[2] + "]" : "";
          return "<tr><td>" + (idx + 1) + "</td><td>" + edge.src_atom + " -> " + edge.dst_atom + shift + "</td><td>" + Number(edge.abs_mae).toExponential(4) + "</td><td>" + Number(edge.rel_mae).toExponential(4) + "</td></tr>";
        }});
        edgesBody.innerHTML = rows.length ? rows.join("") : '<tr><td colspan="4" class="empty">No edges available.</td></tr>';
      }}
      const showAtom = strategyEl.value === "closest_neighbors";
      const showRandom = strategyEl.value === "random";
      atomEl.parentElement.style.display = showAtom ? "flex" : "none";
      randomButton.style.display = showRandom ? "block" : "none";
    }}

    familyEl.addEventListener("change", render);
    strategyEl.addEventListener("change", render);
    atomEl.addEventListener("change", render);
    atomEl.addEventListener("input", render);
    climEl.addEventListener("input", render);
    randomButton.addEventListener("click", function() {{
      randomIndex += 1;
      render();
    }});
    render();
  }})();
  </script>
"""


def _build_interactive_band_section(
    evaluation: dict[str, Any],
    *,
    include_plotlyjs: str | bool,
    div_id_prefix: str,
) -> str:
    settings = evaluation.get("manifest", {}).get("settings", {})
    figure = _build_band_structure_figure(
        evaluation,
        emin_ev=_coerce_optional_float(settings.get("band_emin_ev")),
        emax_ev=_coerce_optional_float(settings.get("band_emax_ev")),
    )
    if figure is None:
        return ""
    band_html = pio.to_html(
        figure,
        include_plotlyjs=include_plotlyjs,
        full_html=False,
        default_width="100%",
        default_height="420px",
        div_id=f"{div_id_prefix}-band",
        config={"responsive": True},
    )
    return f"""
  <h2>Interactive Band Structure</h2>
  <div class="plotly-panel">
    {band_html}
  </div>
"""


def _build_band_structure_figure(
    evaluation: dict[str, Any],
    *,
    emin_ev: float | None = None,
    emax_ev: float | None = None,
) -> go.Figure | None:
    copied_files = evaluation.get("copied_files", {})
    gt_path = copied_files.get("band_structure_gt_gt_overlap.pt") or copied_files.get(
        "band_structure_gt.pt"
    )
    pred_path = copied_files.get(
        "band_structure_pred_gt_overlap.pt"
    ) or copied_files.get("band_structure_pred.pt")
    if gt_path is None and pred_path is None:
        return None
    spectral_title = _evaluation_spectral_title(evaluation)
    plot_style = _evaluation_plot_style(evaluation)

    payloads: list[tuple[str, dict[str, Any], str]] = []
    if gt_path is not None:
        payloads.append(("Ground truth", _load_band_payload(Path(gt_path)), "#FFFFFF"))
    if pred_path is not None:
        payloads.append(
            (
                "Prediction",
                _load_band_payload(Path(pred_path)),
                plot_style["prediction"],
            )
        )

    fig = make_subplots(
        rows=1,
        cols=1,
    )

    for _idx, (trace_name, payload, color) in enumerate(payloads, start=1):
        linear_k = np.asarray(payload["linear_k"], dtype=float)
        energies = _band_energies_ev(payload)
        for band_idx in range(energies.shape[1]):
            fig.add_trace(
                go.Scattergl(
                    x=linear_k,
                    y=energies[:, band_idx],
                    mode="lines",
                    line=dict(
                        color=color,
                        width=1.1,
                        dash="solid" if trace_name == "Ground truth" else "dash",
                    ),
                    opacity=1.0,
                    name=trace_name,
                    hoverinfo="skip",
                    showlegend=band_idx == 0,
                ),
                row=1,
                col=1,
            )
        tick_positions = np.asarray(payload["tick_positions"], dtype=float).tolist()
        tick_labels = [str(label) for label in payload["tick_labels"]]
        fig.update_xaxes(
            tickmode="array",
            tickvals=tick_positions,
            ticktext=tick_labels,
            row=1,
            col=1,
        )
        for xpos in tick_positions:
            fig.add_vline(
                x=xpos,
                line_color="#94a3b8",
                line_width=1,
                row=1,
                col=1,
            )

    if payloads:
        first_payload = payloads[0][1]
        linear_k = np.asarray(first_payload["linear_k"], dtype=float)
        fig.add_trace(
            go.Scattergl(
                x=[float(linear_k[0]), float(linear_k[-1])],
                y=[0.0, 0.0],
                mode="lines",
                line=dict(color="#FFFFFF", width=1.2, dash="dot"),
                name="Fermi level",
                hoverinfo="skip",
                showlegend=True,
            ),
            row=1,
            col=1,
        )

    fig.update_yaxes(title_text="E-E_F (eV)", row=1, col=1)
    if emin_ev is not None and emax_ev is not None:
        fig.update_yaxes(range=[float(emin_ev), float(emax_ev)], row=1, col=1)
    fig.update_layout(
        title=spectral_title,
        height=420,
        autosize=True,
        margin=dict(l=50, r=25, t=55, b=40),
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e8edf7"),
        legend=dict(x=0.99, y=0.99, xanchor="right", yanchor="top"),
    )
    return fig


def _saved_gamma_energies(payload: dict[str, Any]) -> np.ndarray | None:
    """Return the saved band eigenvalues at fractional k = (0, 0, 0)."""
    eigenvalues = _band_energies_ev(payload)
    fractional = payload.get("fractional_kpoints")
    if fractional is None:
        return None
    if torch.is_tensor(fractional):
        fractional = fractional.detach().cpu().numpy()
    fractional = np.asarray(fractional, dtype=float)
    if fractional.ndim != 2 or fractional.shape[0] != eigenvalues.shape[0]:
        return None
    gamma_idx = int(np.argmin(np.linalg.norm(fractional, axis=1)))
    if float(np.linalg.norm(fractional[gamma_idx])) > 1.0e-7:
        return None
    return np.asarray(eigenvalues[gamma_idx], dtype=float)


def _load_dos_eigenvalues(path: Path) -> tuple[np.ndarray, float | None] | None:
    payload = _load_plot_payload(path)
    values = payload.get("eigenvalues_ev")
    if values is None:
        return None
    if torch.is_tensor(values):
        values = values.detach().cpu().numpy()
    values = np.asarray(values, dtype=float)
    if values.ndim != 2:
        return None
    fermi = payload.get("fermi_level_ev")
    return values, None if fermi is None else float(fermi)


def _eigenvalue_correlation_data(
    evaluation: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, str] | None:
    copied_files = evaluation.get("copied_files", {})
    gt_dos = copied_files.get("tetrahedron_dos_cache_gt.pt")
    pred_dos = copied_files.get("tetrahedron_dos_cache_pred.pt")
    if gt_dos is not None and pred_dos is not None:
        gt_payload = _load_dos_eigenvalues(Path(gt_dos))
        pred_payload = _load_dos_eigenvalues(Path(pred_dos))
        if gt_payload is not None and pred_payload is not None:
            gt, gt_fermi = gt_payload
            pred, pred_fermi = pred_payload
            count = min(gt.size, pred.size)
            if count > 0:
                gt = gt.reshape(-1)[:count]
                pred = pred.reshape(-1)[:count]
                if gt_fermi is not None:
                    gt = gt - gt_fermi
                if pred_fermi is not None:
                    pred = pred - pred_fermi
                return gt, pred, "DOS k-mesh"

    gt_path = copied_files.get("band_structure_gt_gt_overlap.pt") or copied_files.get(
        "band_structure_gt.pt"
    )
    pred_path = copied_files.get(
        "band_structure_pred_gt_overlap.pt"
    ) or copied_files.get("band_structure_pred.pt")
    if gt_path is None or pred_path is None:
        return None
    gt = _saved_gamma_energies(_load_band_payload(Path(gt_path)))
    pred = _saved_gamma_energies(_load_band_payload(Path(pred_path)))
    if gt is None or pred is None:
        return None
    count = min(gt.size, pred.size)
    return gt[:count], pred[:count], "Gamma"


def _build_eigenvalue_correlation_figure(
    evaluation: dict[str, Any],
) -> go.Figure | None:
    data = _eigenvalue_correlation_data(evaluation)
    if data is None:
        return None
    target, prediction, source = data
    energy_min = -15.0
    energy_max = 25.0
    finite = (
        np.isfinite(target)
        & np.isfinite(prediction)
        & (target >= energy_min)
        & (target <= energy_max)
        & (prediction >= energy_min)
        & (prediction <= energy_max)
    )
    target = target[finite]
    prediction = prediction[finite]
    if target.size == 0:
        return None
    if target.size > 1 and np.std(target) > 0.0 and np.std(prediction) > 0.0:
        r = float(np.corrcoef(target, prediction)[0, 1])
        r2 = r * r
    else:
        r2 = float("nan")
    mae = float(np.mean(np.abs(prediction - target)))
    counts, x_edges, y_edges = np.histogram2d(
        target,
        prediction,
        bins=60,
        range=((energy_min, energy_max), (energy_min, energy_max)),
    )
    counts = counts.T
    counts[counts <= 0.0] = np.nan
    x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
    y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])
    fig = go.Figure(
        data=[
            go.Heatmap(
                x=x_centers,
                y=y_centers,
                z=counts,
                colorscale=[[0.0, "#FFFFB2"], [0.5, "#FD8D3C"], [1.0, "#BD0026"]],
                colorbar=dict(title="Count"),
                hovertemplate=(
                    "Ground truth=%{x:.5g} eV<br>Prediction=%{y:.5g} eV"
                    "<br>Count=%{z:.0f}<extra></extra>"
                ),
            ),
            go.Scattergl(
                x=target,
                y=prediction,
                mode="markers",
                marker=dict(color="#000000", size=2),
                hoverinfo="skip",
                showlegend=False,
            ),
            go.Scattergl(
                x=[energy_min, energy_max],
                y=[energy_min, energy_max],
                mode="lines",
                line=dict(color="#666666", dash="dash", width=1.0),
                hoverinfo="skip",
                showlegend=False,
            ),
        ]
    )
    r2_text = f"R²={r2:.6f}" if np.isfinite(r2) else "R²=n/a"
    fig.update_layout(
        title=f"Eigenvalue correlation ({source}) | MAE={mae:.4g} eV | {r2_text}",
        xaxis_title="Ground truth E-E_F (eV)",
        yaxis_title="Prediction E-E_F (eV)",
        height=560,
        margin=dict(l=60, r=110, t=60, b=55),
        template="plotly_white",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#FFFFFF",
        font=dict(color="#111111"),
    )
    fig.update_xaxes(range=[energy_min, energy_max])
    fig.update_yaxes(range=[energy_min, energy_max], scaleanchor="x", scaleratio=1)
    return fig


def _build_eigenvalue_correlation_section(
    evaluation: dict[str, Any],
    *,
    include_plotlyjs: str | bool,
    div_id_prefix: str,
) -> str:
    figure = _build_eigenvalue_correlation_figure(evaluation)
    if figure is None:
        return ""
    figure_html = pio.to_html(
        figure,
        include_plotlyjs=include_plotlyjs,
        full_html=False,
        default_width="100%",
        default_height="480px",
        div_id=f"{div_id_prefix}-corr",
        config={"responsive": True},
    )
    return f"""
  <h2>Eigenvalue Correlation</h2>
  <div class="plotly-panel">
    {figure_html}
  </div>
"""


def _build_block_error_section(
    evaluation: dict[str, Any],
    *,
    include_plotlyjs: str | bool,
    div_id_prefix: str,
) -> str:
    copied_files = evaluation.get("copied_files", {})
    block_error_color = _evaluation_plot_style(evaluation)["block_error"]
    metric_files = [
        ("Hamiltonian", copied_files.get("hamiltonian_block_error_metrics.pt")),
        ("Density", copied_files.get("density_block_error_metrics.pt")),
        ("Overlap", copied_files.get("overlap_block_error_metrics.pt")),
    ]
    metric_files = [(name, path) for name, path in metric_files if path is not None]
    if not metric_files:
        return ""

    sections = []
    first_plot_uses_js = True if include_plotlyjs else False
    for idx, (name, path) in enumerate(metric_files, start=1):
        payload = _load_block_error_payload(Path(path))
        section_id = f"{div_id_prefix}-{idx}"
        plots = [
            (
                "Absolute block error vs edge length",
                _make_block_error_figure(
                    payload,
                    x_key="edge_length",
                    y_key="abs_mae",
                    title=f"{name}: absolute block error vs edge length",
                    x_label="Edge length (Angstrom)",
                    y_label="Mean absolute block error",
                    color=block_error_color,
                ),
                False,
                False,
            ),
            (
                "Relative block error vs edge length",
                _make_block_error_figure(
                    payload,
                    x_key="edge_length",
                    y_key="rel_mae",
                    title=f"{name}: relative block error vs edge length",
                    x_label="Edge length (Angstrom)",
                    y_label="Relative block error",
                    color=block_error_color,
                ),
                False,
                False,
            ),
            (
                "Absolute block error vs block magnitude",
                _make_block_error_figure(
                    payload,
                    x_key="block_magnitude",
                    y_key="abs_mae",
                    title=f"{name}: absolute block error vs block magnitude",
                    x_label="Target block magnitude",
                    y_label="Mean absolute block error",
                    color=block_error_color,
                ),
                True,
                False,
            ),
            (
                "Relative block error vs block magnitude",
                _make_block_error_figure(
                    payload,
                    x_key="block_magnitude",
                    y_key="rel_mae",
                    title=f"{name}: relative block error vs block magnitude",
                    x_label="Target block magnitude",
                    y_label="Relative block error",
                    color=block_error_color,
                ),
                True,
                False,
            ),
        ]
        plot_cards = []
        for plot_idx, (label, fig, allow_log_x, allow_log_y) in enumerate(
            plots, start=1
        ):
            plot_div_id = f"{section_id}-plot-{plot_idx}"
            controls_id = f"{plot_div_id}-controls"
            plot_html = pio.to_html(
                fig,
                include_plotlyjs=include_plotlyjs if first_plot_uses_js else False,
                full_html=False,
                default_width="100%",
                default_height="360px",
                div_id=plot_div_id,
                config={"responsive": True},
            )
            first_plot_uses_js = False
            controls = _axis_control_html(
                controls_id,
                allow_log_x=allow_log_x,
                allow_log_y=allow_log_y,
            )
            plot_cards.append(
                f"""
<div class="metric-card">
  <div class="metric-title">{html.escape(label)}</div>
  <div class="axis-controls" id="{html.escape(controls_id)}">
    {controls}
  </div>
  <div class="metric-plot">
    {plot_html}
  </div>
  <script>{_axis_control_script(plot_div_id, controls_id, allow_log_x, allow_log_y)}</script>
</div>
"""
            )
        sections.append(
            f"""
<div class="panel">
  <h2>{html.escape(name)} Block Error Diagnostics</h2>
  <div class="metric-grid">
    {"".join(plot_cards)}
  </div>
</div>
"""
        )
    return "\n".join(sections)


def _load_plot_payload(path: Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise TypeError(f"Unexpected payload type in {path}: {type(payload)!r}")
    return payload


def _build_dos_figure(
    payload: dict[str, Any], *, plot_style: dict[str, str] | None = None
) -> go.Figure | None:
    kind = str(payload.get("kind", ""))
    if kind not in {"dos_comparison", "dos_prediction"}:
        return None
    plot_style = plot_style or {
        "ground_truth": "#000000",
        "prediction": "#D62728",
        "error": "#0072B2",
        "block_error": "#003B73",
    }

    fig = go.Figure()
    if kind == "dos_comparison":
        grid_true = _series_from_payload(payload, "grid_true")
        dos_true = _series_from_payload(payload, "dos_true")
        grid_pred = _series_from_payload(payload, "grid_pred")
        dos_pred = _series_from_payload(payload, "dos_pred")
        dos_error = _series_from_payload(payload, "dos_error")
        fig.add_trace(
            go.Scattergl(
                x=grid_true,
                y=dos_true,
                mode="lines",
                line=dict(color=plot_style["ground_truth"], width=1.9, dash="solid"),
                name="Ground truth",
                showlegend=True,
            )
        )
        fig.add_trace(
            go.Scattergl(
                x=grid_pred,
                y=dos_pred,
                mode="lines",
                line=dict(color=plot_style["prediction"], width=1.6, dash="dash"),
                name="Prediction",
                showlegend=True,
            )
        )
        fig2 = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.08,
            row_heights=[0.72, 0.28],
        )
        for trace in fig.data:
            fig2.add_trace(trace, row=1, col=1)
        fig2.add_trace(
            go.Scattergl(
                x=grid_true,
                y=dos_error,
                mode="lines",
                line=dict(color=plot_style["error"], width=1.4),
                name="Prediction - Ground truth",
                showlegend=False,
            ),
            row=2,
            col=1,
        )
        top_min = float(np.nanmin(np.concatenate([dos_true, dos_pred])))
        top_max = float(np.nanmax(np.concatenate([dos_true, dos_pred])))
        error_min = float(np.nanmin(dos_error))
        error_max = float(np.nanmax(dos_error))
        fig2.add_trace(
            go.Scattergl(
                x=[0.0, 0.0],
                y=[top_min, top_max],
                mode="lines",
                line=dict(color="#FFFFFF", width=1.2, dash="dot"),
                name="Fermi level",
                showlegend=True,
                hoverinfo="skip",
            ),
            row=1,
            col=1,
        )
        fig2.add_trace(
            go.Scattergl(
                x=[0.0, 0.0],
                y=[error_min, error_max],
                mode="lines",
                line=dict(color="#FFFFFF", width=1.2, dash="dot"),
                name="Fermi level",
                showlegend=False,
                hoverinfo="skip",
            ),
            row=2,
            col=1,
        )
        fig2.add_hline(
            y=0.0,
            row=1,
            col=1,
            line=dict(color="#777777", dash="dash", width=1.0),
        )
        fig2.add_hline(
            y=0.0,
            row=2,
            col=1,
            line=dict(color="#777777", dash="dash", width=1.0),
        )
        fig2.update_yaxes(title_text="DOS", row=1, col=1)
        fig2.update_yaxes(title_text="DOS error", row=2, col=1)
        fig2.update_xaxes(
            title_text="E-E_F (eV)",
            row=2,
            col=1,
        )
        fig2.update_layout(
            title=_normalize_spectral_title(payload.get("title")),
            height=560,
            autosize=True,
            margin=dict(l=50, r=25, t=55, b=40),
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#e8edf7"),
            showlegend=True,
            legend=dict(x=0.99, y=0.99, xanchor="right", yanchor="top"),
        )
        x_range = [-15.0, 25.0]
        fig2.update_xaxes(range=x_range, row=1, col=1)
        fig2.update_xaxes(range=x_range, row=2, col=1)
        return fig2

    grid = _series_from_payload(payload, "grid")
    dos = _series_from_payload(payload, "dos")
    fermi = payload.get("fermi_ev")
    if fermi is not None:
        grid = grid - float(fermi)
    fig.add_trace(
        go.Scattergl(
            x=grid,
            y=dos,
            mode="lines",
            line=dict(color=plot_style["prediction"], width=1.8, dash="dash"),
            name="DOS",
            showlegend=False,
        )
    )
    if fermi is not None:
        fig.add_vline(
            x=0.0,
            line=dict(color="#777777", dash="dash", width=1.2),
            annotation_text="$E_F$",
            annotation_position="top left",
        )
    fig.update_layout(
        title=_normalize_spectral_title(payload.get("title")),
        height=420,
        autosize=True,
        margin=dict(l=50, r=25, t=55, b=40),
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e8edf7"),
        showlegend=False,
    )
    fig.update_xaxes(title_text="E-E_F (eV)")
    if fermi is not None:
        fig.update_xaxes(range=[-10.0, 10.0])
    fig.update_yaxes(title_text="DOS")
    return fig


def _normalize_spectral_title(title: Any) -> str:
    text = str(title or "").strip()
    if ":" in text:
        material = text.split(":", 1)[1].strip()
        if material:
            return f"Band structure and DOS: {material}"
    return text or "Band structure and DOS"


def _evaluation_spectral_title(evaluation: dict[str, Any]) -> str:
    copied_files = evaluation.get("copied_files", {})
    for filename in ("dos_comparison.pt", "dos_prediction.pt"):
        path = copied_files.get(filename)
        if path is None:
            continue
        try:
            return _normalize_spectral_title(
                _load_plot_payload(Path(path)).get("title")
            )
        except (OSError, TypeError, KeyError):
            continue
    return "Band structure and DOS"


def _evaluation_plot_style(evaluation: dict[str, Any]) -> dict[str, str]:
    settings = evaluation.get("manifest", {}).get("settings", {})
    return {
        "ground_truth": str(settings.get("ground_truth_color") or "#000000"),
        "prediction": str(settings.get("prediction_color") or "#D62728"),
        "error": str(settings.get("error_color") or "#0072B2"),
        "block_error": str(settings.get("block_error_color") or "#003B73"),
    }


def _build_correlation_figure(payload: dict[str, Any]) -> go.Figure | None:
    if str(payload.get("kind", "")) != "correlation":
        return None
    bound = float(payload.get("bound", 0.0))
    lo = float(payload.get("lo", -bound))
    hi = float(payload.get("hi", bound))
    counts_value = payload.get("histogram_counts")
    x_edges_value = payload.get("x_edges")
    y_edges_value = payload.get("y_edges")
    if (
        counts_value is not None
        and x_edges_value is not None
        and y_edges_value is not None
    ):
        counts = np.asarray(counts_value, dtype=float).T
        x_edges = np.asarray(x_edges_value, dtype=float)
        y_edges = np.asarray(y_edges_value, dtype=float)
    else:
        pred = _series_from_payload(payload, "pred")
        target = _series_from_payload(payload, "target")
        if pred.size == 0 or target.size == 0:
            return None
        if not np.isfinite(bound) or bound <= 0.0:
            bound = float(np.max(np.abs(np.concatenate([pred, target]))))
            lo, hi = -bound, bound
        counts_raw, x_edges, y_edges = np.histogram2d(
            target, pred, bins=60, range=((lo, hi), (lo, hi))
        )
        counts = counts_raw.T
    counts[counts <= 0.0] = np.nan
    x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
    y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])
    corr = payload.get("corr")
    r2 = payload.get("r2")
    mae = payload.get("mae")
    unit = str(payload.get("unit") or "native")
    if r2 is None and corr is not None:
        try:
            r2 = float(corr) * float(corr)
        except Exception:
            r2 = None
    fig = go.Figure(
        data=[
            go.Heatmap(
                x=x_centers,
                y=y_centers,
                z=counts,
                colorscale=[[0.0, "#FFFFB2"], [0.5, "#FD8D3C"], [1.0, "#BD0026"]],
                colorbar=dict(title="Count"),
                hovertemplate=(
                    "Ground truth=%{x:.6g}<br>Prediction=%{y:.6g}"
                    "<br>Count=%{z:.0f}<extra></extra>"
                ),
            )
        ]
    )
    pred = _series_from_payload(payload, "pred")
    target = _series_from_payload(payload, "target")
    if pred.size and target.size:
        count = min(pred.size, target.size)
        fig.add_trace(
            go.Scattergl(
                x=target[:count],
                y=pred[:count],
                mode="markers",
                marker=dict(color="#000000", size=4),
                hoverinfo="skip",
                showlegend=False,
            )
        )
    fig.add_trace(
        go.Scattergl(
            x=[lo, hi],
            y=[lo, hi],
            mode="lines",
            line=dict(color="#666666", dash="dash", width=1.0),
            hoverinfo="skip",
            showlegend=False,
        )
    )
    fig.update_layout(
        title=str(payload.get("title", "Correlation"))
        + (
            f" | MAE = {float(mae):.6g} {unit}"
            if mae is not None and np.isfinite(float(mae))
            else ""
        )
        + (
            f" | R² = {float(r2):.6f}"
            if r2 is not None and np.isfinite(float(r2))
            else ""
        ),
        xaxis_title="Ground truth",
        yaxis_title="Prediction",
        height=560,
        margin=dict(l=55, r=110, t=60, b=50),
        template="plotly_white",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#FFFFFF",
        font=dict(color="#111111"),
    )
    fig.update_xaxes(range=[lo, hi])
    fig.update_yaxes(range=[lo, hi], scaleanchor="x", scaleratio=1)
    return fig


def _fallback_image_html(href: str, alt: str, *, height: str = "420px") -> str:
    return f"""
  <div class="plotly-panel">
    <img src="{html.escape(href)}" alt="{html.escape(alt)}" style="width:100%; height:auto; max-height:{html.escape(height)}; object-fit:contain;">
  </div>
"""


def _build_dos_section(
    evaluation: dict[str, Any],
    *,
    include_plotlyjs: str | bool,
    div_id_prefix: str,
) -> str:
    copied_files = evaluation.get("copied_files", {})
    asset_namespace = Path(str(evaluation.get("asset_dir", ""))).name
    payload_path = copied_files.get("dos_comparison.pt") or copied_files.get(
        "dos_prediction.pt"
    )
    if payload_path is not None:
        payload = _load_plot_payload(Path(payload_path))
        figure = _build_dos_figure(
            payload, plot_style=_evaluation_plot_style(evaluation)
        )
        if figure is not None:
            figure_height = (
                "560px" if str(payload.get("kind")) == "dos_comparison" else "420px"
            )
            figure_html = pio.to_html(
                figure,
                include_plotlyjs=include_plotlyjs,
                full_html=False,
                default_width="100%",
                default_height=figure_height,
                div_id=f"{div_id_prefix}-dos",
                config={"responsive": True},
            )
            return f"""
  <h2>Interactive DOS</h2>
  <div class="plotly-panel">
    {figure_html}
  </div>
"""

    image_href = None
    if copied_files.get("dos_comparison.png") is not None:
        image_href = f"run_assets/{asset_namespace}/dos_comparison.png"
    elif copied_files.get("dos_prediction.png") is not None:
        image_href = f"run_assets/{asset_namespace}/dos_prediction.png"
    if image_href is None:
        return ""
    return f"""
  <h2>Interactive DOS</h2>
  {_fallback_image_html(image_href, "DOS plot", height="560px")}
"""


def _build_correlation_section(
    evaluation: dict[str, Any],
    *,
    include_plotlyjs: str | bool,
    div_id_prefix: str,
) -> str:
    copied_files = evaluation.get("copied_files", {})
    asset_namespace = Path(str(evaluation.get("asset_dir", ""))).name
    items = [
        (
            "Hamiltonian",
            copied_files.get("hamiltonian_correlation.pt"),
            "hamiltonian_correlation.png",
        ),
        (
            "Density",
            copied_files.get("density_correlation.pt"),
            "density_correlation.png",
        ),
        (
            "Overlap",
            copied_files.get("overlap_correlation.pt"),
            "overlap_correlation.png",
        ),
    ]
    sections = []
    first_plot_uses_js = True if include_plotlyjs else False
    for idx, (name, payload_path, fallback_png) in enumerate(items, start=1):
        if payload_path is not None:
            payload = _load_plot_payload(Path(payload_path))
            figure = _build_correlation_figure(payload)
            if figure is None:
                continue
            plot_html = pio.to_html(
                figure,
                include_plotlyjs=include_plotlyjs if first_plot_uses_js else False,
                full_html=False,
                default_width="100%",
                default_height="560px",
                div_id=f"{div_id_prefix}-corr-{idx}",
                config={"responsive": True},
            )
            first_plot_uses_js = False
            sections.append(
                f"""
  <div class="metric-card">
    <div class="metric-title">{html.escape(name)} correlation</div>
    <div class="metric-plot">{plot_html}</div>
  </div>
"""
            )
        elif copied_files.get(fallback_png) is not None:
            sections.append(
                f"""
  <div class="metric-card">
    <div class="metric-title">{html.escape(name)} correlation</div>
    {_fallback_image_html(f"run_assets/{asset_namespace}/{fallback_png}", f"{name} correlation", height="430px")}
  </div>
"""
            )
    if not sections:
        return ""
    return f"""
  <h2>Correlation Diagnostics</h2>
  <div class="metric-grid correlation-grid">
    {"".join(sections)}
  </div>
"""


def _load_block_error_payload(path: Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise TypeError(
            f"Unexpected block error payload type in {path}: {type(payload)!r}"
        )
    return payload


def _coerce_optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _series_from_payload(payload: dict[str, Any], key: str) -> np.ndarray:
    values = payload.get(key, [])
    if torch.is_tensor(values):
        values = values.detach().cpu().numpy()
    return np.asarray(values, dtype=float)


def _make_block_error_figure(
    payload: dict[str, Any],
    *,
    x_key: str,
    y_key: str,
    title: str,
    x_label: str,
    y_label: str,
    color: str = "#003B73",
) -> go.Figure:
    x = _series_from_payload(payload, x_key)
    y = _series_from_payload(payload, y_key)
    hover = [
        f"key={k}<br>"
        f"edge={idx}<br>"
        f"src={src} dst={dst}<br>"
        f"shift=({sx}, {sy}, {sz})<br>"
        f"x={xv:.6g}<br>"
        f"y={yv:.6g}"
        for k, idx, src, dst, sx, sy, sz, xv, yv in zip(
            payload.get("pair_key", []),
            payload.get("edge_index", []),
            payload.get("src_atom", []),
            payload.get("dst_atom", []),
            payload.get("shift_sx", []),
            payload.get("shift_sy", []),
            payload.get("shift_sz", []),
            x,
            y,
        )
    ]
    fig = go.Figure(
        data=[
            go.Scattergl(
                x=x,
                y=y,
                mode="markers",
                marker=dict(size=6, color=color, opacity=1.0, line=dict(width=0)),
                hovertemplate="%{customdata}<extra></extra>",
                customdata=hover,
                showlegend=False,
            )
        ]
    )
    fig.update_layout(
        title=title,
        xaxis_title=x_label,
        yaxis_title=y_label,
        margin=dict(l=45, r=20, t=40, b=45),
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e8edf7"),
    )
    return fig


def _axis_control_html(control_id: str, *, allow_log_x: bool, allow_log_y: bool) -> str:
    parts = [
        '<label class="axis-toggle"><input type="checkbox" data-axis="y" checked> Log Y</label>'
    ]
    if allow_log_x:
        parts.append(
            '<label class="axis-toggle"><input type="checkbox" data-axis="x" checked> Log X</label>'
        )
    return "".join(parts)


def _axis_control_script(
    plot_div_id: str,
    control_id: str,
    allow_log_x: bool,
    allow_log_y: bool,
) -> str:
    relayout_items = ["'yaxis.type': yBox.checked ? 'log' : 'linear'"]
    if allow_log_x:
        relayout_items.append("'xaxis.type': xBox.checked ? 'log' : 'linear'")
    relayout_obj = "{ " + ", ".join(relayout_items) + " }"
    return f"""
(function() {{
  const root = document.getElementById({json.dumps(control_id)});
  const yBox = root ? root.querySelector('input[data-axis="y"]') : null;
  const xBox = root ? root.querySelector('input[data-axis="x"]') : null;
  const plotId = {json.dumps(plot_div_id)};
  function updateAxis() {{
    if (!window.Plotly) return;
    const updates = {relayout_obj};
    Plotly.relayout(document.getElementById(plotId), updates);
  }}
  if (yBox) yBox.addEventListener('change', updateAxis);
  if (xBox) xBox.addEventListener('change', updateAxis);
  updateAxis();
}})();
"""


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
