from __future__ import annotations

import json

from analysis.wandb_sweep_core import RunRecord
from analysis.wandb_sweep_plotly import (
    _available_evaluation_matrices,
    _build_matrix_metrics_section,
    _build_matrix_selector_sections,
    _build_correlation_figure,
    _build_dos_figure,
    _load_evaluation_matrix_metrics,
    build_model_detail_page,
)


def _metrics_payload() -> dict:
    return {
        "schema_version": 1,
        "definition": "element-weighted shift-resolved real-space block error",
        "metrics": {
            "val/hamiltonian_mae": 0.0030126269,
            "val/hamiltonian_mse": 1.2e-5,
            "val/overlap_mae": 0.004,
            "val/overlap_mse": 3.0e-5,
            "val/density_mae": 0.005,
            "val/density_mse": 4.0e-5,
        },
        "matrices": {
            "hamiltonian": {
                "mae": 0.0030126269,
                "mse": 1.2e-5,
                "units": {"mae": "eV", "mse": "eV^2"},
                "scalar_count": 123,
            },
            "overlap": {
                "mae": 0.004,
                "mse": 3.0e-5,
                "units": {"mae": "dimensionless", "mse": "dimensionless^2"},
                "scalar_count": 123,
            },
            "density": {
                "mae": 0.005,
                "mse": 4.0e-5,
                "units": {"mae": "native", "mse": "native^2"},
                "scalar_count": 123,
            },
        },
    }


def test_matrix_selector_and_metric_table_cover_all_predicted_matrices(tmp_path):
    metric_path = tmp_path / "evaluation_matrix_metrics.json"
    metric_path.write_text(json.dumps(_metrics_payload()), encoding="utf-8")
    evaluation = {
        "copied_files": {
            "evaluation_matrix_metrics.json": str(metric_path),
            "hamiltonian_interactive_heatmaps.pt": "ham.pt",
            "overlap_interactive_heatmaps.pt": "overlap.pt",
            "density_interactive_heatmaps.pt": "density.pt",
        }
    }
    metrics = _load_evaluation_matrix_metrics(evaluation)
    names = _available_evaluation_matrices(evaluation, metrics)
    assert names == ["hamiltonian", "overlap", "density"]

    table = _build_matrix_metrics_section(metrics)
    assert "0.00301263 eV" in table
    assert "dimensionless^2" in table
    assert "native^2" in table

    selector, panels = _build_matrix_selector_sections(
        "run/one",
        names,
        {name: f"snapshot-{name}" for name in names},
        {name: f"heatmap-{name}" for name in names},
    )
    assert 'id="evaluation-matrix-run-one"' in selector
    assert selector.count("<option") == 3
    assert panels.count('data-evaluation-matrix="') == 3
    assert 'window.dispatchEvent(new Event("resize"))' in panels


def test_model_report_score_uses_exact_evaluation_hamiltonian_mae(tmp_path):
    metric_path = tmp_path / "evaluation_matrix_metrics.json"
    metric_path.write_text(json.dumps(_metrics_payload()), encoding="utf-8")
    record = RunRecord(
        run_id="example",
        name="Example",
        state="finished",
        score=99.0,
        config={},
    )
    report = build_model_detail_page(
        record,
        include_plotlyjs=False,
        rank_metric="old/dense_metric",
        run_page_index={
            "example": {
                "evaluation": {
                    "source_dir": str(tmp_path),
                    "manifest": {},
                    "image_assets": [],
                    "file_assets": [
                        {"label": "metrics", "href": "evaluation_matrix_metrics.json"}
                    ],
                    "copied_files": {
                        "evaluation_matrix_metrics.json": str(metric_path)
                    },
                }
            }
        },
    )
    assert '<div class="v">val/hamiltonian_mae</div>' in report
    assert '<div class="v">0.00301263</div>' in report
    assert "old/dense_metric" not in report


def test_report_dos_and_correlation_styles_use_requested_contract():
    dos = _build_dos_figure(
        {
            "kind": "dos_comparison",
            "title": "DOS comparison: Silicon",
            "grid_true": [-20.0, 0.0, 30.0],
            "dos_true": [0.0, 1.0, 0.0],
            "grid_pred": [-20.0, 0.0, 30.0],
            "dos_pred": [0.0, 0.8, 0.0],
            "dos_error": [0.0, -0.2, 0.0],
        }
    )
    assert dos is not None
    assert dos.data[0].line.color == "#000000"
    assert dos.data[0].line.dash == "solid"
    assert dos.data[1].line.color == "#D62728"
    assert dos.data[1].line.dash == "dash"
    assert dos.data[2].line.color == "#0072B2"
    assert dos.data[3].name == "Fermi level"
    assert tuple(dos.data[3].x) == (0.0, 0.0)
    assert tuple(dos.layout.xaxis.range) == (-15.0, 25.0)
    assert dos.layout.xaxis2.title.text == "E-E_F (eV)"

    correlation = _build_correlation_figure(
        {
            "kind": "correlation",
            "title": "Hamiltonian correlation",
            "histogram_counts": [[0.0, 1.0], [2.0, 0.0]],
            "x_edges": [-1.0, 0.0, 1.0],
            "y_edges": [-1.0, 0.0, 1.0],
            "lo": -1.0,
            "hi": 1.0,
            "bound": 1.0,
            "mae": 0.003,
            "unit": "eV",
            "r2": 0.99,
            "target": [-0.8, 0.1, 0.9],
            "pred": [-0.7, 0.0, 0.8],
        }
    )
    assert correlation is not None
    assert correlation.data[0].type == "heatmap"
    assert correlation.data[0].colorbar.title.text == "Count"
    assert correlation.data[0].colorscale[0][1] == "#FFFFB2"
    assert correlation.data[0].colorscale[-1][1] == "#BD0026"
    assert correlation.data[1].marker.color == "#000000"
    assert correlation.data[1].marker.size == 4
    assert "MAE = 0.003 eV" in correlation.layout.title.text
