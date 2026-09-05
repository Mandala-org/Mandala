import csv
import json
import sys

import pytest
import torch

from scripts.pair_mappers.finalize_stage5_m0_screen import main


@pytest.mark.unit
def test_report_only_recovery_uses_completed_checkpoints(tmp_path, monkeypatch):
    output = tmp_path / "run"
    models = output / "models"
    models.mkdir(parents=True)
    promotions = [
        {
            "family": "d2",
            "key": f"configuration_{index}",
            "resolution": "compact" if index % 2 == 0 else "high",
            "cutoff_angstrom": 6.5 + index,
            "dimension": 20 + index,
        }
        for index in range(8)
    ]
    (output / "config.json").write_text(
        json.dumps(
            {
                "family": "d2",
                "manifest_hash": "run-hash",
                "promotion_manifest_hash": "promotion-hash",
                "test_shards_read": False,
                "selection_uses_test_hamiltonian": False,
            }
        )
    )
    diagnostics = {}
    metrics = {}
    for item in promotions:
        key = item["key"]
        diagnostics[key] = {
            "onsite": {"O": {"condition_numbers": {"4e": float("inf")}}},
            "offsite": {},
        }
        metrics[key] = {
            "matrix_elements": {"mae": 1.0, "rmse": 2.0, "scalar_count": 3},
            "block_frobenius": {"block_count": 4},
        }
        torch.save({"onsite.weight_l0_e": torch.ones(2)}, models / f"{key}.pt")
    (output / "fit_diagnostics.json").write_text(json.dumps(diagnostics))
    (output / "validation_metrics.json").write_text(json.dumps(metrics))
    descriptor_summary = tmp_path / "descriptor.json"
    descriptor_summary.write_text(
        json.dumps(
            {
                "passed": True,
                "cache_size_bytes": 100,
                "elapsed_seconds": 10.0,
            }
        )
    )
    promotion_path = tmp_path / "promotions.json"
    promotion_path.write_text(
        json.dumps({"manifest_hash": "promotion-hash", "promotions": promotions})
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "finalize_stage5_m0_screen.py",
            "--output-dir",
            str(output),
            "--descriptor-summary",
            str(descriptor_summary),
            "--promotions",
            str(promotion_path),
        ],
    )

    main()

    summary = json.loads((output / "summary.json").read_text())
    assert summary["passed"]
    assert summary["recovered_from_completed_checkpoints"]
    assert summary["fit_stream_seconds"] is None
    with (output / "results.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 8
    assert {row["learned_coefficient_count"] for row in rows} == {"2"}
    assert {row["missing_target_irreps"] for row in rows} == {"4e"}
