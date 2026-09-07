import math

import pytest

from pair_hamiltonian.stage6_validation import assess_stage6_scoped_run
from scripts.pair_mappers.aggregate_stage6_independent import combine_disjoint_metrics
from scripts.pair_mappers.run_stage6_onsite_ridge_screen import (
    _non_log_output_entries,
)


@pytest.mark.unit
def test_disjoint_metric_composition_is_exact():
    combined = combine_disjoint_metrics(
        {"mae": 2.0, "rmse": 3.0, "scalar_count": 10},
        {"mae": 5.0, "rmse": 7.0, "scalar_count": 30},
    )
    assert combined["mae"] == pytest.approx(4.25)
    assert combined["rmse"] == pytest.approx(math.sqrt((9 * 10 + 49 * 30) / 40))
    assert combined["scalar_count"] == 40


@pytest.mark.unit
def test_onsite_screen_allows_launcher_log_precreated_by_tee(tmp_path):
    output = tmp_path / "run"
    output.mkdir()
    (output / "launcher.log").write_text("")
    assert _non_log_output_entries(output) == []
    (output / "partial.json").write_text("{}")
    assert _non_log_output_entries(output) == [output / "partial.json"]


@pytest.mark.unit
def test_stage6_numerical_amendment_accepts_only_narrow_float32_roundoff():
    summary = {
        "completed": True,
        "passed": False,
        "test_shards_read": False,
        "best_validation_matrix_mae_mev": 20.826,
        "final_float32_symmetry_errors": {
            "proper_o3": 2.826e-5,
            "improper_o3": 3.2e-6,
            "projected_homogeneous_reversal": 8e-8,
            "raw_homogeneous_reversal": 0.31,
        },
    }
    assessment = assess_stage6_scoped_run(summary, 2e-5)
    assert assessment["accepted"]
    assert assessment["numerical_tolerance_amendment_used"]
    summary["final_float32_symmetry_errors"]["proper_o3"] = 5.01e-5
    assert not assess_stage6_scoped_run(summary, 2e-5)["accepted"]
