import math

import pytest

from scripts.pair_mappers.aggregate_stage6_independent import combine_disjoint_metrics


@pytest.mark.unit
def test_disjoint_metric_composition_is_exact():
    combined = combine_disjoint_metrics(
        {"mae": 2.0, "rmse": 3.0, "scalar_count": 10},
        {"mae": 5.0, "rmse": 7.0, "scalar_count": 30},
    )
    assert combined["mae"] == pytest.approx(4.25)
    assert combined["rmse"] == pytest.approx(math.sqrt((9 * 10 + 49 * 30) / 40))
    assert combined["scalar_count"] == 40
