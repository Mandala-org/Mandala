from types import SimpleNamespace

import pytest
import torch

from net.operator_metrics import matrix_error_metrics, matrix_error_totals


def matrix(blocks, masks=None):
    return SimpleNamespace(
        basis="e3nn",
        pair_blocks=blocks,
        pair_edges={
            key: torch.arange(len(value) * 5).reshape(5, len(value))
            for key, value in blocks.items()
        },
        coverage_masks=masks or {},
    )


def test_unequal_pair_sizes_are_entry_weighted_and_unit_scaled():
    target = matrix({"A-A": torch.zeros(1, 1, 1), "B-B": torch.zeros(2, 2, 2)})
    prediction = matrix(
        {"A-A": torch.full((1, 1, 1), 10.0), "B-B": torch.ones(2, 2, 2)}
    )
    result = matrix_error_metrics(prediction, target, scale=2.0)
    assert result["mae_elementwise"].item() == pytest.approx(4.0)
    assert result["mse_elementwise"].item() == pytest.approx(48.0)
    assert result["rmse_elementwise"].item() == pytest.approx(48**0.5)


def test_complex_magnitudes_exclude_unobserved_and_unselected_entries():
    target = matrix(
        {"A-A": torch.tensor([[[0j, complex("nan")]], [[0j, 0j]]])},
        {"A-A": torch.tensor([[[True, False]], [[True, True]]])},
    )
    prediction = matrix(
        {"A-A": torch.tensor([[[3 + 4j, complex("nan")]], [[99j, 99j]]])}
    )
    totals = matrix_error_totals(
        prediction, target, edge_selector=lambda edges: torch.tensor([True, False])
    )
    assert totals["count"] == 1
    assert totals["absolute_sum"].item() == 5
    assert totals["squared_sum"].item() == 25
    assert not totals["absolute_sum"].requires_grad


def test_metrics_reject_wrong_edges_basis_and_nonfinite_observations():
    target, prediction = matrix({"A-A": torch.zeros(1, 1, 1)}), matrix(
        {"A-A": torch.ones(1, 1, 1)}
    )
    prediction.pair_edges["A-A"][0] += 1
    with pytest.raises(ValueError, match="support"):
        matrix_error_totals(prediction, target)
    prediction.pair_edges = target.pair_edges
    prediction.basis = "openmx"
    with pytest.raises(ValueError, match="basis"):
        matrix_error_totals(prediction, target)
    prediction.basis = "e3nn"
    prediction.pair_blocks["A-A"][:] = float("inf")
    with pytest.raises(ValueError, match="Nonfinite"):
        matrix_error_totals(prediction, target)
