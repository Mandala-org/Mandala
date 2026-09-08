"""Regression tests for reference-aligned purification accuracy reporting."""

from collections import Counter

import pytest
import torch

from analysis.density_purification import density_errors, evaluate_purification
from core.orbital_irrep_config import OrbitalIrrepConfig
from data.block_matrix import BlockMatrix
from utils.units import HARTREE_TO_EV


def matrix(values, shifts=(0,)):
    edges = [(shift, 0, 0, 0, 0) for shift in shifts]
    return BlockMatrix(
        atoms=("H",),
        atom_counts=Counter(H=1),
        pair_blocks={
            "H-H": torch.tensor(values, dtype=torch.float64).reshape(-1, 1, 1)
        },
        pair_edges={"H-H": torch.tensor(edges).T},
        lookup={edge: ("H-H", i) for i, edge in enumerate(edges)},
        orbital_cfg=OrbitalIrrepConfig.from_dict({"H": "1s"}),
        basis="e3nn",
    )


def test_evaluation_iterates_energy_and_reference_control():
    d, s, ref, h = matrix([0.4]), matrix([2.0]), matrix([0.5]), matrix([-3.0])
    result = evaluate_purification(
        d, s, ref, s, h, predicted_hamiltonian=h, spin_degeneracy=2
    )
    expected = 0.4
    assert result["reference_electrons"] == 2
    for iteration, row in enumerate(result["rows"]):
        assert row["iteration"] == iteration
        assert row["physical_density"]["mae"] == pytest.approx(abs(expected - 0.5))
        assert row["raw_density"] == row["physical_density"]
        assert row["reference_control_density"]["mae"] == pytest.approx(0.0)
        assert row["band_energy_ref_h_abs_error_ev"] == pytest.approx(
            6 * abs(expected - 0.5) * HARTREE_TO_EV
        )
        assert (
            row["band_energy_ref_h_abs_error_ev"]
            == row["band_energy_pred_h_abs_error_ev"]
        )
        assert row["electron_count"] == pytest.approx(4 * expected)
        expected = 6 * expected**2 - 8 * expected**3
    assert result["rows"][-1]["density_mae_improvement_percent"] > 90


def test_density_metrics_include_missing_prediction_edges():
    result = density_errors(matrix([1.0]), matrix([2.0, 1.0, 2.0], shifts=(-1, 0, 1)))
    assert result["mae"] == pytest.approx(4 / 3)
    assert result["rmse"] == pytest.approx((8 / 3) ** 0.5)
    assert result["missing_prediction_edges"] == 2
    assert result["scalar_count"] == 3


def test_reference_control_exposes_support_truncation():
    d = matrix([0.1, 0.4, 0.1], shifts=(-1, 0, 1))
    s = matrix([0.0, 1.0, 0.0], shifts=(-1, 0, 1))
    result = evaluate_purification(d, s, d, s, s)
    for row in result["rows"]:
        assert row["physical_density"] == row["reference_control_density"]
    assert result["rows"][1]["physical_density"]["mae"] > 0
