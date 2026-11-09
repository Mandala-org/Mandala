"""Analytic error tails, population masks, and the paper validation path."""

from types import SimpleNamespace

import pytest
import torch

from net.checkpoint_compat import set_checkpoint_film_semantics
from net.operator_metrics import OperatorErrorDistribution
from scripts.report.evaluate_validation_error_distribution import (
    DistributionModel,
    HARTREE_TO_EV,
    _evaluate_shared_step,
    check_paper_maes,
    project_prediction,
    paper_metadata_path,
    reference_targets,
)


def matrix(values, mask=None, edges=None):
    values = torch.as_tensor(values, dtype=torch.float64).reshape(-1, 1, 1)
    n = len(values)
    return SimpleNamespace(
        basis="e3nn",
        pair_blocks={"H-H": values},
        pair_edges={"H-H": (
            torch.tensor([[1, -1], [0, 0], [0, 0], [0, 1], [1, 0]])[:, :n]
            if edges is None else edges
        )},
        coverage_masks={} if mask is None else {
            "H-H": torch.tensor(mask).reshape(-1, 1, 1)
        },
    )


def test_pooled_tails_and_r2_use_all_elements_not_structure_averages(tmp_path):
    # Structure 1 has one element, structure 2 two reverse periodic elements.
    # Errors in scaled units are [2, 6, 8]; refs are [0, 4, 8].
    with OperatorErrorDistribution(scratch_dir=tmp_path) as distribution:
        first = distribution.update(matrix([1]), matrix([0]), scale=2.0)
        second = distribution.update(matrix([5, 8]), matrix([2, 4]), scale=2.0)
        result = distribution.compute()
    assert (first + second) / 2 == pytest.approx(4.5)  # equal structure MAE
    assert result["mae_elementwise_pooled"] == pytest.approx(16 / 3)
    assert result["rmse"] == pytest.approx((104 / 3) ** 0.5)
    assert result["max_absolute_error"] == 8
    assert result["p95_absolute_error"] == pytest.approx(7.8)
    assert result["p99_absolute_error"] == pytest.approx(7.96)
    assert result["r2"] == pytest.approx(1 - 104 / 32)
    assert result["element_count"] == 3
    assert result["structures"] == 2
    assert list(tmp_path.iterdir()) == []


def test_coverage_selector_and_prefix_exclude_unobserved_elements():
    edges = torch.tensor([[1, -1, 2], [0, 0, 0], [0, 0, 0], [0, 1, 0], [1, 0, 1]])
    target = matrix([1, float("nan")], mask=[True, False], edges=edges[:, :2])
    pred = matrix([4, float("nan"), 999], edges=edges)
    with OperatorErrorDistribution() as distribution:
        distribution.update(pred, target, edge_selector=lambda edges: edges[0] > 0)
        result = distribution.compute()
    assert result["element_count"] == 1
    assert result["rmse"] == 3
    assert result["p99_absolute_error"] == 3
    assert result["r2"] is None


def test_projection_uses_reverse_periodic_blocks(small_angular_snapshot_e3nn):
    raw = small_angular_snapshot_e3nn.hamiltonian
    projected = project_prediction(
        raw, {"pred_reverse_alignment": {"H-H": ("H-H", torch.tensor([0, 1, 3, 2]))}}
    )
    for edge, (pair, index) in raw.lookup.items():
        a, b, c, i, j = edge
        reverse_pair, reverse_index = raw.lookup[(-a, -b, -c, j, i)]
        expected = 0.5 * (
            raw.pair_blocks[pair][index]
            + raw.pair_blocks[reverse_pair][reverse_index].T.conj()
        )
        torch.testing.assert_close(projected.pair_blocks[pair][index], expected)


def test_exact_shared_validation_path_and_legacy_film(small_angular_dataset_e3nn):
    dataset, mapper, cfg = small_angular_dataset_e3nn
    cfg.symmetrize_validation_output = True
    torch.set_num_threads(4)
    torch.manual_seed(44)
    model = DistributionModel(mapper=mapper, cfg=cfg).eval()
    set_checkpoint_film_semantics(model, "legacy_2026_07")
    model.validation_matrices = {}
    x, y = dataset[0]
    _, metrics = _evaluate_shared_step(model, (x, y), batch_idx=0)
    assert set(model.validation_matrices) == set(cfg.matrix_targets)
    for name, (prediction, target) in model.validation_matrices.items():
        with OperatorErrorDistribution() as distribution:
            distribution.update(
                project_prediction(prediction, x), target,
                scale=HARTREE_TO_EV if name == "hamiltonian" else 1.0,
                edge_selector=model._partial_train_mask,
            )
            result = distribution.compute()
        assert result["mae_elementwise_pooled"] == pytest.approx(
            metrics[f"val/{name}_mae_elementwise_projected"], rel=1e-6
        )


def test_paper_mae_gate_accepts_one_percent_and_rejects_mismatch():
    accepted = check_paper_maes({"hamiltonian": {"mae": 0.0030121}}, "ZnCuSnSeS")
    assert accepted["passed"]
    rejected = check_paper_maes({"hamiltonian": {"mae": 0.0030121 * 1.02}}, "ZnCuSnSeS")
    assert not rejected["passed"]
    with pytest.raises(RuntimeError, match="Operator set"):
        check_paper_maes({"hamiltonian": {"mae": 0.00043516}}, "silicon_scales")


def test_paper_metadata_discovery_skips_incomplete_calculations(tmp_path):
    (tmp_path / "HS.out").touch()
    (tmp_path / "log.out").touch()
    (tmp_path / "slurm-123.out").touch()
    assert paper_metadata_path(tmp_path) is None
    (tmp_path / "Si.out").write_text("canonical metadata")
    assert paper_metadata_path(tmp_path) == tmp_path / "Si.out"


def test_paper_density_convention_is_a_fixed_copy(small_angular_snapshot_e3nn):
    native = small_angular_snapshot_e3nn.density
    targets = {"density": native, "other": 123}
    restored = reference_targets(targets, "paper_restricted_total")
    for pair, values in native.pair_blocks.items():
        torch.testing.assert_close(restored["density"].pair_blocks[pair], 2 * values)
    assert targets["density"] is native
    assert restored["other"] == 123
    assert reference_targets(targets, "native_spin0") is targets
