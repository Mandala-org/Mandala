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


@pytest.mark.parametrize("dataset_kind,seed", [("ZnCuSnSeS", 44), ("silicon_scales", 43)])
def test_validation_only_preserves_ordered_split_and_training_mapper(
    tmp_path, monkeypatch, dataset_kind, seed
):
    """Synthetic listing, real discovery/shuffle/factory; no matrix parsing."""
    import random

    import data.factory as factory_module
    from data.factory import DatasetFactory
    from net.common import Config
    from scripts.report.evaluate_validation_error_distribution import (
        build_evaluation_dataset_bundle,
    )

    # Multiple scales exercise the single RNG shared across scale shuffles.
    for scale in (1, 2):
        for index in reversed(range(105)):
            directory = tmp_path / f"scale_{scale}" / f"snapshot_{index}"
            directory.mkdir(parents=True)
            (directory / "HS.out").touch()
            (directory / "Si.out").touch()
        incomplete = tmp_path / f"scale_{scale}" / "incomplete"
        incomplete.mkdir()
        (incomplete / "HS.out").touch()

    args = SimpleNamespace(
        dataset_kind=dataset_kind, data_path=tmp_path, scales="[1, 2]",
        num_train_per_scale=80, num_val_per_scale=20, data_split_seed=seed,
        convention="e3nn", snapshot_cache_dir=None, dataset_device="cpu",
        load_split="all",
    )
    expected = []
    expected_training = set()
    rng = random.Random(seed)
    for scale in (1, 2):
        paths = sorted((tmp_path / f"scale_{scale}").glob("snapshot_*/HS.out"))
        rng.shuffle(paths)
        expected_training.update(paths[:80])
        expected.extend(paths[80:100])
    metadata_reads = []

    def load_info(self, matrix, info):
        metadata_reads.append(matrix)
        # A species present only in training must remain in the mapper.
        orbitals = {"H": "1s", "O": "1s"} if matrix in expected_training else {"H": "1s"}
        return SimpleNamespace(orbital_set=orbitals, is_spinful=False)

    monkeypatch.setattr(DatasetFactory, "_load_info", load_info)
    builds = []

    def dataset(pairs, mapper, cfg, convention, *, preprocessing, split):
        builds.append((split, list(pairs)))
        return list(pairs)

    monkeypatch.setattr(factory_module, "E3GNNDataset", dataset)
    monkeypatch.setattr("scripts.dataset._resolve_single_snapshot_info_path", paper_metadata_path)
    count_full, full, mapper_full = build_evaluation_dataset_bundle(args, Config(verbosity=0))
    assert [split for split, _ in builds] == ["train", "val"]
    training_paths = {matrix for matrix, _ in builds[0][1]}
    builds.clear()
    metadata_reads.clear()
    args.load_split = "validation"
    count_fast, fast, mapper_fast = build_evaluation_dataset_bundle(args, Config(verbosity=0))
    assert builds == [("val", full)]
    assert fast == full
    assert count_fast == count_full == 160
    assert training_paths <= set(metadata_reads)
    assert mapper_fast.orbital_cfg.to_dict() == mapper_full.orbital_cfg.to_dict()

    assert set(mapper_fast.orbital_cfg.elements()) == {"H", "O"}
    assert [matrix for matrix, _ in fast] == expected
