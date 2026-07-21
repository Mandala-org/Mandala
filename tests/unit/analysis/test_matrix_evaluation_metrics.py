from types import SimpleNamespace

import pytest
import torch

from analysis.evaluation import (
    _save_heatmap_triptych,
    build_hamiltonian_interactive_heatmap_payload,
    compute_shift_resolved_matrix_metrics,
    save_correlation_plot,
    save_eigenvalue_correlation_plot,
)


def _matrix(blocks: dict[str, torch.Tensor], edges: dict[str, torch.Tensor]):
    return SimpleNamespace(pair_blocks=blocks, pair_edges=edges)


def test_shift_resolved_metrics_are_globally_element_weighted() -> None:
    # A one-element block and a four-element block must contribute according to
    # their scalar counts, rather than as two equally weighted pair means.
    edges = {
        "H-H": torch.tensor([[0], [0], [0], [0], [0]]),
        "Si-Si": torch.tensor([[1], [0], [0], [0], [1]]),
    }
    target = _matrix(
        {
            "H-H": torch.zeros((1, 1, 1)),
            "Si-Si": torch.zeros((1, 2, 2)),
        },
        edges,
    )
    pred = _matrix(
        {
            "H-H": torch.ones((1, 1, 1)),
            "Si-Si": torch.full((1, 2, 2), 2.0),
        },
        edges,
    )

    metrics = compute_shift_resolved_matrix_metrics(pred, target)

    assert metrics["scalar_count"] == 5
    assert metrics["edge_count"] == 2
    assert metrics["mae"] == pytest.approx(9.0 / 5.0)
    assert metrics["mse"] == pytest.approx(17.0 / 5.0)
    assert metrics["absolute_error_sum"] == pytest.approx(9.0)
    assert metrics["squared_error_sum"] == pytest.approx(17.0)


def test_shift_resolved_metrics_reject_misaligned_periodic_images() -> None:
    blocks = {"Si-Si": torch.zeros((1, 1, 1))}
    target = _matrix(
        blocks,
        {"Si-Si": torch.tensor([[0], [0], [0], [0], [1]])},
    )
    pred = _matrix(
        blocks,
        {"Si-Si": torch.tensor([[1], [0], [0], [0], [1]])},
    )

    with pytest.raises(ValueError, match="shift-resolved edges"):
        compute_shift_resolved_matrix_metrics(pred, target)


def test_heatmap_auto_clim_uses_all_displayed_values(tmp_path) -> None:
    output_path = tmp_path / "auto.png"
    cutout = {
        "gt": [[-0.2, 0.1], [0.0, 0.05]],
        "pred": [[-0.1, 0.6], [0.0, 0.05]],
        "diff": [[0.1, 0.5], [0.0, 0.0]],
        "axes": {},
    }
    _save_heatmap_triptych(
        cutout,
        output_path,
        title="Hamiltonian: worst absolute error fragment",
        clim="auto",
    )
    assert output_path.is_file()


def test_correlation_uses_all_shift_resolved_elements_and_correct_mae(tmp_path) -> None:
    edges = {"Si-Si": torch.tensor([[0], [0], [0], [0], [0]])}
    target = _matrix({"Si-Si": torch.zeros((1, 2, 2))}, edges)
    pred = _matrix({"Si-Si": torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])}, edges)
    output_path = tmp_path / "correlation.png"
    save_correlation_plot(
        pred,
        target,
        output_path,
        title="Hamiltonian correlation",
        max_points=1,
        alpha=0.01,
        seed=123,
        value_scale=2.0,
        value_unit="eV",
        bins=8,
    )
    payload = torch.load(
        output_path.with_suffix(".pt"), map_location="cpu", weights_only=False
    )
    assert payload["point_count"] == 4
    assert payload["mae"] == pytest.approx(5.0)
    assert payload["unit"] == "eV"
    assert payload["pred"].numel() == 4
    assert payload["target"].numel() == 4
    assert output_path.is_file()


def test_eigenvalue_correlation_mae_uses_only_fixed_energy_window(tmp_path) -> None:
    output_path = tmp_path / "eigenvalue.png"
    metrics = save_eigenvalue_correlation_plot(
        torch.tensor([-20.0, -15.0, 0.0, 25.0, 30.0]),
        torch.tensor([-20.0, -14.0, 2.0, 24.0, 30.0]),
        output_path,
        title="Eigenvalue correlation",
        bins=8,
    )
    assert metrics is not None
    assert metrics["point_count"] == 3
    assert metrics["mae_ev"] == pytest.approx(4.0 / 3.0)
    assert metrics["energy_min_ev"] == -15.0
    assert metrics["energy_max_ev"] == 25.0


def test_interactive_heatmap_all_atoms_is_shift_resolved_only() -> None:
    edges = {"Si-Si": torch.tensor([[0], [0], [0], [0], [0]])}
    orbital_cfg = SimpleNamespace(element_to_irreps={"Si": SimpleNamespace(dim=1)})
    common = {
        "pair_edges": edges,
        "atoms": ("Si",),
        "orbital_cfg": orbital_cfg,
        "lookup": {(0, 0, 0, 0, 0): ("Si-Si", 0)},
        "keys": lambda: edges.keys(),
    }
    target = SimpleNamespace(pair_blocks={"Si-Si": torch.zeros((1, 1, 1))}, **common)
    pred = SimpleNamespace(pair_blocks={"Si-Si": torch.ones((1, 1, 1))}, **common)
    payload = build_hamiltonian_interactive_heatmap_payload(
        pred,
        target,
        positions=torch.zeros((1, 3)),
        box=torch.eye(3),
        default_clim=0.1,
        include_all_atoms=True,
        random_count=0,
    )
    assert "sum_pbc" not in payload
    assert set(payload["shift_resolved"]) == {"all"}
    assert payload["shift_resolved"]["all"]["selection_label"].startswith("All atoms")
