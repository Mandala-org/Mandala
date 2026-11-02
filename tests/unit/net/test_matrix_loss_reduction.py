"""Analytic objective controls through the shared E3GNN graph step."""

from collections import Counter

import pytest
import torch

from core.block_irrep_mapper import BlockIrrepMapper
from core.orbital_irrep_config import OrbitalIrrepConfig
from data.block_matrix import BlockMatrix
from data.structure_inference import build_model_input_from_structure
from net.common import Config
from net.configuration import validate_configuration_sections
from net.e3gnn import E3GNN
from utils.units import HARTREE_TO_EV


def _config(**overrides):
    values = dict(
        l_max=0,
        hidden_irreps="2x0e",
        n_radial=2,
        radial_layers=[2],
        num_layers_gnn=0,
        neck_depth=1,
        head_e3mlp_layers=1,
        dtype=torch.float64,
        device="cpu",
        verbosity=0,
        enable_energy=False,
        enable_num_electrons=False,
        train_observables_on_gt=False,
        log_partial_gt_observables=False,
        log_per_irrep_metrics=False,
        log_hamiltonian_irrep_contrib_metrics=False,
        log_hamiltonian_pair_contrib_metrics=False,
        log_train_metrics=True,
        data={
            "representation": "orbital_blocks",
            "graph_source": "radius",
            "graph_radius": 2.0,
        },
    )
    values.update(overrides)
    return Config(**values)


def _case(*, single=False, **overrides):
    cfg = _config(**overrides)
    orbital_cfg = OrbitalIrrepConfig.from_dict(
        {"C": "2s"} if single else {"H": "1s", "O": "2s"}
    )
    mapper = BlockIrrepMapper(orbital_cfg, dtype=torch.float64)
    model = E3GNN(mapper, cfg).double()
    atoms = ("C", "C") if single else ("H", "O")
    x = build_model_input_from_structure(
        atoms=atoms,
        positions=torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.2, 0.1]], dtype=torch.float64),
        box=torch.eye(3, dtype=torch.float64) * 20,
        cfg=cfg,
        mapper=mapper,
    )
    all_edges = torch.cat([x["edge_shift"], x["edge_index"]], dim=0)
    edges = {
        key: all_edges[:, partition["global_idx"]]
        for key, partition in x["edge_partitions"].items()
        if len(partition["global_idx"])
    }
    errors = {"H-H": 10.0, "H-O": 3.0, "O-H": 4.0, "O-O": 1.0, "C-C": 2.0}
    leaves = {
        key: torch.full(
            (edge.shape[1], *orbital_cfg.block_dims(key)),
            errors[key],
            dtype=torch.float64,
            requires_grad=True,
        )
        for key, edge in edges.items()
    }
    lookup = {
        tuple(edge.tolist()): (key, i)
        for key, values in edges.items()
        for i, edge in enumerate(values.T)
    }

    def matrix(blocks):
        return BlockMatrix(
            atoms=atoms,
            atom_counts=Counter(atoms),
            pair_blocks=blocks,
            pair_edges=edges,
            lookup=lookup,
            orbital_cfg=orbital_cfg,
            basis="e3nn",
        )

    target = matrix({key: torch.zeros_like(value) for key, value in leaves.items()})
    prediction = matrix(leaves).to_vectors(mapper)
    return model, x, target, prediction, leaves


@pytest.mark.parametrize("kind", ["mae", "mse"])
@pytest.mark.parametrize("reduction", ["pair_sum", "pair_mean", "elements"])
def test_unequal_species_blocks_have_declared_objective_and_fixed_diagnostics(
    kind, reduction
):
    model, x, target, prediction, leaves = _case(
        matrix_loss_reduction=reduction, objective={"kind": kind}
    )
    loss, metrics, _ = model._graph_step(
        (x, {"hamiltonian": target}), 0, "train", {"hamiltonian": prediction}
    )
    # Four directed species pairs have 1, 2, 2 and 4 observed entries.
    expected = {
        "mae": {"pair_sum": 18.0, "pair_mean": 4.5, "elements": 28 / 9},
        "mse": {"pair_sum": 126.0, "pair_mean": 31.5, "elements": 154 / 9},
    }
    assert loss.item() == pytest.approx(expected[kind][reduction])
    assert metrics["train/hamiltonian_mae"].item() == pytest.approx(18 * HARTREE_TO_EV)
    assert metrics["train/hamiltonian_mae_elementwise"].item() == pytest.approx(
        28 / 9 * HARTREE_TO_EV
    )
    loss.backward()
    for key, value in leaves.items():
        denominator = (
            9
            if reduction == "elements"
            else value.numel() * (4 if reduction == "pair_mean" else 1)
        )
        expected_gradient = (
            torch.ones_like(value) if kind == "mae" else 2 * value.detach()
        ) / denominator
        torch.testing.assert_close(value.grad, expected_gradient)
    # The opposite directions have independent gradients under raw supervision.
    if kind == "mse":
        assert not torch.equal(leaves["H-O"].grad, leaves["O-H"].grad.transpose(-1, -2))


@pytest.mark.parametrize("kind", ["mae", "mse"])
def test_single_species_graphene_control_has_identical_loss_and_gradient(kind):
    results = []
    for reduction in ("pair_sum", "pair_mean", "elements"):
        model, x, target, prediction, leaves = _case(
            single=True, matrix_loss_reduction=reduction, objective={"kind": kind}
        )
        loss, *_ = model._graph_step(
            (x, {"hamiltonian": target}), 0, "train", {"hamiltonian": prediction}
        )
        loss.backward()
        results.append((loss.detach(), leaves["C-C"].grad))
    for loss, gradient in results[1:]:
        torch.testing.assert_close(loss, results[0][0], rtol=0, atol=0)
        torch.testing.assert_close(gradient, results[0][1], rtol=0, atol=0)


@pytest.mark.parametrize("reduction", ["pair_mean", "elements"])
def test_observed_masks_exclude_missing_values_and_partial_edges_from_raw_gradient(
    reduction,
):
    model, x, target, prediction, leaves = _case(
        matrix_loss_reduction=reduction,
        objective={"kind": "mse"},
        partial_train="offdiag",
    )
    target._coverage_masks = {
        "H-O": torch.ones_like(target.pair_blocks["H-O"], dtype=torch.bool)
    }
    target._coverage_masks["H-O"][0, 0, 1] = False
    target.pair_blocks["H-O"][0, 0, 1] = float("nan")
    loss, metrics, _ = model._graph_step(
        (x, {"hamiltonian": target}), 0, "train", {"hamiltonian": prediction}
    )
    # One H-O entry at 3 and two O-H entries at 4; onsite entries do not count.
    assert loss.item() == pytest.approx(12.5 if reduction == "pair_mean" else 41 / 3)
    assert metrics["train/hamiltonian_mae_elementwise"].item() == pytest.approx(
        11 / 3 * HARTREE_TO_EV
    )
    loss.backward()
    assert leaves["H-H"].grad is None or not leaves["H-H"].grad.any()
    assert leaves["O-O"].grad is None or not leaves["O-O"].grad.any()
    assert leaves["H-O"].grad[0, 0, 1] == 0
    denominator = 3 if reduction == "elements" else 2
    assert leaves["H-O"].grad[0, 0, 0] == pytest.approx(6 / denominator)
    assert torch.isfinite(leaves["H-O"].grad).all()


@pytest.mark.parametrize("reduction", ["pair_mean", "elements"])
def test_diagnostic_flags_do_not_change_objective_or_gradients(reduction):
    results = []
    for enabled in (False, True):
        model, x, target, prediction, leaves = _case(
            matrix_loss_reduction=reduction,
            objective={"kind": "mae"},
            log_hamiltonian_irrep_contrib_metrics=enabled,
            log_hamiltonian_pair_contrib_metrics=enabled,
        )
        loss, metrics, _ = model._graph_step(
            (x, {"hamiltonian": target}), 0, "train", {"hamiltonian": prediction}
        )
        loss.backward()
        results.append(
            (
                loss.detach(),
                metrics["train/hamiltonian_mae_elementwise"],
                {key: value.grad for key, value in leaves.items()},
            )
        )
    for a, b in zip(results[0][:2], results[1][:2]):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    for key in results[0][2]:
        torch.testing.assert_close(
            results[0][2][key], results[1][2][key], rtol=0, atol=0
        )


@pytest.mark.parametrize(
    "override,match",
    [
        ({"matrix_loss_reduction": "entry"}, "matrix_loss_reduction"),
        ({"train_on_irrep_parts": True}, "irrep"),
        ({"train_target": "irreps"}, "irrep"),
        ({"loss_weighting_mode": "distance_short_range_bias"}, "loss_weighting_mode"),
        ({"data": {"representation": "packed_shells"}}, "packed_shells"),
    ],
)
def test_incompatible_reductions_fail_instead_of_changing_another_objective(
    override, match
):
    values = {"matrix_loss_reduction": "elements", **override}
    with pytest.raises(ValueError, match=match):
        _case(**values)


def test_invalid_reduction_is_rejected_by_configuration_validation():
    assert Config().matrix_loss_reduction == "pair_sum"
    with pytest.raises(ValueError, match="matrix_loss_reduction"):
        validate_configuration_sections({"matrix_loss_reduction": "entry"})
