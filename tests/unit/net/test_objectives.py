"""Objectives are expressions over registered terms."""

from types import SimpleNamespace

import pytest
import torch

from net.common import Config
from net.objectives import (
    Objective,
    ObjectiveContext,
    default_objective,
    objective_catalog,
    parse_expression,
    register_objective_term,
)


def _context(**components):
    model = SimpleNamespace(
        device="cpu", objective_l1_fraction=lambda: 0.25, parameters=lambda: []
    )
    return ObjectiveContext(
        model=model, x={}, y={}, stage="train", components=components
    )


def test_expression_language_is_restricted():
    _, labels = parse_expression("H + 0.5 * max(S, 2 * D) - sqrt(abs(E)) / 3")
    assert labels == {"H", "S", "D", "E"}
    for invalid in ("H ** 2", "__import__('os')", "H if S else D", "max(H)", "H.real"):
        with pytest.raises(ValueError):
            parse_expression(invalid)


def test_matrix_terms_mix_mse_and_mae_and_expression_combines_them():
    objective = Objective(
        {
            "terms": {
                "H": {"term": "matrix_error", "operator": "hamiltonian"},
                "S": {
                    "term": "matrix_error",
                    "operator": "overlap",
                    "l1_fraction": 1.0,
                },
            },
            "expression": "H + 0.5 * S",
        }
    )
    context = _context(
        **{
            "matrix_mse:hamiltonian": torch.tensor(4.0),
            "matrix_mae:hamiltonian": torch.tensor(8.0),
            "matrix_mse:overlap": torch.tensor(1.0),
            "matrix_mae:overlap": torch.tensor(3.0),
        }
    )
    total, values = objective.evaluate(context)
    assert float(values["H"]) == pytest.approx(0.75 * 4 + 0.25 * 8)
    assert float(values["S"]) == pytest.approx(3.0)
    assert float(total) == pytest.approx(5.0 + 1.5)


def test_flat_configuration_lowers_to_its_historical_weighting():
    cfg = Config(
        matrix_targets=["hamiltonian", "overlap"],
        objective={"weights": {"overlap": 0.1}},
        train_on_energy=True,
        loss_coef_observables=0.01,
        l2_reg_coef=1e-6,
    )
    objective = default_objective(cfg)
    assert set(objective["terms"]) == {
        "matrix_hamiltonian",
        "matrix_overlap",
        "energy",
        "l2",
    }
    assert objective["expression"] == (
        "1.0 * matrix_hamiltonian + 0.1 * matrix_overlap + 0.01 * energy + 1e-06 * l2"
    )
    Objective(objective)


def test_custom_terms_register_and_reject_undefined_labels():
    @register_objective_term("test_constant", category="custom", summary="A constant")
    def constant(context):
        return torch.tensor(2.0)

    assert any(term["name"] == "test_constant" for term in objective_catalog())
    total, _ = Objective(
        {"terms": {"c": {"term": "test_constant"}}, "expression": "3 * c"}
    ).evaluate(_context())
    assert float(total) == 6.0
    with pytest.raises(ValueError, match=r"uses d, not a term of the objective \(terms: c\)"):
        Objective({"terms": {"c": {"term": "test_constant"}}, "expression": "c + d"})
    with pytest.raises(ValueError, match="Unknown objective term"):
        Objective({"terms": {"c": {"term": "no_such_term"}}})
    with pytest.raises(ValueError, match="needs"):
        Objective({"terms": {"s": {"term": "spectral"}}}).evaluate(_context())


def test_terms_declare_the_quantities_they_need():
    objective = Objective(
        {"terms": {"F": {"term": "forces"}, "E": {"term": "band_energy"}}}
    )
    assert objective.requirements() == {"forces", "energy"}
    with pytest.raises(ValueError, match="enable_forces"):
        objective.check_requirements(Config(enable_forces=False, enable_energy=True))
    objective.check_requirements(Config(enable_forces=True, enable_energy=True))


def test_operator_ports_are_declared_options():
    from net.objectives import register_objective_term
    from net.registry import Option

    with pytest.raises(ValueError, match="must be declared options"):
        register_objective_term("bad_ports", operators=("left",))(lambda c: c)
    register_objective_term(
        "operator_gap",
        operators=("left", "right"),
        options=[Option("left", "str"), Option("right", "str")],
        override=True,
    )(lambda context, left, right: context.zero())
    from net.objectives import _TERMS

    term = _TERMS["operator_gap"]
    assert term.to_json()["operators"] == ["left", "right"]
    assert term.operator_inputs({"left": "hamiltonian", "right": None}) == {
        "left": "hamiltonian"
    }
