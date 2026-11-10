"""Independent trace oracle and registration contract for objective plugins."""

from collections import Counter
from dataclasses import replace

import pytest
import torch
from e3nn import o3

from core.operator_layout import OrbitalFrameRotation
from core.orbital_irrep_config import OrbitalIrrepConfig
from data.block_matrix import BlockMatrix
from examples.layer_cookbook.new_observable import (
    hamiltonian_trace_loss,
    onsite_trace_per_atom,
)
from net.objectives import build_objective_terms, register_objective_term


@pytest.fixture
def matrices():
    # Deliberately interleave onsite, shifted self-image and offsite edges.
    keys = [(1, 0, 0, 0, 0), (0, 0, 0, 1, 1), (0, 0, 0, 0, 1), (0, 0, 0, 0, 0)]
    blocks = torch.full((4, 4, 4), 1000.0, dtype=torch.float64)
    blocks[1] = torch.diag(torch.tensor([5.0, 6.0, 7.0, 8.0], dtype=torch.float64))
    blocks[3] = torch.diag(torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float64))
    blocks.requires_grad_()
    predicted = BlockMatrix(
        atoms=("H", "H"),
        atom_counts=Counter({"H": 2}),
        pair_blocks={"H-H": blocks},
        pair_edges={"H-H": torch.tensor(keys).T},
        lookup={key: ("H-H", row) for row, key in enumerate(keys)},
        orbital_cfg=OrbitalIrrepConfig.from_dict({"H": "1s1p"}),
        basis="e3nn",
    )
    reference = replace(predicted, pair_blocks={"H-H": torch.zeros_like(blocks)})
    return predicted, reference


def test_trace_labels_gradients_and_periodic_images(matrices):
    predicted, reference = matrices
    # (10 + 26) / 2 = 18; the huge shifted-self/offsite blocks contribute zero.
    assert onsite_trace_per_atom(predicted).item() == 18.0
    loss = hamiltonian_trace_loss(
        predictions={"hamiltonian": predicted},
        targets={"hamiltonian": reference},
        inputs={},
    )
    assert loss.item() == 324.0
    loss.backward()
    expected = torch.zeros((4, 4, 4), dtype=torch.float64)
    expected[1] = expected[3] = torch.eye(4, dtype=torch.float64) * 18.0
    torch.testing.assert_close(predicted.pair_blocks["H-H"].grad, expected)
    reference.pair_blocks["H-H"][[1, 3]] = torch.eye(4, dtype=torch.float64)
    changed = hamiltonian_trace_loss(
        predictions={"hamiltonian": predicted},
        targets={"hamiltonian": reference},
        inputs={},
    )
    assert changed.item() == 196.0  # (18 - 4)^2


@pytest.mark.parametrize("parity", [1, -1])
def test_trace_is_invariant_under_orbital_frame_rotation(matrices, parity):
    predicted, _ = matrices
    generator = OrbitalFrameRotation(dtype=torch.float64)
    rotation = o3.rand_matrix(dtype=torch.float64) * parity
    rotated = replace(
        predicted,
        pair_blocks={
            "H-H": generator.rotate_blocks(
                predicted.pair_blocks["H-H"], rotation, [0, 1], [0, 1]
            )
        },
    )
    torch.testing.assert_close(
        onsite_trace_per_atom(rotated), onsite_trace_per_atom(predicted)
    )


@pytest.mark.parametrize("problem", ["missing", "masked", "nonfinite"])
def test_trace_rejects_incomplete_onsite_reference(matrices, problem):
    _, reference = matrices
    if problem == "missing":
        reference.lookup = {
            key: value
            for key, value in reference.lookup.items()
            if key != (0, 0, 0, 0, 0)
        }
    elif problem == "masked":
        mask = torch.ones_like(reference.pair_blocks["H-H"], dtype=torch.bool)
        mask[3, 0, 0] = False
        reference._coverage_masks = {"H-H": mask}
    else:
        reference.pair_blocks["H-H"][3, 0, 0] = torch.nan
    with pytest.raises(ValueError, match="onsite"):
        onsite_trace_per_atom(reference)


def test_registration_options_and_autograd(monkeypatch):
    from net import objectives

    monkeypatch.setattr(objectives, "_OBJECTIVE_TERMS", {})

    @register_objective_term("squared_scalar")
    def squared(*, predictions, targets, inputs, offset=0.0):
        return (predictions["scalar"] - targets["scalar"] - offset).square()

    (term,) = build_objective_terms(
        {
            "terms": [
                {"name": "squared_scalar", "weight": 0.25, "options": {"offset": 1.0}}
            ]
        }
    )
    scalar = torch.tensor(4.0, requires_grad=True)
    loss = term.weight * term(
        predictions={"scalar": scalar}, targets={"scalar": 1.0}, inputs={}
    )
    loss.backward()
    assert loss.item() == 1.0
    assert scalar.grad.item() == 1.0
    with pytest.raises(ValueError, match="already registered"):
        register_objective_term("squared_scalar")(squared)


@pytest.mark.parametrize(
    "terms,match",
    [
        ("hamiltonian_trace", "list"),
        (["hamiltonian_trace"], "mapping"),
        ([{"name": "unregistered"}], "import its registration"),
        ([{"name": "hamiltonian_trace", "wieght": 1.0}], "Unknown"),
        ([{"name": "hamiltonian_trace", "weight": -1.0}], "nonnegative"),
        ([{"name": "hamiltonian_trace", "weight": float("nan")}], "finite"),
        ([{"name": "hamiltonian_trace", "options": []}], "mapping"),
        ([{"name": "hamiltonian_trace", "options": {"inputs": {}}}], "replace"),
        ([{"name": "hamiltonian_trace"}] * 2, "Duplicate"),
    ],
)
def test_invalid_term_configuration_fails(terms, match):
    with pytest.raises(ValueError, match=match):
        build_objective_terms({"terms": terms})


@pytest.mark.parametrize(
    "value", [1.0, torch.ones(2), torch.tensor(1j), torch.tensor(float("nan"))]
)
def test_bad_plugin_return_fails(monkeypatch, value):
    from net import objectives

    monkeypatch.setattr(objectives, "_OBJECTIVE_TERMS", {"bad": lambda **kwargs: value})
    (term,) = build_objective_terms({"terms": [{"name": "bad"}]})
    with pytest.raises(ValueError, match="scalar|nonfinite"):
        term(predictions={}, targets={}, inputs={})


def test_existing_objectives_need_no_plugin():
    assert build_objective_terms(None) == []
    assert build_objective_terms({"kind": "mse"}) == []
