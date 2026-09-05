import pytest
import torch

from core.orbital_irrep_config import OrbitalIrrepConfig
from pair_hamiltonian.hermiticity import (
    directed_hermiticity_relative_error,
    project_directed_irreps,
    project_onsite_irreps,
)
from pair_hamiltonian.output_schema import FullBlockIrrepTransform


@pytest.fixture
def transform():
    return FullBlockIrrepTransform(
        OrbitalIrrepConfig.from_dict({"A": "1s1p", "B": "1s1p"}),
        dtype=torch.float64,
    )


@pytest.mark.unit
def test_projection_uses_two_raw_directed_predictions(transform):
    pair_names = ("A-A", "A-B", "B-A", "B-B")
    pair_types = torch.tensor([1, 2, 0, 0])
    inverse = torch.tensor([1, 0, 3, 2])
    raw = torch.randn(4, transform.irreps("A-A").dim, dtype=torch.float64)
    projected = project_directed_irreps(transform, pair_names, raw, pair_types, inverse)
    assert torch.allclose(
        projected[1], transform.reverse(("A", "B"), projected[0]), atol=1e-12
    )
    assert torch.allclose(
        projected[3], transform.reverse(("A", "A"), projected[2]), atol=1e-12
    )
    # Changing only the independently evaluated reverse direction changes both
    # projected members; one direction is not generated from the other.
    changed = raw.clone()
    changed[1] += 1.0
    changed_projected = project_directed_irreps(
        transform, pair_names, changed, pair_types, inverse
    )
    assert not torch.equal(changed_projected[0], projected[0])
    assert not torch.equal(changed_projected[1], projected[1])
    assert (
        directed_hermiticity_relative_error(
            transform, pair_names, projected, pair_types, inverse
        )
        < 1e-14
    )


@pytest.mark.unit
def test_onsite_projection_is_evaluation_only_operation(transform):
    raw = torch.randn(5, transform.irreps("A-A").dim, dtype=torch.float64)
    projected = project_onsite_irreps(transform, "A", raw)
    assert not torch.equal(raw, projected)
    assert torch.allclose(
        projected, transform.reverse(("A", "A"), projected), atol=1e-12
    )


@pytest.mark.unit
def test_projection_preserves_prediction_dtype_with_high_precision_transform(transform):
    raw = torch.randn(2, transform.irreps("A-B").dim, dtype=torch.float32)
    projected = project_directed_irreps(
        transform,
        ("A-A", "A-B", "B-A", "B-B"),
        raw,
        torch.tensor([1, 2]),
        torch.tensor([1, 0]),
    )
    assert projected.dtype == torch.float32
    assert torch.allclose(
        projected[1],
        transform.reverse(("A", "B"), projected[0]).float(),
        atol=2e-6,
    )


@pytest.mark.unit
def test_directed_projection_rejects_incomplete_reverse_pairs(transform):
    raw = torch.randn(2, transform.irreps("A-B").dim, dtype=torch.float64)
    with pytest.raises(ValueError, match="reverse type"):
        project_directed_irreps(
            transform,
            ("A-A", "A-B", "B-A", "B-B"),
            raw,
            torch.tensor([1, 1]),
            torch.tensor([1, 0]),
        )
