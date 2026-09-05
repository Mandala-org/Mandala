import pytest
import torch
from e3nn import o3

from core.orbital_irrep_config import OrbitalIrrepConfig
from pair_hamiltonian.hamgnn_sio2 import HARTREE_TO_MEV
from pair_hamiltonian.metrics import FullBlockMetricAccumulator
from pair_hamiltonian.output_schema import FullBlockIrrepTransform


@pytest.mark.unit
def test_headline_metric_is_reconstructed_matrix_element_weighted():
    config = OrbitalIrrepConfig.from_dict({"X": "1s1p"})
    transform = FullBlockIrrepTransform(config, dtype=torch.float64)
    target_blocks = torch.zeros(2, 4, 4, dtype=torch.float64)
    predicted_blocks = target_blocks.clone()
    predicted_blocks[0] = 1.0
    predicted_blocks[1] = 3.0
    target = transform.blocks_to_irreps("X-X", target_blocks)
    prediction = transform.blocks_to_irreps("X-X", predicted_blocks)
    metrics = FullBlockMetricAccumulator(transform, distance_bin_width_angstrom=1.0)
    metrics.update(
        "X-X",
        prediction,
        target,
        onsite=False,
        distances_angstrom=torch.tensor([0.5, 1.5]),
    )
    result = metrics.compute()
    assert result["matrix_elements"]["mae"] == pytest.approx(2.0 * HARTREE_TO_MEV)
    assert result["matrix_elements"]["scalar_count"] == 32
    assert result["block_frobenius"]["mae"] == pytest.approx(8.0 * HARTREE_TO_MEV)
    assert result["block_frobenius"]["rmse"] == pytest.approx(
        80.0**0.5 * HARTREE_TO_MEV
    )
    assert result["block_frobenius"]["block_count"] == 2
    assert len(result["by_target_irrep_copy"]) == len(transform.schema("X-X").copies)
    assert len(result["by_target_irrep_copy_invariant_norm"]) == len(
        transform.schema("X-X").copies
    )
    assert set(result["by_distance"]) == {
        "0.000-1.000_angstrom",
        "1.000-2.000_angstrom",
    }


@pytest.mark.unit
def test_irrep_norm_metrics_are_rotation_invariant():
    config = OrbitalIrrepConfig.from_dict({"X": "1s1p"})
    transform = FullBlockIrrepTransform(config, dtype=torch.float64)
    target = torch.randn(6, transform.irreps("X-X").dim, dtype=torch.float64)
    prediction = torch.randn_like(target)
    first = FullBlockMetricAccumulator(transform)
    first.update("X-X", prediction, target, onsite=False)
    rotation = o3.rand_matrix(dtype=torch.float64)
    action = transform.output_action("X-X", rotation)
    second = FullBlockMetricAccumulator(transform)
    second.update("X-X", prediction @ action.T, target @ action.T, onsite=False)
    a = first.compute()["by_target_irrep_invariant_norm"]
    b = second.compute()["by_target_irrep_invariant_norm"]
    for label in a:
        assert a[label]["mae"] == pytest.approx(b[label]["mae"], rel=1e-12)
        assert a[label]["rmse"] == pytest.approx(b[label]["rmse"], rel=1e-12)
