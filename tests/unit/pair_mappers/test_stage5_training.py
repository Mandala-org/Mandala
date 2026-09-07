from types import SimpleNamespace

import pytest
import torch

from core.orbital_irrep_config import OrbitalIrrepConfig
from pair_hamiltonian.output_schema import FullBlockIrrepTransform
from scripts.pair_mappers.train_stage5_neural_calibration import (
    PAIRS,
    SPECIES,
    _invariant_onsite_means,
    _training_loss,
)


class _ZeroMapper:
    def predict_onsite(self, species, descriptor):
        del species
        return torch.zeros(descriptor.shape[0], 1)

    def predict_offsite(self, pair, descriptor_i, displacement, descriptor_j):
        del pair, displacement, descriptor_j
        return torch.zeros(descriptor_i.shape[0], 1)


class _ScopedMapper(_ZeroMapper):
    def __init__(self, forbidden):
        self.forbidden = forbidden

    def predict_onsite(self, species, descriptor):
        if self.forbidden == "onsite":
            raise AssertionError("onsite path must not be called")
        return super().predict_onsite(species, descriptor)

    def predict_offsite(self, pair, descriptor_i, displacement, descriptor_j):
        if self.forbidden == "offsite":
            raise AssertionError("offsite path must not be called")
        return super().predict_offsite(pair, descriptor_i, displacement, descriptor_j)


@pytest.mark.unit
def test_calibration_loss_matches_block_count_weighted_global_mse():
    counts = [1, 2, 3, 4, 5, 6]
    targets = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    data = {
        "descriptor": torch.zeros(1, 1),
        "onsite": {},
        "offsite": {},
    }
    for species, count, target in zip(SPECIES, counts[:2], targets[:2]):
        data["onsite"][species] = {
            "descriptor": torch.zeros(count, 1),
            "target": torch.full((count, 1), target),
        }
    for pair, count, target in zip(PAIRS, counts[2:], targets[2:]):
        data["offsite"][pair] = {
            "source": torch.zeros(count, dtype=torch.long),
            "target_index": torch.zeros(count, dtype=torch.long),
            "displacement": torch.ones(count, 3),
            "target": torch.full((count, 1), target),
            "envelope": torch.logspace(-4, -1, count),
        }
    args = SimpleNamespace(
        onsite_batch_size=10,
        offsite_batch_size=10,
        range_loss_mode="physical_mse",
        onsite_loss_weight=None,
    )
    losses = []
    for mode in ("physical_mse", "weighted_normalized_mse"):
        args.range_loss_mode = mode
        loss, sampled = _training_loss(
            _ZeroMapper(),
            data,
            args,
            torch.Generator().manual_seed(1),
            torch.device("cpu"),
        )
        losses.append(loss)
    expected = sum(count * target**2 for count, target in zip(counts, targets)) / sum(
        counts
    )
    assert float(losses[0]) == pytest.approx(expected)
    assert torch.allclose(losses[0], losses[1], rtol=1e-6, atol=1e-6)
    assert sampled == sum(counts)

    args.range_loss_mode = "physical_mse"
    args.onsite_loss_weight = 0.5
    balanced, _ = _training_loss(
        _ZeroMapper(),
        data,
        args,
        torch.Generator().manual_seed(1),
        torch.device("cpu"),
    )
    onsite_expected = sum(
        count * target**2 for count, target in zip(counts[:2], targets[:2])
    ) / sum(counts[:2])
    offsite_expected = sum(
        count * target**2 for count, target in zip(counts[2:], targets[2:])
    ) / sum(counts[2:])
    assert float(balanced) == pytest.approx(0.5 * (onsite_expected + offsite_expected))


@pytest.mark.unit
@pytest.mark.parametrize(
    ("scope", "forbidden", "expected"),
    (("onsite", "offsite", 2.5), ("offsite", "onsite", 21.5)),
)
def test_independent_training_loss_never_calls_the_other_scope(
    scope, forbidden, expected
):
    data = {
        "descriptor": torch.zeros(1, 1),
        "onsite": {
            "O": {"descriptor": torch.zeros(1, 1), "target": torch.ones(1, 1)},
            "Si": {"descriptor": torch.zeros(1, 1), "target": torch.full((1, 1), 2.0)},
        },
        "offsite": {
            pair: {
                "source": torch.zeros(1, dtype=torch.long),
                "target_index": torch.zeros(1, dtype=torch.long),
                "displacement": torch.ones(1, 3),
                "target": torch.full((1, 1), float(index + 3)),
                "envelope": torch.ones(1),
            }
            for index, pair in enumerate(PAIRS)
        },
    }
    args = SimpleNamespace(
        onsite_batch_size=1,
        offsite_batch_size=1,
        range_loss_mode="physical_mse",
        onsite_loss_weight=None,
        target_scope=scope,
    )
    loss, sampled = _training_loss(
        _ScopedMapper(forbidden),
        data,
        args,
        torch.Generator().manual_seed(1),
        torch.device("cpu"),
    )
    assert float(loss) == pytest.approx(expected)
    assert sampled == (2 if scope == "onsite" else 4)


@pytest.mark.unit
def test_onsite_mean_baseline_is_invariant_and_hermitian():
    transform = FullBlockIrrepTransform(
        OrbitalIrrepConfig.from_dict({"O": "1s1p", "Si": "1s1p"}),
        dtype=torch.float64,
    )
    data = {"onsite": {}}
    for species in SPECIES:
        dimension = transform.irreps((species, species)).dim
        data["onsite"][species] = {
            "target": torch.randn(9, dimension, dtype=torch.float64)
        }
    means = _invariant_onsite_means(data, transform)
    for species, mean in means.items():
        offset = 0
        for multiplicity, irrep in transform.irreps((species, species)):
            width = multiplicity * irrep.dim
            if not (irrep.l == 0 and irrep.p == 1):
                assert torch.count_nonzero(mean[offset : offset + width]) == 0
            offset += width
        assert torch.allclose(
            mean,
            transform.reverse((species, species), mean),
            atol=1e-12,
            rtol=1e-12,
        )
