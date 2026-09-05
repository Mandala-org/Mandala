from types import SimpleNamespace

import pytest
import torch

from scripts.pair_mappers.train_stage5_neural_calibration import (
    PAIRS,
    SPECIES,
    _training_loss,
)


class _ZeroMapper:
    def predict_onsite(self, species, descriptor):
        del species
        return torch.zeros(descriptor.shape[0], 1)

    def predict_offsite(self, pair, descriptor_i, displacement, descriptor_j):
        del pair, displacement, descriptor_j
        return torch.zeros(descriptor_i.shape[0], 1)


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
