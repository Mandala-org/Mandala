import numpy as np
import pytest
import torch

from data.envelope import evaluate_slater_soft_cutoff
from pair_hamiltonian.range_factorization import (
    RangeEnvelopeAccumulator,
    envelope_manifest,
)


@pytest.mark.unit
def test_train_only_range_fit_recovers_smooth_magnitude_curve():
    distances = np.linspace(0.55, 6.45, 600, dtype=np.float64)
    theta = torch.tensor([-1.2, -0.3, 0.7, 6.2, -1.5], dtype=torch.float64)
    magnitude = evaluate_slater_soft_cutoff(
        torch.from_numpy(distances), theta.expand(distances.size, -1)
    ).numpy()
    targets = np.repeat(magnitude[:, None], 9, axis=1)
    accumulator = RangeEnvelopeAccumulator(6.5, 0.1)
    accumulator.update(distances, targets)
    fit = accumulator.fit("O-Si")
    prediction = fit.evaluate(torch.from_numpy(distances)).numpy()
    log_error = np.sqrt(np.mean(np.square(np.log(prediction) - np.log(magnitude))))
    assert fit.fitted_block_count == distances.size
    assert log_error < 0.08


@pytest.mark.unit
def test_envelope_manifest_is_deterministic_and_marks_training_partition():
    distances = np.linspace(0.55, 6.45, 600, dtype=np.float64)
    targets = np.exp(-distances)[:, None]
    accumulator = RangeEnvelopeAccumulator(6.5, 0.1)
    accumulator.update(distances, targets)
    fit = accumulator.fit("O-O")
    first = envelope_manifest([fit], dataset_fingerprint="dataset", split_hash="split")
    second = envelope_manifest([fit], dataset_fingerprint="dataset", split_hash="split")
    assert first == second
    assert first["fit_partition"] == "train"
    assert len(first["content_hash"]) == 64
