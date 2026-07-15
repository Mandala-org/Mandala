import pytest
import torch

import net.spectral_loss as spectral_loss
from net.spectral_loss import compute_spectral_eigenvalue_loss
from utils.units import HARTREE_TO_EV


def _payload():
    return {
        "spectral_kpoints_abs": torch.zeros(1, 3),
        "spectral_shifts": torch.zeros(1, 3, dtype=torch.long),
        "spectral_gt_overlap_k": torch.eye(2).unsqueeze(0),
        "spectral_gt_eigs_ev": torch.tensor([[0.0, 2.0]]),
        "spectral_gt_fermi_ev": torch.tensor(0.0),
        "spectral_window_weights": torch.tensor([[1.0, 3.0]]),
    }


@pytest.mark.parametrize(
    ("loss_kind", "expected"),
    [("mae", 1.75), ("mse", 3.25), ("huber", 0.17)],
)
@pytest.mark.unit
def test_spectral_loss_kinds_always_use_ground_truth_overlap(
    monkeypatch, loss_kind, expected
):
    payload = _payload()
    seen = {}
    monkeypatch.setattr(
        spectral_loss,
        "block_matrix_to_shiftspace_dense",
        lambda pred, shifts: torch.zeros(1),
    )
    monkeypatch.setattr(
        spectral_loss,
        "shiftspace_to_kspace_dense",
        lambda shift, **kwargs: torch.zeros(1),
    )

    def fake_eigenvalues(hamiltonian, overlap, **kwargs):
        seen["overlap"] = overlap
        return torch.tensor([[1.0, 4.0]]) / HARTREE_TO_EV

    monkeypatch.setattr(
        spectral_loss, "_generalized_eigenvalues_kspace", fake_eigenvalues
    )

    loss, stats = compute_spectral_eigenvalue_loss(
        pred_hamiltonian=object(),
        spectral_payload=payload,
        box=torch.eye(3),
        loss_kind=loss_kind,
        huber_delta_ev=0.1,
        overlap_psd_cleanup=True,
        overlap_jitter=False,
    )

    assert seen["overlap"] is payload["spectral_gt_overlap_k"]
    assert loss.item() == pytest.approx(expected, abs=1e-6)
    assert stats["spectral_mae_ev"].item() == pytest.approx(1.75)
