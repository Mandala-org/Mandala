from types import SimpleNamespace

import pytest
import torch

import net.observable_metrics as observable_metrics
from net.observable_metrics import (
    observable_loss,
    rescale_density_prediction_to_num_electrons,
)


@pytest.mark.parametrize(
    ("loss_kind", "expected_raw"),
    [("mse", 4.0), ("mae", 2.0)],
)
@pytest.mark.unit
def test_observable_loss_kind_for_fully_predicted_energy(loss_kind, expected_raw):
    cfg = SimpleNamespace(
        train_on_energy=True,
        train_on_num_electrons=False,
        train_observables_on_gt=False,
        loss_coef_observables=0.25,
        observable_loss_kind=loss_kind,
    )

    energy_loss, electron_loss = observable_loss(
        cfg=cfg,
        observable_values={"energy": torch.tensor(3.0)},
        energy_target=torch.tensor(1.0),
        num_electrons_target=None,
        mse_fn=torch.nn.functional.mse_loss,
        device=torch.device("cpu"),
    )

    assert energy_loss.item() == pytest.approx(0.25 * expected_raw)
    assert electron_loss.item() == 0.0


@pytest.mark.unit
def test_observable_mae_half_gt_averages_both_one_sided_energies():
    cfg = SimpleNamespace(
        train_on_energy=True,
        train_on_num_electrons=False,
        train_observables_on_gt=True,
        loss_coef_observables=0.5,
        observable_loss_kind="mae",
    )

    energy_loss, _ = observable_loss(
        cfg=cfg,
        observable_values={
            "energy_gt_hamiltonian": torch.tensor(4.0),
            "energy_gt_density": torch.tensor(0.0),
        },
        energy_target=torch.tensor(1.0),
        num_electrons_target=None,
        mse_fn=torch.nn.functional.mse_loss,
        device=torch.device("cpu"),
    )

    assert energy_loss.item() == pytest.approx(0.5 * ((3.0 + 1.0) / 2.0))


@pytest.mark.unit
def test_density_rescaling_uses_predicted_overlap_and_detaches_metric_scale(
    monkeypatch,
):
    class ScalableDensity:
        def __init__(self, scale=None):
            self.scale = scale

        def __mul__(self, scale):
            return ScalableDensity(scale)

    density = ScalableDensity()
    predicted_overlap = object()
    predicted_count = torch.tensor(5.0, requires_grad=True)
    seen = {}

    def fake_trace(density_arg, overlap_arg, alignment_arg):
        seen["density"] = density_arg
        seen["overlap"] = overlap_arg
        seen["alignment"] = alignment_arg
        return predicted_count

    monkeypatch.setattr(
        observable_metrics, "trace_matmul_sparse_block_matrix_aligned", fake_trace
    )
    scaled, count = rescale_density_prediction_to_num_electrons(
        {"density": density, "overlap": predicted_overlap},
        trace_alignment="alignment",
        num_electrons_target=torch.tensor(10.0),
        overlap_true=object(),
    )

    assert seen == {
        "density": density,
        "overlap": predicted_overlap,
        "alignment": "alignment",
    }
    assert count is predicted_count
    assert scaled["density"].scale == pytest.approx(2.0)
    assert isinstance(scaled["density"].scale, float)
