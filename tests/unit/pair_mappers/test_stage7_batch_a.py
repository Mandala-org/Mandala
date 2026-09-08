import json
import sys

import pytest
import torch

from scripts.pair_mappers.audit_stage7_onsite_identity_gauge import (
    _optimal_identity_shifts,
)
from scripts.pair_mappers.freeze_stage7_batch_a import DESCRIPTORS, main as freeze_main
from scripts.pair_mappers.train_stage5_neural_calibration import (
    _restore_generator_state,
    _scheduled_learning_rate,
)


def test_constant_then_cosine_schedule_preserves_first_10k_and_hits_floor():
    settings = {
        "total_steps": 30_000,
        "base_learning_rate": 1.0e-3,
        "minimum_learning_rate": 1.0e-4,
        "decay_start_step": 10_000,
        "schedule": "constant_then_cosine",
    }
    assert _scheduled_learning_rate(1, **settings) == pytest.approx(1.0e-3)
    assert _scheduled_learning_rate(10_000, **settings) == pytest.approx(1.0e-3)
    assert _scheduled_learning_rate(20_000, **settings) == pytest.approx(5.5e-4)
    assert _scheduled_learning_rate(30_000, **settings) == pytest.approx(1.0e-4)


def test_generator_state_restore_uses_portable_cpu_byte_state():
    source = torch.Generator(device="cpu").manual_seed(17)
    state = source.get_state()
    expected = torch.rand(5, generator=source)
    restored = torch.Generator(device="cpu")
    _restore_generator_state(restored, state)
    assert torch.equal(torch.rand(5, generator=restored), expected)


def test_identity_gauge_solver_recovers_common_and_atom_shifts():
    identity = torch.tensor([[1.0, 0.0, 2.0], [0.0, 2.0, 1.0]])
    atom_shifts = torch.tensor([0.3, -0.1])
    residual = atom_shifts[:, None] * identity
    common, atoms = _optimal_identity_shifts(residual, identity)
    assert atoms == pytest.approx(atom_shifts)
    assert common == pytest.approx(0.1)


def test_stage7_freeze_has_four_comparable_offsite_tasks(tmp_path, monkeypatch):
    stage6 = tmp_path / "stage6"
    aggregate = stage6 / "aggregate_v1"
    aggregate.mkdir(parents=True)
    (aggregate / "summary.json").write_text(
        json.dumps({"passed": True, "test_shards_read": False})
    )
    (aggregate / "selected_bundle.json").write_text(
        json.dumps(
            {
                "test_shards_read": False,
                "manifest_hash": "selection",
                "offsite": {"model_id": "offsite_m5_large_lr1em3"},
                "onsite": {"family": "d4", "model_id": "onsite"},
            }
        )
    )
    promotions = tmp_path / "promotions.json"
    promotions.write_text(
        json.dumps(
            {
                "promotions": [
                    {"family": family, "key": key}
                    for family, key in DESCRIPTORS.items()
                ]
            }
        )
    )
    output = tmp_path / "stage7"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "freeze_stage7_batch_a.py",
            "--stage6-root",
            str(stage6),
            "--promotions",
            str(promotions),
            "--output-dir",
            str(output),
        ],
    )
    freeze_main()
    manifest = json.loads(
        (output / "offsite" / "calibration_manifest.json").read_text()
    )
    assert len(manifest["tasks"]) == 4
    assert {task["family"] for task in manifest["tasks"]} == set(DESCRIPTORS)
    assert manifest["steps"] == 30_000
    assert manifest["decay_start_step"] == 10_000
    assert manifest["early_stopping_evaluations"] == 61
    assert [task["warm_start_task_id"] for task in manifest["tasks"]] == [
        "offsite_m5_large_lr1em3",
        None,
        None,
        None,
    ]
