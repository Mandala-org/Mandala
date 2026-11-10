import json
from pathlib import Path

import pytest

from scripts.reproduce_examples import train

RECIPES = Path(__file__).resolve().parents[3] / "scripts/reproduce_examples"


@pytest.mark.parametrize("name", ["sio2", "zncusnses"])
def test_envelope_checksum_survives_json_formatting(tmp_path, name):
    recipe = json.loads((RECIPES / f"{name}.json").read_text())
    envelope = json.loads((RECIPES / recipe["envelope_file"]).read_text())
    path = tmp_path / "recipe.json"
    path.write_text(json.dumps(recipe))
    target = tmp_path / recipe["envelope_file"]
    for text in [
        json.dumps(envelope),
        json.dumps(envelope, indent=4) + "\n",
        json.dumps(envelope, sort_keys=True) + "\r\n",
    ]:
        target.write_text(text)
        assert train.load_recipe(path)["config"]["hamiltonian_envelope_path"] == str(
            target
        )
    pair = next(iter(envelope["pairs"].values()))
    pair["theta"][0] += 0.01
    target.write_text(json.dumps(envelope))
    with pytest.raises(ValueError, match="checksum mismatch"):
        train.load_recipe(path)


def test_silicon_uses_all_96_available_snapshots():
    recipe = train.load_recipe(RECIPES / "silicon.json")
    assert recipe["training"]["num_train_per_scale"] == 80
    assert recipe["training"]["num_val_per_scale"] == 16
    assert recipe["config"]["data_split_seed"] == 43
    assert recipe["config"]["allow_incomplete_dataset"] is False


def test_failed_setup_can_retry_but_existing_run_is_protected(tmp_path, monkeypatch):
    import torch
    from scripts import train as training

    recipe = json.loads((RECIPES / "silicon.json").read_text())
    recipe["training"]["checkpoint_dir"] = str(tmp_path / "checkpoints")
    recipe["training"]["num_val_per_scale"] = 20
    path = tmp_path / "recipe.json"
    path.write_text(json.dumps(recipe))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda: "test GPU")
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda: (8, 0))

    def fail_setup(*args, **kwargs):
        raise ValueError("insufficient snapshots")

    monkeypatch.setattr(training, "run_training", fail_setup)
    with pytest.raises(ValueError, match="insufficient snapshots"):
        train.main(["--recipe", str(path)])
    recipe["training"]["num_val_per_scale"] = 16
    path.write_text(json.dumps(recipe))
    calls = []
    monkeypatch.setattr(training, "run_training", lambda values: calls.append(values))
    train.main(["--recipe", str(path)])
    assert calls[0]["resume_from_checkpoint"] is None
    assert len(list((tmp_path / "checkpoints").glob("*.failed_setup_*.json"))) == 1
    (tmp_path / "checkpoints/silicon").mkdir()
    recipe["config"]["lr"] *= 2
    path.write_text(json.dumps(recipe))
    with pytest.raises(RuntimeError, match="without a recovery checkpoint"):
        train.main(["--recipe", str(path)])


def test_completed_recovery_checkpoint_skips_training(tmp_path, monkeypatch):
    import torch
    from net import evaluation
    from scripts import train as training

    recipe = json.loads((RECIPES / "silicon_v2.json").read_text())
    recipe["training"]["checkpoint_dir"] = str(tmp_path / "checkpoints")
    path = tmp_path / "recipe.json"
    path.write_text(json.dumps(recipe))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda: "test GPU")
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda: (8, 0))
    calls = []
    monkeypatch.setattr(training, "run_training", lambda values: calls.append(values))
    # Establish exactly the same contract as a fresh first segment.
    train.main(["--recipe", str(path)])
    run_dir = tmp_path / "checkpoints/silicon"
    run_dir.mkdir()
    (run_dir / "latest_checkpoint.pt").touch()
    monkeypatch.setattr(evaluation, "load_checkpoint", lambda path: {
        "epoch_boundary_resume": {"safe_to_resume": True},
        "epoch": recipe["config"]["max_epochs"],
    })
    train.main(["--recipe", str(path)])
    assert len(calls) == 1
