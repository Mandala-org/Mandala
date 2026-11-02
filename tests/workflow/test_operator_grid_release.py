"""Synthetic provenance and scheduler tests; never invoke Slurm or real data."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import release_operator_accuracy as release
from scripts import submit_operator_grid as grid


@pytest.fixture
def certified(tmp_path, monkeypatch):
    records, original = tmp_path / "grid", tmp_path / "certification"
    for root in (records, original):
        (root / "source").mkdir(parents=True)
        (root / "source/SOURCE_MANIFEST.json").write_text(
            json.dumps({"files": {"src/a.py": "same", "scripts/train.py": "train"}})
        )
    (records / "batch").mkdir()
    (original / "results").mkdir()
    (records / "batch/registration.json").write_text('{"runtime_profile":"mandala"}')
    for material, counts in (("graphene", (270, 90)), ("mos2", (300, 100))):
        manifest = records / "batch" / f"{material}_pilot_run_manifest.json"
        manifest.write_text("{}")
        document = {
            "passed": True,
            "test_inputs": 0,
            "input_hashes_match_pilot": True,
            "predictor_configuration_matches_pilot": True,
            "pilot_run_manifest_sha256": release.sha256(manifest),
            "splits": {
                name: {"structures": value}
                for name, value in zip(("train", "val"), counts)
            },
            "checks": {
                key: 0.0
                for key in (
                    "roundtrip_ev",
                    "pipeline_target_change_ev",
                    "energy_roundtrip_ev",
                    "cartesian_once_angstrom",
                    "pipeline_positions_angstrom",
                    "raw_target_hermiticity_max_ev",
                )
            },
            "module_origins": {
                "data.sample": str(original / "source/src/data/sample.py")
            },
        }
        (original / "results" / f"{material}_audit.json").write_text(
            json.dumps(document)
        )
    monkeypatch.setattr(release, "verify", lambda root: None)
    return records, original


def test_release_binds_only_identical_science_and_copies_audit_hashes(certified):
    records, original = certified
    result = release.certify(records, original)
    assert result["decision"] == "release_registered_screen"
    assert result["scientific_source_files_identical"] == 2
    assert result["audit_sha256"]["graphene"] == release.sha256(
        records / "results/graphene_audit.json"
    )
    path = records / "source/SOURCE_MANIFEST.json"
    document = json.loads(path.read_text())
    document["files"]["src/a.py"] = "changed"
    path.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="Scientific source changed"):
        release.certify(records, original)


@pytest.mark.parametrize(
    "mutation", ["passed", "test_inputs", "counts", "units", "origin"]
)
def test_failed_or_incomplete_certification_stays_held(certified, mutation):
    records, original = certified
    path = original / "results/graphene_audit.json"
    document = json.loads(path.read_text())
    if mutation == "passed":
        document["passed"] = False
    elif mutation == "test_inputs":
        document["test_inputs"] = 90
    elif mutation == "counts":
        document["splits"]["val"]["structures"] = 89
    elif mutation == "units":
        document["checks"]["energy_roundtrip_ev"] = 1.0
    else:
        document["module_origins"]["data.sample"] = "/stale/data/sample.py"
    path.write_text(json.dumps(document))
    with pytest.raises(ValueError):
        release.certify(records, original)


def test_legacy_gate_requires_two_material_runtime_checks(certified):
    records, original = certified
    jobs = [
        {"material": material, "treatment": "baseline", "config": material + ".yaml"}
        for material in ("graphene", "mos2")
    ]
    (records / "batch/registration.json").write_text(
        json.dumps({"runtime_profile": "mandala_legacy", "jobs": jobs})
    )
    with pytest.raises(ValueError, match="both material"):
        release.certify(records, original)
    probes = []
    for material in ("graphene", "mos2"):
        config = records / "batch" / (material + ".yaml")
        config.write_text(material)
        path = original / (material + "_runtime.json")
        path.write_text(
            json.dumps(
                {
                    "passed": True,
                    "test_inputs": 0,
                    "config_sha256": release.sha256(config),
                    "model_probe": {
                        "training_backward_finite": True,
                        "raw_validation_outputs_finite": True,
                        "optimizer_steps": 0,
                    },
                    "runtime": {
                        "torch": "2.5.1+cu121",
                        "cuda": "12.1",
                        "gpu": {"name": "Tesla V100-SXM2-32GB"},
                    },
                }
            )
        )
        probes.append(path)
    assert len(release.certify(records, original, probes)["runtime_probes"]) == 2
    invalid = json.loads(probes[0].read_text())
    invalid["runtime"]["cuda"] = "12.8"
    probes[0].write_text(json.dumps(invalid))
    with pytest.raises(ValueError, match="Incorrect V100"):
        release.certify(records, original, probes)


def job(name):
    return {
        "id": name,
        "allocation": {
            "account": "casus",
            "partition": "gpu-a100",
            "cpus": 12,
            "mem": "128G",
            "exclude": "ga[006-009]",
        },
        "allocation_hours": 48,
        "segments": 2,
        "launchers": [name + "_1.sh", name + "_2.sh"],
    }


def test_single_gpu_commands_hold_then_depend_on_success():
    first = grid.planned_command(Path("/frozen"), job("baseline"), 1)
    second = grid.planned_command(Path("/frozen"), job("baseline"), 2, previous="111")
    assert "--hold" in first
    assert "--dependency=afterok:111" in second
    assert "--nodes=1" in first and "--gres=gpu:1" in first
    assert "--exclude=ga[006-009]" in first
    assert not any(value.startswith("--nodelist=") for value in first)


def test_grid_submissions_are_exclusive_and_review_depends_on_certification(
    tmp_path, monkeypatch
):
    records = tmp_path / "grid"
    (records / "batch").mkdir(parents=True)
    (records / "batch/registration.json").write_text(
        json.dumps(
            {
                "jobs": [job("baseline"), job("mse")],
                "runtime_profile": "mandala",
                "budget": {"max_gpu_hours": 384},
            }
        )
    )
    monkeypatch.setattr(grid, "verify", lambda root: None)
    commands = []

    def run(command, **kwargs):
        commands.append(command)
        return SimpleNamespace(
            returncode=0,
            stdout=str(100 + len(commands)),
            stderr="",
            check_returncode=lambda: None,
        )

    monkeypatch.setattr(grid.subprocess, "run", run)
    grid.submit(records, tmp_path / "cert", ["7", "8"])
    assert len(commands) == 5
    assert "--dependency=afterok:101" in commands[1]
    assert "--dependency=afterok:103" in commands[3]
    assert "--dependency=afterok:7:8" in commands[4]
    with pytest.raises(FileExistsError):
        grid.submit(records, tmp_path / "cert", ["7", "8"])
    assert len(commands) == 5
