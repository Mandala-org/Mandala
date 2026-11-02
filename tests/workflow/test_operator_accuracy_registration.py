"""Small-file-only controls for immutable, validation-only registrations."""

import base64
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

from omegaconf import OmegaConf
import pytest

from scripts import collect_operator_pilot_evidence as collector
from scripts import prepare_operator_accuracy as registration
from scripts import submit_operator_accuracy as submitter


@pytest.fixture
def inputs(tmp_path, monkeypatch):
    repository = Path(__file__).resolve().parents[2]
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    manifest = {"source_snapshot_sha256": "f" * 64, "files": {}}
    for material in ("graphene", "mos2"):
        relative = Path("configs/pilots") / f"{material}_mandala.yaml"
        local = tmp_path / relative
        local.parent.mkdir(parents=True, exist_ok=True)
        local.write_bytes((repository / relative).read_bytes())
        manifest["files"][str(relative)] = registration.sha256(local)
        directory = evidence / material
        directory.mkdir()
        splits = {
            name: [{"matrix": f"/bigdata/{material}/{name}_ONLY/hamiltonians.h5"}]
            for name in ("train", "val", "test")
        }
        (directory / "split_manifest.json").write_text(
            json.dumps({"splits": splits, "split_hash_sha256": material * 8})
        )
        (directory / "run_manifest.json").write_text('{"frozen_pilot": true}')
    (evidence / "source_manifest.json").write_text(json.dumps(manifest))
    monkeypatch.chdir(tmp_path)
    return evidence, tmp_path / "records"


@pytest.fixture
def prepared(inputs):
    evidence, records = inputs
    batch = records / "batch"
    registration.prepare(evidence, batch, Path("/remote/records"))
    source = records / "source"
    source.mkdir()
    (source / "frozen.py").write_text("# frozen source\n")
    (source / "SOURCE_MANIFEST.json").write_text(
        json.dumps({"files": {"frozen.py": registration.sha256(source / "frozen.py")}})
    )
    return records


def load_registration(records):
    return json.loads((records / "batch/registration.json").read_text())


def test_exact_registered_grid_has_only_prespecified_changes(prepared):
    document = load_registration(prepared)
    common = {
        "baseline": {},
        "mse": {"loss_l1_fraction": 0.0},
        "rollback_off": {"revert_on_spike": False},
        "mse_rollback_off": {"loss_l1_fraction": 0.0, "revert_on_spike": False},
        "adamw_decay0": {"optimization.optimizer.weight_decay": 0.0},
        "radial_copy": {"radial_weight_granularity": "copy"},
    }
    assert len(document["jobs"]) == 14
    assert document["budget"]["max_gpu_hours"] == 84
    assert document["budget"]["array_concurrency"] == 2
    for material in ("graphene", "mos2"):
        expected = deepcopy(common)
        if material == "mos2":
            expected.update(
                pair_mean={"matrix_loss_reduction": "pair_mean"},
                elements={"matrix_loss_reduction": "elements"},
            )
        jobs = [j for j in document["jobs"] if j["material"] == material]
        assert {j["treatment"] for j in jobs} == set(expected)
        baseline = OmegaConf.load(prepared / "batch" / f"{material}_baseline_s42.yaml")
        for job in jobs:
            assert job["changes_from_baseline"] == expected[job["treatment"]]
            controlled = deepcopy(baseline)
            controlled.run_name = job["id"]
            for key, value in job["changes_from_baseline"].items():
                OmegaConf.update(controlled, key, value)
            actual = OmegaConf.load(prepared / "batch" / job["config"])
            assert OmegaConf.to_container(actual) == OmegaConf.to_container(controlled)
            assert (
                actual.max_epochs
                == actual.training.max_epochs
                == job["max_epochs"]
                == 48
            )
            assert job["internal_seconds"] == 18000
            assert job["allocation_hours"] == 6
            assert job["seed"] == job["data_split_seed"] == 42
            assert actual.seed == actual.training.seed == 42
            assert actual.data_split_seed == actual.training.data_split_seed == 42


def test_executable_configs_exclude_reserved_test_and_share_controller(prepared):
    document = load_registration(prepared)
    expected_scheduler = None
    expected_optimizer = None
    reserved = json.loads((prepared / "batch/reserved_membership.json").read_text())
    assert "test_ONLY" in json.dumps(reserved)
    for path in (prepared / "batch").glob("*.yaml"):
        text = path.read_text()
        assert "test_ONLY" not in text
        config = OmegaConf.load(path)
        assert config.data.splits.test == []
        assert "split" not in config.training
        assert len(config.data.splits.train) == len(config.data.splits.val) == 1
        assert "train_ONLY" in config.data.splits.train[0]
        assert "val_ONLY" in config.data.splits.val[0]
        scheduler = OmegaConf.to_container(config.optimization.scheduler)
        optimizer = OmegaConf.to_container(config.optimization.optimizer)
        optimizer.pop("weight_decay")
        expected_scheduler = expected_scheduler or scheduler
        expected_optimizer = expected_optimizer or optimizer
        assert scheduler == expected_scheduler
        assert optimizer == expected_optimizer
        assert config.checkpoint_monitor == "val/hamiltonian_mae_elementwise"
        assert (
            scheduler["monitor"]
            == config.lr_scheduler_target
            == config.checkpoint_monitor
        )
    for job in document["jobs"]:
        launcher = (prepared / "batch" / (job["id"] + ".sh")).read_text()
        assert "--evaluate-final-test" not in launcher
        assert "--segment 1 --budget-seconds 18000" in launcher
    assert " gate --records " in (prepared / "batch/screen_array.sh").read_text()


@pytest.mark.parametrize("inherited_pythonpath", [False, True])
def test_all_launchers_import_frozen_source_before_competing_packages(
    inputs, tmp_path, inherited_pythonpath
):
    evidence, _ = inputs
    records = tmp_path / "revision with spaces"
    batch = records / "batch"
    registration.prepare(evidence, batch, records)
    export_lines = []
    for launcher in sorted(batch.glob("*.sh")):
        lines = launcher.read_text().splitlines()
        exports = [line for line in lines if line.startswith("export PYTHONPATH=")]
        assert len(exports) == 1
        export_lines.extend(exports)
        assert lines.index("source ~/scripts/profiles/mandala.sh") < lines.index(
            exports[0]
        )
        assert lines.index(exports[0]) < next(
            index for index, line in enumerate(lines) if line.startswith("python ")
        )
    assert len(export_lines) == 17  # two audits, fourteen arms, one array.
    assert len(set(export_lines)) == 1

    frozen_src, frozen_root = records / "source/src", records / "source"
    competing = tmp_path / "competing packages"
    packages = {
        frozen_src / "data": "frozen-src",
        frozen_root / "scripts": "frozen-root",
        competing / "data": "stale",
        competing / "scripts": "stale",
    }
    for package, origin in packages.items():
        package.mkdir(parents=True, exist_ok=True)
        # Match the repository's namespace packages, including the stale tree.
        (package / "sample.py").write_text(f"ORIGIN = {origin!r}\n")
    environment = os.environ.copy()
    if inherited_pythonpath:
        environment["PYTHONPATH"] = str(competing)
    else:
        environment.pop("PYTHONPATH", None)
    probe = (
        "import json, os, data.sample, scripts.sample; "
        "print(json.dumps({'data': data.sample.ORIGIN, 'scripts': scripts.sample.ORIGIN, "
        "'data_file': data.sample.__file__, 'scripts_file': scripts.sample.__file__, "
        "'pythonpath': os.environ['PYTHONPATH'].split(':')}))"
    )
    command = export_lines[0] + "\nexec " + shlex.join([sys.executable, "-c", probe])
    result = subprocess.run(
        ["bash", "-c", command],
        cwd=tmp_path,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    actual = json.loads(result.stdout)
    assert actual["data"] == "frozen-src"
    assert actual["scripts"] == "frozen-root"
    assert Path(actual["data_file"]) == frozen_src / "data/sample.py"
    assert Path(actual["scripts_file"]) == frozen_root / "scripts/sample.py"
    expected_pythonpath = [str(frozen_src), str(frozen_root)]
    if inherited_pythonpath:
        expected_pythonpath.append(str(competing))
    assert actual["pythonpath"] == expected_pythonpath


def test_preparation_is_immutable_and_digests_cover_every_output(prepared, inputs):
    document = load_registration(prepared)
    actual = {
        p.name for p in (prepared / "batch").iterdir() if p.name != "registration.json"
    }
    assert set(document["files"]) == actual
    before = {p.name: p.read_bytes() for p in (prepared / "batch").iterdir()}
    registration.verify(prepared)
    with pytest.raises(FileExistsError):
        registration.prepare(inputs[0], prepared / "batch", Path("/remote/records"))
    assert before == {p.name: p.read_bytes() for p in (prepared / "batch").iterdir()}


def test_changed_local_pilot_is_rejected_before_destination_creation(inputs):
    evidence, records = inputs
    with Path("configs/pilots/mos2_mandala.yaml").open("a") as stream:
        stream.write("\n# changed after pilot freeze\n")
    with pytest.raises(ValueError, match="differs from frozen remote"):
        registration.prepare(evidence, records / "batch", Path("/remote/records"))
    assert not (records / "batch").exists()


def test_explicit_supplemental_source_manifest_is_supported(inputs):
    evidence, records = inputs
    supplemental = evidence.parent / "supplemental_source.json"
    (evidence / "source_manifest.json").rename(supplemental)
    registration.main(
        [
            "prepare",
            "--evidence",
            str(evidence),
            "--destination",
            str(records / "batch"),
            "--remote-records",
            "/remote/records",
            "--source-manifest",
            str(supplemental),
        ]
    )
    assert len(load_registration(records)["jobs"]) == 14


@pytest.mark.parametrize(
    "changed",
    [
        "source/frozen.py",
        "batch/mos2_baseline_s42.yaml",
        "batch/screen_array.sh",
        "batch/reserved_membership.json",
    ],
)
def test_verification_rejects_changed_frozen_input(prepared, changed):
    with (prepared / changed).open("a") as stream:
        stream.write("\n# mutation\n")
    with pytest.raises(ValueError, match="Frozen .* changed"):
        registration.verify(prepared)


def install_release(records):
    (records / "workflow").mkdir()
    (records / "results").mkdir()
    review = {
        "decision": "release_registered_screen",
        "registration_sha256": registration.sha256(records / "batch/registration.json"),
        "audit_sha256": {},
    }
    for material in ("graphene", "mos2"):
        path = records / "results" / f"{material}_audit.json"
        path.write_text('{"passed": true}')
        review["audit_sha256"][material] = registration.sha256(path)
    (records / "workflow/certification_review.json").write_text(json.dumps(review))
    return review


def test_gate_requires_explicit_release_and_rechecks_frozen_source(prepared):
    with pytest.raises(FileNotFoundError):
        registration.gate(prepared)
    install_release(prepared)
    registration.gate(prepared)
    (prepared / "source/frozen.py").write_text("changed\n")
    with pytest.raises(ValueError, match="Frozen source changed"):
        registration.gate(prepared)


@pytest.mark.parametrize(
    "failure",
    ["decision", "registration", "audit_hash", "audit_failed", "truthy_not_boolean"],
)
def test_gate_fails_closed_for_missing_or_changed_certification(prepared, failure):
    review = install_release(prepared)
    if failure == "decision":
        review["decision"] = "pending"
    elif failure == "registration":
        review["registration_sha256"] = "0" * 64
    elif failure == "audit_hash":
        review["audit_sha256"]["graphene"] = "0" * 64
    else:
        path = prepared / "results/graphene_audit.json"
        path.write_text(
            json.dumps({"passed": False if failure == "audit_failed" else "false"})
        )
        review["audit_sha256"]["graphene"] = registration.sha256(path)
    (prepared / "workflow/certification_review.json").write_text(json.dumps(review))
    with pytest.raises(ValueError):
        registration.gate(prepared)


def test_collect_preserves_exact_bytes_and_never_overwrites(tmp_path, monkeypatch):
    raw = b"\xff bounded UTF-8 tail\n"
    payload = {
        "captured_utc": "2026-10-29T17:19:27+00:00",
        "files": {
            "log_tails/a.txt": {
                "source": "/remote/log",
                "sha256": hashlib.sha256(raw).hexdigest(),
                "content_base64": base64.b64encode(raw).decode(),
                "tail_bytes": 32768,
            }
        },
        "workflow": {},
    }
    calls = []

    def fake_run(command, **kwargs):
        calls.append(command)
        assert "best_test_evaluation" not in kwargs["input"]
        return subprocess.CompletedProcess(command, 0, stdout=json.dumps(payload))

    monkeypatch.setattr(collector.subprocess, "run", fake_run)
    destination = tmp_path / "capture"
    collector.collect("fixture-host", destination)
    assert (destination / "evidence/log_tails/a.txt").read_bytes() == raw
    with pytest.raises(FileExistsError):
        collector.collect("fixture-host", destination)
    assert len(calls) == 1


@pytest.mark.parametrize("failure", ["path_escape", "checksum"])
def test_collect_rejects_invalid_remote_record(tmp_path, monkeypatch, failure):
    payload = {
        "captured_utc": "2026-10-29T17:19:27+00:00",
        "files": {
            "../escaped" if failure == "path_escape" else "data.csv": {
                "sha256": "0" * 64,
                "content_base64": base64.b64encode(b"data").decode(),
            }
        },
        "workflow": {},
    }
    monkeypatch.setattr(
        collector.subprocess,
        "run",
        lambda command, **kwargs: subprocess.CompletedProcess(
            command, 0, stdout=json.dumps(payload)
        ),
    )
    with pytest.raises(ValueError):
        collector.collect("fixture-host", tmp_path / "capture")
    assert not (tmp_path / "escaped").exists()


@pytest.mark.parametrize("failure", [None, "missing_manifest", "source_identity"])
def test_remote_collection_requires_frozen_manifests_and_excludes_test_artifacts(
    tmp_path, monkeypatch, capsys, failure
):
    pilots = tmp_path / "repo/researchops/runs/operator_benchmark_pilots_v1"
    pilots.mkdir(parents=True)
    models = tmp_path / "models"
    frozen = tmp_path / "frozen"
    frozen.mkdir()
    (pilots / "jobs.json").write_text(
        json.dumps(
            {"pilot_source": {"path": str(frozen), "source_snapshot_sha256": "f" * 64}}
        )
    )
    (pilots / "datasets.json").write_text("{}")
    (frozen / "SOURCE_MANIFEST.json").write_text(
        json.dumps(
            {
                "source_snapshot_sha256": (
                    "0" * 64 if failure == "source_identity" else "f" * 64
                )
            }
        )
    )
    for material in ("graphene", "mos2"):
        metrics = pilots / material / "metrics/version_0"
        metrics.mkdir(parents=True)
        (metrics / "metrics.csv").write_text("val/mae\n1\n")
        (pilots / material / "segment_1.json").write_text("{}")
        (pilots / material / "best_val_evaluation.json").write_text("{}")
        (pilots / material / "best_test_evaluation.json").write_text('{"secret": 9}')
        (pilots / material / "unapproved.csv").write_text("secret\n9\n")
        directory = models / material
        directory.mkdir(parents=True)
        for name in ("split_manifest.json", "run_manifest.json"):
            (directory / name).write_text("{}")
    (pilots / "graphene_s2_99.out").write_text("secret final test results")
    if failure == "missing_manifest":
        (models / "mos2/run_manifest.json").unlink()
    code = collector.REMOTE.replace(
        'root = Path.home() / "casus/mandala"',
        f"root = Path({str(tmp_path / 'repo')!r})",
    )
    code = code.replace(
        'Path("/bigdata/casus/wdm/hamiltonian_learning/models/checkpoints/operator_benchmark_pilots_v1")',
        f"Path({str(models)!r})",
    )
    monkeypatch.setattr(
        collector.subprocess,
        "run",
        lambda command, **kwargs: subprocess.CompletedProcess(
            command, 0, stdout="", stderr=""
        ),
    )
    if failure:
        with pytest.raises((FileNotFoundError, ValueError)):
            exec(compile(code, "<remote collection fixture>", "exec"), {})
    else:
        exec(compile(code, "<remote collection fixture>", "exec"), {})
        captured = json.loads(capsys.readouterr().out)
        assert captured["files"]["source_manifest.json"]["source"] == str(
            frozen / "SOURCE_MANIFEST.json"
        )
        assert all(
            "test_evaluation" not in path
            and "s2_" not in path
            and "unapproved" not in path
            for path in captured["files"]
        )


def test_submission_commands_preserve_registered_resources_without_submitting(
    prepared, monkeypatch, capsys
):
    audits = submitter.commands(prepared, "audits")
    screens = submitter.commands(prepared, "screen")
    assert len(audits) == 2
    assert len(screens) == 1
    assert {Path(c[-1]).name for c in audits} == {"audit_graphene.sh", "audit_mos2.sh"}
    for command in audits + screens:
        assert command[:2] == ["sbatch", "--parsable"]
        assert "--account=casus" in command
        assert "--partition=gpu-h100-casus" in command
        assert "--gres=gpu:1" in command
        assert f"--chdir={prepared / 'source'}" in command
    assert all("--time=02:00:00" in c for c in audits)
    assert "--time=06:00:00" in screens[0]
    assert "--array=0-13%2" in screens[0]
    assert "--signal=INT@600" in screens[0]
    assert 14 * 6 == load_registration(prepared)["budget"]["max_gpu_hours"]

    def no_submission(*args, **kwargs):
        raise AssertionError("A dry run must not invoke sbatch")

    monkeypatch.setattr(submitter.subprocess, "run", no_submission)
    submitter.submit(prepared, "audits", dry_run=True)
    assert json.loads(capsys.readouterr().out) == audits
    install_release(prepared)
    submitter.submit(prepared, "screen", dry_run=True)
    assert json.loads(capsys.readouterr().out) == screens


def test_long_grid_prespecifies_all_seeds_segments_and_full_upper_budget(inputs):
    evidence, records = inputs
    batch = records / "batch"
    registration.prepare(
        evidence,
        batch,
        Path("/remote/long records"),
        seeds=(42, 43, 44),
        max_epochs=500,
        segments=2,
        allocation_hours=48,
        internal_hours=45,
        endpoints=(48, 200, 500),
        namespace="mandala_accuracy_v1_long",
    )
    document = load_registration(records)
    assert document["id"] == "mandala_accuracy_v1_long"
    assert document["initialization_seeds"] == [42, 43, 44]
    assert document["data_split_seed"] == 42
    assert document["budget"]["candidates"] == len(document["jobs"]) == 42
    assert document["budget"]["max_gpu_hours"] == 4032
    assert document["budget"]["max_internal_gpu_hours"] == 3780
    assert document["budget"]["segments_per_candidate"] == 2
    assert document["budget"]["array_concurrency"] is None
    assert not (batch / "screen_array.sh").exists()
    assert "no validation-selected promotion" in document["purpose"]
    for material, count in (("graphene", 6), ("mos2", 8)):
        for seed in (42, 43, 44):
            jobs = [
                job
                for job in document["jobs"]
                if job["material"] == material and job["seed"] == seed
            ]
            assert len(jobs) == count
            baseline = OmegaConf.load(batch / f"{material}_baseline_s{seed}.yaml")
            for job in jobs:
                assert job["hardware_block"] == f"{material}_s{seed}"
                assert job["max_epochs"] == 500
                assert job["segments"] == 2
                assert job["internal_seconds"] == 162000
                assert job["max_internal_seconds"] == 324000
                assert job["allocation_hours"] == 48
                assert job["max_gpu_hours"] == 96
                assert job["primary_endpoints_epochs"] == [48, 200, 500]
                config = OmegaConf.load(batch / job["config"])
                controlled = deepcopy(baseline)
                controlled.run_name = job["id"]
                for key, value in job["changes_from_baseline"].items():
                    OmegaConf.update(controlled, key, value)
                assert OmegaConf.to_container(config) == OmegaConf.to_container(
                    controlled
                )
                assert config.seed == config.training.seed == seed
                assert config.data_split_seed == config.training.data_split_seed == 42
                assert config.max_epochs == config.training.max_epochs == 500
                assert config.max_wall_clock_seconds == 162000
                assert config.data.splits.test == []
                assert "test_ONLY" not in (batch / job["config"]).read_text()
                assert config.save_dir.endswith("/mandala_accuracy_v1_long")
                assert len(job["launchers"]) == 2
                for segment, filename in enumerate(job["launchers"], start=1):
                    launcher = (batch / filename).read_text()
                    assert (
                        f"--segments 2 --segment {segment} --budget-seconds 162000"
                        in launcher
                    )
                    assert f"mandala_accuracy_v1_long/{job['id']}" in launcher
                    assert f"export PYTHONHASHSEED={seed}" in launcher
                    assert "export CUBLAS_WORKSPACE_CONFIG=:4096:8" in launcher
                    assert (
                        "export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1"
                        in launcher
                    )
                    assert "exec python -u -m scripts.run_training_segment" in launcher
                    assert " gate --records " in launcher
                    assert "--evaluate-final-test" not in launcher
                    assert "This frozen launcher takes no arguments." in launcher
                    subprocess.run(["bash", "-n", str(batch / filename)], check=True)
    for material in ("graphene", "mos2"):
        audit = (batch / f"audit_{material}.sh").read_text()
        assert f"/mandala_accuracy_v1_long/pilot_snapshots/{material}.pt" in audit
        assert "test_ONLY" not in audit
    actual = {path.name for path in batch.iterdir() if path.name != "registration.json"}
    assert set(document["files"]) == actual
    assert all(
        document["files"][name] == registration.sha256(batch / name) for name in actual
    )
    with pytest.raises(ValueError, match="Multi-segment registrations"):
        submitter.commands(records, "screen")


@pytest.mark.parametrize(
    "options",
    [
        {"seeds": ()},
        {"seeds": (42, 42)},
        {"seeds": (-1,)},
        {"seeds": (2**32,)},
        {"seeds": (42.0,)},
        {"max_epochs": 0},
        {"segments": 0},
        {"allocation_hours": 49},
        {"allocation_hours": 0},
        {"internal_hours": 6},
        {"internal_hours": 0},
        {"internal_hours": float("nan")},
        {"endpoints": (24, 12)},
        {"endpoints": (12, 12)},
        {"endpoints": ()},
        {"endpoints": (12, 49)},
        {"namespace": "../escape"},
        {"namespace": "/absolute"},
        {"namespace": ""},
        {"namespace": "contains space"},
    ],
)
def test_invalid_optional_registration_policy_fails_before_creating_outputs(
    inputs, options
):
    evidence, records = inputs
    with pytest.raises(ValueError):
        registration.prepare(evidence, records / "batch", Path("/remote"), **options)
    assert not (records / "batch").exists()


def test_long_registration_optional_parameters_are_exposed_by_cli(inputs):
    evidence, records = inputs
    registration.main(
        [
            "prepare",
            "--evidence",
            str(evidence),
            "--destination",
            str(records / "batch"),
            "--remote-records",
            "/remote/long",
            "--seeds",
            "42",
            "43",
            "44",
            "--max-epochs",
            "500",
            "--segments",
            "2",
            "--allocation-hours",
            "48",
            "--internal-hours",
            "45",
            "--endpoints",
            "48",
            "200",
            "500",
            "--namespace",
            "mandala_accuracy_v1_long",
        ]
    )
    document = load_registration(records)
    assert len(document["jobs"]) == 42
    assert document["budget"]["max_gpu_hours"] == 4032


def test_submission_hardware_overrides_preserve_literal_node_ranges_and_defaults(
    prepared,
):
    commands = submitter.commands(
        prepared,
        "audits",
        partition="gpu-a100-casus",
        nodelist="ga[010-014]",
        account="casus",
    )
    assert len(commands) == 2
    for command in commands:
        assert "--partition=gpu-a100-casus" in command
        assert "--nodelist=ga[010-014]" in command
        assert "--account=casus" in command
        assert "--gres=gpu:1" in command
        assert "--time=02:00:00" in command
    assert all(
        not any(value.startswith("--nodelist=") for value in command)
        for command in submitter.commands(prepared, "audits")
    )


def test_submission_cli_passes_optional_hardware_without_submitting(
    prepared, monkeypatch
):
    calls = []
    monkeypatch.setattr(
        submitter, "submit", lambda *args, **kwargs: calls.append((args, kwargs))
    )
    submitter.main(
        [
            "--records",
            str(prepared),
            "--phase",
            "audits",
            "--dry-run",
            "--partition",
            "gpu-h100-casus",
            "--nodelist",
            "gh[001-003]",
            "--account",
            "casus",
        ]
    )
    assert calls == [
        (
            (prepared.resolve(), "audits"),
            {
                "dry_run": True,
                "partition": "gpu-h100-casus",
                "nodelist": "gh[001-003]",
                "account": "casus",
            },
        )
    ]


def test_separate_legacy_runtime_registration_pins_all_launchers(inputs):
    evidence, records = inputs
    registration.prepare(
        evidence,
        records / "batch",
        Path("/remote/legacy"),
        seeds=(45,),
        namespace="mandala_accuracy_v1_v100",
        runtime_profile="mandala_legacy",
        concurrency=14,
    )
    document = load_registration(records)
    assert document["runtime_profile"] == "mandala_legacy"
    assert document["budget"]["array_concurrency"] == 14
    for path in (records / "batch").glob("*.sh"):
        assert "source ~/scripts/profiles/mandala_legacy.sh" in path.read_text()
    assert "0-13" in " ".join(submitter.commands(records, "screen")[0])
    assert any(
        value.endswith("%14") for value in submitter.commands(records, "screen")[0]
    )


@pytest.mark.parametrize(
    "options", [{"runtime_profile": "unknown"}, {"concurrency": 0}]
)
def test_runtime_registration_invalid_before_output(inputs, options):
    evidence, records = inputs
    with pytest.raises(ValueError):
        registration.prepare(evidence, records / "batch", Path("/remote"), **options)
    assert not records.exists()
