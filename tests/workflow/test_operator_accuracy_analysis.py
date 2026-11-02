"""Synthetic, small-CSV tests for the separately versioned screen analysis."""

import csv
import hashlib
import json
from pathlib import Path

import pytest

from scripts import analyze_operator_accuracy as analysis


def write_metrics(path, rows, *, extra_fields=()):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "epoch",
                "step",
                analysis.RAW,
                analysis.PROJECTED,
                *extra_fields,
            ],
        )
        writer.writeheader()
        for epoch, raw, projected in rows:
            writer.writerow(
                {
                    "epoch": epoch,
                    "step": (epoch + 1) * 10 - 1,
                    analysis.RAW: raw / 1000,
                    analysis.PROJECTED: "" if projected is None else projected / 1000,
                }
            )


def fixture_registration(tmp_path):
    factors = {"baseline": 1, "mse": 0.5, "rollback_off": 0.8, "mse_rollback_off": 0.2}
    jobs = []
    results = tmp_path / "results"
    for treatment, scale in factors.items():
        name = f"graphene_{treatment}_s42"
        jobs.append(
            {
                "id": name,
                "material": "graphene",
                "treatment": treatment,
                "seed": 42,
                "primary_endpoints_epochs": [12, 24, 48],
            }
        )
        rows = [
            (0, 10 * scale, 9 * scale),
            (10, scale, 0.8 * scale),
            (11, 2 * scale, 0.1 * scale),
            (23, 3 * scale, 0.3 * scale),
            (47, 4 * scale, 0.4 * scale),
        ]
        write_metrics(results / name / "metrics/version_0/metrics.csv", rows)
    registration = tmp_path / "registration.json"
    registration.write_text(
        json.dumps(
            {
                "id": "synthetic",
                "jobs": jobs,
                "contrasts": ["mse-baseline", "mse_rollback_off-mse"],
            }
        )
    )
    return registration, results


def test_exact_horizons_raw_selection_and_registered_contrasts(tmp_path):
    registration, results = fixture_registration(tmp_path)
    report = analysis.analyze(registration, results)
    baseline = report["arms"]["graphene_baseline_s42"]
    assert set(baseline["endpoints"]) == {"12", "24", "48"}
    twelve = baseline["endpoints"]["12"]
    assert twelve["status"] == "complete"
    assert twelve["raw_endpoint"]["epoch_zero_based"] == 11
    assert twelve["raw_endpoint"]["raw_mae_mev"] == pytest.approx(2)
    assert twelve["best_raw_through_endpoint"]["epoch_zero_based"] == 10
    assert twelve["best_raw_through_endpoint"][
        "projected_mae_mev_at_same_epoch"
    ] == pytest.approx(0.8)
    # The lower projected score at epoch 11 never selects a checkpoint.
    assert (
        twelve["best_raw_through_endpoint"]["projected_mae_mev_at_same_epoch"]
        != twelve["raw_endpoint"]["projected_mae_mev_at_same_epoch"]
    )
    difference = report["baseline_differences"]["graphene_mse_s42"]["12"]
    assert difference["raw_endpoint_delta_mev"] == pytest.approx(-1)
    assert difference["best_raw_delta_mev"] == pytest.approx(-0.5)
    assert difference["projected_at_raw_best_delta_mev"] == pytest.approx(-0.4)
    contrast = report["registered_contrasts"]["mse-baseline"]["graphene"]["12"]
    assert contrast == difference
    interaction = report["mae_by_rollback_interactions"]["graphene"]["12"]
    assert interaction["status"] == "complete"
    assert interaction["raw_endpoint_interaction_mev"] == pytest.approx(-0.2)
    assert interaction["best_raw_interaction_mev"] == pytest.approx(-0.1)


def test_missing_and_resource_incomplete_endpoints_never_get_substitutes(tmp_path):
    registration, results = fixture_registration(tmp_path)
    path = results / "graphene_mse_s42/metrics/version_0/metrics.csv"
    write_metrics(path, [(0, 5, 4), (10, 0.5, 0.4)])
    report = analysis.analyze(registration, results)
    incomplete = report["arms"]["graphene_mse_s42"]["endpoints"]["12"]
    assert incomplete["missing"] and incomplete["resource_incomplete"]
    assert incomplete["raw_endpoint"] is None
    assert incomplete["best_raw_through_endpoint"] is None
    assert (
        report["baseline_differences"]["graphene_mse_s42"]["12"][
            "raw_endpoint_delta_mev"
        ]
        is None
    )
    assert (
        report["mae_by_rollback_interactions"]["graphene"]["12"][
            "raw_endpoint_interaction_mev"
        ]
        is None
    )
    path.unlink()
    assert analysis.analyze(registration, results)["arms"]["graphene_mse_s42"][
        "metrics_missing"
    ]


def test_a_gap_before_later_epochs_is_explicitly_missing_not_resource_incomplete(
    tmp_path,
):
    registration, results = fixture_registration(tmp_path)
    write_metrics(
        results / "graphene_mse_s42/metrics/version_0/metrics.csv",
        [(10, 1, 0.9), (23, 2, 1.8)],
    )
    report = analysis.analyze(registration, results)
    endpoints = report["arms"]["graphene_mse_s42"]["endpoints"]
    assert endpoints["12"]["status"] == "missing_endpoint"
    assert not endpoints["12"]["resource_incomplete"]
    assert endpoints["24"]["complete"]
    assert (
        report["baseline_differences"]["graphene_mse_s42"]["24"]["status"] == "complete"
    )
    assert (
        report["baseline_differences"]["graphene_mse_s42"]["12"]["status"]
        == "incomplete"
    )


def test_missing_projected_at_raw_best_never_uses_another_epoch(tmp_path):
    registration, results = fixture_registration(tmp_path)
    write_metrics(
        results / "graphene_mse_s42/metrics/version_0/metrics.csv",
        [(10, 0.5, None), (11, 1, 0.1)],
    )
    report = analysis.analyze(registration, results)
    difference = report["baseline_differences"]["graphene_mse_s42"]["12"]
    assert difference["status"] == "complete"
    assert difference["best_raw_delta_mev"] == pytest.approx(-0.5)
    assert difference["projected_at_raw_best_delta_mev"] is None


def test_actual_input_and_analysis_hashes_and_no_other_reads(tmp_path, monkeypatch):
    registration, results = fixture_registration(tmp_path)
    (results / "graphene_baseline_s42/best_test_evaluation.json").write_text(
        "must never read"
    )
    permitted = {
        registration.resolve(),
        Path(analysis.__file__).resolve(),
        *(p.resolve() for p in results.glob("*/metrics/version*/metrics.csv")),
    }
    actual_reads = set()
    original = Path.read_bytes

    def checked_read(path):
        assert path.resolve() in permitted
        actual_reads.add(path.resolve())
        return original(path)

    monkeypatch.setattr(Path, "read_bytes", checked_read)
    report = analysis.analyze(registration, results)
    assert actual_reads == permitted
    assert (
        report["registration_input"]["sha256"]
        == hashlib.sha256(original(registration)).hexdigest()
    )
    assert (
        report["analysis_source"]["sha256"]
        == hashlib.sha256(original(Path(analysis.__file__))).hexdigest()
    )
    for arm in report["arms"].values():
        assert len(arm["inputs"]) == 1
        item = arm["inputs"][0]
        assert (
            item["sha256"] == hashlib.sha256(original(Path(item["path"]))).hexdigest()
        )


@pytest.mark.parametrize(
    "mutation",
    [
        "duplicate_id",
        "duplicate_treatment",
        "adaptive_horizon",
        "invalid_seed",
        "path_escape",
    ],
)
def test_ambiguous_or_unregistered_analysis_is_rejected(tmp_path, mutation):
    registration, results = fixture_registration(tmp_path)
    document = json.loads(registration.read_text())
    if mutation == "duplicate_id":
        document["jobs"][1]["id"] = document["jobs"][0]["id"]
    elif mutation == "duplicate_treatment":
        document["jobs"][1]["treatment"] = "baseline"
    elif mutation == "adaptive_horizon":
        document["jobs"][1]["primary_endpoints_epochs"] = [12, 24, 72]
    elif mutation == "invalid_seed":
        document["jobs"][1]["seed"] = -1
    else:
        document["jobs"][1]["id"] = "../test_artifact"
    registration.write_text(json.dumps(document))
    with pytest.raises(ValueError):
        analysis.analyze(registration, results)


def test_multiple_seeds_and_hardware_keep_paired_controls_separate(tmp_path):
    from copy import deepcopy

    registration, results = fixture_registration(tmp_path)
    document = json.loads(registration.read_text())
    additional = []
    for job in document["jobs"]:
        job["hardware"] = {"class": "H100"}
        other = deepcopy(job)
        other.update(
            id=job["id"].replace("s42", "s43"), seed=43, hardware={"class": "A100"}
        )
        additional.append(other)
        values = {"baseline": 11, "mse": 3, "rollback_off": 9, "mse_rollback_off": 1}
        write_metrics(
            results / other["id"] / "metrics/version_0/metrics.csv",
            [(11, values[other["treatment"]], 0.5)],
        )
    document["jobs"].extend(additional)
    registration.write_text(json.dumps(document))
    report = analysis.analyze(registration, results)
    assert len(report["comparison_groups"]) == 2
    assert report["baseline_differences"]["graphene_mse_s42"]["12"][
        "raw_endpoint_delta_mev"
    ] == pytest.approx(-1)
    assert report["baseline_differences"]["graphene_mse_s43"]["12"][
        "raw_endpoint_delta_mev"
    ] == pytest.approx(-8)
    assert (
        report["baseline_differences"]["graphene_mse_s43"]["24"]["status"]
        == "incomplete"
    )
    document["jobs"][1]["hardware"] = {"class": "V100"}
    registration.write_text(json.dumps(document))
    assert (
        analysis.analyze(registration, results)["baseline_differences"][
            "graphene_mse_s42"
        ]["12"]["status"]
        == "incomplete"
    )


def test_long_registered_horizons_require_exact_epochs(tmp_path):
    registration, results = fixture_registration(tmp_path)
    document = json.loads(registration.read_text())
    document["primary_endpoints_epochs"] = [48, 200, 500]
    for job in document["jobs"]:
        job["primary_endpoints_epochs"] = [48, 200, 500]
        write_metrics(
            results / job["id"] / "metrics/version_0/metrics.csv",
            [(47, 3, 2.8), (198, 2, 1.8), (499, 1, 0.8)],
        )
    registration.write_text(json.dumps(document))
    report = analysis.analyze(registration, results)
    assert report["metric_contract"]["horizons_completed_epochs"] == [48, 200, 500]
    assert (
        report["arms"]["graphene_baseline_s42"]["endpoints"]["200"]["status"]
        == "missing_endpoint"
    )
    assert (
        report["baseline_differences"]["graphene_mse_s42"]["200"]["status"]
        == "incomplete"
    )
    assert (
        report["baseline_differences"]["graphene_mse_s42"]["500"]["status"]
        == "complete"
    )


def test_duplicate_versions_must_agree_and_ties_use_earliest_epoch(tmp_path):
    registration, results = fixture_registration(tmp_path)
    original = results / "graphene_baseline_s42/metrics/version_0/metrics.csv"
    duplicate = results / "graphene_baseline_s42/metrics/version_1/metrics.csv"
    duplicate.parent.mkdir()
    duplicate.write_bytes(original.read_bytes())
    assert (
        analysis.analyze(registration, results)["arms"]["graphene_baseline_s42"][
            "endpoints"
        ]["12"]["best_raw_through_endpoint"]["epoch_zero_based"]
        == 10
    )
    write_metrics(duplicate, [(10, 1.01, 0.8)])
    with pytest.raises(ValueError, match="Conflicting validation epoch"):
        analysis.analyze(registration, results)
    duplicate.unlink()
    write_metrics(original, [(5, 1, 0.9), (10, 1, 0.8), (11, 2, 0.1)])
    assert (
        analysis.analyze(registration, results)["arms"]["graphene_baseline_s42"][
            "endpoints"
        ]["12"]["best_raw_through_endpoint"]["epoch_zero_based"]
        == 5
    )


@pytest.mark.parametrize("invalid", ["nan", "inf", "-1"])
def test_nonfinite_or_negative_validation_errors_fail_closed(tmp_path, invalid):
    registration, results = fixture_registration(tmp_path)
    path = results / "graphene_mse_s42/metrics/version_0/metrics.csv"
    path.write_text(f"epoch,step,{analysis.RAW}\n11,119,{invalid}\n")
    with pytest.raises(ValueError, match="Invalid val/"):
        analysis.analyze(registration, results)


def test_unterminated_csv_tail_is_hashed_but_not_used_as_endpoint(tmp_path):
    registration, results = fixture_registration(tmp_path)
    path = results / "graphene_mse_s42/metrics/version_0/metrics.csv"
    write_metrics(path, [(10, 0.5, 0.4), (11, 1, 0.1)])
    path.write_bytes(path.read_bytes().rstrip(b"\r\n"))
    arm = analysis.analyze(registration, results)["arms"]["graphene_mse_s42"]
    assert arm["endpoints"]["12"]["missing"]
    assert arm["inputs"][0]["ignored_unterminated_tail_bytes"] > 0


def test_test_columns_and_escaped_metric_symlinks_are_rejected(tmp_path):
    registration, results = fixture_registration(tmp_path)
    path = results / "graphene_mse_s42/metrics/version_0/metrics.csv"
    write_metrics(path, [(11, 1, 0.9)], extra_fields=("test/mae",))
    with pytest.raises(ValueError, match="test columns"):
        analysis.analyze(registration, results)
    outside = tmp_path / "outside.csv"
    path.replace(outside)
    path.symlink_to(outside)
    with pytest.raises(ValueError, match="escapes"):
        analysis.analyze(registration, results)


def test_material_prefixed_registered_contrast(tmp_path):
    registration, results = fixture_registration(tmp_path)
    document = json.loads(registration.read_text())
    for treatment, raw in (("baseline", 2), ("pair_mean", 1.5), ("elements", 1)):
        name = f"mos2_{treatment}_s42"
        document["jobs"].append(
            {
                "id": name,
                "material": "mos2",
                "treatment": treatment,
                "seed": 42,
                "primary_endpoints_epochs": [12, 24, 48],
            }
        )
        write_metrics(
            results / name / "metrics/version_0/metrics.csv", [(11, raw, raw)]
        )
    document["contrasts"].append("mos2_elements-pair_mean (weighting)")
    registration.write_text(json.dumps(document))
    contrast = analysis.analyze(registration, results)["registered_contrasts"][
        "mos2_elements-pair_mean (weighting)"
    ]
    assert set(contrast) == {"mos2"}
    assert contrast["mos2"]["12"]["raw_endpoint_delta_mev"] == pytest.approx(-0.5)
    assert contrast["mos2"]["24"]["status"] == "incomplete"


def test_cli_writes_finite_bounded_summary_and_refuses_overwrite(tmp_path):
    registration, results = fixture_registration(tmp_path)
    output = tmp_path / "analysis.json"
    arguments = [
        "--registration",
        str(registration),
        "--results",
        str(results),
        "--output",
        str(output),
    ]
    analysis.main(arguments)
    assert output.stat().st_size < 100000
    assert json.loads(output.read_text())["metric_contract"][
        "horizons_completed_epochs"
    ] == [12, 24, 48]
    original = output.read_bytes()
    with pytest.raises(FileExistsError):
        analysis.main(arguments)
    assert output.read_bytes() == original
