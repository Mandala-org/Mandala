#!/usr/bin/env python3
"""Freeze the geometry-only descriptor promotions after the Stage-3 gate."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any


RADII = (6.5, 7.5, 8.5, 10.5)
FAMILIES = ("d1", "d2", "d3", "d4")


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _radius_tag(radius: float) -> str:
    return f"r{radius:.1f}".replace(".", "p")


def _selected_keys(radius: float) -> dict[str, dict[str, str]]:
    tag = _radius_tag(radius)
    return {
        "d1": {
            "compact": f"{tag}_spherical_bessel_compact_n4_l3",
            "high": f"{tag}_spherical_bessel_high_n8_l6",
        },
        "d2": {
            "compact": f"{tag}_d2_degree8_o8_l8",
            "high": f"{tag}_d2_degree10_o10_l10",
        },
        "d3": {
            "compact": f"{tag}_d3_compact_o4_l3",
            "high": f"{tag}_d3_high_o8_l6",
        },
        "d4": {
            "compact": f"{tag}_d4_spherical_bessel_compact_n4_l3_b2",
            "high": f"{tag}_d4_spherical_bessel_high_n8_l6_b3",
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts-root", type=Path, default=Path("artifacts"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/pair_stage3/sio2_descriptor_promotions_v1"),
    )
    parser.add_argument("--synthetic-success-threshold", type=float, default=0.99)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stage2 = args.artifacts_root / "pair_stage2"
    stage3 = args.artifacts_root / "pair_stage3"
    confirmation = stage3 / "sio2_reconstruction_confirmation_v1"
    cases = stage3 / "sio2_confirmation_cases_v1"
    information = stage3 / "sio2_information_suite_v1"

    confirmation_summary = _read_json(confirmation / "summary.json")
    confirmation_config = _read_json(confirmation / "config.json")
    cases_summary = _read_json(cases / "summary.json")
    if not confirmation_summary["completed"] or not cases_summary["passed"]:
        raise RuntimeError("Stage-3 confirmation did not complete successfully")
    if confirmation_summary["selection_uses_hamiltonian"]:
        raise RuntimeError("Descriptor selection must not use Hamiltonian targets")
    if _sha256(cases / "cases.json") != confirmation_config["cases_sha256"]:
        raise RuntimeError("Confirmation case registry hash mismatch")

    task_files = sorted((confirmation / "tasks").glob("task_*.json"))
    with (confirmation / "results.csv").open(newline="") as stream:
        results = list(csv.DictReader(stream))
    if len(task_files) != confirmation_summary["task_count"] or len(results) != len(
        task_files
    ):
        raise RuntimeError("Incomplete Stage-3 task registry")
    if any(
        row["manifest_hash"] != confirmation_summary["manifest_hash"] for row in results
    ):
        raise RuntimeError("Mixed manifests in Stage-3 results")

    with (confirmation / "configuration_metrics.csv").open(newline="") as stream:
        reconstruction = {
            (row["family"], row["key"]): row for row in csv.DictReader(stream)
        }
    with (information / "configuration_metrics.csv").open(newline="") as stream:
        info = {(row["family"], row["key"]): row for row in csv.DictReader(stream)}
    with (stage2 / "sio2_descriptor_certification_v1" / "descriptor_metrics.csv").open(
        newline=""
    ) as stream:
        certification = {
            (row["family"], row["key"]): row for row in csv.DictReader(stream)
        }

    schemas: dict[str, dict[str, dict[str, Any]]] = {}
    source_files: list[Path] = [
        confirmation / "summary.json",
        confirmation / "config.json",
        confirmation / "results.csv",
        confirmation / "configuration_metrics.csv",
        cases / "cases.json",
        cases / "summary.json",
        information / "configuration_metrics.csv",
        stage2 / "sio2_descriptor_certification_v1" / "descriptor_metrics.csv",
    ]
    for family in FAMILIES:
        family_dir = stage2 / f"{family}_sio2_precompute_v1"
        family_summary = _read_json(family_dir / "summary.json")
        if not family_summary["passed"] or not family_summary["full_run"]:
            raise RuntimeError(f"Stage-2 cache did not pass for {family}")
        schema_path = family_dir / "descriptor_schemas.json"
        schemas[family] = {item["key"]: item for item in _read_json(schema_path)}
        source_files.extend((family_dir / "summary.json", schema_path))

    selected: list[dict[str, Any]] = []
    for radius in RADII:
        for family, levels in _selected_keys(radius).items():
            for resolution, key in levels.items():
                schema = schemas[family][key]
                cert = certification[(family, key)]
                symmetry_passed = (
                    float(cert["proper_o3_relative_error"]) <= 1.0e-9
                    and float(cert["improper_o3_relative_error"]) <= 1.0e-9
                    and cert["finite"] == "True"
                    and cert["metadata_hash_matches"] == "True"
                )
                if not symmetry_passed:
                    raise RuntimeError(f"Symmetry/certification gate failed for {key}")

                base_family = family
                base_key = key
                if family == "d4":
                    base_family = "d1"
                    base_key = schema["d1_key"]
                reconstruction_row = reconstruction[(base_family, base_key)]
                info_row = info[(family, key)]
                synthetic_fraction = float(
                    reconstruction_row["synthetic_reconstruction_success_fraction"]
                )
                real_fraction = float(
                    reconstruction_row["real_reconstruction_success_fraction"]
                )
                if (
                    resolution == "high"
                    and synthetic_fraction < args.synthetic_success_threshold
                ):
                    raise RuntimeError(
                        f"Preferred high resolution misses reconstruction gate: {key}"
                    )
                if float(info_row["rank_deficiency_fraction"]) > 0.001:
                    raise RuntimeError(f"Unexpected rank deficiency for {key}")
                if int(info_row["candidate_collision_count"]) != 0:
                    raise RuntimeError(f"Unresolved candidate collision for {key}")

                selected.append(
                    {
                        "family": family,
                        "resolution": resolution,
                        "cutoff_angstrom": radius,
                        "key": key,
                        "content_hash": schema["content_hash"],
                        "dimension": len_from_irreps(schema["irreps_out"]),
                        "reconstruction_evidence_family": base_family,
                        "reconstruction_evidence_key": base_key,
                        "synthetic_reconstruction_success_fraction": synthetic_fraction,
                        "real_subset_reconstruction_success_fraction": real_fraction,
                        "empirically_stably_reconstructive": synthetic_fraction
                        >= args.synthetic_success_threshold,
                        "rank_deficiency_fraction": float(
                            info_row["rank_deficiency_fraction"]
                        ),
                        "candidate_collision_count": int(
                            info_row["candidate_collision_count"]
                        ),
                        "completeness_status": (
                            "empirically_stably_reconstructive_on_registered_synthetic_domain"
                            if synthetic_fraction >= args.synthetic_success_threshold
                            else "basis-limit-only_with_documented_finite_failures"
                        ),
                    }
                )

    manifest = {
        "convention": "mandala-sio2-stage3-descriptor-promotions-v1",
        "selection_uses_hamiltonian": False,
        "selection_policy": {
            "radii_angstrom": list(RADII),
            "levels_per_family_and_radius": ["compact", "high"],
            "preferred_high_synthetic_success_threshold": args.synthetic_success_threshold,
            "d1_basis": "spherical_bessel",
            "d1_basis_reason": "only D1 compact basis with 408/408 synthetic recoveries",
            "d2_compact_caveat": (
                "degree-8 is retained as the compact information/compute point; "
                "finite reconstruction failures are explicit and no completeness is claimed"
            ),
            "d4_rule": "degree-2 compact and degree-3 high lifts inheriting selected D1",
        },
        "source_sha256": {str(path): _sha256(path) for path in source_files},
        "promotions": selected,
    }
    encoded = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    manifest["manifest_hash"] = hashlib.sha256(encoded).hexdigest()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    _atomic_json(args.output_dir / "promotions.json", manifest)
    summary = {
        "passed": True,
        "manifest_hash": manifest["manifest_hash"],
        "promotion_count": len(selected),
        "family_count": len(FAMILIES),
        "cutoff_count": len(RADII),
        "preferred_high_count": sum(x["resolution"] == "high" for x in selected),
        "preferred_high_gate_pass_count": sum(
            x["resolution"] == "high" and x["empirically_stably_reconstructive"]
            for x in selected
        ),
        "compact_stable_count": sum(
            x["resolution"] == "compact" and x["empirically_stably_reconstructive"]
            for x in selected
        ),
        "confirmation_real_subset_success_count": (
            confirmation_summary["reconstruction_success_count"]
            - confirmation_summary["synthetic_reconstruction_success_count"]
        ),
        "confirmation_real_subset_case_count": (
            confirmation_summary["task_count"]
            - confirmation_summary["synthetic_case_count"]
        ),
        "selection_uses_hamiltonian": False,
    }
    _atomic_json(args.output_dir / "summary.json", summary)
    report_lines = [
        "# Stage 3 descriptor promotions",
        "",
        "Selection used geometry-only certification; no Hamiltonian target was read.",
        "",
        "| Family | Level | $R_D$ (Å) | Dimension | Synthetic | Real subset | Status |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for item in selected:
        report_lines.append(
            f"| {item['family'].upper()} | {item['resolution']} | "
            f"{item['cutoff_angstrom']:.1f} | {item['dimension']} | "
            f"{item['synthetic_reconstruction_success_fraction']:.3%} | "
            f"{item['real_subset_reconstruction_success_fraction']:.3%} | "
            f"{item['completeness_status']} |"
        )
    report_lines.extend(
        (
            "",
            "All preferred high resolutions pass the pre-registered 99% synthetic "
            "reconstruction target. Compact D2 failures remain visible and D2 compact "
            "is not described as finite-truncation complete.",
            "",
            "The real confirmation subset recovered only 30/256 mapper/configuration "
            "tasks. Those cases are eight centers from structure 0000, reused at every "
            "cutoff, with physical radii concentrated well inside the larger cutoffs. "
            "The failures are consistent with radial inverse conditioning and are not "
            "a representative-dataset success estimate. They remain an explicit threat "
            "to practical finite-truncation completeness; promotion means the descriptor "
            "may enter Hamiltonian screening, not that real-neighborhood injectivity was proved.",
            "",
        )
    )
    (args.output_dir / "report.md").write_text("\n".join(report_lines))
    print(json.dumps(summary, indent=2, sort_keys=True))


def len_from_irreps(specification: str) -> int:
    """Return the flat dimension without importing torch/e3nn in this audit CLI."""
    dimension = 0
    for term in specification.split("+"):
        multiplicity, irrep = term.split("x")
        angular_momentum = int(irrep[:-1])
        dimension += int(multiplicity) * (2 * angular_momentum + 1)
    return dimension


if __name__ == "__main__":
    main()
