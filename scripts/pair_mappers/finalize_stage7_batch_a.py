#!/usr/bin/env python3
"""Combine Stage-7 Batch-A outputs into the next result-dependent gate."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage7-root", type=Path, required=True)
    parser.add_argument("--stage6-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def main() -> None:
    args = parse_args()
    root = args.stage7_root.resolve()
    stage6 = args.stage6_root.resolve()
    output = args.output_dir.resolve()
    if output.exists() and [
        path for path in output.iterdir() if path.name != "launcher.log"
    ]:
        raise FileExistsError(f"refusing to overwrite nonempty {output}")
    output.mkdir(parents=True, exist_ok=True)

    freeze = json.loads((root / "summary.json").read_text())
    offsite_summary = json.loads(
        (root / "offsite" / "aggregate_v1" / "summary.json").read_text()
    )
    ridge_summary = json.loads(
        (root / "onsite_ridge_extension" / "summary.json").read_text()
    )
    gauge = json.loads((root / "onsite_gauge_audit" / "summary.json").read_text())
    if not all(
        item["passed"] for item in (freeze, offsite_summary, ridge_summary, gauge)
    ):
        raise ValueError("at least one Stage-7 component did not pass")
    if any(
        item["test_shards_read"]
        for item in (freeze, offsite_summary, ridge_summary, gauge)
    ):
        raise ValueError("Stage-7 gate detected test-target access")

    offsite_rows = _read_csv(root / "offsite" / "aggregate_v1" / "results.csv")
    if len(offsite_rows) != 4:
        raise ValueError("expected exactly four offsite descriptor results")
    offsite_rows.sort(key=lambda row: float(row["validation_matrix_mae_mev"]))
    stage6_ridge = _read_csv(stage6 / "onsite_ridge" / "d4" / "results.csv")
    stage7_ridge = _read_csv(root / "onsite_ridge_extension" / "results.csv")
    ridge_rows = stage6_ridge + stage7_ridge
    ridge_rows.sort(key=lambda row: float(row["validation_matrix_mae_mev"]))
    best_ridge = ridge_rows[0]
    best_offsite = offsite_rows[0]
    if not all(
        math.isfinite(float(row["validation_matrix_mae_mev"]))
        for row in (*offsite_rows, *ridge_rows)
    ):
        raise ValueError("non-finite validation metric in Stage-7 gate")

    gate = {
        "convention": "mandala-stage7-batch-a-result-gate-v1",
        "completed": True,
        "passed": True,
        "selection_partition": "validation",
        "test_shards_read": False,
        "best_offsite_task_id": best_offsite["task_id"],
        "best_offsite_family": best_offsite["family"],
        "best_offsite_descriptor_key": best_offsite["descriptor_key"],
        "best_offsite_validation_mae_mev": float(
            best_offsite["validation_matrix_mae_mev"]
        ),
        "best_offsite_step": int(best_offsite["best_step"]),
        "best_onsite_ridge_model_id": best_ridge["model_id"],
        "best_onsite_ridge_validation_mae_mev": float(
            best_ridge["validation_matrix_mae_mev"]
        ),
        "gauge_audit": {
            "uncorrected_validation_mae_mev": gauge["uncorrected_validation_mae_mev"],
            "structure_identity_oracle_validation_mae_mev": gauge[
                "structure_identity_oracle_validation_mae_mev"
            ],
            "structure_identity_oracle_mae_reduction_fraction": gauge[
                "structure_identity_oracle_mae_reduction_fraction"
            ],
            "energy_zero_or_fermi_metadata_available_in_cache": gauge[
                "energy_zero_or_fermi_metadata_available_in_cache"
            ],
            "diagnostic_only": True,
        },
        "next_gate": (
            "Use the selected offsite descriptor for richer bond-basis/rank screens. "
            "Choose onsite energy-reference treatment only after interpreting the "
            "diagnostic oracle reduction; never deploy the oracle target-derived shift."
        ),
        "source_hashes": {
            "freeze_summary": _sha256(root / "summary.json"),
            "offsite_aggregate": _sha256(
                root / "offsite" / "aggregate_v1" / "summary.json"
            ),
            "ridge_summary": _sha256(root / "onsite_ridge_extension" / "summary.json"),
            "gauge_summary": _sha256(root / "onsite_gauge_audit" / "summary.json"),
        },
    }
    gate["manifest_hash"] = hashlib.sha256(
        json.dumps(gate, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    _write(output / "summary.json", gate)
    lines = [
        "# Stage 7 Batch A result gate",
        "",
        "No test target was read. All rankings below use the frozen validation split.",
        "",
        "## Offsite descriptor comparison",
        "",
        "| Rank | Family | Task | Best step | Projected MAE (meV) | Raw MAE (meV) |",
        "|---:|---|---|---:|---:|---:|",
    ]
    for rank, row in enumerate(offsite_rows, 1):
        lines.append(
            f"| {rank} | {row['family'].upper()} | `{row['task_id']}` | "
            f"{int(row['best_step']):,} | {float(row['validation_matrix_mae_mev']):.6f} | "
            f"{float(row['validation_raw_matrix_mae_mev']):.6f} |"
        )
    lines.extend(
        [
            "",
            "## Onsite ridge and identity-gauge diagnostic",
            "",
            f"Best combined old+extended ridge candidate: `{best_ridge['model_id']}` at "
            f"{float(best_ridge['validation_matrix_mae_mev']):.6f} meV.",
            "",
            f"The frozen Stage-6 onsite model changes from "
            f"{float(gauge['uncorrected_validation_mae_mev']):.6f} to "
            f"{float(gauge['structure_identity_oracle_validation_mae_mev']):.6f} meV "
            "under a target-derived per-structure identity shift. This is diagnostic only, "
            "not an inference-time correction.",
            "",
            "The next architecture batch must be frozen only after these results are interpreted.",
        ]
    )
    (output / "report.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(gate, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
