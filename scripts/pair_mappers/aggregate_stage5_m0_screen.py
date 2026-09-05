#!/usr/bin/env python3
"""Aggregate the four validation-only M0 descriptor-family screens."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


FAMILIES = ("d1", "d2", "d3", "d4")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--family-result",
        nargs=2,
        action="append",
        metavar=("FAMILY", "DIRECTORY"),
        required=True,
    )
    parser.add_argument("--promotions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def _atomic_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _hash(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _pareto(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    result = []
    for candidate in rows:
        dimension = int(candidate["descriptor_dimension"])
        error = float(candidate["validation_matrix_mae_mev"])
        dominated = any(
            int(other["descriptor_dimension"]) <= dimension
            and float(other["validation_matrix_mae_mev"]) <= error
            and (
                int(other["descriptor_dimension"]) < dimension
                or float(other["validation_matrix_mae_mev"]) < error
            )
            for other in rows
        )
        if not dominated:
            result.append(candidate)
    return sorted(
        result,
        key=lambda row: (
            int(row["descriptor_dimension"]),
            float(row["validation_matrix_mae_mev"]),
        ),
    )


def main() -> None:
    args = parse_args()
    supplied = {family: Path(directory) for family, directory in args.family_result}
    if set(supplied) != set(FAMILIES) or len(args.family_result) != len(FAMILIES):
        raise ValueError("provide each of d1, d2, d3, and d4 exactly once")
    output = args.output_dir.resolve()
    existing = [] if not output.exists() else list(output.iterdir())
    if existing:
        raise FileExistsError(f"refusing to overwrite nonempty {output}")
    output.mkdir(parents=True, exist_ok=True)

    promotions = json.loads(args.promotions.read_text())
    rows: list[dict[str, object]] = []
    source_manifests = {}
    for family in FAMILIES:
        directory = supplied[family]
        summary = json.loads((directory / "summary.json").read_text())
        config = json.loads((directory / "config.json").read_text())
        if not summary["completed"] or not summary["passed"]:
            raise ValueError(f"{family} screen did not pass")
        if summary["test_shards_read"] or config["test_shards_read"]:
            raise ValueError(f"{family} screen accessed test shards")
        if config["promotion_manifest_hash"] != promotions["manifest_hash"]:
            raise ValueError(f"{family} used a different promotion manifest")
        if summary["manifest_hash"] != config["manifest_hash"]:
            raise ValueError(f"{family} config/summary manifest mismatch")
        source_manifests[family] = config["manifest_hash"]
        with (directory / "results.csv").open(newline="") as stream:
            family_rows = list(csv.DictReader(stream))
        if len(family_rows) != 8 or any(row["family"] != family for row in family_rows):
            raise ValueError(f"{family} must contain exactly eight rows")
        rows.extend(family_rows)

    if len({row["key"] for row in rows}) != 32:
        raise ValueError("expected 32 unique promoted configurations")
    for row in rows:
        row["cutoff_angstrom"] = float(row["cutoff_angstrom"])
        row["descriptor_dimension"] = int(row["descriptor_dimension"])
        row["uncompressed_bytes_per_atom"] = int(row["uncompressed_bytes_per_atom"])
        row["validation_matrix_mae_mev"] = float(row["validation_matrix_mae_mev"])
        row["validation_matrix_rmse_mev"] = float(row["validation_matrix_rmse_mev"])
        row["validation_matrix_element_count"] = int(
            row["validation_matrix_element_count"]
        )
        row["validation_block_count"] = int(row["validation_block_count"])
        row["finite"] = row["finite"].lower() == "true"
    rows.sort(
        key=lambda row: (
            float(row["validation_matrix_mae_mev"]),
            int(row["descriptor_dimension"]),
            str(row["key"]),
        )
    )

    best_by_family_resolution = {}
    for family in FAMILIES:
        for resolution in ("compact", "high"):
            subset = [
                row
                for row in rows
                if row["family"] == family and row["resolution"] == resolution
            ]
            best_by_family_resolution[f"{family}_{resolution}"] = min(
                subset,
                key=lambda row: (
                    float(row["validation_matrix_mae_mev"]),
                    int(row["descriptor_dimension"]),
                ),
            )["key"]
    pareto = _pareto(rows)
    config = {
        "convention": "mandala-stage5-m0-validation-aggregate-v1",
        "promotion_manifest_hash": promotions["manifest_hash"],
        "source_manifests": source_manifests,
        "selection_partition": "validation",
        "test_shards_read": False,
    }
    config["manifest_hash"] = _hash(config)
    _atomic_json(output / "config.json", config)

    fieldnames = ["aggregate_manifest_hash", *rows[0].keys()]
    with (output / "results.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({"aggregate_manifest_hash": config["manifest_hash"], **row})

    colors = {"d1": "#0072B2", "d2": "#D55E00", "d3": "#009E73", "d4": "#CC79A7"}
    markers = {"compact": "o", "high": "s"}
    figure, axis = plt.subplots(figsize=(8.0, 5.2), constrained_layout=True)
    for family in FAMILIES:
        for resolution in ("compact", "high"):
            subset = [
                row
                for row in rows
                if row["family"] == family and row["resolution"] == resolution
            ]
            axis.scatter(
                [row["descriptor_dimension"] for row in subset],
                [row["validation_matrix_mae_mev"] for row in subset],
                c=colors[family],
                marker=markers[resolution],
                label=f"{family.upper()} {resolution}",
                alpha=0.85,
            )
    axis.plot(
        [row["descriptor_dimension"] for row in pareto],
        [row["validation_matrix_mae_mev"] for row in pareto],
        color="black",
        linewidth=1.0,
        linestyle="--",
        label="validation Pareto frontier",
    )
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlabel("Descriptor dimension")
    axis.set_ylabel("Validation matrix-element MAE (meV)")
    axis.grid(True, which="both", alpha=0.2)
    axis.legend(ncol=2, fontsize=8)
    figure.savefig(output / "mae_vs_dimension.png", dpi=180)
    plt.close(figure)

    summary = {
        "completed": True,
        "passed": all(bool(row["finite"]) for row in rows),
        "manifest_hash": config["manifest_hash"],
        "configuration_count": len(rows),
        "test_shards_read": False,
        "best_overall_key": rows[0]["key"],
        "best_overall_validation_matrix_mae_mev": rows[0]["validation_matrix_mae_mev"],
        "best_by_family_resolution": best_by_family_resolution,
        "pareto_keys": [row["key"] for row in pareto],
    }
    _atomic_json(output / "summary.json", summary)
    lines = [
        "# Stage 5 M0 validation aggregate",
        "",
        "Training and validation partitions only; no test shard was read. Rankings are exploratory validation evidence, not test results.",
        "",
        "| Rank | Family | Resolution | $R_D$ (Å) | Dimension | Validation MAE (meV) | RMSE (meV) |",
        "|---:|---|---|---:|---:|---:|---:|",
    ]
    for rank, row in enumerate(rows, start=1):
        lines.append(
            f"| {rank} | {str(row['family']).upper()} | {row['resolution']} | "
            f"{float(row['cutoff_angstrom']):.1f} | {int(row['descriptor_dimension'])} | "
            f"{float(row['validation_matrix_mae_mev']):.6f} | "
            f"{float(row['validation_matrix_rmse_mev']):.6f} |"
        )
    (output / "report.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
