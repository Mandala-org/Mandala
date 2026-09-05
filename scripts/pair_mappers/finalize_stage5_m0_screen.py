#!/usr/bin/env python3
"""Recover Stage 5 M0 reports from completed fit/validation checkpoints."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--descriptor-summary", type=Path, required=True)
    parser.add_argument("--promotions", type=Path, required=True)
    return parser.parse_args()


def _atomic_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _coefficient_count(path: Path) -> int:
    state = torch.load(path, map_location="cpu", weights_only=True)
    return sum(value.numel() for name, value in state.items() if ".weight_" in name)


def main() -> None:
    args = parse_args()
    output = args.output_dir.resolve()
    required = (
        output / "config.json",
        output / "fit_diagnostics.json",
        output / "validation_metrics.json",
    )
    if not all(path.is_file() for path in required):
        raise FileNotFoundError("fit and validation checkpoints are incomplete")
    if (output / "summary.json").exists():
        raise FileExistsError("summary already exists; recovery is unnecessary")

    config = json.loads((output / "config.json").read_text())
    diagnostics = json.loads((output / "fit_diagnostics.json").read_text())
    detailed = json.loads((output / "validation_metrics.json").read_text())
    descriptor_summary = json.loads(args.descriptor_summary.read_text())
    promotion_manifest = json.loads(args.promotions.read_text())
    family = config["family"]
    if family not in ("d1", "d2", "d3", "d4"):
        raise ValueError(f"unexpected family {family!r}")
    if not descriptor_summary["passed"]:
        raise ValueError("descriptor precomputation did not pass")
    if config["promotion_manifest_hash"] != promotion_manifest["manifest_hash"]:
        raise ValueError("promotion manifest changed since the recovered run")
    if config["test_shards_read"] or config["selection_uses_test_hamiltonian"]:
        raise ValueError("refusing to recover a run that accessed test targets")
    promotions = [
        item for item in promotion_manifest["promotions"] if item["family"] == family
    ]
    if len(promotions) != 8:
        raise ValueError(f"expected eight promoted {family} configurations")

    rows = []
    for item in promotions:
        key = item["key"]
        if key not in diagnostics or key not in detailed:
            raise KeyError(f"missing completed checkpoint for {key}")
        model_path = output / "models" / f"{key}.pt"
        if not model_path.is_file():
            raise FileNotFoundError(model_path)
        headline = detailed[key]["matrix_elements"]
        missing = sorted(
            {
                irrep
                for site in diagnostics[key].values()
                for pair in site.values()
                for irrep, condition in pair["condition_numbers"].items()
                if math.isinf(condition)
            }
        )
        rows.append(
            {
                "manifest_hash": config["manifest_hash"],
                "family": family,
                "key": key,
                "resolution": item["resolution"],
                "cutoff_angstrom": item["cutoff_angstrom"],
                "descriptor_dimension": item["dimension"],
                "uncompressed_bytes_per_atom": 4 * int(item["dimension"]),
                "learned_coefficient_count": _coefficient_count(model_path),
                "descriptor_cache_size_bytes": descriptor_summary["cache_size_bytes"],
                "descriptor_precompute_seconds": descriptor_summary["elapsed_seconds"],
                "descriptor_neighbor_contributions_per_second": descriptor_summary.get(
                    "neighbor_contributions_per_second"
                ),
                "validation_matrix_mae_mev": headline["mae"],
                "validation_matrix_rmse_mev": headline["rmse"],
                "validation_matrix_element_count": headline["scalar_count"],
                "validation_block_count": detailed[key]["block_frobenius"][
                    "block_count"
                ],
                "missing_target_irreps": ";".join(missing),
                "finite": bool(
                    math.isfinite(headline["mae"]) and math.isfinite(headline["rmse"])
                ),
            }
        )
    with (output / "results.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "completed": True,
        "passed": all(row["finite"] for row in rows),
        "manifest_hash": config["manifest_hash"],
        "family": family,
        "configuration_count": len(rows),
        "finite_count": sum(row["finite"] for row in rows),
        "test_shards_read": False,
        "recovered_from_completed_checkpoints": True,
        "timings_available": False,
        "fit_stream_seconds": None,
        "fit_structures_per_second": None,
        "validation_seconds": None,
        "validation_directed_blocks_per_second_across_eight_models": None,
        "peak_cuda_memory_bytes": None,
    }
    _atomic_json(output / "summary.json", summary)
    lines = [
        f"# Stage 5 M0 validation screen: {family.upper()}",
        "",
        "Recovered report-only from completed fit and validation checkpoints after a metadata-field failure. No training or validation was repeated. Timing data was not recoverable.",
        "",
        "Test shards were not read. All offsite targets use the frozen train-only range envelope.",
        "",
        "| Resolution | $R_D$ (Å) | Dimension | Validation MAE (meV) | RMSE (meV) | Missing types |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['resolution']} | {float(row['cutoff_angstrom']):.1f} | "
            f"{row['descriptor_dimension']} | {float(row['validation_matrix_mae_mev']):.6f} | "
            f"{float(row['validation_matrix_rmse_mev']):.6f} | "
            f"{row['missing_target_irreps'] or 'none'} |"
        )
    (output / "report.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
