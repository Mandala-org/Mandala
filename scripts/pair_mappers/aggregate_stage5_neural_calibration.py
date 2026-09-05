#!/usr/bin/env python3
"""Aggregate the frozen 12-run M3/M5 neural calibration grid."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def _atomic_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def main() -> None:
    args = parse_args()
    output = args.output_dir.resolve()
    if output.exists() and list(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite nonempty {output}")
    output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((args.grid_dir / "calibration_manifest.json").read_text())
    rows = []
    for task in manifest["tasks"]:
        run = args.grid_dir / "runs" / task["task_id"]
        config = json.loads((run / "config.json").read_text())
        summary = json.loads((run / "summary.json").read_text())
        if not summary["completed"] or not summary["passed"]:
            raise ValueError(f"incomplete calibration task {task['task_id']}")
        if summary["test_shards_read"] or config["test_shards_read"]:
            raise ValueError(f"test access recorded for {task['task_id']}")
        if config["calibration_manifest_hash"] != manifest["manifest_hash"]:
            raise ValueError(f"calibration manifest mismatch for {task['task_id']}")
        if config["manifest_hash"] != summary["manifest_hash"]:
            raise ValueError(f"run manifest mismatch for {task['task_id']}")
        rows.append(
            {
                "task_id": task["task_id"],
                "run_manifest_hash": summary["manifest_hash"],
                "architecture": task["architecture"],
                "resource_band": task["resource_band"],
                "learning_rate": task["learning_rate"],
                "parameter_count": summary["parameter_count"],
                "best_step": summary["best_step"],
                "validation_matrix_mae_mev": summary["best_validation_matrix_mae_mev"],
                "validation_matrix_rmse_mev": summary[
                    "best_validation_matrix_rmse_mev"
                ],
                "training_seconds": summary["training_seconds"],
                "sampled_training_blocks_per_second": summary[
                    "sampled_training_blocks_per_second"
                ],
                "peak_cuda_memory_bytes": summary["peak_cuda_memory_bytes"],
                "stopped_early": summary["stopped_early"],
            }
        )
    if len(rows) != 12 or any(
        not math.isfinite(float(row["validation_matrix_mae_mev"])) for row in rows
    ):
        raise ValueError("calibration grid must have 12 finite results")
    rows.sort(
        key=lambda row: (
            float(row["validation_matrix_mae_mev"]),
            int(row["parameter_count"]),
        )
    )
    with (output / "results.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    best_by_architecture = {
        architecture: next(
            row["task_id"] for row in rows if row["architecture"] == architecture
        )
        for architecture in ("m3", "m5")
    }
    summary = {
        "completed": True,
        "passed": True,
        "calibration_manifest_hash": manifest["manifest_hash"],
        "configuration_count": len(rows),
        "best_overall_task_id": rows[0]["task_id"],
        "best_overall_validation_matrix_mae_mev": rows[0]["validation_matrix_mae_mev"],
        "best_by_architecture": best_by_architecture,
        "test_shards_read": False,
    }
    _atomic_json(output / "summary.json", summary)

    colors = {"m3": "#0072B2", "m5": "#D55E00"}
    markers = {3.0e-4: "o", 1.0e-3: "s"}
    figure, axis = plt.subplots(figsize=(7.5, 5.0), constrained_layout=True)
    for architecture in ("m3", "m5"):
        for learning_rate in (3.0e-4, 1.0e-3):
            subset = [
                row
                for row in rows
                if row["architecture"] == architecture
                and float(row["learning_rate"]) == learning_rate
            ]
            axis.scatter(
                [row["parameter_count"] for row in subset],
                [row["validation_matrix_mae_mev"] for row in subset],
                color=colors[architecture],
                marker=markers[learning_rate],
                s=55,
                label=f"{architecture.upper()}, lr={learning_rate:g}",
            )
            for row in subset:
                axis.annotate(
                    str(row["resource_band"])[0].upper(),
                    (row["parameter_count"], row["validation_matrix_mae_mev"]),
                    xytext=(4, 3),
                    textcoords="offset points",
                    fontsize=8,
                )
    axis.axhline(
        float(manifest["m0_gate"]["best_validation_matrix_mae_mev"]),
        color="black",
        linestyle="--",
        linewidth=1.0,
        label="best M0 validation",
    )
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlabel("Learnable parameters")
    axis.set_ylabel("Validation matrix-element MAE (meV)")
    axis.grid(True, which="both", alpha=0.2)
    axis.legend(fontsize=8)
    figure.savefig(output / "validation_mae_vs_parameters.png", dpi=180)
    plt.close(figure)

    lines = [
        "# Stage 5 M3/M5 neural calibration",
        "",
        "One seed, 25% of training structures, full validation partition, and no test-target access.",
        "",
        "| Rank | Task | Parameters | Best step | Validation MAE (meV) | RMSE (meV) | blocks/s | Peak GPU (GiB) |",
        "|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for rank, row in enumerate(rows, start=1):
        lines.append(
            f"| {rank} | {row['task_id']} | {int(row['parameter_count']):,} | "
            f"{row['best_step']} | {float(row['validation_matrix_mae_mev']):.6f} | "
            f"{float(row['validation_matrix_rmse_mev']):.6f} | "
            f"{float(row['sampled_training_blocks_per_second']):.1f} | "
            f"{float(row['peak_cuda_memory_bytes']) / 2**30:.3f} |"
        )
    (output / "report.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
