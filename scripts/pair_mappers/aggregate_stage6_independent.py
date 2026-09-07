#!/usr/bin/env python3
"""Select independent validation leaders and compose their disjoint headline metric."""

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
    parser.add_argument("--stage6-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def _atomic_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def combine_disjoint_metrics(
    *metrics: dict[str, float | int]
) -> dict[str, float | int]:
    """Combine MAE/RMSE from disjoint scalar sets without rereading targets."""
    count = sum(int(item["scalar_count"]) for item in metrics)
    if count <= 0:
        raise ValueError("cannot combine empty metric sets")
    absolute = sum(float(item["mae"]) * int(item["scalar_count"]) for item in metrics)
    square = sum(
        float(item["rmse"]) ** 2 * int(item["scalar_count"]) for item in metrics
    )
    return {
        "mae": absolute / count,
        "rmse": math.sqrt(square / count),
        "scalar_count": count,
    }


def main() -> None:
    args = parse_args()
    root = args.stage6_root.resolve()
    output = args.output_dir.resolve()
    if output.exists() and list(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite nonempty {output}")
    output.mkdir(parents=True, exist_ok=True)
    root_summary = json.loads((root / "summary.json").read_text())
    if not root_summary["passed"] or root_summary["test_shards_read"]:
        raise ValueError("Stage-6 freeze did not pass cleanly")

    candidates: list[dict[str, object]] = []
    for scope in ("offsite", "onsite_neural"):
        grid = root / scope
        aggregate = json.loads((grid / "aggregate_v1" / "summary.json").read_text())
        if not aggregate["passed"] or aggregate["test_shards_read"]:
            raise ValueError(f"{scope} neural aggregate did not pass cleanly")
        for row in _read_csv(grid / "aggregate_v1" / "results.csv"):
            task_id = row["task_id"]
            metrics = json.loads(
                (grid / "runs" / task_id / "validation_metrics.json").read_text()
            )["projected"]["matrix_elements"]
            expected_scope = "offsite" if scope == "offsite" else "onsite"
            if row["target_scope"] != expected_scope:
                raise ValueError(f"scope mismatch for {task_id}")
            candidates.append(
                {
                    "scope": expected_scope,
                    "kind": "neural",
                    "model_id": task_id,
                    "artifact": str((grid / "runs" / task_id).relative_to(root)),
                    "method": row["architecture"],
                    "family": row["family"],
                    "descriptor_key": row["descriptor_key"],
                    "onsite_baseline": row["onsite_baseline"],
                    "mae": float(metrics["mae"]),
                    "rmse": float(metrics["rmse"]),
                    "scalar_count": int(metrics["scalar_count"]),
                }
            )
    ridge_root = root / "onsite_ridge"
    for family in ("d1", "d2", "d3", "d4"):
        family_dir = ridge_root / family
        summary = json.loads((family_dir / "summary.json").read_text())
        config = json.loads((family_dir / "config.json").read_text())
        if (
            not summary["passed"]
            or summary["test_shards_read"]
            or config["test_shards_read"]
        ):
            raise ValueError(f"onsite ridge {family} did not pass cleanly")
        detailed = json.loads((family_dir / "validation_metrics.json").read_text())
        for row in _read_csv(family_dir / "results.csv"):
            model_id = row["model_id"]
            metrics = detailed[model_id]["projected"]["matrix_elements"]
            candidates.append(
                {
                    "scope": "onsite",
                    "kind": "closed_form",
                    "model_id": model_id,
                    "artifact": str(family_dir.relative_to(root)),
                    "method": row["method"],
                    "family": family,
                    "descriptor_key": row["descriptor_key"],
                    "onsite_baseline": (
                        "invariant_mean"
                        if row["method"] == "invariant_mean"
                        else "none"
                    ),
                    "mae": float(metrics["mae"]),
                    "rmse": float(metrics["rmse"]),
                    "scalar_count": int(metrics["scalar_count"]),
                }
            )
    if not candidates or any(
        not math.isfinite(float(row["mae"])) for row in candidates
    ):
        raise ValueError(
            "all independent candidates must have finite validation metrics"
        )
    candidates.sort(key=lambda row: (str(row["scope"]), float(row["mae"])))
    with (output / "candidates.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(candidates[0]))
        writer.writeheader()
        writer.writerows(candidates)
    best = {
        scope: min(
            (row for row in candidates if row["scope"] == scope),
            key=lambda row: float(row["mae"]),
        )
        for scope in ("onsite", "offsite")
    }
    combined = combine_disjoint_metrics(
        {key: best["onsite"][key] for key in ("mae", "rmse", "scalar_count")},
        {key: best["offsite"][key] for key in ("mae", "rmse", "scalar_count")},
    )
    bundle = {
        "convention": "mandala-stage6-independent-composition-v1",
        "selection_partition": "validation",
        "test_shards_read": False,
        "selection_uses_test_hamiltonian": False,
        "parameter_sharing_between_scopes": False,
        "joint_optimization": False,
        "onsite": best["onsite"],
        "offsite": best["offsite"],
        "combined_validation_matrix_elements": combined,
        "composition": (
            "onsite predictions come only from the selected onsite artifact; "
            "directed offsite predictions come only from the selected offsite artifact"
        ),
        "evaluation_projection": (
            "onsite transpose projection and global reverse-pair offsite projection, "
            "both strictly after independent fitting"
        ),
    }
    bundle["manifest_hash"] = hashlib.sha256(
        json.dumps(bundle, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    _atomic_json(output / "selected_bundle.json", bundle)
    summary = {
        "completed": True,
        "passed": True,
        "manifest_hash": bundle["manifest_hash"],
        "candidate_count": len(candidates),
        "best_onsite_model_id": best["onsite"]["model_id"],
        "best_offsite_model_id": best["offsite"]["model_id"],
        "combined_validation_matrix_mae_mev": combined["mae"],
        "combined_validation_matrix_rmse_mev": combined["rmse"],
        "test_shards_read": False,
    }
    _atomic_json(output / "summary.json", summary)
    (output / "report.md").write_text(
        "# Stage 6 independent composition gate\n\n"
        f"Selected onsite: `{best['onsite']['model_id']}` at {float(best['onsite']['mae']):.6f} meV.\n\n"
        f"Selected offsite: `{best['offsite']['model_id']}` at {float(best['offsite']['mae']):.6f} meV.\n\n"
        f"Exact disjoint-union validation headline: {float(combined['mae']):.6f} meV MAE, "
        f"{float(combined['rmse']):.6f} meV RMSE. No test target was read.\n"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
