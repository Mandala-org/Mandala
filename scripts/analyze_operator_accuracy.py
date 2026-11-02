"""Summarize fixed preregistered validation horizons from small CSV files.

This analysis is separate from the frozen scientific execution snapshot. It
reads only the registration and each registered arm's metrics/version*/metrics.csv.
It never opens datasets, checkpoints, test evaluations or workflow artifacts.
"""

from __future__ import annotations

# Exact integer checks reject JSON booleans in frozen seeds and budgets.

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
import math
from pathlib import Path
import re

RAW = "val/hamiltonian_mae_elementwise"
PROJECTED = RAW + "_projected"
MAX_BYTES = 8 * 1024**2


def digest(value):
    return hashlib.sha256(value).hexdigest()


def read_small(path):
    if path.stat().st_size > MAX_BYTES:
        raise ValueError(f"Refusing large analysis input: {path}")
    value = path.read_bytes()
    if len(value) > MAX_BYTES:
        raise ValueError(f"Analysis input grew beyond its size limit: {path}")
    return value


def integer(value, label):
    number = float(value)
    if not math.isfinite(number) or number < 0 or not number.is_integer():
        raise ValueError(f"Invalid {label}: {value}")
    return int(number)


def optional_number(value, label):
    if value is None or value == "":
        return None
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise ValueError(f"Invalid {label}: {value}")
    return number


def read_arm(results, job, horizons, context):
    arm = results / job["id"]
    paths = sorted((arm / "metrics").glob("version*/metrics.csv"))
    if len(paths) > 32:
        raise ValueError(f"Too many metric versions for {job['id']}")
    epochs, inputs = {}, []
    for path in paths:
        if not path.resolve().is_relative_to(results.resolve()):
            raise ValueError(f"Metrics path escapes the results directory: {path}")
        value = read_small(path)
        # A writer may be in the middle of appending a record. Do not interpret
        # a partial last row as an endpoint; hash the actual captured bytes.
        complete = value if value.endswith(b"\n") else value[: value.rfind(b"\n") + 1]
        inputs.append(
            {
                "path": str(path),
                "sha256": digest(value),
                "bytes": len(value),
                "ignored_unterminated_tail_bytes": len(value) - len(complete),
            }
        )
        reader = csv.DictReader(io.StringIO(complete.decode("utf-8")))
        if not reader.fieldnames:
            continue
        if any(name and name.startswith("test/") for name in reader.fieldnames):
            raise ValueError(f"Refusing CSV containing test columns: {path}")
        if not {"epoch", RAW}.issubset(reader.fieldnames):
            raise ValueError(f"Missing validation metric columns: {path}")
        for index, row in enumerate(reader):
            if index >= 10000:
                raise ValueError(f"Too many rows in screen metrics: {path}")
            raw = optional_number(row.get(RAW), RAW)
            if raw is None:
                continue
            epoch = integer(row["epoch"], "epoch")
            record = {
                "epoch_zero_based": epoch,
                "step_zero_based": (
                    integer(row["step"], "step") if row.get("step") else None
                ),
                "raw_mae_mev": raw * 1000,
                "projected_mae_mev_at_same_epoch": None,
            }
            projected = optional_number(row.get(PROJECTED), PROJECTED)
            if projected is not None:
                record["projected_mae_mev_at_same_epoch"] = projected * 1000
            if epoch in epochs:
                previous = epochs[epoch]
                for key, current in record.items():
                    if (
                        current is not None
                        and previous[key] is not None
                        and current != previous[key]
                    ):
                        raise ValueError(
                            f"Conflicting validation epoch {epoch} for {job['id']}"
                        )
                    if previous[key] is None:
                        previous[key] = current
            else:
                epochs[epoch] = record
    endpoints = {}
    for endpoint in horizons:
        target = endpoint - 1
        complete = target in epochs
        resource_incomplete = not epochs or max(epochs) < target
        status = (
            "complete"
            if complete
            else "resource_incomplete" if resource_incomplete else "missing_endpoint"
        )
        selected = (
            min(
                (epochs[e] for e in sorted(epochs) if e <= target),
                key=lambda r: r["raw_mae_mev"],
                default=None,
            )
            if complete
            else None
        )
        endpoints[str(endpoint)] = {
            "requested_completed_epochs": endpoint,
            "required_csv_epoch_zero_based": target,
            "status": status,
            "complete": complete,
            "missing": not complete,
            "resource_incomplete": resource_incomplete,
            "raw_endpoint": epochs.get(target),
            "best_raw_through_endpoint": selected,
        }
    return {
        "material": job["material"],
        "treatment": job["treatment"],
        "seed": job["seed"],
        "comparison_context": context,
        "metrics_missing": not paths,
        "validation_rows_missing": not epochs,
        "maximum_observed_validation_epoch_zero_based": max(epochs) if epochs else None,
        "inputs": inputs,
        "endpoints": endpoints,
    }


def comparison(arms, left_id, right_id, endpoint):
    ids = [left_id, right_id]
    missing = [
        name
        for name in ids
        if name is None or not arms[name]["endpoints"][endpoint]["complete"]
    ]
    result = {
        "left_arm": left_id,
        "right_arm": right_id,
        "status": "incomplete" if missing else "complete",
        "missing_or_incomplete_arms": missing,
        "raw_endpoint_delta_mev": None,
        "best_raw_delta_mev": None,
        "projected_at_raw_best_delta_mev": None,
    }
    if not missing:
        left, right = [arms[name]["endpoints"][endpoint] for name in ids]
        result["raw_endpoint_delta_mev"] = (
            left["raw_endpoint"]["raw_mae_mev"] - right["raw_endpoint"]["raw_mae_mev"]
        )
        result["best_raw_delta_mev"] = (
            left["best_raw_through_endpoint"]["raw_mae_mev"]
            - right["best_raw_through_endpoint"]["raw_mae_mev"]
        )
        projected = [
            side["best_raw_through_endpoint"]["projected_mae_mev_at_same_epoch"]
            for side in (left, right)
        ]
        if all(value is not None for value in projected):
            result["projected_at_raw_best_delta_mev"] = projected[0] - projected[1]
    return result


def interaction(arms, ids, endpoint):
    missing = [
        name
        for name in ids
        if name is None or not arms[name]["endpoints"][endpoint]["complete"]
    ]
    result = {
        "arms_in_formula_order": ids,
        "status": "incomplete" if missing else "complete",
        "missing_or_incomplete_arms": missing,
        "raw_endpoint_interaction_mev": None,
        "best_raw_interaction_mev": None,
        "projected_at_raw_best_interaction_mev": None,
    }
    if not missing:
        points = [arms[name]["endpoints"][endpoint] for name in ids]
        for key, source, metric in (
            ("raw_endpoint_interaction_mev", "raw_endpoint", "raw_mae_mev"),
            ("best_raw_interaction_mev", "best_raw_through_endpoint", "raw_mae_mev"),
            (
                "projected_at_raw_best_interaction_mev",
                "best_raw_through_endpoint",
                "projected_mae_mev_at_same_epoch",
            ),
        ):
            values = [point[source][metric] for point in points]
            if all(value is not None for value in values):
                result[key] = values[0] - values[1] - values[2] + values[3]
    return result


def registered_horizons(job):
    values = job["primary_endpoints_epochs"]
    if not isinstance(values, list) or not values or len(values) > 16:
        raise ValueError("Register 1–16 fixed validation horizons")
    if any(type(value) is not int or value <= 0 for value in values):  # noqa: E721
        raise ValueError("Registered horizons must be positive integer epoch counts")
    if values != sorted(set(values)):
        raise ValueError("Registered horizons must be strictly increasing")
    return tuple(values)


def comparison_context(registration, job):
    # A hardware_block is a matching label, not proof of a GPU/device class.
    # Runtime/device declarations participate in matching whenever supplied;
    # never open an external workflow assignment or infer equivalence from it.
    fields = (
        "hardware_block",
        "hardware",
        "runtime",
        "runtime_identity",
        "runtime_profile",
    )
    return {
        field: job[field] if field in job else registration.get(field)
        for field in fields
    }


def analyze(registration_path, results):
    value = read_small(registration_path)
    registration = json.loads(value)
    jobs = registration["jobs"]
    if not jobs or len(jobs) > 128:
        raise ValueError("Registration must contain 1–128 arms")
    horizons = registered_horizons(jobs[0])
    if (
        "primary_endpoints_epochs" in registration
        and tuple(registration["primary_endpoints_epochs"]) != horizons
    ):
        raise ValueError("Registration and arm horizons disagree")
    groups, arms = {}, {}
    for job in jobs:
        name = job["id"]
        if not re.fullmatch(r"[A-Za-z0-9_-]+", name) or name in arms:
            raise ValueError(f"Invalid or duplicate arm identity: {name}")
        if registered_horizons(job) != horizons:
            raise ValueError("Every registered arm must share the same fixed horizons")
        if type(job["seed"]) is not int or job["seed"] < 0:  # noqa: E721
            raise ValueError("Initialization seeds must be nonnegative integers")
        context = comparison_context(registration, job)
        context_json = json.dumps(context, sort_keys=True, separators=(",", ":"))
        group_key = (job["material"], job["seed"], context_json)
        group = groups.setdefault(
            group_key,
            {
                "material": job["material"],
                "seed": job["seed"],
                "comparison_context": context,
                "treatments": {},
            },
        )
        treatments = group["treatments"]
        if job["treatment"] in treatments:
            raise ValueError(
                "Ambiguous material/seed/context/treatment in registration"
            )
        treatments[job["treatment"]] = name
        arms[name] = read_arm(results, job, horizons, context)
    # Retain the old material keys when there is only one registered block for
    # that material. Multiple seeds or hardware/runtime blocks get explicit IDs.
    by_group = {}
    for (material, seed, context_json), group in groups.items():
        count = sum(other[0] == material for other in groups)
        group_id = (
            material
            if count == 1
            else f"{material}__s{seed}__{digest(context_json.encode())[:12]}"
        )
        by_group[group_id] = group
        for name in group["treatments"].values():
            arms[name]["comparison_group"] = group_id
    baselines, interactions, contrasts = {}, {}, {}
    for group_id, group in by_group.items():
        treatments = group["treatments"]
        baseline = treatments.get("baseline")
        for treatment, name in treatments.items():
            if treatment != "baseline":
                baselines[name] = {
                    str(e): comparison(arms, name, baseline, str(e)) for e in horizons
                }
        ids = [
            treatments.get(t)
            for t in ("mse_rollback_off", "mse", "rollback_off", "baseline")
        ]
        interactions[group_id] = {
            str(e): interaction(arms, ids, str(e)) for e in horizons
        }
    for descriptor in registration.get("contrasts", []):
        expression = descriptor.split(" (", 1)[0]
        applicable = list(by_group)
        for material in sorted(
            {group["material"] for group in by_group.values()}, key=len, reverse=True
        ):
            if expression.startswith(material + "_"):
                applicable = [
                    name
                    for name, group in by_group.items()
                    if group["material"] == material
                ]
                expression = expression[len(material) + 1 :]
                break
        if expression.count("-") != 1:
            raise ValueError(f"Unsupported registered contrast: {descriptor}")
        left, right = expression.split("-")
        contrasts[descriptor] = {
            group_id: {
                str(e): comparison(
                    arms,
                    by_group[group_id]["treatments"].get(left),
                    by_group[group_id]["treatments"].get(right),
                    str(e),
                )
                for e in horizons
            }
            for group_id in applicable
        }
    return {
        "schema_version": 2,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "analysis_source": {
            "path": str(Path(__file__).resolve()),
            "sha256": digest(Path(__file__).read_bytes()),
            "separate_from_frozen_execution_source": True,
        },
        "registration_input": {"path": str(registration_path), "sha256": digest(value)},
        "registration_id": registration.get("id"),
        "metric_contract": {
            "units": "meV",
            "primary": "raw global structure-mean observed-entry validation MAE",
            "horizons_completed_epochs": list(horizons),
            "projection": "at the same raw-selected epoch; never independently selected",
            "best_raw_ties": "earliest epoch",
            "differences": "left minus right; negative means lower observed MAE",
            "interaction_formula": "mse_rollback_off - mse - rollback_off + baseline",
            "incomplete": "No substituted/adaptive horizons. Resource-incomplete means the required epoch is not yet observed; termination cause is unknown from CSV alone.",
            "matching": "Within material, initialization seed and declared hardware/runtime context only; no automatic pooling across seeds, hardware or runtimes.",
            "hardware_identity": "hardware_block is a matching label only. Missing device/runtime declarations remain unknown; no hardware equivalence is inferred.",
            "limitations": "Descriptive per-seed comparisons only, with no significance, convergence, or causal component claim.",
        },
        "comparison_groups": by_group,
        "arms": arms,
        "baseline_differences": baselines,
        "registered_contrasts": contrasts,
        "mae_by_rollback_interactions": interactions,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registration", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError(args.output)
    result = analyze(args.registration, args.results)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "arms": len(result["arms"]),
                "registered_horizons": result["metric_contract"][
                    "horizons_completed_epochs"
                ],
            }
        )
    )


if __name__ == "__main__":
    main()
