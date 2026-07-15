from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate and aggregate a controlled multi-seed W&B ablation."
    )
    parser.add_argument("--entity", default="b-brzoza")
    parser.add_argument("--project", required=True)
    parser.add_argument("--group", required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--expected-seeds", type=int, default=5)
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=[
            "test/hamiltonian_mae",
            "test/spectral_mae_ev",
            "test/energy_mae",
        ],
    )
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def _t_critical_95(df: int) -> float:
    table = {
        1: 12.706,
        2: 4.303,
        3: 3.182,
        4: 2.776,
        5: 2.571,
        6: 2.447,
        7: 2.365,
        8: 2.306,
        9: 2.262,
        10: 2.228,
        15: 2.131,
        20: 2.086,
        30: 2.042,
    }
    if df in table:
        return table[df]
    if df > 30:
        return 1.96
    lower = max(key for key in table if key < df)
    upper = min(key for key in table if key > df)
    weight = (df - lower) / (upper - lower)
    return table[lower] + weight * (table[upper] - table[lower])


def _summary(values: list[float]) -> dict[str, float | int]:
    n = len(values)
    mean = statistics.fmean(values)
    std = statistics.stdev(values) if n > 1 else 0.0
    half_width = _t_critical_95(n - 1) * std / math.sqrt(n) if n > 1 else 0.0
    return {
        "n": n,
        "mean": mean,
        "std": std,
        "ci95_low": mean - half_width,
        "ci95_high": mean + half_width,
    }


def _config_value(config: dict[str, Any], name: str) -> Any:
    return config.get(name, config.get(name.replace("_", "-")))


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    import wandb

    api = wandb.Api()
    runs = list(
        api.runs(
            f"{args.entity}/{args.project}",
            filters={"group": args.group},
        )
    )
    if not runs:
        raise ValueError(f"No runs found for group={args.group!r}")

    records: list[dict[str, Any]] = []
    for run in runs:
        config = dict(run.config)
        summary = dict(run.summary)
        if run.state != "finished":
            raise ValueError(f"Run {run.id} is not finished (state={run.state!r})")
        setting = _config_value(config, "ablation_setting")
        seed = _config_value(config, "seed")
        split_hash = summary.get("data/split_hash_sha256") or config.get(
            "repro/split_hash_sha256"
        )
        if setting in (None, "") or seed is None or not split_hash:
            raise ValueError(
                f"Run {run.id} lacks ablation_setting, seed, or split hash metadata"
            )
        metrics = {
            metric: float(summary[metric])
            for metric in args.metrics
            if summary.get(metric) is not None
        }
        records.append(
            {
                "run_id": run.id,
                "name": run.name,
                "setting": str(setting),
                "seed": int(seed),
                "split_hash": str(split_hash),
                "metrics": metrics,
            }
        )

    split_hashes = {record["split_hash"] for record in records}
    if len(split_hashes) != 1:
        raise ValueError(f"Runs do not share one fixed split: {sorted(split_hashes)}")

    by_setting: dict[str, dict[int, dict[str, Any]]] = {}
    for record in records:
        seed_map = by_setting.setdefault(record["setting"], {})
        if record["seed"] in seed_map:
            raise ValueError(
                f"Duplicate seed={record['seed']} for setting={record['setting']!r}"
            )
        seed_map[record["seed"]] = record

    if args.baseline not in by_setting:
        raise ValueError(f"Baseline setting {args.baseline!r} was not found")
    expected_seed_set = set(by_setting[args.baseline])
    if len(expected_seed_set) != args.expected_seeds:
        raise ValueError(
            f"Baseline has {len(expected_seed_set)} seeds; expected {args.expected_seeds}"
        )
    for setting, seed_map in by_setting.items():
        if set(seed_map) != expected_seed_set:
            raise ValueError(
                f"Setting {setting!r} does not have the same seeds as the baseline"
            )

    aggregate: dict[str, Any] = {}
    paired: dict[str, Any] = {}
    baseline = by_setting[args.baseline]
    for setting, seed_map in sorted(by_setting.items()):
        aggregate[setting] = {}
        for metric in args.metrics:
            values = [
                seed_map[seed]["metrics"][metric]
                for seed in sorted(seed_map)
                if metric in seed_map[seed]["metrics"]
            ]
            if len(values) == args.expected_seeds:
                aggregate[setting][metric] = _summary(values)
        if setting == args.baseline:
            continue
        paired[setting] = {}
        for metric in args.metrics:
            deltas = []
            for seed in sorted(expected_seed_set):
                if (
                    metric not in seed_map[seed]["metrics"]
                    or metric not in baseline[seed]["metrics"]
                ):
                    break
                deltas.append(
                    seed_map[seed]["metrics"][metric]
                    - baseline[seed]["metrics"][metric]
                )
            if len(deltas) == args.expected_seeds:
                paired[setting][metric] = {
                    **_summary(deltas),
                    "interpretation": "negative favors treatment",
                }

    return {
        "entity": args.entity,
        "project": args.project,
        "group": args.group,
        "baseline": args.baseline,
        "split_hash_sha256": next(iter(split_hashes)),
        "seeds": sorted(expected_seed_set),
        "aggregate": aggregate,
        "paired_delta_vs_baseline": paired,
        "runs": records,
    }


def main() -> None:
    args = _parse_args()
    report = build_report(args)
    rendered = json.dumps(report, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
