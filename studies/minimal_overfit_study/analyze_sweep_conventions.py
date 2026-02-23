"""Analyze convention sweep runs from Weights & Biases.

Focuses on mae_H and irrep_metrics/*_l2_block_rel. Prints a summary and flags
settings that appear definitively wrong vs inconclusive.
"""

from __future__ import annotations

import argparse
import math
from collections import defaultdict
from statistics import median

import yaml

try:
    import wandb
except Exception as exc:  # pragma: no cover - informative failure
    raise SystemExit(
        "wandb is required for this script. Install with `pip install wandb`."
    ) from exc


def safe_float(x):
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None


def load_sweep_yaml(path: str | None):
    if not path:
        return {}
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


def get_run_metric(run, key_candidates):
    for k in key_candidates:
        if k in run.summary:
            return safe_float(run.summary[k])
    return None


def collect_irrep_l2_block_rel(run):
    values = {}
    for k, v in run.summary.items():
        if k.startswith("irrep_metrics/") and k.endswith("_l2_block_rel"):
            values[k] = safe_float(v)
    return values


def classify_setting(stats, best_mae, best_irrep):
    """Heuristic classification.

    Definitely wrong: median mae_H >= 10x best AND median irrep rel >= 10x best
    for both available and values finite.
    """
    med_mae = stats.get("median_mae_H")
    med_irrep = stats.get("median_irrep_l2_block_rel")
    if med_mae is None or med_irrep is None:
        return "inconclusive"
    if best_mae is None or best_irrep is None:
        return "inconclusive"
    if med_mae >= 10 * best_mae and med_irrep >= 10 * best_irrep:
        return "definitely wrong"
    return "uncertain"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("sweep_id", type=str)
    parser.add_argument(
        "--project",
        type=str,
        default=None,
        help="WandB project (overrides sweep YAML)",
    )
    parser.add_argument(
        "--entity",
        type=str,
        default=None,
        help="WandB entity/org (optional)",
    )
    parser.add_argument(
        "--sweep-yaml",
        type=str,
        default=None,
        help="Path to sweep YAML for metadata",
    )
    args = parser.parse_args()

    sweep_yaml = load_sweep_yaml(args.sweep_yaml)
    project = args.project or sweep_yaml.get("project")
    if not project:
        raise SystemExit("Project not provided and not found in sweep YAML.")

    api = wandb.Api()
    sweep_path = (
        f"{args.entity}/{project}/{args.sweep_id}"
        if args.entity
        else f"{project}/{args.sweep_id}"
    )
    sweep = api.sweep(sweep_path)

    runs = list(sweep.runs)
    print(f"Sweep: {sweep_path}")
    print(f"Total runs: {len(runs)}")
    by_state = defaultdict(int)
    for r in runs:
        by_state[r.state] += 1
    print("Run states:")
    for k in sorted(by_state):
        print(f"  {k}: {by_state[k]}")

    # Collect per-run metrics
    run_rows = []
    all_mae = []
    all_irrep_vals = []
    for r in runs:
        mae = get_run_metric(r, ["mae_H", "final/mae_H"])
        irrep_vals = collect_irrep_l2_block_rel(r)
        if mae is not None:
            all_mae.append(mae)
        if irrep_vals:
            all_irrep_vals.extend(v for v in irrep_vals.values() if v is not None)
        run_rows.append(
            {
                "id": r.id,
                "name": r.name,
                "state": r.state,
                "mae_H": mae,
                "irrep_l2_block_rel_vals": irrep_vals,
                "config": r.config,
            }
        )

    best_mae = min(all_mae) if all_mae else None
    best_irrep = min(all_irrep_vals) if all_irrep_vals else None

    # Group by key settings
    key_fields = ["convention", "xyz-permutation", "change-box", "box-convention"]
    grouped = defaultdict(list)
    for row in run_rows:
        cfg = row["config"]
        key = tuple(cfg.get(k) for k in key_fields)
        grouped[key].append(row)

    def summarize_group(rows):
        maes = [r["mae_H"] for r in rows if r["mae_H"] is not None]
        irrep_vals = []
        for r in rows:
            irrep_vals.extend(
                v for v in r["irrep_l2_block_rel_vals"].values() if v is not None
            )
        return {
            "n": len(rows),
            "median_mae_H": median(maes) if maes else None,
            "best_mae_H": min(maes) if maes else None,
            "median_irrep_l2_block_rel": median(irrep_vals) if irrep_vals else None,
            "best_irrep_l2_block_rel": min(irrep_vals) if irrep_vals else None,
        }

    print(
        "\nSummary by setting (convention, xyz-permutation, change-box, box-convention):"
    )
    setting_rows = []
    for key, rows in grouped.items():
        stats = summarize_group(rows)
        classification = classify_setting(stats, best_mae, best_irrep)
        setting_rows.append((key, stats, classification))

    # Sort by median mae_H (None last)
    def sort_key(item):
        med = item[1]["median_mae_H"]
        return math.inf if med is None else med

    for key, stats, classification in sorted(setting_rows, key=sort_key):
        print(f"\n  Setting: {dict(zip(key_fields, key))}")
        print(f"    runs: {stats['n']}")
        print(f"    median mae_H: {stats['median_mae_H']}")
        print(f"    best mae_H: {stats['best_mae_H']}")
        print(f"    median irrep l2_block_rel: {stats['median_irrep_l2_block_rel']}")
        print(f"    best irrep l2_block_rel: {stats['best_irrep_l2_block_rel']}")
        print(f"    classification: {classification}")

    # Overall best runs
    print("\nTop runs by mae_H (lower is better):")
    top = sorted(
        [r for r in run_rows if r["mae_H"] is not None], key=lambda r: r["mae_H"]
    )[:10]
    for r in top:
        cfg = r["config"]
        print(
            f"  {r['name']} ({r['id']}): mae_H={r['mae_H']} "
            f"cfg={{convention={cfg.get('convention')}, xyz-permutation={cfg.get('xyz-permutation')}, "
            f"change-box={cfg.get('change-box')}, box-convention={cfg.get('box-convention')}}}"
        )

    # Inconclusive vs wrong summary
    wrong = [s for s in setting_rows if s[2] == "definitely wrong"]
    unsure = [s for s in setting_rows if s[2] == "uncertain"]
    print("\nClassification summary:")
    print(f"  definitely wrong: {len(wrong)}")
    print(f"  uncertain: {len(unsure)}")


if __name__ == "__main__":
    main()
