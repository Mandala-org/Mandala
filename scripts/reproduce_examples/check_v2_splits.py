"""Check publication v2 membership on ROSI without loading matrices or training.

Use the training entrypoint's dataset dispatch, including explicit-manifest
loading, intercepting only the final expensive dataset construction.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from scripts import dataset, train
from scripts.reproduce_examples.train import digest, load_recipe

MATERIALS = ("silicon", "sio2", "zncusnses")
CHECKPOINTS = Path("/bigdata/casus/wdm/hamiltonian_learning/models/checkpoints")


def pair_payload(train_pairs, val_pairs, test_pairs, cfg, *, convention):
    return {
        name: [
            {"matrix": str(matrix.resolve()), "info": str(info.resolve())}
            for matrix, info in pairs
        ]
        for name, pairs in (
            ("train", train_pairs), ("val", val_pairs), ("test", test_pairs or [])
        )
    }


def construct_splits(path):
    recipe = load_recipe(path)
    values = recipe["config"] | recipe["training"]
    values["_explicit_args"] = set(values)
    args = SimpleNamespace(**values)
    cfg = train._populate_config_from_args(args)
    with patch.object(dataset, "_create_datasets_from_pairs", pair_payload):
        return train._build_dataset_bundle(args, cfg, None)


def identities(rows):
    return {(row["matrix"], row["info"]) for row in rows}


def compare(splits, reference, *, reduced_training=False):
    current = {name: identities(rows) for name, rows in splits.items()}
    old = {name: identities(rows) for name, rows in reference["splits"].items()}
    expected_train = reference["splits"]["train"][:80] if reduced_training else reference["splits"]["train"]
    expected_val = reference["splits"]["train"][80:] if reduced_training else reference["splits"]["val"][:len(splits["val"])]
    expected_test = reference["splits"]["val"] if reduced_training else reference["splits"]["val"][len(splits["val"]):]
    return {
        "training_members_match_expected": current["train"] == identities(expected_train),
        "training_order_matches_expected": splits["train"] == expected_train,
        "training_subset_of_previous_training": current["train"] <= old["train"],
        "validation_matches_expected": splits["val"] == expected_val,
        "test_matches_expected": splits["test"] == expected_test,
        "all_previous_members_retained": set.union(*current.values()) == set.union(*old.values()),
        "splits_disjoint": not (
            current["train"] & current["val"]
            or current["train"] & current["test"]
            or current["val"] & current["test"]
        ),
        "no_duplicate_members": all(
            len(current[name]) == len(rows) for name, rows in splits.items()
        ),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-root", type=Path, default=CHECKPOINTS)
    parser.add_argument(
        "--output", type=Path, default=Path(__file__).with_name("v2_split_check.json")
    )
    args = parser.parse_args(argv)
    # Original selected runs, not subsequent ablations/additional-seed runs.
    manifest_paths = sorted(
        set(args.checkpoint_root.glob("*/split_manifest.json"))
        | set(args.checkpoint_root.glob("*/*/split_manifest.json"))
        | set(args.checkpoint_root.glob("paper_models/**/split_manifest.json"))
    )
    report = {
        "schema_version": 1,
        "host": platform.node(),
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "method": "scripts.train._build_dataset_bundle with explicit v1-derived lists; intercept only "
        "scripts.dataset._create_datasets_from_pairs before matrix loading",
        "selection_source_sha256": {
            str(path.relative_to(ROOT)): digest(path)
            for path in (ROOT / "scripts/dataset.py", ROOT / "scripts/train.py")
        },
        "materials": {},
    }
    passed = True
    for name in MATERIALS:
        path = Path(__file__).with_name(f"{name}_v2.json")
        splits = construct_splits(path)
        reference_path = (
            args.checkpoint_root / "publication_retrain_v1" / name / "split_manifest.json"
        )
        reference = json.loads(reference_path.read_text())
        checks = compare(splits, reference, reduced_training=name == "sio2")
        try:
            previous = construct_splits(path.with_name(f"{name}.json"))
        except (ValueError, FileNotFoundError) as error:
            reconstruction = {"status": "discovery_failed", "reason": str(error)}
        else:
            reconstructed = previous == reference["splits"]
            reconstruction = {"status": "compared", "identical": reconstructed}
            checks["v1_recipe_reconstructs_recorded_members_and_order"] = reconstructed
        recipe = load_recipe(path)
        counts = {key: len(rows) for key, rows in splits.items()}
        training = recipe["training"]
        suffix = "_per_scale" if "scales" in training else ""
        multiplier = len(training["scales"]) if suffix else 1
        checks["requested_counts_match"] = all(
            counts[key] == training[f"num_{key}{suffix}"] * multiplier
            for key in ("train", "val", "test")
        )
        selected = json.loads(path.with_name(f"{name}_source.json").read_text())
        selected_run = selected["selected_config"]["run_name"]
        paper_paths = [
            p for p in manifest_paths
            if selected_run in p.parts
            or ("paper_models" in p.parts and name in str(p).lower())
        ]
        paper_checks = []
        for paper_path in paper_paths:
            comparison = compare(splits, json.loads(paper_path.read_text()), reduced_training=name == "sio2")
            paper_checks.append({"path": str(paper_path), "checks": comparison})
            passed = passed and all(comparison.values())
        canonical = json.dumps(splits, sort_keys=True, separators=(",", ":"))
        report["materials"][name] = {
            "recipe_sha256": digest(path),
            "counts": counts,
            "v1_manifest": str(reference_path),
            "v1_manifest_sha256": digest(reference_path),
            "v1_checks": checks,
            "v1_recipe_reconstruction": reconstruction,
            "explicit_manifest_sha256": digest(Path(training["dataset_manifest"])),
            "paper_manifest_search": {
                "checkpoint_root": str(args.checkpoint_root),
                "selected_run": selected_run,
                "patterns": ["*/split_manifest.json", "*/*/split_manifest.json",
                             "paper_models/**/split_manifest.json"],
                "status": "compared" if paper_paths else "not_found",
                "comparisons": paper_checks,
            },
            "split_hash_sha256": hashlib.sha256(canonical.encode()).hexdigest(),
            "splits": splits,
        }
        passed = passed and all(checks.values())
        print(f"{name}: counts={counts}, checks={checks}, paper_manifests={len(paper_paths)}")
    report["passed"] = passed
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    if not passed:
        raise SystemExit("Split comparison failed; inspect the report before submitting.")


if __name__ == "__main__":
    main()
