"""Prepare a frozen SiOx split and train four MANDALA configurations on ROSI.

No scheduler submission occurs here. Launch from the repository root with
PYTHONPATH=src:.; the bash entry points set the reproducibility environment.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import json
from pathlib import Path
import random
import time

ROOT = Path(__file__).resolve().parents[2]
BUNDLE = Path(__file__).resolve().parent
LARGE = Path("/bigdata/casus/wdm/hamiltonian_learning")
RECORDS = ROOT / "researchops/runs/siox_models_v1"
MODELS = ("deeph", "deeph_e3", "mace_h", "mandala_paper")


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024**2), b""):
            result.update(chunk)
    return result.hexdigest()


def write_json(path, document):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def immutable_json(path, document):
    if path.exists():
        if json.loads(path.read_text()) != document:
            raise ValueError(f"Frozen specification changed: {path}")
    else:
        write_json(path, document)


@contextmanager
def exclusive(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def source_files(matrix, info):
    return {
        path
        for path in (
            matrix,
            info,
            *matrix.parent.glob("*.cif"),
            *matrix.parent.glob("*.dat"),
        )
        if path.is_file()
    }


def select_inputs(data_root, *, seed=42, num_train=90, num_val=10):
    from scripts.dataset import discover_siox_snapshot_pairs, _split_shuffled_pairs

    # Identical discovery order and Python shuffle to the frozen paper recipe.
    pairs = discover_siox_snapshot_pairs(data_root)
    random.Random(seed).shuffle(pairs)
    train, val, test = _split_shuffled_pairs(
        pairs, num_train=num_train, num_val=num_val, num_test=0
    )
    splits = {}
    for split, selected in (("train", train), ("val", val), ("test", test)):
        splits[split] = []
        for matrix, info in selected:
            sources = source_files(matrix, info)
            splits[split].append(
                {
                    "matrix_path": str(matrix),
                    "info_path": str(info),
                    "source_sha256": {str(p): digest(p) for p in sorted(sources)},
                }
            )
    return {"schema_version": 1, "seed": seed, "splits": splits}


def validate_sources(row):
    sources = source_files(Path(row["matrix_path"]), Path(row["info_path"]))
    if {str(p) for p in sources} != set(row["source_sha256"]):
        raise ValueError(f"OpenMX source file inventory changed: {row['matrix_path']}")
    for path, expected in row["source_sha256"].items():
        if digest(path) != expected:
            raise ValueError(f"OpenMX source changed since preparation: {path}")


def prepare(args):
    args.records.mkdir(parents=True, exist_ok=True)
    with exclusive(args.records / ".prepare.lock"):
        selection = select_inputs(
            args.data_root.resolve(),
            seed=42,
            num_train=args.num_train,
            num_val=args.num_val,
        )
        rows = [row for values in selection["splits"].values() for row in values]
        manifest = selection
        immutable_json(args.records / "dataset.json", manifest)
    print(
        f"Frozen {len(rows)} OpenMX snapshot references: {args.records / 'dataset.json'}",
        flush=True,
    )


def validate_manifest(path):
    manifest = json.loads(path.read_text())
    if manifest.get("schema_version") != 1 or manifest.get("seed") != 42:
        raise ValueError("Expected the frozen seed-42 SiOx manifest")
    paths = []
    for split in ("train", "val", "test"):
        for row in manifest["splits"][split]:
            validate_sources(row)
            paths.append(str(Path(row["matrix_path"]).resolve()))
    if len(paths) != len(set(paths)):
        raise ValueError("Manifest contains duplicate or overlapping snapshots")
    if not manifest["splits"]["train"] or not manifest["splits"]["val"]:
        raise ValueError("Manifest needs nonempty training and validation splits")
    return manifest


def model_settings(model, manifest, manifest_path, checkpoint_root):
    """Apply only dataset/IO overrides to the frozen scientific configurations."""
    from omegaconf import OmegaConf

    document = OmegaConf.to_container(
        OmegaConf.load(ROOT / f"configs/reference/{model}.yaml"), resolve=False
    )
    document["data"]["splits"] = {
        split: [
            str(Path(row["matrix_path"]).parent) for row in manifest["splits"][split]
        ]
        for split in ("train", "val", "test")
    }
    document["training"].update(seed=42, data_split_seed=42)
    # Explicit lists above are authoritative, not these source-default ratios.
    document["training"]["split"] = {
        f"{split}_size": len(paths)
        for split, paths in document["data"]["splits"].items()
    }
    return document


def recovery_checkpoint(run_dir, *, action):
    import torch

    latest = run_dir / "latest_checkpoint.pt"
    if action == "start" and run_dir.exists():
        raise FileExistsError(f"Start refuses existing run: {run_dir}")
    if action == "continue" and not latest.is_file():
        raise FileNotFoundError(f"No recovery checkpoint: {latest}")
    if run_dir.exists() and not latest.is_file():
        raise RuntimeError(
            f"Incomplete run has no safe checkpoint: {run_dir}; inspect it before retrying"
        )
    if not latest.is_file():
        return None
    checkpoint = torch.load(latest, map_location="cpu", weights_only=False)
    if not checkpoint.get("epoch_boundary_resume", {}).get("safe_to_resume"):
        raise ValueError(f"Checkpoint is not a safe epoch boundary: {latest}")
    return latest


def train(args):
    manifest_path = (args.records / "dataset.json").resolve()
    manifest = validate_manifest(manifest_path)
    settings = model_settings(args.model, manifest, manifest_path, args.checkpoint_root)
    if args.plan:
        print(json.dumps(settings, indent=2))
        return
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError(
            "ROSI training requires a CUDA GPU; use --plan for inspection"
        )
    run_dir = args.checkpoint_root / args.model
    with exclusive(args.checkpoint_root / f".{args.model}.lock"):
        latest = recovery_checkpoint(run_dir, action=args.action)
        completion = args.records / f"{args.model}.complete.json"
        if completion.is_file():
            print(f"Training already completed: {completion}")
            return
        started = time.perf_counter()
        from omegaconf import OmegaConf
        from scripts.train import main

        config = args.records / f"{args.model}.yaml"
        content = OmegaConf.to_yaml(OmegaConf.create(settings), resolve=False)
        if config.exists() and config.read_text() != content:
            raise ValueError(f"Frozen model configuration changed: {config}")
        config.write_text(content)
        cli = [
            "--config",
            str(config),
            "--output",
            str(run_dir),
            "--accelerator",
            "gpu",
            "--max-wall-clock-seconds",
            "162000",
            "--log-dir",
            str(args.records / args.model / "metrics"),
        ]
        if latest:
            cli.extend(["--resume", str(latest)])
        main(cli)
        max_epochs = settings.get("training", {}).get(
            "max_epochs", settings.get("max_epochs", 10000)
        )
        min_lr = settings.get("training", {}).get("min_lr")
        latest = recovery_checkpoint(run_dir, action="continue")
        checkpoint = torch.load(latest, map_location="cpu", weights_only=False)
        epoch = int(checkpoint["epoch"])
        learning_rates = [
            group["lr"]
            for opt in checkpoint["optimizer_states"]
            for group in opt["param_groups"]
        ]
        complete = epoch >= max_epochs or (
            min_lr is not None and max(learning_rates) <= min_lr
        )
        status = {
            "model": args.model,
            "epoch": epoch,
            "learning_rates": learning_rates,
            "complete": complete,
            "checkpoint": str(latest),
            "seconds": time.perf_counter() - started,
        }
        write_json(args.records / f"{args.model}.last_segment.json", status)
        if complete:
            write_json(completion, status)
        print(json.dumps(status, indent=2), flush=True)
        if not complete:
            print("Segment saved. Resubmit the same launcher to continue.", flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="operation", required=True)
    preparation = sub.add_parser("prepare")
    preparation.add_argument("--data-root", type=Path, default=LARGE / "data/SiOx_new")
    preparation.add_argument("--records", type=Path, default=RECORDS)
    preparation.add_argument("--num-train", type=int, default=90)
    preparation.add_argument("--num-val", type=int, default=10)
    training = sub.add_parser("train")
    training.add_argument("--model", choices=MODELS, required=True)
    training.add_argument("--records", type=Path, default=RECORDS)
    training.add_argument(
        "--checkpoint-root",
        type=Path,
        default=LARGE / "models/checkpoints/siox_models_v1",
    )
    training.add_argument(
        "--action", choices=("auto", "start", "continue"), default="auto"
    )
    training.add_argument("--plan", action="store_true")
    args = parser.parse_args(argv)
    if args.operation == "prepare":
        if args.num_train < 1 or args.num_val < 1:
            parser.error("Training and validation counts must be positive")
        prepare(args)
    else:
        args.checkpoint_root = args.checkpoint_root.resolve()
        train(args)


if __name__ == "__main__":
    main()
