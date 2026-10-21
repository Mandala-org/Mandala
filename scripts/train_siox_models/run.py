"""Prepare a frozen SiOx split and train four MANDALA configurations on ROSI.

No scheduler submission occurs here. Launch from the repository root with
PYTHONPATH=src:.; the bash entry points set the reproducibility environment.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
import fcntl
import hashlib
import importlib.metadata
import json
import multiprocessing
import os
from pathlib import Path
import platform
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


def select_inputs(data_root, native_root, *, seed=42, num_train=90, num_val=10):
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
                    "native_path": str(
                        native_root / matrix.parent.relative_to(data_root)
                    ),
                }
            )
    return {"schema_version": 1, "seed": seed, "splits": splits}


def native_hashes(directory):
    files = {
        "hamiltonians.h5",
        "overlaps.h5",
        "info.json",
        "element.dat",
        "orbital_types.dat",
        "site_positions.dat",
        "lat.dat",
        "rlat.dat",
        "R_list.dat",
    }
    return {name: digest(directory / name) for name in sorted(files)}


def validate_sources(row):
    sources = source_files(Path(row["matrix_path"]), Path(row["info_path"]))
    if {str(p) for p in sources} != set(row["source_sha256"]):
        raise ValueError(f"OpenMX source file inventory changed: {row['matrix_path']}")
    for path, expected in row["source_sha256"].items():
        if digest(path) != expected:
            raise ValueError(f"OpenMX source changed since preparation: {path}")


def convert_one(row):
    from data.deeph_e3_exporter import export_openmx_snapshot_to_deeph_e3

    destination = Path(row["native_path"])
    receipt = destination / "conversion.json"
    validate_sources(row)
    if destination.exists():
        if not receipt.is_file():
            raise RuntimeError(
                f"Unverified conversion directory: {destination}. Inspect and move it "
                "aside before retrying preparation; existing data are never overwritten."
            )
        record = json.loads(receipt.read_text())
        if record["input"] != row or record["native_sha256"] != native_hashes(
            destination
        ):
            raise ValueError(f"Converted input or provenance changed: {destination}")
        return record
    summary = export_openmx_snapshot_to_deeph_e3(
        row["matrix_path"],
        row["info_path"],
        destination,
        include_overlap=True,
        include_density=False,
        validate=True,
    )
    validate_sources(row)
    record = {
        "input": row,
        "native_sha256": native_hashes(destination),
        "validation": summary.to_dict(),
    }
    write_json(receipt, record)
    return record


def prepare(args):
    args.records.mkdir(parents=True, exist_ok=True)
    with exclusive(args.records / ".prepare.lock"):
        selection = select_inputs(
            args.data_root.resolve(),
            args.native_root.resolve(),
            seed=42,
            num_train=args.num_train,
            num_val=args.num_val,
        )
        immutable_json(args.records / "selection.json", selection)
        rows = [row for values in selection["splits"].values() for row in values]
        if args.workers == 1:
            converted = [convert_one(row) for row in rows]
        else:
            with ProcessPoolExecutor(
                max_workers=args.workers,
                mp_context=multiprocessing.get_context("spawn"),
            ) as pool:
                converted = list(pool.map(convert_one, rows))
        manifest = selection | {"conversions": converted}
        immutable_json(args.records / "dataset.json", manifest)
    print(
        f"Prepared {len(rows)} validated snapshots: {args.records / 'dataset.json'}",
        flush=True,
    )


def validate_manifest(path):
    manifest = json.loads(path.read_text())
    if manifest.get("schema_version") != 1 or manifest.get("seed") != 42:
        raise ValueError("Expected the frozen seed-42 SiOx manifest")
    paths = []
    conversions = {
        item["input"]["native_path"]: item for item in manifest["conversions"]
    }
    for split in ("train", "val", "test"):
        for row in manifest["splits"][split]:
            validate_sources(row)
            paths.append(str(Path(row["matrix_path"]).resolve()))
            conversion = conversions[row["native_path"]]
            if conversion["input"] != row or conversion[
                "native_sha256"
            ] != native_hashes(Path(row["native_path"])):
                raise ValueError(f"Native snapshot changed: {row['native_path']}")
    if len(paths) != len(set(paths)):
        raise ValueError("Manifest contains duplicate or overlapping snapshots")
    if not manifest["splits"]["train"] or not manifest["splits"]["val"]:
        raise ValueError("Manifest needs nonempty training and validation splits")
    return manifest


def model_settings(model, manifest, manifest_path, checkpoint_root):
    """Apply only dataset/IO overrides to the frozen scientific configurations."""
    if model == "mandala_paper":
        recipe = json.loads((BUNDLE / "mandala_paper.json").read_text())
        envelope = (BUNDLE / recipe["envelope_file"]).resolve()
        if digest(envelope) != recipe["envelope_sha256"]:
            raise ValueError("Frozen paper radial envelope checksum mismatch")
        recipe["config"]["hamiltonian_envelope_path"] = str(envelope)
        values = recipe["config"] | recipe["training"]
        values.update(
            dataset_manifest=str(manifest_path),
            checkpoint_dir=str(checkpoint_root),
            save_dir=str(checkpoint_root),
            run_name=model,
        )
        return values
    from omegaconf import OmegaConf

    document = OmegaConf.to_container(
        OmegaConf.load(ROOT / f"configs/reference/{model}.yaml"), resolve=False
    )
    document["data"]["splits"] = {
        split: [row["native_path"] for row in manifest["splits"][split]]
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


def runtime_contract(settings, manifest_path):
    import torch

    sources = sorted((ROOT / "src").rglob("*.py")) + sorted(
        (ROOT / "scripts").rglob("*.py")
    )
    artifacts = [p for p in BUNDLE.iterdir() if p.suffix in {".sh", ".json"}]
    artifacts.append(ROOT / "scripts/reproduce_examples/sio2_envelope.json")
    return {
        "settings": settings,
        "dataset_sha256": digest(manifest_path),
        "source_sha256": {str(p.relative_to(ROOT)): digest(p) for p in sources},
        "recipe_sha256": {
            str(p.relative_to(ROOT)): digest(p) for p in sorted(artifacts)
        },
        "environment": {
            key: os.environ.get(key)
            for key in (
                "PYTHONHASHSEED",
                "CUBLAS_WORKSPACE_CONFIG",
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
            )
        },
        "python": platform.python_version(),
        "packages": {
            d.metadata["Name"]: d.version for d in importlib.metadata.distributions()
        },
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "gpu": torch.cuda.get_device_name(),
        "gpu_capability": list(torch.cuda.get_device_capability()),
        "torch_num_threads": torch.get_num_threads(),
    }


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
        contract = args.records / f"{args.model}.contract.json"
        if latest and not contract.is_file():
            raise FileNotFoundError(
                f"Recovery requires the original contract: {contract}"
            )
        immutable_json(contract, runtime_contract(settings, manifest_path))
        completion = args.records / f"{args.model}.complete.json"
        if completion.is_file():
            print(f"Training already completed: {completion}")
            return
        started = time.perf_counter()
        if args.model == "mandala_paper":
            from scripts.train import run_training

            values = dict(settings)
            values.update(
                resume_from_checkpoint=str(latest) if latest else None,
                resume_mode="latest",
            )
            values["_explicit_args"] = set(values)
            run_training(values)
            max_epochs = values["max_epochs"]
            min_lr = values["lr_scheduler_min_lr"]
        else:
            from omegaconf import OmegaConf
            from scripts.train_modular import main

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
            max_epochs = settings["training"]["max_epochs"]
            min_lr = settings["training"].get("min_lr")
        latest = recovery_checkpoint(run_dir, action="continue")
        checkpoint = torch.load(latest, map_location="cpu", weights_only=False)
        epoch = int(checkpoint["epoch"])
        learning_rates = [
            group["lr"]
            for opt in checkpoint["optimizer_states"]
            for group in opt["param_groups"]
        ]
        complete = epoch >= max_epochs or (
            args.model != "mandala_paper"
            and min_lr is not None
            and max(learning_rates) <= min_lr
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
    preparation.add_argument(
        "--native-root", type=Path, default=LARGE / "data/DeepH-E3/SiOx_models_v1"
    )
    preparation.add_argument("--records", type=Path, default=RECORDS)
    preparation.add_argument("--workers", type=int, default=8)
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
        if args.workers < 1 or args.num_train < 1 or args.num_val < 1:
            parser.error("Worker, training and validation counts must be positive")
        prepare(args)
    else:
        args.checkpoint_root = args.checkpoint_root.resolve()
        train(args)


if __name__ == "__main__":
    main()
