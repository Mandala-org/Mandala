"""Train any configured operator pipeline through the single MANDALA E3GNN.

Run from the repository root with PYTHONPATH=src:. and an explicit data manifest
or data.root. No architecture selection, imported upstream runtime, or cluster
submission is hidden in this command.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
from pathlib import Path
import platform

import numpy as np
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from core.block_irrep_mapper import BlockIrrepMapper
from core.orbital_irrep_config import OrbitalIrrepConfig
from data.operator_dataset import (
    OperatorDataset,
    read_native_metadata,
    load_operator_sample,
    collate_operator_samples,
    statistics_records,
)
from net.common import Config
from net.configuration import resolve_configuration
from net.e3gnn import E3GNN
from net.restart import EpochBoundaryCheckpoint, EpochShuffleSampler
from net.training_policies import SpeciesIrrepAffine, masked_error
from net.time_budget import WallClockBudgetCallback


def validate_training_options(document):
    training = document.get("training", {})
    allowed = {
        "seed",
        "data_split_seed",
        "dtype",
        "batch_size",
        "max_epochs",
        "min_lr",
        "grad_clip_val",
        "accumulate_grad_batches",
        "split",
    }
    unknown = set(training) - allowed
    if unknown:
        raise ValueError(f"Unknown training options: {sorted(unknown)}")
    for key in (
        "seed",
        "data_split_seed",
        "batch_size",
        "max_epochs",
        "accumulate_grad_batches",
    ):
        if key in training:
            value = training[key]
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < (0 if "seed" in key else 1)
            ):
                raise ValueError(
                    f"training.{key} must be an integer in its valid positive range"
                )
    for key in ("min_lr", "grad_clip_val"):
        if training.get(key) is not None:
            value = training[key]
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not np.isfinite(value)
                or value <= 0
            ):
                raise ValueError(f"training.{key} must be finite and positive")
    settings = training.get("split", {})
    allowed_split = {
        f"{name}_{kind}"
        for name in ("train", "val", "test")
        for kind in ("size", "ratio")
    }
    if set(settings) - allowed_split:
        raise ValueError(
            f"Unknown split options: {sorted(set(settings) - allowed_split)}"
        )
    for key, value in settings.items():
        if key.endswith("_size"):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{key} must be a nonnegative integer")
        elif (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not np.isfinite(value)
            or not 0 <= value <= 1
        ):
            raise ValueError(f"{key} must be a finite ratio in [0,1]")


def dataset_paths(document):
    validate_training_options(document)
    data, training = document["data"], document.get("training", {})
    graph_source = data.get("graph_source", "stored")
    if graph_source not in {"stored", "radius"}:
        raise ValueError("graph_source must be stored or radius")
    if graph_source == "radius":
        radius = data.get("graph_radius")
        if (
            isinstance(radius, bool)
            or not isinstance(radius, (int, float))
            or not np.isfinite(radius)
            or radius <= 0
        ):
            raise ValueError("Radius graph requires a finite positive graph_radius")
    if "splits" in data:
        splits = {
            key: [Path(p) for p in data["splits"].get(key, [])]
            for key in ("train", "val", "test")
        }
    else:
        root = Path(data["root"])
        paths = sorted(
            {
                p.parent
                for p in root.rglob("info.json")
                if graph_source == "radius"
                or (p.parent / data.get("support_file", "hamiltonians.h5")).is_file()
            }
        )
        if not paths:
            raise ValueError(f"No native operator snapshots under {root}")
        order = np.random.RandomState(
            training.get("data_split_seed", training.get("seed", 42))
        ).permutation(len(paths))
        settings = training.get("split", {})
        ntrain = settings.get(
            "train_size", int(settings.get("train_ratio", 0.67) * len(paths))
        )
        nval = settings.get(
            "val_size", int(settings.get("val_ratio", 0.33) * len(paths))
        )
        ntest = settings.get(
            "test_size", int(settings.get("test_ratio", 0.0) * len(paths))
        )
        if min(ntrain, nval, ntest) < 0 or ntrain + nval + ntest > len(paths):
            raise ValueError("Requested split sizes exceed available snapshots")
        splits = {
            "train": [paths[i] for i in order[:ntrain]],
            "val": [paths[i] for i in order[ntrain : ntrain + nval]],
            "test": [paths[i] for i in order[ntrain + nval : ntrain + nval + ntest]],
        }
    flat = [p.resolve() for values in splits.values() for p in values]
    if len(flat) != len(set(flat)):
        raise ValueError("Dataset splits overlap or contain duplicate snapshots")
    if not splits["train"] or not splits["val"]:
        raise ValueError("Both training and validation splits must be nonempty")
    return splits


def input_manifest(splits):
    result = {}
    for split, paths in splits.items():
        result[split] = []
        for path in paths:
            hashes = {}
            for source in sorted(path.iterdir()):
                if (
                    source.suffix not in {".dat", ".json", ".h5"}
                    or not source.is_file()
                ):
                    continue
                digest = hashlib.sha256()
                with source.open("rb") as handle:
                    for chunk in iter(lambda: handle.read(1024**2), b""):
                        digest.update(chunk)
                hashes[source.name] = digest.hexdigest()
            result[split].append({"path": str(path.resolve()), "sha256": hashes})
    return result


def build_components(document, *, resume=False):
    splits = dataset_paths(document)
    train_meta = [read_native_metadata(path) for path in splits["train"]]
    shells = {}
    for meta in train_meta:
        for species, ls in meta["species_shells"].items():
            if species in shells and shells[species] != ls:
                raise ValueError(f"Training structures disagree on {species} orbitals")
            shells[species] = ls
    spinful = train_meta[0]["spinful"]
    if any(meta["spinful"] != spinful for meta in train_meta):
        raise ValueError("A training run must use a single spin layout")
    resolved, codec = resolve_configuration(
        document, species_shells=shells, spinful=spinful
    )
    data = resolved["data"]
    species = list(resolved["species"]["order"])
    dtype = torch.get_default_dtype()
    samples = {
        split: [
            load_operator_sample(
                path,
                codec,
                species=species,
                split=split,
                matrix_files=data.get("matrix_files"),
                support_file=data.get("support_file", "hamiltonians.h5"),
                local_coordinates=data.get("local_coordinates", False),
                num_l=data.get("num_l", 5),
                frame_source=data.get("frame_source", "derive"),
                target_frame=data.get("target_frame", "global"),
                graph_source=data.get("graph_source", "stored"),
                graph_radius=data.get("graph_radius"),
                dtype=dtype,
            )
            for path in paths
        ]
        for split, paths in splits.items()
    }
    orbital_spec = {
        atom: [f"1x{l}{'e' if l % 2 == 0 else 'o'}" for l in ls]
        for atom, ls in shells.items()
    }
    mapper = BlockIrrepMapper(OrbitalIrrepConfig.from_dict(orbital_spec), dtype=dtype)
    optimization = dict(resolved.get("optimization") or {})
    if resolved["training"].get("min_lr") is not None:
        optimization["stop_lr"] = resolved["training"]["min_lr"]
    cfg = Config(
        pipeline=resolved["pipeline"],
        objective=resolved.get("objective"),
        optimization=optimization,
        verbosity=0,
    )
    # Dataset preparation and mapper construction must not shift the declared
    # network initialization stream.
    pl.seed_everything(int(resolved["training"].get("seed", 42)), workers=True)
    model = E3GNN(mapper, cfg)
    if not resume:
        for transform in model.pipeline.transforms.values():
            if isinstance(transform, SpeciesIrrepAffine):
                transform.fit(
                    statistics_records(
                        samples["train"], codec, target=transform.target
                    ),
                    training_ids=[sample["id"] for sample in samples["train"]],
                )
    return (
        resolved,
        codec,
        model,
        {key: OperatorDataset(value) for key, value in samples.items()},
        input_manifest(splits),
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--accelerator", default="auto")
    parser.add_argument("--max-wall-clock-seconds", type=float)
    parser.add_argument("--log-dir", type=Path)
    args = parser.parse_args(argv)
    document = OmegaConf.to_container(
        OmegaConf.merge(OmegaConf.load(args.config), OmegaConf.from_dotlist(args.set)),
        resolve=False,
    )
    dtype_name = document.get("training", {}).get("dtype", "float32")
    if dtype_name not in {"float32", "float64"}:
        raise ValueError("Supported training dtypes are float32 and float64")
    torch.set_default_dtype(getattr(torch, dtype_name))
    resolved, codec, model, datasets, manifest = build_components(
        document, resume=args.resume is not None
    )
    if args.dry_run:
        if args.resume:
            checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
            model.load_state_dict(checkpoint["state_dict"])
        batch = collate_operator_samples([datasets["train"][0]])
        predictions = model(batch[0])
        objective = resolved.get("objective", {})
        loss = sum(
            objective.get("weights", {}).get(name, 1.0)
            * masked_error(
                value,
                batch[1][name]["values"],
                batch[1][name]["mask"],
                kind=objective.get("kind", "mse"),
                complex_reduction=objective.get("complex_reduction", "magnitude"),
            )
            for name, value in predictions.items()
        )
        loss.backward()
        print(
            json.dumps(
                {
                    "parameters": sum(p.numel() for p in model.parameters()),
                    "loss": float(loss.detach()),
                    "split_sizes": {k: len(v) for k, v in datasets.items()},
                },
                indent=2,
            )
        )
        return
    args.output.mkdir(parents=True, exist_ok=True)
    training_ids = [sample["id"] for sample in datasets["train"].samples]
    statistics_manifest = {
        name: {
            "target": transform.target,
            "training_ids": training_ids,
            "training_manifest_hash": hashlib.sha256(
                json.dumps(tuple(training_ids)).encode()
            ).hexdigest(),
        }
        for name, transform in model.pipeline.transforms.items()
        if isinstance(transform, SpeciesIrrepAffine)
    }
    specification = {
        "configuration": resolved,
        "inputs": manifest,
        "statistics": statistics_manifest,
        "runtime": {"python": platform.python_version(), "torch": torch.__version__},
    }
    record = args.output / "specification.json"
    if record.exists():
        if args.resume is None or json.loads(record.read_text()) != specification:
            raise ValueError(
                "Output run already exists or its immutable specification changed"
            )
    elif args.resume:
        raise ValueError("Resume requires the original run specification")
    else:
        record.write_text(json.dumps(specification, indent=2) + "\n")
    training = resolved["training"]
    batch_size = int(training.get("batch_size", 1))
    train_loader = DataLoader(
        datasets["train"],
        batch_size=batch_size,
        sampler=EpochShuffleSampler(datasets["train"], training.get("seed", 42)),
        collate_fn=collate_operator_samples,
    )
    val_loader = DataLoader(
        datasets["val"], batch_size=batch_size, collate_fn=collate_operator_samples
    )
    callback = EpochBoundaryCheckpoint(args.output, monitor="val/loss")
    callbacks = [callback]
    if args.max_wall_clock_seconds is not None:
        callbacks.append(
            WallClockBudgetCallback(budget_seconds=args.max_wall_clock_seconds)
        )
    trainer = pl.Trainer(
        max_epochs=int(training.get("max_epochs", 3000)),
        accelerator=args.accelerator,
        devices=1,
        precision="64-true" if dtype_name == "float64" else "32-true",
        gradient_clip_val=training.get("grad_clip_val"),
        accumulate_grad_batches=training.get("accumulate_grad_batches", 1),
        deterministic=True,
        callbacks=callbacks,
        logger=CSVLogger(str(args.log_dir), name="") if args.log_dir else False,
        enable_checkpointing=False,
        num_sanity_val_steps=0,
        default_root_dir=args.output,
    )
    trainer.fit(
        model,
        train_loader,
        val_loader,
        ckpt_path=str(args.resume) if args.resume else None,
        **(
            {"weights_only": False}
            if "weights_only" in inspect.signature(trainer.fit).parameters
            else {}
        ),
    )


if __name__ == "__main__":
    main()
