"""Run a frozen recipe segment and score only the terminal training state."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from collections import defaultdict
from pathlib import Path

from omegaconf import OmegaConf
import torch

from scripts.audit_operator_accuracy import (
    aggregate_errors,
    error_record,
    frozen_module_origins,
)


def score(config, checkpoint, split, output):
    from core.basis_converter import OpenMXE3NNConverter
    from data.snapshot import Snapshot
    from scripts.train import build_components

    document = OmegaConf.load(config)
    dtype = getattr(
        torch,
        document.get("training", {}).get("dtype", document.get("dtype", "float32")),
    )
    torch.set_default_dtype(dtype)
    _, _, model, datasets, inputs = build_components(document, resume=True)
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(state["state_dict"], strict=True)
    model.to("cuda").eval()
    rows = defaultdict(list)
    support = hashlib.sha256()
    for (path, _), (x, _) in zip(
        datasets[split].snapshot_paths, datasets[split], strict=True
    ):
        with torch.no_grad():
            pred = model.predict_matrices(x)["hamiltonian"]
            if pred.basis not in ("e3nn", "openmx"):
                raise ValueError(f"Unexpected prediction basis {pred.basis}")
            if pred.basis == "e3nn":
                pred = OpenMXE3NNConverter(pred.orbital_cfg).matrix_to_openmx(pred)
            target = Snapshot.from_deeph_e3(
                path.parent,
                convention="openmx",
                dtype=dtype,
                source_basis="openmx",
                source_cartesian_axes="xyz",
            ).hamiltonian.to(model.device)
            if (
                pred.atoms != target.atoms
                or pred.orbital_cfg.to_dict() != target.orbital_cfg.to_dict()
            ):
                raise ValueError("Prediction orbital layout differs from common target")
            order = {}
            for pair, edges in target.pair_edges.items():
                keys = [tuple(e) for e in edges.T.tolist()]
                indices = [pred.lookup[key] for key in keys]
                if any(p != pair for p, _ in indices):
                    raise ValueError("Prediction species identity mismatch")
                order[pair] = torch.tensor(
                    [i for _, i in indices], device=model.device, dtype=torch.long
                )
                pmask = pred.coverage_masks.get(pair)
                tmask = target.coverage_masks.get(pair)
                if pmask is not None:
                    missing = ~pmask[order[pair]]
                    if (missing if tmask is None else missing & tmask).any():
                        raise ValueError(
                            "Prediction does not cover all observed target entries"
                        )
                support.update(
                    json.dumps(
                        [str(path), pair, keys, list(target.pair_blocks[pair].shape)]
                    ).encode()
                )
                if tmask is not None:
                    support.update(tmask.cpu().numpy().tobytes())
            raw = pred.reorder_edges(order)
            # Verify all scored reverse edges exist before projecting, with no
            # implicit completion or zero-fill for missing predictions.
            for sx, sy, sz, i, j in target.lookup:
                if (-sx, -sy, -sz, j, i) not in target.lookup:
                    raise ValueError("Common target support is not reverse-closed")
            projected = (pred + pred.adjoint()) * 0.5
            rows["raw"].append(error_record(raw, target))
            rows["projected"].append(
                error_record(projected.reorder_edges(order), target)
            )
    metrics = {name: aggregate_errors(items) for name, items in rows.items()}
    if not rows or any(
        not math.isfinite(x["structure_mean_mae_mev"]) for x in metrics.values()
    ):
        raise ValueError("Missing or nonfinite scores")
    write(
        output,
        {
            "split": split,
            "checkpoint": str(checkpoint),
            "epoch": int(state["epoch"]),
            "updates": int(state["global_step"]),
            "support_sha256": support.hexdigest(),
            "inputs_sha256": hashlib.sha256(
                json.dumps(inputs[split], sort_keys=True).encode()
            ).hexdigest(),
            "metrics": metrics,
        },
    )


def write(path, value):
    with path.open("x") as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write("\n")


def main():
    from scripts.train import main as train

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--records", type=Path, required=True)
    p.add_argument("--segment", type=int, required=True)
    p.add_argument("--probe", action="store_true")
    args = p.parse_args()
    if not os.getenv("SLURM_JOB_ID"):
        raise RuntimeError("Run through Slurm")
    frozen_module_origins()
    args.records.mkdir(parents=True, exist_ok=True)
    config = OmegaConf.load(args.config)
    planned_epochs = int(
        config.get("training", {}).get("max_epochs", config.get("max_epochs", 3000))
    )
    stop_lr = config.get("training", {}).get("min_lr")
    marker = args.records / "terminal_test.json"
    if marker.exists():
        return
    train_config = args.config
    if args.probe:
        config.data.splits.train = list(config.data.splits.train)[:1]
        config.data.splits.val = list(config.data.splits.val)[:1]
        config.data.splits.test = []
        config.training.max_epochs = 1
        config.max_epochs = 1
        train_config = args.records / "probe.yaml"
        OmegaConf.save(config, train_config)
    else:
        config.data.splits.test = []
        train_config = args.records / f"train_segment_{args.segment}.yaml"
        OmegaConf.save(config, train_config)
    ckpt = args.output / "latest_checkpoint.pt"
    if ckpt.exists() != (args.segment > 1):
        raise RuntimeError("Fresh/continuation checkpoint contract violated")
    cli = [
        "--config",
        str(train_config),
        "--output",
        str(args.output),
        "--accelerator",
        "gpu",
        "--max-wall-clock-seconds",
        str(2700 if args.probe else 162000),
        "--wall-clock-scope",
        "session",
        "--log-dir",
        str(args.records / "metrics"),
    ]
    if args.segment > 1:
        cli.extend(["--resume", str(ckpt)])
    train(cli)
    state = torch.load(ckpt, map_location="cpu", weights_only=False)
    if not state["epoch_boundary_resume"]["safe_to_resume"]:
        raise ValueError("Latest checkpoint is not an epoch boundary")
    lr = max(g["lr"] for opt in state["optimizer_states"] for g in opt["param_groups"])
    epoch = int(state["epoch"])
    terminal = epoch >= planned_epochs or (stop_lr is not None and lr <= stop_lr)
    write(
        args.records / f"segment_{args.segment}.json",
        {
            "epochs": epoch,
            "lr": lr,
            "terminal": terminal,
            "updates": int(state["global_step"]),
            "gpu": torch.cuda.get_device_name(),
        },
    )
    if args.probe:
        if epoch != 1:
            raise ValueError("Probe did not complete exactly one epoch")
        score(train_config, ckpt, "val", args.records / "probe_score.json")
    elif terminal:
        score(args.config, ckpt, "test", marker)


if __name__ == "__main__":
    main()
