"""Evaluate a shared E3GNN checkpoint on a frozen configuration and split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from omegaconf import OmegaConf
import torch


def evaluate(config, checkpoint, *, split, accelerator):
    from scripts.train import build_components
    from data.sample import move_tree
    from net.common import Config

    document = OmegaConf.load(config)
    dtype = Config.from_mapping(document).dtype
    if dtype not in {torch.float32, torch.float64}:
        raise ValueError("Supported evaluation dtypes are float32 and float64")
    torch.set_default_dtype(dtype)
    _, _, model, datasets, _ = build_components(document, resume=True)
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(state["state_dict"], strict=True)
    model.to(device="cuda" if accelerator == "gpu" else "cpu", dtype=dtype).eval()
    records = []
    if not datasets[split]:
        raise ValueError(f"Empty requested evaluation split: {split}")
    with torch.no_grad():
        for sample in datasets[split]:
            _, metrics, *_ = model._graph_step(
                move_tree(sample, model.device, dtype), 0, "val"
            )
            records.append(
                {
                    key.removeprefix("val/"): float(value)
                    for key, value in metrics.items()
                    if "elementwise" in key
                }
            )
    result = {
        "checkpoint": str(checkpoint),
        "checkpoint_epoch": int(state["epoch"]),
        "split": split,
        "dtype": str(dtype).removeprefix("torch."),
        "structures": len(records),
        "reduction": "mean over structures of unweighted observed directed physical matrix-entry errors",
        "hamiltonian_units": {"mae": "eV", "mse": "eV^2", "rmse": "eV"},
        "metrics": {
            key: sum(row[key] for row in records) / len(records) for key in records[0]
        },
        "per_structure": records,
    }
    if not all(torch.isfinite(torch.tensor(v)) for v in result["metrics"].values()):
        raise ValueError("Nonfinite checkpoint evaluation")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "val", "test"), default="val")
    parser.add_argument("--accelerator", choices=("cpu", "gpu"), default="gpu")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = evaluate(
        args.config, args.checkpoint, split=args.split, accelerator=args.accelerator
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {key: value for key, value in result.items() if key != "per_structure"},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
