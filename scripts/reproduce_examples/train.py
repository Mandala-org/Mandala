"""Frozen publication recipes: start from scratch, then recover the same run.

This wrapper never loads the historical selected models. Their configurations
are the provenance of the JSON recipes, not runtime dependencies.
"""

from __future__ import annotations

import argparse
import dataclasses
import fcntl
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_recipe(path):
    recipe = json.loads(path.read_text())
    config = recipe["config"]
    if config.get("compatibility") or recipe["training"].get("resume_from_checkpoint"):
        raise ValueError("Publication recipes must initialize from scratch.")
    if config["hamiltonian_envelope_mode"] != "off":
        envelope = path.parent / recipe["envelope_file"]
        if digest(envelope) != recipe["envelope_sha256"]:
            raise ValueError("Frozen radial envelope checksum mismatch.")
        config["hamiltonian_envelope_path"] = str(envelope.resolve())
    return recipe


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", type=Path, required=True)
    parser.add_argument(
        "--action", choices=["auto", "start", "continue"], default="auto"
    )
    parser.add_argument(
        "--plan",
        action="store_true",
        help="Print the recipe without training or creating run files.",
    )
    args = parser.parse_args(argv)
    path = args.recipe.resolve()
    recipe = load_recipe(path)
    if args.plan:
        print(json.dumps(recipe, indent=2))
        return

    import torch
    from net.common import Config
    from net.evaluation import load_checkpoint
    from scripts.train import run_training

    values = recipe["config"] | recipe["training"]
    unknown = set(recipe["config"]) - {
        field.name for field in dataclasses.fields(Config)
    }
    if unknown:
        raise ValueError(f"Unknown configuration fields: {sorted(unknown)}")
    if not torch.cuda.is_available():
        raise RuntimeError("These approved recipes require a CUDA GPU.")
    checkpoint_root = Path(values["checkpoint_dir"])
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    run_dir = checkpoint_root / values["run_name"]
    latest = run_dir / "latest_checkpoint.pt"
    # One writer per model, including across separate scheduler jobs.
    with (checkpoint_root / f".{values['run_name']}.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        continuing = latest.is_file()
        if args.action == "start" and run_dir.exists():
            raise FileExistsError("Start refuses to overwrite an existing run.")
        if args.action == "continue" and not continuing:
            raise FileNotFoundError(f"No recovery checkpoint: {latest}")
        if run_dir.exists() and not continuing:
            raise RuntimeError(
                "Incomplete run directory without a recovery checkpoint; inspect it explicitly."
            )

        code = (
            sorted((ROOT / "src").rglob("*.py"))
            + sorted((ROOT / "scripts").rglob("*.py"))
            + sorted(path.parent.glob("*.sh"))
        )
        contract = {
            "recipe_sha256": digest(path),
            "source_sha256": {str(p.relative_to(ROOT)): digest(p) for p in code},
            "python": platform.python_version(),
            "packages": {
                dist.metadata["Name"]: dist.version
                for dist in importlib.metadata.distributions()
            },
            "cuda_runtime": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "gpu": torch.cuda.get_device_name(),
            "gpu_capability": list(torch.cuda.get_device_capability()),
            "torch_num_threads": torch.get_num_threads(),
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            "pythonhashseed": os.environ.get("PYTHONHASHSEED"),
        }
        contract_path = checkpoint_root / f"{values['run_name']}.contract.json"
        if contract_path.exists():
            if json.loads(contract_path.read_text()) != contract:
                raise RuntimeError(
                    "Recipe, source, or runtime changed; refusing an untracked continuation."
                )
        elif continuing:
            raise FileNotFoundError(
                "Recovery checkpoint has no original reproducibility contract."
            )
        else:
            contract_path.write_text(
                json.dumps(contract, indent=2, sort_keys=True) + "\n"
            )
        if continuing:
            checkpoint = load_checkpoint(latest)
            if not checkpoint.get("epoch_boundary_resume", {}).get("safe_to_resume"):
                raise ValueError("Latest checkpoint is not a safe epoch boundary.")
            if int(checkpoint["epoch"]) >= int(values["max_epochs"]):
                print("The frozen total epoch limit has already been reached.")
                return
            del checkpoint
        values["resume_from_checkpoint"] = str(latest) if continuing else None
        values["resume_mode"] = "latest"
        values["_explicit_args"] = set(values)
        # Offline W&B records per-segment logs; checkpoints/manifests are the
        # authoritative continuation state and do not require network access.
        run_training(values)


if __name__ == "__main__":
    main()
