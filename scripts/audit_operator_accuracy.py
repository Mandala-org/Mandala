"""Certify frozen train/validation operator inputs and common physical metrics.

Execute on an allocated compute node. Test membership must be empty. This CLI
uses the shared DatasetFactory/E3GNN path and writes bounded aggregate JSON.
Optional checkpoint evaluation snapshots a live pilot checkpoint into bigdata;
it never modifies the pilot and never selects from test data.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import importlib
import json
import os
from pathlib import Path
import shutil

import numpy as np
from omegaconf import OmegaConf
import torch


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024**2), b""):
            digest.update(chunk)
    return digest.hexdigest()


def frozen_module_origins():
    """Certify what Python imported before opening scientific input files."""
    source_root = Path(__file__).resolve().parents[1] / "src"
    origins = {}
    for name in ("data.sample", "data.factory", "data.snapshot", "net.e3gnn"):
        module = importlib.import_module(name)
        actual = getattr(module, "__file__", None)
        expected = source_root.joinpath(*name.split(".")).with_suffix(".py").resolve()
        if actual is None or Path(actual).resolve() != expected:
            raise ValueError(
                f"Imported module {name} is outside frozen source: {actual}; expected {expected}"
            )
        origins[name] = str(expected)
    return origins


def aligned_max_error(left, right, *, scale=1.0):
    """Compare complete support by physical edge identity, never storage order."""
    if (
        left.basis != right.basis
        or left.atoms != right.atoms
        or left.orbital_cfg.to_dict() != right.orbital_cfg.to_dict()
        or set(left.lookup) != set(right.lookup)
    ):
        raise ValueError("Basis or complete directed support changed")
    maximum = 0.0
    for key, values in left.pair_blocks.items():
        indices = [
            right.lookup[tuple(edge)][1] for edge in left.pair_edges[key].T.tolist()
        ]
        actual = right.pair_blocks[key][indices]
        maximum = max(maximum, float((values - actual).abs().max()) * scale)
    return maximum


def error_record(prediction, target, *, edge_selector=None):
    """Common global physical errors, plus directed species-pair denominators."""
    from net.operator_metrics import matrix_error_totals
    from utils.units import HARTREE_TO_EV

    totals = matrix_error_totals(prediction, target, edge_selector=edge_selector)
    row = {
        "absolute_sum_ev": float(totals["absolute_sum"]) * HARTREE_TO_EV,
        "squared_sum_ev2": float(totals["squared_sum"]) * HARTREE_TO_EV**2,
        "entries": totals["count"],
        "pairs": {},
    }
    for key, values in target.pair_blocks.items():
        n = len(values)
        difference = (prediction.pair_blocks[key][:n] - values).abs().double()
        mask = getattr(target, "coverage_masks", {}).get(key)
        if mask is None:
            mask = torch.ones_like(values, dtype=torch.bool)
        else:
            mask = mask.clone()
        if edge_selector is not None:
            selected = edge_selector(target.pair_edges[key])
            if selected is not None:
                mask &= selected.reshape((-1,) + (1,) * (values.ndim - 1))
        observed = difference[mask]
        if observed.numel():
            row["pairs"][key] = {
                "absolute_sum_ev": float(observed.sum()) * HARTREE_TO_EV,
                "squared_sum_ev2": float(observed.square().sum()) * HARTREE_TO_EV**2,
                "entries": observed.numel(),
            }
    return row


def aggregate_errors(rows):
    def reduce(values):
        count = sum(row["entries"] for row in values)
        mse = sum(row["squared_sum_ev2"] for row in values) / count
        return {
            "structures": len(values),
            "entries": count,
            "structure_mean_mae_mev": float(
                np.mean([row["absolute_sum_ev"] / row["entries"] for row in values])
            )
            * 1000,
            "dataset_entry_mae_mev": sum(row["absolute_sum_ev"] for row in values)
            / count
            * 1000,
            "structure_mean_rmse_mev": float(
                np.mean(
                    [(row["squared_sum_ev2"] / row["entries"]) ** 0.5 for row in values]
                )
            )
            * 1000,
            "dataset_entry_rmse_mev": mse**0.5 * 1000,
        }

    result = reduce(rows)
    pairs = defaultdict(list)
    for row in rows:
        for key, values in row["pairs"].items():
            pairs[key].append(values)
    result["directed_species_pairs"] = {
        key: reduce(values) for key, values in pairs.items()
    }
    return result


def snapshot_checkpoint(source, destination):
    """Exclusively copy one stable checkpoint generation without overwriting it."""
    source, destination = Path(source), Path(destination)
    if source.resolve() == destination.resolve():
        raise ValueError("Checkpoint copy must differ from the source")
    destination.parent.mkdir(parents=True, exist_ok=True)

    def identity(stat):
        return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns

    created = False
    try:
        with source.open("rb") as original:
            before = identity(os.fstat(original.fileno()))
            with destination.open("xb") as copied:
                created = True
                shutil.copyfileobj(original, copied)
                copied.flush()
                os.fsync(copied.fileno())
            after = identity(os.fstat(original.fileno()))
        if before != after or after != identity(source.stat()):
            raise RuntimeError(
                "Checkpoint changed during snapshot; retry a stable generation"
            )
    except BaseException:
        if created:
            destination.unlink(missing_ok=True)
        raise


def predictor_signature(configuration):
    """Normalize old defaults and ignore policies that cannot affect inference."""
    from net.common import Config

    ignored = {
        "objective",
        "optimization",
        "training",
        "save_dir",
        "run_name",
        "device",
        "gpus",
        "num_workers",
        "dataset_device",
        "snapshot_cache_dir",
        "verbosity",
        "bench_verbosity",
        "max_wall_clock_seconds",
        "matrix_loss_reduction",
    }
    defaults = Config().to_mapping()

    def clean_config(config):
        merged = defaults | dict(config)
        result = {}
        for key, value in merged.items():
            if key in ignored or key.startswith("log_") or key.startswith("print_"):
                continue
            if key == "data":
                value = {k: v for k, v in value.items() if k not in {"root", "splits"}}
            result[key] = clean(value)
        return result

    def clean(value):
        if isinstance(value, dict):
            return {
                key: clean_config(item) if key == "config" else clean(item)
                for key, item in value.items()
            }
        if isinstance(value, (list, tuple)):
            return [clean(item) for item in value]
        if isinstance(value, torch.dtype):
            return str(value).removeprefix("torch.")
        return value.removeprefix("torch.") if isinstance(value, str) else value

    return clean_config(configuration)


def audit(
    config_path,
    expected_manifest,
    *,
    checkpoint=None,
    checkpoint_copy=None,
    accelerator="cpu",
):
    module_origins = frozen_module_origins()
    from data.sample import move_tree
    from data.snapshot import Snapshot
    from data.deeph_e3_parser import load_deeph_e3_block_matrix, load_deeph_e3_metadata
    from scripts.dataset import dataset_paths, input_manifest
    from scripts.train import build_components
    from utils.units import HARTREE_TO_EV

    document = OmegaConf.load(config_path)
    splits = dataset_paths(document)
    if splits["test"]:
        raise ValueError("Accuracy investigation refuses test inputs")
    if document["data"].get("representation") != "orbital_blocks":
        raise ValueError("This initial certification requires orbital_blocks")
    expected_document = json.loads(Path(expected_manifest).read_text())
    expected = expected_document["specification"]["inputs"]
    actual_inputs = input_manifest(splits)
    for split in ("train", "val"):
        if actual_inputs[split] != expected[split]:
            raise ValueError(
                f"{split} file identities/content hashes differ from frozen pilot"
            )
    dtype = getattr(
        torch,
        document.get("training", {}).get("dtype", document.get("dtype", "float32")),
    )
    torch.set_default_dtype(dtype)
    _, _, model, datasets, loaded_inputs = build_components(document, resume=True)
    for split in ("train", "val"):
        if loaded_inputs[split] != actual_inputs[split]:
            raise ValueError(f"{split} inputs changed during dataset construction")
    if predictor_signature(model.cfg.to_mapping()) != predictor_signature(
        expected_document["config"]
    ):
        raise ValueError("Prediction configuration differs from frozen pilot")
    result = {
        "config_sha256": file_hash(config_path),
        "pilot_run_manifest_sha256": file_hash(expected_manifest),
        "test_inputs": 0,
        "input_hashes_match_pilot": True,
        "predictor_configuration_matches_pilot": True,
        "module_origins": module_origins,
        "parameters": sum(p.numel() for p in model.parameters()),
        "splits": {},
        "checks": {},
    }
    maxima = defaultdict(float)
    for split in ("train", "val"):
        dataset = datasets[split]
        zero_rows, hermitian_rows = [], []
        counts = defaultdict(int)
        radius_max = 0.0
        for (matrix_path, _), sample in zip(dataset.snapshot_paths, dataset):
            x, y = sample
            target = y["hamiltonian"]
            if target.basis != "e3nn":
                raise ValueError("Target basis must be global e3nn")
            source_options = {
                key: document["data"].get(key)
                for key in ("source_basis", "source_cartesian_axes", "matrix_files")
            }
            snapshot = Snapshot.from_deeph_e3(
                matrix_path.parent, convention="openmx", dtype=dtype, **source_options
            )
            converted = snapshot.to_e3nn()
            if converted.to_e3nn() is not converted:
                raise ValueError("Repeated basis conversion is not idempotent")
            maxima["roundtrip_ev"] = max(
                maxima["roundtrip_ev"],
                aligned_max_error(
                    snapshot.hamiltonian,
                    converted.to_openmx().hamiltonian,
                    scale=HARTREE_TO_EV,
                ),
            )
            maxima["pipeline_target_change_ev"] = max(
                maxima["pipeline_target_change_ev"],
                aligned_max_error(converted.hamiltonian, target, scale=HARTREE_TO_EV),
            )
            metadata = load_deeph_e3_metadata(
                matrix_path.parent, dtype=dtype, **source_options
            )
            original_ev = load_deeph_e3_block_matrix(
                matrix_path,
                metadata,
                matrix_name="hamiltonian",
                dtype=dtype,
                energy_unit="eV",
            )
            if original_ev.basis != "openmx":
                from core.basis_converter import OpenMXE3NNConverter

                original_ev = OpenMXE3NNConverter(
                    original_ev.orbital_cfg
                ).matrix_to_openmx(original_ev)
            maxima["energy_roundtrip_ev"] = max(
                maxima["energy_roundtrip_ev"],
                aligned_max_error(original_ev, snapshot.hamiltonian * HARTREE_TO_EV),
            )
            maxima["cartesian_once_angstrom"] = max(
                maxima["cartesian_once_angstrom"],
                float(
                    (converted.positions - snapshot.positions[:, [1, 2, 0]]).abs().max()
                ),
            )
            maxima["pipeline_positions_angstrom"] = max(
                maxima["pipeline_positions_angstrom"],
                float((x["positions"] - converted.positions).abs().max()),
            )
            maxima["raw_target_hermiticity_max_ev"] = max(
                maxima["raw_target_hermiticity_max_ev"],
                aligned_max_error(
                    snapshot.hamiltonian,
                    snapshot.hamiltonian.adjoint(),
                    scale=HARTREE_TO_EV,
                ),
            )
            hermitian_rows.append(
                error_record(snapshot.hamiltonian.adjoint(), snapshot.hamiltonian)
            )
            zero_rows.append(error_record(target * 0.0, target))
            for key, values in target.pair_blocks.items():
                counts[key] += values.numel()
            radius_max = max(radius_max, float(x["edge_length"].max()))
        result["splits"][split] = {
            "structures": len(dataset),
            "entries_by_pair": dict(counts),
            "max_stored_edge_length_angstrom": radius_max,
            "zero_predictor": aggregate_errors(zero_rows),
            "raw_target_hermiticity": aggregate_errors(hermitian_rows),
        }
    result["checks"] = dict(maxima)
    thresholds = {name: 5e-5 for name in maxima}
    thresholds["pipeline_target_change_ev"] = 5e-6
    result["check_thresholds"] = thresholds
    result["passed"] = all(value <= thresholds[name] for name, value in maxima.items())
    if checkpoint is not None:
        if checkpoint_copy is None:
            raise ValueError("Checkpoint snapshot destination is required")
        checkpoint_copy = Path(checkpoint_copy)
        snapshot_checkpoint(checkpoint, checkpoint_copy)
        state = torch.load(checkpoint_copy, map_location="cpu", weights_only=False)
        model.load_state_dict(state["state_dict"], strict=True)
        model.to("cuda" if accelerator == "gpu" else "cpu").eval()
        prediction_rows = defaultdict(list)
        for sample in datasets["val"]:
            x, y = move_tree(sample, model.device, dtype)
            raw = model.predict_matrices(x)["hamiltonian"]
            target = y["hamiltonian"]
            projected = 0.5 * (raw + raw.adjoint())
            prediction_rows["raw"].append(error_record(raw, target))
            prediction_rows["projected"].append(error_record(projected, target))
            prediction_rows["raw_hermiticity"].append(error_record(raw.adjoint(), raw))

            def onsite(edges):
                return (edges[3] == edges[4]) & (edges[:3] == 0).all(dim=0)

            for label, prediction in (("raw", raw), ("projected", projected)):
                prediction_rows[label + "_onsite"].append(
                    error_record(prediction, target, edge_selector=onsite)
                )
                prediction_rows[label + "_offsite"].append(
                    error_record(
                        prediction, target, edge_selector=lambda edges: ~onsite(edges)
                    )
                )
        result["validation_checkpoint"] = {
            "source": str(checkpoint),
            "snapshot": str(checkpoint_copy),
            "sha256": file_hash(checkpoint_copy),
            "epoch": int(state["epoch"]),
            "metrics": {
                key: aggregate_errors(rows) for key, rows in prediction_rows.items()
            },
        }
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--expected-manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--checkpoint-copy", type=Path)
    parser.add_argument("--accelerator", choices=("cpu", "gpu"), default="cpu")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError(args.output)
    result = audit(
        args.config,
        args.expected_manifest,
        checkpoint=args.checkpoint,
        checkpoint_copy=args.checkpoint_copy,
        accelerator=args.accelerator,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "passed": result["passed"],
                "checks": result["checks"],
            },
            indent=2,
        )
    )
    if not result["passed"]:
        raise SystemExit("Data certification failed; inspect before any training batch")


if __name__ == "__main__":
    main()
