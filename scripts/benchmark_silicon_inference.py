#!/usr/bin/env python
"""Benchmark full-dataset GPU inference and energy evaluation.

The benchmark deliberately keeps every prepared sample on the selected device.
It measures preparation, device transfer, warmup, matrix inference, and the
end-to-end energy path separately.  Results are written both to stdout and to
the requested text file.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
import time
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from data.factory import DatasetFactory  # noqa: E402
from net.e3gnn import E3GNN  # noqa: E402


DEFAULT_CHECKPOINT = (
    REPO_ROOT
    / "checkpoints/silicon_perturbed_hamiltonian/sleek-sweep-8-restart-lr2em3/best_model.pt"
)
DEFAULT_DATA_ROOT = Path(
    "/bigdata/casus/wdm/hamiltonian_learning/data/perturbed_snapshots_Si"
)


class TeeLogger:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = path.open("w", encoding="utf-8")

    def log(self, message: str = "") -> None:
        print(message, flush=True)
        self.handle.write(message + "\n")
        self.handle.flush()

    def close(self) -> None:
        self.handle.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark Silicon perturbed inference with the full dataset on GPU."
    )
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument(
        "--scales",
        type=str,
        default="2",
        help="Comma-separated scale directories; default '2' has 100 snapshots.",
    )
    parser.add_argument("--num-snapshots", type=int, default=100)
    parser.add_argument("--num-passes", type=int, default=10)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=REPO_ROOT / "eval_outputs/silicon_benchmark_snapshot_cache",
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        default=REPO_ROOT / "eval_outputs/silicon_inference_benchmark.txt",
    )
    return parser.parse_args()


def load_eval_module():
    path = REPO_ROOT / "scripts/evaluate_checkpoint_materials.py"
    spec = importlib.util.spec_from_file_location("benchmark_eval_module", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load evaluation helpers from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def discover_pairs(root: Path, scales: str, limit: int) -> list[tuple[Path, Path]]:
    scale_names = [item.strip() for item in scales.split(",") if item.strip()]
    if not scale_names:
        raise ValueError("--scales must contain at least one scale")

    pairs: list[tuple[Path, Path]] = []
    for scale in scale_names:
        scale_dir = root / f"scale_{scale}"
        if not scale_dir.is_dir():
            raise FileNotFoundError(f"Scale directory not found: {scale_dir}")
        for snapshot_dir in sorted(
            path for path in scale_dir.iterdir() if path.is_dir()
        ):
            matrix = snapshot_dir / "HS.out"
            info = snapshot_dir / "Si.out"
            if matrix.exists() and info.exists():
                pairs.append((matrix, info))
    if len(pairs) < limit:
        raise ValueError(
            f"Found only {len(pairs)} usable Silicon snapshot pairs, but "
            f"--num-snapshots={limit} was requested."
        )
    return pairs[:limit]


def move_to_device(value: Any, device: torch.device) -> Any:
    if isinstance(value, dict):
        return {key: move_to_device(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [move_to_device(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(move_to_device(item, device) for item in value)
    if hasattr(value, "to"):
        try:
            return value.to(device)
        except TypeError:
            return value.to(device=device)
    return value


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def elapsed(start: float, end: float, count: int) -> tuple[float, float]:
    total = end - start
    return total, total / count


def main() -> None:
    args = parse_args()
    logger = TeeLogger(args.log_file)
    try:
        if args.num_snapshots <= 0 or args.num_passes <= 0:
            raise ValueError("--num-snapshots and --num-passes must be positive")
        if args.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested, but torch.cuda.is_available() is false"
            )
        device = torch.device(args.device)
        eval_mod = load_eval_module()

        logger.log("=== Mandala Silicon inference benchmark ===")
        logger.log(f"checkpoint: {args.checkpoint.expanduser().resolve()}")
        logger.log(f"dataset_root: {args.dataset_root.resolve()}")
        logger.log(f"scales: {args.scales}; snapshots: {args.num_snapshots}")
        logger.log(f"passes: {args.num_passes}; device: {device}")
        if device.type == "cuda":
            logger.log(f"GPU: {torch.cuda.get_device_name(device)}")

        if not args.checkpoint.exists():
            raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

        t0 = time.perf_counter()
        pairs = discover_pairs(args.dataset_root, args.scales, args.num_snapshots)
        checkpoint = eval_mod._load_checkpoint(args.checkpoint)
        cfg = eval_mod._restore_config(checkpoint)
        cfg.dataset_device = None
        cfg.snapshot_cache_dir = str(args.cache_dir)
        cfg.spectral_loss_enabled = False
        cfg.spectral_fermi_cache_path = None
        factory = DatasetFactory(cfg, convention="e3nn")
        for matrix, info in pairs:
            factory.add_snapshot(matrix, info, purpose="train")
        dataset, _, mapper = factory.create()
        t1 = time.perf_counter()
        logger.log(
            f"data preparation: {t1 - t0:.3f} s "
            f"({len(dataset)} samples, {len(dataset) / (t1 - t0):.3f} samples/s)"
        )

        t2 = time.perf_counter()
        samples = [
            (
                move_to_device(dataset[index][0], device),
                move_to_device(dataset[index][1], device),
            )
            for index in tqdm(range(len(dataset)), desc="Moving samples to device")
        ]
        synchronize(device)
        t3 = time.perf_counter()
        logger.log(
            f"device transfer: {t3 - t2:.3f} s "
            f"({len(samples) / (t3 - t2):.3f} samples/s)"
        )

        model = E3GNN(mapper=mapper, cfg=cfg)
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        model.to(device)
        model.eval()
        logger.log("model loaded and resident on device")

        warmup_count = min(5, len(samples))
        logger.log(f"warmup: {warmup_count} forward passes")
        synchronize(device)
        warmup_start = time.perf_counter()
        with torch.no_grad():
            for index in range(warmup_count):
                model(samples[index][0])
        synchronize(device)
        warmup_end = time.perf_counter()
        logger.log(
            f"warmup time: {warmup_end - warmup_start:.3f} s "
            f"({(warmup_end - warmup_start) / warmup_count:.3f} s/pass)"
        )

        total_forwards = args.num_passes * len(samples)
        synchronize(device)
        forward_start = time.perf_counter()
        with torch.no_grad():
            for pass_index in range(args.num_passes):
                for x, _ in tqdm(
                    samples,
                    desc=f"Forward pass {pass_index + 1}/{args.num_passes}",
                    leave=False,
                ):
                    model(x)
        synchronize(device)
        forward_end = time.perf_counter()
        forward_total, forward_each = elapsed(
            forward_start, forward_end, total_forwards
        )
        logger.log(
            f"model forward total: {forward_total:.3f} s for {total_forwards} passes"
        )
        logger.log(
            f"model forward per snapshot: {forward_each:.6f} s "
            f"({1.0 / forward_each:.3f} samples/s)"
        )

        if "density" not in cfg.matrix_targets:
            raise RuntimeError(
                "Energy benchmark requires a checkpoint with density in matrix_targets; "
                f"this checkpoint has {cfg.matrix_targets!r}."
            )
        logger.log(
            "benchmarking end-to-end energy evaluation: forward + block conversion + Tr(DH)"
        )
        synchronize(device)
        energy_start = time.perf_counter()
        energy_values = []
        with torch.no_grad():
            for pass_index in range(args.num_passes):
                for x, _ in tqdm(
                    samples,
                    desc=f"Energy pass {pass_index + 1}/{args.num_passes}",
                    leave=False,
                ):
                    predictions = model(x)
                    snapshot = model.predictions_to_snapshot(
                        predictions,
                        x["positions"],
                        x["box"],
                        x=x,
                    )
                    energy_values.append(snapshot.get_energy())
        synchronize(device)
        energy_end = time.perf_counter()
        energy_total, energy_each = elapsed(energy_start, energy_end, total_forwards)
        logger.log(
            f"energy evaluation total: {energy_total:.3f} s for {total_forwards} passes"
        )
        logger.log(
            f"energy evaluation per snapshot: {energy_each:.6f} s "
            f"({1.0 / energy_each:.3f} samples/s)"
        )
        logger.log(
            f"energy overhead over forward: {energy_each - forward_each:.6f} s/snapshot"
        )
        logger.log(f"energy outputs computed: {len(energy_values)}")
        logger.log(f"wrote log: {args.log_file.resolve()}")
    finally:
        logger.close()


if __name__ == "__main__":
    main()
