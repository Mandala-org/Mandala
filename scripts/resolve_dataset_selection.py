#!/usr/bin/env python

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from scripts.dataset import (  # noqa: E402
    discover_scale_snapshot_pairs,
    discover_single_snapshot_pairs,
)


def str_to_list(raw: str | None) -> list[int] | None:
    if raw is None:
        return None
    stripped = raw.strip()
    if not stripped:
        return []
    return [int(part.strip()) for part in stripped.split(",") if part.strip()]


def setup_argparse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Resolve which snapshot pairs a training configuration will select, "
            "without loading matrices or building datasets."
        )
    )
    parser.add_argument(
        "--dataset-kind",
        type=str,
        required=True,
        choices=["siox", "ZnCuSnSeS_small", "ZnCuSnSeS"],
        help="Dataset kind as passed to wandb_run.py/train.py.",
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        required=True,
        help="Dataset root path.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Shuffle seed used by the dataset split.",
    )
    parser.add_argument(
        "--num-train",
        type=int,
        default=None,
        help="Global num_train for single-snapshot datasets.",
    )
    parser.add_argument(
        "--num-val",
        type=int,
        default=None,
        help="Global num_val for single-snapshot datasets.",
    )
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=0.2,
        help="Val fraction when num_train/num_val are omitted for single-snapshot datasets.",
    )
    parser.add_argument(
        "--scales",
        type=str_to_list,
        default=None,
        help="Comma-separated scales for dataset-kind ZnCuSnSeS. Default: 1,2,3",
    )
    parser.add_argument(
        "--num-train-per-scale",
        type=int,
        default=40,
        help="Train count per scale for dataset-kind ZnCuSnSeS.",
    )
    parser.add_argument(
        "--num-val-per-scale",
        type=int,
        default=10,
        help="Val count per scale for dataset-kind ZnCuSnSeS.",
    )
    return parser.parse_args()


def _print_pairs(label: str, pairs: list[tuple[Path, Path]]) -> None:
    print(f"{label}: {len(pairs)}", flush=True)
    for idx, (matrix_path, info_path) in enumerate(pairs):
        print(f"  [{idx}] matrix={matrix_path}", flush=True)
        print(f"      info={info_path}", flush=True)


def _resolve_single_snapshot_split(
    *,
    data_path: Path,
    dataset_label: str,
    seed: int,
    num_train: int | None,
    num_val: int | None,
    val_fraction: float,
) -> tuple[list[tuple[Path, Path]], list[tuple[Path, Path]], list[tuple[Path, Path]]]:
    all_pairs = discover_single_snapshot_pairs(data_path, label=dataset_label)
    rng = random.Random(seed)
    rng.shuffle(all_pairs)
    if not all_pairs:
        raise ValueError(f"No snapshots found under {data_path}")

    if num_train is not None or num_val is not None:
        if num_train is None or num_val is None:
            raise ValueError("Must provide both --num-train and --num-val together.")
    else:
        inferred_num_val = int(len(all_pairs) * val_fraction)
        if len(all_pairs) > 1:
            inferred_num_val = max(1, inferred_num_val)
        num_val = inferred_num_val
        num_train = len(all_pairs) - num_val

    assert num_train is not None and num_val is not None
    if len(all_pairs) < num_train + num_val:
        raise ValueError(
            f"Requested train+val={num_train + num_val} but found only {len(all_pairs)} snapshots"
        )
    train_pairs = all_pairs[:num_train]
    val_pairs = all_pairs[num_train : num_train + num_val]
    return all_pairs, train_pairs, val_pairs


def _resolve_scaled_split(
    *,
    data_path: Path,
    seed: int,
    scales: list[int],
    num_train_per_scale: int,
    num_val_per_scale: int,
) -> tuple[
    dict[int, list[tuple[Path, Path]]], list[tuple[Path, Path]], list[tuple[Path, Path]]
]:
    pairs_by_scale = discover_scale_snapshot_pairs(
        data_path, scales=scales, label="ZnCuSnSeS"
    )
    rng = random.Random(seed)
    train_pairs: list[tuple[Path, Path]] = []
    val_pairs: list[tuple[Path, Path]] = []
    for scale in scales:
        scale_pairs = list(pairs_by_scale[scale])
        rng.shuffle(scale_pairs)
        required = num_train_per_scale + num_val_per_scale
        if len(scale_pairs) < required:
            raise ValueError(
                f"Requested train+val={required} per scale but found only {len(scale_pairs)} under scale_{scale}"
            )
        train_pairs.extend(scale_pairs[:num_train_per_scale])
        val_pairs.extend(
            scale_pairs[num_train_per_scale : num_train_per_scale + num_val_per_scale]
        )
        pairs_by_scale[scale] = scale_pairs
    return pairs_by_scale, train_pairs, val_pairs


def main() -> None:
    args = setup_argparse()
    print(f"dataset_kind={args.dataset_kind}", flush=True)
    print(f"data_path={args.data_path.resolve()}", flush=True)
    print(f"seed={args.seed}", flush=True)
    print("", flush=True)

    if args.dataset_kind in {"siox", "ZnCuSnSeS_small"}:
        dataset_label = args.dataset_kind
        all_pairs, train_pairs, val_pairs = _resolve_single_snapshot_split(
            data_path=args.data_path.resolve(),
            dataset_label=dataset_label,
            seed=args.seed,
            num_train=args.num_train,
            num_val=args.num_val,
            val_fraction=args.val_fraction,
        )
        _print_pairs("all_pairs_after_shuffle", all_pairs)
        print("", flush=True)
        _print_pairs("train_pairs", train_pairs)
        print("", flush=True)
        _print_pairs("val_pairs", val_pairs)
        return

    scales = args.scales if args.scales is not None else [1, 2, 3]
    pairs_by_scale, train_pairs, val_pairs = _resolve_scaled_split(
        data_path=args.data_path.resolve(),
        seed=args.seed,
        scales=scales,
        num_train_per_scale=args.num_train_per_scale,
        num_val_per_scale=args.num_val_per_scale,
    )
    for scale in scales:
        print(
            f"scale_{scale}_pairs_after_shuffle: {len(pairs_by_scale[scale])}",
            flush=True,
        )
        for idx, (matrix_path, info_path) in enumerate(pairs_by_scale[scale]):
            print(f"  [{idx}] matrix={matrix_path}", flush=True)
            print(f"      info={info_path}", flush=True)
        print("", flush=True)
    _print_pairs("train_pairs", train_pairs)
    print("", flush=True)
    _print_pairs("val_pairs", val_pairs)


if __name__ == "__main__":
    main()
