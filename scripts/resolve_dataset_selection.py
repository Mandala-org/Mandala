#!/usr/bin/env python

from __future__ import annotations

import argparse
import random
from pathlib import Path

from data.openmx_info_parser import parse_info_out


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
            "List the exact train/val snapshot pairs selected by the dataset "
            "split logic, without loading matrices."
        )
    )
    parser.add_argument(
        "--dataset-kind",
        type=str,
        required=True,
        choices=["silicon", "silicon_scales", "siox", "ZnCuSnSeS_small", "ZnCuSnSeS"],
    )
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-temp", type=int, default=300)
    parser.add_argument("--max-temp", type=int, default=3000)
    parser.add_argument("--temp-step", type=int, default=300)
    parser.add_argument("--n-snapshots-per-temp", type=int, default=50)
    parser.add_argument("--val-temp", type=int, default=1500)
    parser.add_argument("--val-n-snapshots", type=int, default=None)
    parser.add_argument("--num-train", type=int, default=None)
    parser.add_argument("--num-val", type=int, default=None)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--scales", type=str_to_list, default=None)
    parser.add_argument("--num-train-per-scale", type=int, default=40)
    parser.add_argument("--num-val-per-scale", type=int, default=10)
    parser.add_argument(
        "--allow-log-fallback",
        action="store_true",
        help=(
            "Reproduce the legacy training-time behavior that accepted log.out "
            "when the expected info file was missing."
        ),
    )
    parser.add_argument(
        "--require-parseable-info",
        action="store_true",
        help=(
            "Drop snapshots whose info file does not contain a parseable "
            "<coordinates.forces> block, matching training-time skipping."
        ),
    )
    return parser.parse_args()


def _is_parseable_info(path: Path) -> bool:
    try:
        parse_info_out(path)
        return True
    except Exception:
        return False


def _resolve_info_file(
    sample_dir: Path,
    *,
    allow_log_fallback: bool,
    require_parseable_info: bool,
) -> Path | None:
    for name in ("Si.out", "SiO2.out", "ZnCuSeS.out", "info.dat", "info.txt"):
        candidate = sample_dir / name
        if candidate.exists():
            if require_parseable_info and not _is_parseable_info(candidate):
                return None
            return candidate
    if allow_log_fallback:
        candidate = sample_dir / "log.out"
        if candidate.exists():
            if require_parseable_info and not _is_parseable_info(candidate):
                return None
            return candidate
    return None


def _discover_pairs(
    root: Path,
    *,
    allow_log_fallback: bool,
    require_parseable_info: bool,
) -> list[tuple[Path, Path]]:
    pairs: list[tuple[Path, Path]] = []
    if root.is_dir():
        matrix_path = root / "HS.out"
        if matrix_path.exists():
            info_path = _resolve_info_file(
                root,
                allow_log_fallback=allow_log_fallback,
                require_parseable_info=require_parseable_info,
            )
            if info_path is not None:
                pairs.append((matrix_path.resolve(), info_path.resolve()))
        for sample_dir in sorted(path for path in root.iterdir() if path.is_dir()):
            matrix_path = sample_dir / "HS.out"
            if not matrix_path.exists():
                continue
            info_path = _resolve_info_file(
                sample_dir,
                allow_log_fallback=allow_log_fallback,
                require_parseable_info=require_parseable_info,
            )
            if info_path is None:
                continue
            pairs.append((matrix_path.resolve(), info_path.resolve()))
    return pairs


def _discover_scale_pairs(
    root: Path,
    scales: list[int],
    *,
    allow_log_fallback: bool,
    require_parseable_info: bool,
) -> dict[int, list[tuple[Path, Path]]]:
    pairs_by_scale: dict[int, list[tuple[Path, Path]]] = {}
    for scale in scales:
        scale_dir = root / f"scale_{scale}"
        if not scale_dir.is_dir():
            raise FileNotFoundError(f"Missing scale directory: {scale_dir}")
        pairs_by_scale[scale] = _discover_pairs(
            scale_dir,
            allow_log_fallback=allow_log_fallback,
            require_parseable_info=require_parseable_info,
        )
    return pairs_by_scale


def _print_pairs(label: str, pairs: list[tuple[Path, Path]]) -> None:
    print(f"{label}: {len(pairs)}")
    for idx, (matrix_path, info_path) in enumerate(pairs):
        print(f"  [{idx}] matrix={matrix_path}")
        print(f"      info={info_path}")


def _split_single_snapshot(
    *,
    data_path: Path,
    seed: int,
    allow_log_fallback: bool,
    require_parseable_info: bool,
    num_train: int | None,
    num_val: int | None,
    val_fraction: float,
    min_temp: int | None = None,
    max_temp: int | None = None,
    temp_step: int | None = None,
    n_snapshots_per_temp: int | None = None,
    val_temp: int | None = None,
    val_n_snapshots: int | None = None,
) -> tuple[list[tuple[Path, Path]], list[tuple[Path, Path]]]:
    rng = random.Random(seed)

    if min_temp == max_temp == val_temp and min_temp is not None:
        root = data_path / f"{val_temp}K"
        all_pairs = _discover_pairs(
            root,
            allow_log_fallback=allow_log_fallback,
            require_parseable_info=require_parseable_info,
        )
        rng.shuffle(all_pairs)
        if not all_pairs:
            raise ValueError(f"No snapshots found under {root}")
        if num_train is None or num_val is None:
            raise ValueError(
                "Single-temp global split requires --num-train and --num-val."
            )
        if len(all_pairs) < num_train + num_val:
            raise ValueError(
                f"Requested train+val={num_train + num_val} but found only {len(all_pairs)} snapshots under {root}"
            )
        return all_pairs[:num_train], all_pairs[num_train : num_train + num_val]

    if num_train is not None or num_val is not None:
        if num_train is None or num_val is None:
            raise ValueError("Global split requires both --num-train and --num-val.")
        all_temps = list(
            range(min_temp or 300, (max_temp or 3000) + 1, temp_step or 300)
        )
        if not all_temps:
            raise ValueError("No temperatures selected.")
        if num_train % len(all_temps) != 0 or num_val % len(all_temps) != 0:
            raise ValueError(
                "Balanced split requires num_train and num_val to be divisible by the number of temperatures."
            )
        train_per_temp = num_train // len(all_temps)
        val_per_temp = num_val // len(all_temps)
        train_pairs: list[tuple[Path, Path]] = []
        val_pairs: list[tuple[Path, Path]] = []
        for temp in all_temps:
            temp_pairs = _discover_pairs(
                data_path / f"{temp}K",
                allow_log_fallback=allow_log_fallback,
                require_parseable_info=require_parseable_info,
            )
            shuffled = list(temp_pairs)
            rng.shuffle(shuffled)
            if len(shuffled) < train_per_temp + val_per_temp:
                raise ValueError(
                    f"Requested train+val per temp={train_per_temp + val_per_temp} but found only {len(shuffled)} snapshots under {data_path / f'{temp}K'}"
                )
            train_pairs.extend(shuffled[:train_per_temp])
            val_pairs.extend(shuffled[train_per_temp : train_per_temp + val_per_temp])
        return train_pairs, val_pairs

    if min_temp is None or max_temp is None or temp_step is None or val_temp is None:
        raise ValueError(
            "Temperature split requires min_temp, max_temp, temp_step, and val_temp."
        )

    all_temps = list(range(min_temp, max_temp + 1, temp_step))
    train_temps = [
        temp
        for temp in all_temps
        if temp != val_temp or min_temp == max_temp == val_temp
    ]
    train_pairs: list[tuple[Path, Path]] = []
    for temp in train_temps:
        temp_pairs = _discover_pairs(
            data_path / f"{temp}K",
            allow_log_fallback=allow_log_fallback,
            require_parseable_info=require_parseable_info,
        )
        num_to_sample = min(len(temp_pairs), n_snapshots_per_temp or 50)
        shuffled = list(temp_pairs)
        rng.shuffle(shuffled)
        train_pairs.extend(shuffled[:num_to_sample])

    val_pairs = _discover_pairs(
        data_path / f"{val_temp}K",
        allow_log_fallback=allow_log_fallback,
        require_parseable_info=require_parseable_info,
    )
    num_val_to_sample = min(len(val_pairs), n_snapshots_per_temp or 50)
    if val_n_snapshots is not None:
        num_val_to_sample = min(num_val_to_sample, val_n_snapshots)
    shuffled_val = list(val_pairs)
    rng.shuffle(shuffled_val)
    return train_pairs, shuffled_val[:num_val_to_sample]


def _split_multi_snapshot(
    *,
    data_path: Path,
    seed: int,
    allow_log_fallback: bool,
    require_parseable_info: bool,
    scales: list[int],
    num_train_per_scale: int,
    num_val_per_scale: int,
) -> tuple[
    list[tuple[Path, Path]], list[tuple[Path, Path]], dict[int, list[tuple[Path, Path]]]
]:
    rng = random.Random(seed)
    pairs_by_scale = _discover_scale_pairs(
        data_path,
        scales,
        allow_log_fallback=allow_log_fallback,
        require_parseable_info=require_parseable_info,
    )
    train_pairs: list[tuple[Path, Path]] = []
    val_pairs: list[tuple[Path, Path]] = []
    for scale in scales:
        shuffled = list(pairs_by_scale[scale])
        rng.shuffle(shuffled)
        required = num_train_per_scale + num_val_per_scale
        if len(shuffled) < required:
            raise ValueError(
                f"Requested train+val={required} per scale but found only {len(shuffled)} snapshots under scale_{scale}"
            )
        train_pairs.extend(shuffled[:num_train_per_scale])
        val_pairs.extend(
            shuffled[num_train_per_scale : num_train_per_scale + num_val_per_scale]
        )
        pairs_by_scale[scale] = shuffled
    return train_pairs, val_pairs, pairs_by_scale


def main() -> None:
    args = setup_argparse()
    data_path = args.data_path.expanduser().resolve()

    if args.dataset_kind == "silicon":
        train_pairs, val_pairs = _split_single_snapshot(
            data_path=data_path,
            seed=args.seed,
            allow_log_fallback=args.allow_log_fallback,
            require_parseable_info=args.require_parseable_info,
            num_train=args.num_train,
            num_val=args.num_val,
            val_fraction=args.val_fraction,
            min_temp=args.min_temp,
            max_temp=args.max_temp,
            temp_step=args.temp_step,
            n_snapshots_per_temp=args.n_snapshots_per_temp,
            val_temp=args.val_temp,
            val_n_snapshots=args.val_n_snapshots,
        )
    elif args.dataset_kind == "silicon_scales":
        scales = args.scales if args.scales is not None else [1]
        train_pairs, val_pairs, _ = _split_multi_snapshot(
            data_path=data_path,
            seed=args.seed,
            allow_log_fallback=args.allow_log_fallback,
            require_parseable_info=args.require_parseable_info,
            scales=[int(x) for x in scales],
            num_train_per_scale=args.num_train_per_scale,
            num_val_per_scale=args.num_val_per_scale,
        )
    elif args.dataset_kind == "siox":
        all_pairs = _discover_pairs(
            data_path,
            allow_log_fallback=args.allow_log_fallback,
            require_parseable_info=args.require_parseable_info,
        )
        rng = random.Random(args.seed)
        rng.shuffle(all_pairs)
        if not all_pairs:
            raise ValueError(f"No snapshots found under {data_path}")
        if args.num_train is not None or args.num_val is not None:
            if args.num_train is None or args.num_val is None:
                raise ValueError(
                    "Global split requires both --num-train and --num-val."
                )
            if len(all_pairs) < args.num_train + args.num_val:
                raise ValueError(
                    f"Requested train+val={args.num_train + args.num_val} but found only {len(all_pairs)} snapshots under {data_path}"
                )
            train_pairs = all_pairs[: args.num_train]
            val_pairs = all_pairs[args.num_train : args.num_train + args.num_val]
        else:
            num_val = int(len(all_pairs) * args.val_fraction)
            if len(all_pairs) > 1:
                num_val = max(1, num_val)
            train_pairs = all_pairs[: len(all_pairs) - num_val]
            val_pairs = all_pairs[len(all_pairs) - num_val :]
    elif args.dataset_kind == "ZnCuSnSeS_small":
        all_pairs = _discover_pairs(
            data_path,
            allow_log_fallback=args.allow_log_fallback,
            require_parseable_info=args.require_parseable_info,
        )
        rng = random.Random(args.seed)
        rng.shuffle(all_pairs)
        if not all_pairs:
            raise ValueError(f"No snapshots found under {data_path}")
        if args.num_train is not None or args.num_val is not None:
            if args.num_train is None or args.num_val is None:
                raise ValueError(
                    "Global split requires both --num-train and --num-val."
                )
            if len(all_pairs) < args.num_train + args.num_val:
                raise ValueError(
                    f"Requested train+val={args.num_train + args.num_val} but found only {len(all_pairs)} snapshots under {data_path}"
                )
            train_pairs = all_pairs[: args.num_train]
            val_pairs = all_pairs[args.num_train : args.num_train + args.num_val]
        else:
            num_val = int(len(all_pairs) * args.val_fraction)
            if len(all_pairs) > 1:
                num_val = max(1, num_val)
            train_pairs = all_pairs[: len(all_pairs) - num_val]
            val_pairs = all_pairs[len(all_pairs) - num_val :]
    elif args.dataset_kind == "ZnCuSnSeS":
        scales = args.scales if args.scales is not None else [1, 2, 3]
        train_pairs, val_pairs, _ = _split_multi_snapshot(
            data_path=data_path,
            seed=args.seed,
            allow_log_fallback=args.allow_log_fallback,
            require_parseable_info=args.require_parseable_info,
            scales=[int(x) for x in scales],
            num_train_per_scale=args.num_train_per_scale,
            num_val_per_scale=args.num_val_per_scale,
        )
    else:
        raise ValueError(f"Unsupported dataset kind: {args.dataset_kind}")

    _print_pairs("train_pairs", train_pairs)
    print()
    _print_pairs("val_pairs", val_pairs)


if __name__ == "__main__":
    main()
