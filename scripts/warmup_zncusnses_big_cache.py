#!/usr/bin/env python
"""Warm raw and preprocessed caches for sharded ZnCuSnSeS datasets."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from data.factory import DatasetFactory  # noqa: E402
from net.common import Config  # noqa: E402
from scripts.dataset import discover_scale_snapshot_pairs  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build Mandala snapshot and preprocessed-sample caches without "
            "constructing a model or starting training."
        )
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path("/bigdata/casus/wdm/hamiltonian_learning/data/ZnCuSnSeS_big"),
    )
    parser.add_argument(
        "--snapshot-cache-dir",
        type=Path,
        default=Path(
            "/bigdata/casus/wdm/hamiltonian_learning/data/"
            "ZnCuSnSeS_big/snapshot_cache"
        ),
    )
    parser.add_argument("--scales", default="1,2")
    parser.add_argument("--expected-per-scale", type=int, default=400)
    parser.add_argument("--num-shards", type=int, default=64)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument(
        "--envelope-path",
        type=Path,
        default=Path(
            "eval_outputs/zncusnses_radial_fit_study/"
            "slater_soft_cutoff_envelope.json"
        ),
    )
    return parser.parse_args()


def parse_scales(value: str) -> list[int]:
    try:
        scales = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise ValueError(f"Invalid --scales value: {value!r}") from exc
    if not scales or len(scales) != len(set(scales)):
        raise ValueError("--scales must contain unique comma-separated integers")
    return scales


def build_cache_config(args: argparse.Namespace) -> Config:
    envelope_path = args.envelope_path.expanduser().resolve()
    if not envelope_path.is_file():
        raise FileNotFoundError(f"Envelope artifact not found: {envelope_path}")

    cfg = Config()
    cfg.dtype = torch.float32
    cfg.cutoff_radius = 11.0
    cfg.l_max = 6
    cfg.n_radial = 128
    cfg.radial_layers = (128, 128, 128)
    cfg.radial_embedding_scale = "none"
    cfg.matrix_targets = ["hamiltonian"]
    cfg.train_target = "matrix"
    cfg.apply_cutoff_to_targets = True
    cfg.symmetrize_hamiltonian_targets = True
    cfg.require_exact_edge_match = True
    cfg.precompute_edge_features = True
    cfg.separate_shifted_self = True
    cfg.hamiltonian_envelope_path = str(envelope_path)
    cfg.hamiltonian_envelope_mode = "multiply_prediction"
    cfg.pair_distance_normalization = "off"
    cfg.loss_weighting_mode = "off"
    cfg.spectral_loss_enabled = False
    cfg.spectral_fermi_cache_path = None
    cfg.snapshot_cache_dir = str(args.snapshot_cache_dir.expanduser().resolve())
    cfg.dataset_device = None
    cfg.allow_incomplete_dataset = False
    cfg.shuffle_snapshot_load_order = False
    cfg.verbosity = 0
    return cfg


def main() -> None:
    args = parse_args()
    if args.num_shards <= 0:
        raise ValueError("--num-shards must be positive")
    if args.shard_index < 0 or args.shard_index >= args.num_shards:
        raise ValueError(
            f"--shard-index must be in [0, {args.num_shards}), got {args.shard_index}"
        )
    if args.expected_per_scale <= 0:
        raise ValueError("--expected-per-scale must be positive")

    scales = parse_scales(args.scales)
    data_path = args.data_path.expanduser().resolve()
    if not data_path.is_dir():
        raise FileNotFoundError(f"Dataset root not found: {data_path}")

    print("=== ZnCuSnSeS_big cache warmup ===", flush=True)
    print(f"data_path={data_path}", flush=True)
    print(
        f"snapshot_cache_dir={args.snapshot_cache_dir.expanduser().resolve()}",
        flush=True,
    )
    print(f"scales={scales}", flush=True)
    print(
        f"shard={args.shard_index}/{args.num_shards} "
        f"expected_per_scale={args.expected_per_scale}",
        flush=True,
    )

    pairs_by_scale = discover_scale_snapshot_pairs(
        data_path,
        scales=scales,
        label="ZnCuSnSeS_big",
    )
    all_pairs: list[tuple[Path, Path]] = []
    for scale in scales:
        pairs = pairs_by_scale[scale]
        if len(pairs) != args.expected_per_scale:
            raise ValueError(
                f"Expected exactly {args.expected_per_scale} valid snapshots in "
                f"scale_{scale}, found {len(pairs)}. Refusing partial cache warmup."
            )
        all_pairs.extend(pairs)

    shard_pairs = all_pairs[args.shard_index :: args.num_shards]
    if not shard_pairs:
        raise ValueError(
            f"Shard {args.shard_index}/{args.num_shards} contains no snapshots"
        )
    print(
        f"total_pairs={len(all_pairs)} shard_pairs={len(shard_pairs)}",
        flush=True,
    )
    print(f"first_pair={shard_pairs[0][0].parent}", flush=True)
    print(f"last_pair={shard_pairs[-1][0].parent}", flush=True)

    cfg = build_cache_config(args)
    factory = DatasetFactory(cfg, convention="e3nn")
    for matrix_path, info_path in shard_pairs:
        factory.add_snapshot(matrix_path, info_path, purpose="train")

    started = time.perf_counter()
    dataset, _, _ = factory.create()
    elapsed = time.perf_counter() - started
    if len(dataset) != len(shard_pairs):
        raise RuntimeError(
            f"Cache warmup loaded {len(dataset)} samples, expected {len(shard_pairs)}"
        )

    print("=== Cache warmup complete ===", flush=True)
    print(f"shard={args.shard_index}/{args.num_shards}", flush=True)
    print(f"snapshots={len(dataset)}", flush=True)
    print(f"elapsed_seconds={elapsed:.3f}", flush=True)
    print(f"seconds_per_snapshot={elapsed / len(dataset):.3f}", flush=True)
    print(
        "snapshot_cache="
        f"hits:{dataset.snapshot_cache_hits} "
        f"misses:{dataset.snapshot_cache_misses}",
        flush=True,
    )
    print(
        "preprocessed_cache="
        f"hits:{dataset.preprocessed_cache_hits} "
        f"misses:{dataset.preprocessed_cache_misses}",
        flush=True,
    )


if __name__ == "__main__":
    main()
