#!/usr/bin/env python

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch
from tqdm.auto import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from analysis import evaluation as analysis_eval  # noqa: E402
from data.snapshot import Snapshot  # noqa: E402
from data.spectral_fermi_cache import (  # noqa: E402
    SPECTRAL_FERMI_CACHE_VERSION,
    load_spectral_fermi_cache,
    spectral_fermi_cache_hash,
    spectral_fermi_cache_key,
)
from net.common import Config  # noqa: E402
from scripts.dataset import (  # noqa: E402
    discover_scale_snapshot_pairs,
    discover_single_snapshot_pairs,
    discover_siox_snapshot_pairs,
    discover_silicon_snapshot_pairs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Precompute DOS-derived Fermi levels for a dataset once, with shard support "
            "for later spectral fine-tuning."
        )
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="compute",
        choices=["compute", "merge"],
    )
    parser.add_argument(
        "--dataset-kind",
        type=str,
        default="ZnCuSnSeS",
        choices=["silicon", "silicon_scales", "siox", "ZnCuSnSeS_small", "ZnCuSnSeS"],
    )
    parser.add_argument("--data-path", type=Path, default=None)
    parser.add_argument("--scales", type=str, default="[1]")
    parser.add_argument("--convention", type=str, default="e3nn")
    parser.add_argument("--cutoff-radius", type=float, default=None)
    parser.add_argument("--symmetrize-hamiltonian-targets", type=str, default="true")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, default=None)
    parser.add_argument("--dos-cache-dir", type=Path, default=None)
    parser.add_argument("--dos-kmesh", type=str, default="4x4x4")
    parser.add_argument("--dos-energy-min", type=float, default=-10.0)
    parser.add_argument("--dos-energy-max", type=float, default=15.0)
    parser.add_argument("--dos-bin-width", type=float, default=0.1)
    parser.add_argument("--tetra-batch-size", type=int, default=256)
    parser.add_argument("--chunk-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--overlap-psd-cleanup", action="store_true")
    parser.add_argument("--overlap-jitter", action="store_true")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--show-inner-progress", action="store_true")
    parser.add_argument("--force-refresh", action="store_true")
    parser.add_argument("--strict", action="store_true")
    return parser.parse_args()


def _str_to_bool(value: str) -> bool:
    lowered = str(value).strip().lower()
    if lowered in {"1", "true", "t", "yes", "y"}:
        return True
    if lowered in {"0", "false", "f", "no", "n"}:
        return False
    raise ValueError(f"Expected boolean string, got {value!r}")


def _parse_scales(value: str) -> list[int]:
    parsed = json.loads(value)
    if not isinstance(parsed, list) or not parsed:
        raise ValueError("--scales must decode to a non-empty JSON list")
    return [int(item) for item in parsed]


def _discover_pairs(args: argparse.Namespace) -> list[tuple[Path, Path]]:
    if args.data_path is None:
        raise ValueError("--data-path is required in compute mode")
    root = args.data_path.expanduser().resolve()
    if args.dataset_kind == "silicon":
        return discover_silicon_snapshot_pairs(root)
    if args.dataset_kind == "siox":
        return discover_siox_snapshot_pairs(root)
    if args.dataset_kind == "ZnCuSnSeS_small":
        return discover_single_snapshot_pairs(root, label="ZnCuSnSeS_small")
    if args.dataset_kind in {"ZnCuSnSeS", "silicon_scales"}:
        scales = _parse_scales(args.scales)
        pairs_by_scale = discover_scale_snapshot_pairs(
            root,
            scales=scales,
            label=args.dataset_kind,
        )
        pairs: list[tuple[Path, Path]] = []
        for scale in scales:
            pairs.extend(pairs_by_scale[scale])
        return pairs
    raise ValueError(f"Unsupported dataset_kind={args.dataset_kind!r}")


def _select_shard(
    pairs: list[tuple[Path, Path]],
    *,
    num_shards: int,
    shard_index: int,
) -> list[tuple[Path, Path]]:
    if num_shards <= 0:
        raise ValueError("--num-shards must be positive")
    if shard_index < 0 or shard_index >= num_shards:
        raise ValueError("--shard-index must be in [0, num_shards)")
    return [pair for idx, pair in enumerate(pairs) if idx % num_shards == shard_index]


def _shard_cache_path(output_dir: Path, *, num_shards: int, shard_index: int) -> Path:
    return (
        output_dir
        / f"spectral_fermi_cache_shard_{shard_index:02d}_of_{num_shards:02d}.pt"
    )


def _shard_summary_path(output_dir: Path, *, num_shards: int, shard_index: int) -> Path:
    return (
        output_dir
        / f"spectral_fermi_cache_shard_{shard_index:02d}_of_{num_shards:02d}.json"
    )


def _dos_cache_path(dos_cache_dir: Path, matrix_path: Path, info_path: Path) -> Path:
    return (
        dos_cache_dir
        / f"{matrix_path.stem}_{spectral_fermi_cache_hash(matrix_path, info_path)}.pt"
    )


def _build_snapshot(
    matrix_path: Path,
    info_path: Path,
    *,
    convention: str,
    cutoff_radius: float | None,
    symmetrize_hamiltonian_targets: bool,
) -> Snapshot:
    cfg = Config(
        dtype=torch.float32,
        cutoff_radius=(7.0 if cutoff_radius is None else float(cutoff_radius)),
    )
    snapshot = Snapshot.from_openmx(
        matrix_path=matrix_path,
        info_path=info_path,
        convention=convention,
        symmetrize_density=True,
        cutoff_radius=None,
        cfg=cfg,
    )
    if cutoff_radius is not None:
        snapshot = snapshot.filter_by_distance(float(cutoff_radius))
    snapshot = snapshot.symmetrize_matrices(
        hamiltonian=bool(symmetrize_hamiltonian_targets),
        overlap=True,
        density=True,
    )
    return snapshot


def _record_for_snapshot(
    matrix_path: Path,
    info_path: Path,
    *,
    args: argparse.Namespace,
    dos_cache_dir: Path,
) -> dict[str, Any]:
    snapshot = _build_snapshot(
        matrix_path,
        info_path,
        convention=args.convention,
        cutoff_radius=args.cutoff_radius,
        symmetrize_hamiltonian_targets=_str_to_bool(
            args.symmetrize_hamiltonian_targets
        ),
    )
    grid_ev, dos, num_electrons, dos_target, fermi_level_ev = (
        analysis_eval.compute_tetrahedron_dos_and_fermi(
            snapshot,
            kmesh_spec=args.dos_kmesh,
            chunk_size=args.chunk_size,
            num_workers=args.num_workers,
            psd_cleanup=args.overlap_psd_cleanup,
            allow_jitter=args.overlap_jitter,
            bin_width=args.dos_bin_width,
            tetra_batch_size=args.tetra_batch_size,
            e_min=args.dos_energy_min,
            e_max=args.dos_energy_max,
            show_progress=args.show_inner_progress,
            cache_path=_dos_cache_path(dos_cache_dir, matrix_path, info_path),
            progress_label=matrix_path.parent.name,
        )
    )
    return {
        "matrix_path": str(matrix_path),
        "info_path": str(info_path),
        "fermi_level_ev": float(fermi_level_ev),
        "fermi_level_hartree": float(fermi_level_ev / analysis_eval.HARTREE_TO_EV),
        "num_electrons": float(num_electrons),
        "dos_electron_target": (None if dos_target is None else float(dos_target)),
        "grid_size": int(grid_ev.numel()),
        "dos_cache_path": str(_dos_cache_path(dos_cache_dir, matrix_path, info_path)),
        "has_info_fermi_level": bool(
            getattr(getattr(snapshot, "info", None), "fermi_level", None) is not None
        ),
    }


def _save_payload(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def _save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def compute_mode(args: argparse.Namespace) -> None:
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    dos_cache_dir = (
        args.dos_cache_dir.expanduser().resolve()
        if args.dos_cache_dir is not None
        else (output_dir / "dos_cache")
    )
    dos_cache_dir.mkdir(parents=True, exist_ok=True)

    all_pairs = _discover_pairs(args)
    shard_pairs = _select_shard(
        all_pairs,
        num_shards=args.num_shards,
        shard_index=args.shard_index,
    )
    shard_path = _shard_cache_path(
        output_dir,
        num_shards=args.num_shards,
        shard_index=args.shard_index,
    )
    summary_path = _shard_summary_path(
        output_dir,
        num_shards=args.num_shards,
        shard_index=args.shard_index,
    )

    records: dict[str, Any] = {}
    failures: list[dict[str, Any]] = []
    if shard_path.exists() and not args.force_refresh:
        existing = load_spectral_fermi_cache(shard_path)
        records.update(existing["records"])
        failures.extend(existing["failures"])
        print(
            f"--- Resuming existing shard cache {shard_path}: "
            f"records={len(records)}, failures={len(failures)} ---"
        )

    print(
        "--- Spectral Fermi precompute ---\n"
        f"dataset_kind={args.dataset_kind}\n"
        f"data_path={args.data_path}\n"
        f"pairs_total={len(all_pairs)}\n"
        f"pairs_in_shard={len(shard_pairs)}\n"
        f"shard={args.shard_index + 1}/{args.num_shards}\n"
        f"output_shard={shard_path}\n"
        f"dos_cache_dir={dos_cache_dir}"
    )

    progress = tqdm(
        shard_pairs,
        total=len(shard_pairs),
        desc=f"Shard {args.shard_index + 1}/{args.num_shards}",
    )
    processed = 0
    for pair_idx, (matrix_path, info_path) in enumerate(progress, start=1):
        key = spectral_fermi_cache_key(matrix_path, info_path)
        if key in records:
            progress.set_postfix_str("cached")
            continue
        label = f"{matrix_path.parent.name} [{pair_idx}/{len(shard_pairs)}]"
        print(f"\n--- Processing {label} ---", flush=True)
        t0 = time.perf_counter()
        try:
            record = _record_for_snapshot(
                matrix_path,
                info_path,
                args=args,
                dos_cache_dir=dos_cache_dir,
            )
            records[key] = record
            processed += 1
            elapsed = time.perf_counter() - t0
            progress.set_postfix_str(
                f"ok {record['fermi_level_ev']:.3f} eV, {elapsed:.1f}s"
            )
            print(
                f"--- Finished {label}: "
                f"fermi={record['fermi_level_ev']:.6f} eV, "
                f"num_electrons={record['num_electrons']:.3f}, "
                f"elapsed={elapsed:.2f}s ---",
                flush=True,
            )
        except Exception as exc:
            failure = {
                "matrix_path": str(matrix_path),
                "info_path": str(info_path),
                "error": str(exc),
            }
            failures.append(failure)
            progress.set_postfix_str("failed")
            print(f"!!! FAILED {label}: {exc}", flush=True)
            if args.strict:
                raise
        payload = {
            "version": SPECTRAL_FERMI_CACHE_VERSION,
            "records": records,
            "failures": failures,
            "settings": {
                "dataset_kind": args.dataset_kind,
                "data_path": str(args.data_path),
                "scales": (
                    _parse_scales(args.scales)
                    if args.dataset_kind in {"ZnCuSnSeS", "silicon_scales"}
                    else None
                ),
                "convention": args.convention,
                "cutoff_radius": args.cutoff_radius,
                "symmetrize_hamiltonian_targets": _str_to_bool(
                    args.symmetrize_hamiltonian_targets
                ),
                "dos_kmesh": args.dos_kmesh,
                "dos_energy_min": args.dos_energy_min,
                "dos_energy_max": args.dos_energy_max,
                "dos_bin_width": args.dos_bin_width,
                "tetra_batch_size": args.tetra_batch_size,
                "chunk_size": args.chunk_size,
                "num_workers": args.num_workers,
                "overlap_psd_cleanup": args.overlap_psd_cleanup,
                "overlap_jitter": args.overlap_jitter,
                "num_shards": args.num_shards,
                "shard_index": args.shard_index,
            },
        }
        _save_payload(shard_path, payload)
        _save_json(
            summary_path,
            {
                "records": len(records),
                "failures": len(failures),
                "processed_this_run": processed,
                "output_shard": str(shard_path),
            },
        )
    print(
        f"--- Shard complete: records={len(records)}, failures={len(failures)}, "
        f"output={shard_path} ---"
    )


def merge_mode(args: argparse.Namespace) -> None:
    output_dir = args.output_dir.expanduser().resolve()
    shard_paths = sorted(output_dir.glob("spectral_fermi_cache_shard_*_of_*.pt"))
    if not shard_paths:
        raise FileNotFoundError(f"No shard caches found in {output_dir}")
    merged_records: dict[str, Any] = {}
    merged_failures: list[dict[str, Any]] = []
    settings_ref: dict[str, Any] | None = None
    for shard_path in tqdm(shard_paths, desc="Merging spectral Fermi shards"):
        payload = load_spectral_fermi_cache(shard_path)
        shard_settings = payload.get("settings", {})
        if settings_ref is None:
            settings_ref = dict(shard_settings)
        overlap = set(merged_records).intersection(payload["records"])
        if overlap:
            raise ValueError(
                f"Duplicate keys encountered while merging shards: "
                f"{sorted(list(overlap))[:3]}"
            )
        merged_records.update(payload["records"])
        merged_failures.extend(payload["failures"])
    output_path = (
        args.output_path.expanduser().resolve()
        if args.output_path is not None
        else (output_dir / "spectral_fermi_cache_merged.pt")
    )
    merged = {
        "version": SPECTRAL_FERMI_CACHE_VERSION,
        "records": merged_records,
        "failures": merged_failures,
        "settings": settings_ref or {},
    }
    _save_payload(output_path, merged)
    _save_json(
        output_path.with_suffix(".json"),
        {
            "records": len(merged_records),
            "failures": len(merged_failures),
            "output_path": str(output_path),
            "shards": [str(path) for path in shard_paths],
        },
    )
    print(
        f"--- Merged spectral Fermi cache saved to {output_path}: "
        f"records={len(merged_records)}, failures={len(merged_failures)} ---"
    )


def main() -> None:
    args = parse_args()
    if args.mode == "compute":
        compute_mode(args)
        return
    merge_mode(args)


if __name__ == "__main__":
    main()
