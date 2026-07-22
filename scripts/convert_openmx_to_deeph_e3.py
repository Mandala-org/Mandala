#!/usr/bin/env python3
"""Convert OpenMX snapshot trees to DeepH-E3's processed dataset format."""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from data.deeph_e3_exporter import (  # noqa: E402
    DeepHE3ExportSummary,
    export_openmx_snapshot_to_deeph_e3,
)


@dataclass(frozen=True)
class ConversionTask:
    matrix_path: Path
    info_path: Path
    output_dir: Path


def _resolve_info_path(snapshot_dir: Path, dataset_kind: str) -> Path:
    expected_names = {
        "siox": ("SiO2.out",),
        "zncusnses": ("ZnCuSeS.out",),
        "auto": ("SiO2.out", "ZnCuSeS.out", "Si.out"),
    }[dataset_kind]
    matches = [
        snapshot_dir / name
        for name in expected_names
        if (snapshot_dir / name).is_file()
    ]
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected exactly one OpenMX info file from {expected_names} in "
            f"{snapshot_dir}, found {[path.name for path in matches]}"
        )
    return matches[0]


def _discover_tasks(args: argparse.Namespace) -> tuple[list[ConversionTask], bool]:
    input_dir = args.input_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input dataset directory not found: {input_dir}")

    if (input_dir / "HS.out").is_file():
        info_path = _resolve_info_path(input_dir, args.dataset_kind)
        return [ConversionTask(input_dir / "HS.out", info_path, output_dir)], True

    search_roots: list[Path]
    if args.dataset_kind == "zncusnses" and args.scales:
        search_roots = []
        for scale in args.scales:
            scale_dir = input_dir / f"scale_{scale}"
            if not scale_dir.is_dir():
                raise FileNotFoundError(
                    f"Requested scale directory not found: {scale_dir}"
                )
            search_roots.append(scale_dir)
    else:
        search_roots = [input_dir]

    snapshot_dirs = sorted(
        {
            matrix_path.parent
            for search_root in search_roots
            for matrix_path in search_root.rglob("HS.out")
        }
    )
    if not snapshot_dirs:
        raise ValueError(f"No HS.out snapshots found under {input_dir}")

    tasks = []
    for snapshot_dir in snapshot_dirs:
        relative_dir = snapshot_dir.relative_to(input_dir)
        tasks.append(
            ConversionTask(
                matrix_path=snapshot_dir / "HS.out",
                info_path=_resolve_info_path(snapshot_dir, args.dataset_kind),
                output_dir=output_dir / relative_dir,
            )
        )
    return tasks, False


def _convert_task(
    task: ConversionTask,
    *,
    include_overlap: bool,
    include_density: bool,
    validate: bool,
) -> DeepHE3ExportSummary:
    return export_openmx_snapshot_to_deeph_e3(
        task.matrix_path,
        task.info_path,
        task.output_dir,
        include_overlap=include_overlap,
        include_density=include_density,
        validate=validate,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert OpenMX HS.out snapshots to the processed HDF5/text format "
            "consumed by DeepH-E3. Existing output snapshots are never overwritten."
        )
    )
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--dataset-kind",
        choices=("siox", "zncusnses", "auto"),
        required=True,
    )
    parser.add_argument(
        "--scales",
        type=int,
        nargs="+",
        help="For ZnCuSnSeS, convert only these scale_N directories.",
    )
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument(
        "--include-overlap",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--include-density",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--validate",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Strictly reload and numerically compare every exported matrix.",
    )
    args = parser.parse_args()
    if args.num_workers < 1:
        parser.error("--num-workers must be positive")
    if args.dataset_kind != "zncusnses" and args.scales:
        parser.error("--scales is only valid with --dataset-kind zncusnses")
    return args


def main() -> None:
    args = _parse_args()
    tasks, single_snapshot = _discover_tasks(args)
    existing = [task.output_dir for task in tasks if task.output_dir.exists()]
    if existing:
        examples = "\n".join(f"  {path}" for path in existing[:10])
        raise FileExistsError(
            f"Refusing to overwrite {len(existing)} existing output snapshot(s):\n"
            f"{examples}"
        )

    print("=== OpenMX -> DeepH-E3 conversion ===", flush=True)
    print(f"input_dir: {args.input_dir.expanduser().resolve()}")
    print(f"output_dir: {args.output_dir.expanduser().resolve()}")
    print(f"dataset_kind: {args.dataset_kind}")
    print(f"snapshots: {len(tasks)}")
    print(f"workers: {args.num_workers}")
    print(f"include_overlap: {args.include_overlap}")
    print(f"include_density: {args.include_density}")
    print(f"strict_validation: {args.validate}", flush=True)

    summaries: list[DeepHE3ExportSummary] = []
    if args.num_workers == 1:
        for task in tqdm(tasks, desc="Converting snapshots", unit="snapshot"):
            summaries.append(
                _convert_task(
                    task,
                    include_overlap=args.include_overlap,
                    include_density=args.include_density,
                    validate=args.validate,
                )
            )
    else:
        with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
            futures = {
                executor.submit(
                    _convert_task,
                    task,
                    include_overlap=args.include_overlap,
                    include_density=args.include_density,
                    validate=args.validate,
                ): task
                for task in tasks
            }
            with tqdm(
                total=len(tasks), desc="Converting snapshots", unit="snapshot"
            ) as progress:
                for future in as_completed(futures):
                    task = futures[future]
                    try:
                        summaries.append(future.result())
                    except BaseException as exc:
                        for pending in futures:
                            pending.cancel()
                        raise RuntimeError(
                            f"Conversion failed for {task.matrix_path}"
                        ) from exc
                    progress.update(1)

    summaries.sort(key=lambda summary: summary.source_dir)
    manifest = {
        "format": "deeph_e3_processed",
        "source_root": str(args.input_dir.expanduser().resolve()),
        "dataset_kind": args.dataset_kind,
        "include_overlap": args.include_overlap,
        "include_density": args.include_density,
        "strict_validation": args.validate,
        "snapshots": [summary.to_dict() for summary in summaries],
    }
    manifest_path = (
        args.output_dir.with_name(args.output_dir.name + "_conversion_manifest.json")
        if single_snapshot
        else args.output_dir / "conversion_manifest.json"
    )
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")

    total_blocks = sum(summary.num_hamiltonian_blocks for summary in summaries)
    max_error = max(summary.hamiltonian_roundtrip_max_abs_ha for summary in summaries)
    print("\n=== Conversion complete ===")
    print(f"converted snapshots: {len(summaries)}")
    print(f"Hamiltonian blocks: {total_blocks}")
    print(f"max Hamiltonian round-trip error: {max_error:.6e} Ha")
    print(f"manifest: {manifest_path}")


if __name__ == "__main__":
    main()
