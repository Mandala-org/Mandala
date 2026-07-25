#!/usr/bin/env python
"""Generate deterministic diamond-Si supercells for inference benchmarks."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ase.build import bulk
from ase.io import write


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "benchmark_data/silicon_scaling"
SILICON_LATTICE_CONSTANT_ANGSTROM = 5.4437
REPETITIONS = (1, 2, 4, 8, 16, 32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate 8, 64, 512, 4096, 32768, and 262144 atom diamond-Si " "CIF files."
        )
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--lattice-constant",
        type=float,
        default=SILICON_LATTICE_CONSTANT_ANGSTROM,
    )
    parser.add_argument(
        "--repetitions",
        type=str,
        default=",".join(str(value) for value in REPETITIONS),
        help="Comma-separated cubic repeat factors selected from 1,2,4,8,16,32.",
    )
    return parser.parse_args()


def parse_repetitions(value: str) -> tuple[int, ...]:
    try:
        repetitions = tuple(
            int(item.strip()) for item in value.split(",") if item.strip()
        )
    except ValueError as exc:
        raise ValueError(f"Invalid --repetitions value: {value!r}") from exc
    if not repetitions or len(set(repetitions)) != len(repetitions):
        raise ValueError("--repetitions must contain unique integer values")
    unsupported = sorted(set(repetitions).difference(REPETITIONS))
    if unsupported:
        raise ValueError(
            f"Unsupported repetitions {unsupported}; available={REPETITIONS}"
        )
    return tuple(repeat for repeat in REPETITIONS if repeat in repetitions)


def main() -> None:
    args = parse_args()
    if args.lattice_constant <= 0.0:
        raise ValueError("--lattice-constant must be positive")
    repetitions = parse_repetitions(args.repetitions)

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    conventional_cell = bulk(
        "Si",
        crystalstructure="diamond",
        a=args.lattice_constant,
        cubic=True,
    )
    if len(conventional_cell) != 8:
        raise RuntimeError(
            f"Expected an 8-atom conventional diamond cell, got {len(conventional_cell)}"
        )

    print(f"output_dir={output_dir}", flush=True)
    print(f"lattice_constant_angstrom={args.lattice_constant:.8f}", flush=True)
    for repeat in repetitions:
        structure = conventional_cell.repeat((repeat, repeat, repeat))
        structure.pbc = True
        structure.wrap()
        atom_count = len(structure)
        expected_count = 8 * repeat**3
        if atom_count != expected_count:
            raise RuntimeError(
                f"Supercell {repeat}x{repeat}x{repeat} has {atom_count} atoms; "
                f"expected {expected_count}"
            )
        path = output_dir / f"silicon_{atom_count:05d}_atoms.cif"
        write(path, structure, format="cif")
        print(
            f"wrote={path} atoms={atom_count} repeat={repeat}x{repeat}x{repeat} "
            f"cell_lengths_angstrom={structure.cell.lengths().tolist()}",
            flush=True,
        )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr, flush=True)
        raise
