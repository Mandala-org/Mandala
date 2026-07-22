#!/usr/bin/env python
"""Reconstruct OpenMX/Mandala density matrices on an OpenMX real-space grid."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from analysis.openmx_density_grid import (  # noqa: E402
    evaluate_density_grids,
    load_openmx_snapshot,
    parse_openmx_grid,
    parse_openmx_pao,
    write_density_outputs,
)


def _dtype(name: str) -> torch.dtype:
    return {"float32": torch.float32, "float64": torch.float64}[name]


def _device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested, but CUDA is unavailable")
    return torch.device(name)


def _parse_paos(values: list[str], dtype: torch.dtype):
    result = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"--pao must have the form ELEMENT=PATH, got {value!r}")
        element, raw_path = value.split("=", 1)
        element = element.strip()
        if not element or element in result:
            raise ValueError(f"Invalid or duplicate PAO element in {value!r}")
        result[element] = parse_openmx_pao(raw_path, dtype=dtype)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--matrix-path", type=Path, required=True, help="Ground-truth or sole HS.out"
    )
    parser.add_argument("--predicted-matrix-path", type=Path, default=None)
    parser.add_argument(
        "--info-path",
        type=Path,
        required=True,
        help="OpenMX input/output with geometry and basis",
    )
    parser.add_argument(
        "--grid-info-path",
        type=Path,
        action="append",
        default=[],
        help="OpenMX output containing Grid_Origin and grid vectors; may be repeated",
    )
    parser.add_argument(
        "--pao",
        action="append",
        required=True,
        metavar="ELEMENT=PATH",
        help="OpenMX PAO file for one element; repeat for multi-element systems",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--point-chunk-size", type=int, default=4096)
    parser.add_argument("--edge-batch-size", type=int, default=32)
    parser.add_argument("--expected-electrons", type=float, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dtype = _dtype(args.dtype)
    device = _device(args.device)
    grid_paths = [args.info_path, *args.grid_info_path]
    if not args.grid_info_path:
        sibling_log = args.info_path.parent / "log.out"
        if sibling_log.is_file():
            grid_paths.append(sibling_log)
    print(
        f"--- Loading OpenMX grid metadata from: {[str(path) for path in grid_paths]} ---"
    )
    grid = parse_openmx_grid(grid_paths, dtype=dtype)
    paos = _parse_paos(args.pao, dtype=dtype)
    print(f"--- Loading ground-truth density matrix: {args.matrix_path} ---")
    ground_truth = load_openmx_snapshot(args.matrix_path, args.info_path)
    snapshots = {"ground_truth": ground_truth}
    if args.predicted_matrix_path is not None:
        print(f"--- Loading predicted density matrix: {args.predicted_matrix_path} ---")
        snapshots["prediction"] = load_openmx_snapshot(
            args.predicted_matrix_path,
            args.info_path,
        )
    print(f"--- Evaluating on device={device}, dtype={dtype} ---")
    result = evaluate_density_grids(
        snapshots,
        grid,
        paos,
        device=device,
        dtype=dtype,
        point_chunk_size=args.point_chunk_size,
        edge_batch_size=args.edge_batch_size,
        expected_electrons=args.expected_electrons,
    )
    write_density_outputs(args.output_dir, result, grid, ground_truth)
    print(
        f"--- Wrote density reconstruction outputs to {args.output_dir.resolve()} ---"
    )
    for name, metrics in result.metrics["matrices"].items():
        print(
            f"{name}: integrated N={metrics['integrated_electrons']:.9f}, "
            f"Tr(D S_gt)={metrics['trace_density_ground_truth_overlap']:.9f}, "
            f"Tr(D S_own)={metrics['trace_density_own_overlap']:.9f}"
        )
    if "comparison" in result.metrics:
        comparison = result.metrics["comparison"]
        print(
            "comparison: "
            f"MAE={comparison['mae_e_per_bohr3']:.6e} e/bohr^3, "
            f"RMSE={comparison['rmse_e_per_bohr3']:.6e} e/bohr^3, "
            f"integrated_abs_error={comparison['integrated_abs_error_e']:.6e} e"
        )


if __name__ == "__main__":
    main()
