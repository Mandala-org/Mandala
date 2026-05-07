#!/usr/bin/env python

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-codex")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from ase.io import read as ase_read

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from core.block_irrep_mapper import BlockIrrepMapper  # noqa: E402
from core.orbital_irrep_config import OrbitalIrrepConfig  # noqa: E402
from data.openmx_info_parser import parse_info_out  # noqa: E402
from data.snapshot import _compute_atom_image_shifts  # noqa: E402


HEADER_RE = re.compile(
    r"global index=(\d+)\s+local index=\d+\s+\(global=(\d+),\s*Rn=([-]?\d+)\)"
)
HEADER_OVERLAP_POS_X_RE = re.compile(
    r"global index=(\d+)\s+local index=\d+\s+\(global=(\d+),\s*Rn=([-]?\d+) ([-]?\d+) ([-]?\d+) ([-]?\d+)\)"
)
SECTION_RE = re.compile(
    r"^(Kohn-Sham Hamiltonian spin=0|Overlap matrix|Density matrix spin=0|Overlap matrix with position operator x)$"
)
SKIP_OVERLAP_RE = re.compile(r"^Overlap matrix with (position|momentum) operator")


def setup_argparse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Scan OpenMX block headers without building tensors, then export raw "
            "edge-order diagnostics for one matrix/key."
        )
    )
    parser.add_argument(
        "--snapshot-path",
        type=Path,
        default=Path("data/small/ZnCu2Sn_SeS_2_scale_1_010"),
        help="Snapshot directory, or a direct matrix file path.",
    )
    parser.add_argument(
        "--matrix-path",
        type=Path,
        default=None,
        help="Optional explicit matrix path. Overrides --snapshot-path discovery.",
    )
    parser.add_argument(
        "--info-path",
        type=Path,
        default=None,
        help="Optional explicit info path. Overrides --snapshot-path discovery.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("eval_outputs/openmx_header_scan"),
        help="Directory where CSV/JSON/PNG outputs are saved.",
    )
    parser.add_argument(
        "--matrix-name",
        type=str,
        default="hamiltonian",
        choices=["hamiltonian", "overlap", "density"],
        help="Which matrix section to export in detail.",
    )
    parser.add_argument(
        "--key",
        type=str,
        default="Zn-Se",
        help="Element-pair key for focused detailed output.",
    )
    parser.add_argument(
        "--top-keys",
        type=int,
        default=15,
        help="How many keys to show in the summary plot.",
    )
    return parser.parse_args()


def _discover_snapshot_paths(
    snapshot_path: Path, matrix_path: Path | None, info_path: Path | None
) -> tuple[Path, Path]:
    if matrix_path is not None and info_path is not None:
        return matrix_path.resolve(), info_path.resolve()
    if matrix_path is not None or info_path is not None:
        raise ValueError("Provide both --matrix-path and --info-path together.")

    if snapshot_path.is_file():
        matrix_candidate = snapshot_path.resolve()
        info_candidate = _discover_info_file(snapshot_path.parent)
        if info_candidate is None:
            raise FileNotFoundError(
                f"Could not infer info file next to matrix file: {snapshot_path}"
            )
        return matrix_candidate, info_candidate.resolve()

    if not snapshot_path.is_dir():
        raise FileNotFoundError(f"Snapshot path does not exist: {snapshot_path}")

    matrix_candidate = (snapshot_path / "HS.out").resolve()
    if not matrix_candidate.exists():
        raise FileNotFoundError(f"Matrix file not found: {matrix_candidate}")

    info_candidate = _discover_info_file(snapshot_path)
    if info_candidate is None:
        raise FileNotFoundError(
            f"Could not infer info file under snapshot directory: {snapshot_path}"
        )
    return matrix_candidate, info_candidate.resolve()


def _discover_info_file(snapshot_dir: Path) -> Path | None:
    preferred = ["ZnCuSeS.out", "SiO2.out", "info.dat", "info.txt"]
    for name in preferred:
        candidate = snapshot_dir / name
        if candidate.exists():
            return candidate
    for candidate in sorted(snapshot_dir.glob("*.out")):
        if candidate.name not in {"HS.out", "log.out"}:
            return candidate
    fallback = snapshot_dir / "log.out"
    if fallback.exists():
        return fallback
    return None


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str))


def _edge_distance(
    edge: tuple[int, int, int, int, int],
    positions: torch.Tensor | None,
    box: torch.Tensor | None,
) -> float | None:
    if positions is None or box is None:
        return None
    sx, sy, sz, src, dst = edge
    shift = torch.tensor([sx, sy, sz], dtype=positions.dtype, device=positions.device)
    disp = positions[dst] - positions[src] + shift @ box
    return float(torch.linalg.norm(disp).item())


def _plot_key_count_summary(
    output_path: Path,
    rows: list[dict[str, Any]],
    *,
    top_keys: int,
    matrix_name: str,
) -> None:
    top = sorted(rows, key=lambda row: (-int(row["count"]), row["key"]))[:top_keys]
    if not top:
        return
    fig, ax = plt.subplots(figsize=(12, 6), constrained_layout=True)
    keys = [row["key"] for row in top][::-1]
    counts = [row["count"] for row in top][::-1]
    ax.barh(keys, counts, color="#2c7fb8", alpha=0.85)
    ax.set_xlabel("Header count")
    ax.set_title(f"Raw OpenMX header counts per key for {matrix_name}")
    ax.grid(axis="x", alpha=0.2)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _plot_selected_key_rank_distance(
    output_path: Path,
    rows: list[dict[str, Any]],
    *,
    matrix_name: str,
    key: str,
) -> None:
    if not rows:
        return
    parser_ranks = [
        int(row["parser_rank"]) for row in rows if row["distance_cif"] is not None
    ]
    parser_dists = [
        row["distance_cif"] for row in rows if row["distance_cif"] is not None
    ]
    if not parser_ranks:
        return
    fig, ax = plt.subplots(figsize=(14, 6), constrained_layout=True)
    ax.plot(parser_ranks, parser_dists, marker="o", color="#d95f02")
    ax.set_xlabel("Parser rank within key after sort by (i, j, rn)")
    ax.set_ylabel("Distance in CIF geometry (A)")
    ax.set_title(f"Parser-order distance profile for {matrix_name}:{key}")
    ax.grid(alpha=0.2)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main() -> None:
    args = setup_argparse()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    matrix_path, info_path = _discover_snapshot_paths(
        args.snapshot_path, args.matrix_path, args.info_path
    )
    print(f"[INFO] matrix={matrix_path}", flush=True)
    print(f"[INFO] info={info_path}", flush=True)

    info = parse_info_out(info_path, dtype=torch.float64)
    atoms = list(info.elements)
    orbital_cfg = OrbitalIrrepConfig.from_dict(info.orbital_set)
    mapper = BlockIrrepMapper(
        orbital_cfg, diagonal=False, device="cpu", dtype=torch.float32
    )

    cif_path = info_path.with_suffix(".cif")
    cif_positions: torch.Tensor | None = None
    cif_box: torch.Tensor | None = None
    atom_image_shifts: torch.Tensor | None = None
    if cif_path.exists():
        cif = ase_read(cif_path)
        cif_atoms = list(cif.get_chemical_symbols())
        if cif_atoms != atoms:
            raise RuntimeError(
                f"CIF atom order mismatch for {cif_path}: {cif_atoms[:8]} != {atoms[:8]}"
            )
        cif_positions = torch.tensor(cif.get_positions(), dtype=torch.float64)
        cif_box = torch.tensor(cif.cell.array, dtype=torch.float64)
        atom_image_shifts = _compute_atom_image_shifts(
            info.positions.to(dtype=torch.float64),
            cif_positions,
            cif_box,
        )

    rn_shift_map: dict[int, tuple[int, int, int]] = {}
    raw_records: list[dict[str, Any]] = []
    current: str | None = None
    density_seen = False
    file_order = 0

    with matrix_path.open() as fh:
        line_iter = iter(fh)
        for raw_line in line_iter:
            line = raw_line.strip()
            if SECTION_RE.match(line):
                if line.startswith("Density"):
                    if density_seen:
                        current = None
                        continue
                    density_seen = True
                    current = "density"
                elif line.startswith("Kohn-Sham"):
                    current = "hamiltonian"
                elif line.startswith("Overlap matrix with position operator x"):
                    current = "overlap position x"
                else:
                    current = "overlap"
                continue

            if SKIP_OVERLAP_RE.match(line) and current != "overlap position x":
                current = None
                continue
            if current is None:
                continue

            if current == "overlap position x":
                match = HEADER_OVERLAP_POS_X_RE.match(line)
                if match:
                    rn = int(match.group(3))
                    rn_shift_map[rn] = (
                        int(match.group(4)),
                        int(match.group(5)),
                        int(match.group(6)),
                    )
                continue

            match = HEADER_RE.match(line)
            if match is None:
                continue

            src = int(match.group(1)) - 1
            dst = int(match.group(2)) - 1
            rn = int(match.group(3))
            key = f"{atoms[src]}-{atoms[dst]}"
            d_i, _d_j = mapper.block_dims(key)
            raw_records.append(
                {
                    "file_order": file_order,
                    "matrix_name": current,
                    "key": key,
                    "src": src,
                    "dst": dst,
                    "rn": rn,
                }
            )
            file_order += 1

            rows_read = 0
            while rows_read < d_i:
                data_line = next(line_iter).strip()
                if data_line:
                    rows_read += 1

    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in raw_records:
        grouped[(record["matrix_name"], record["key"])].append(record)

    detailed_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for (matrix_name, key), records in grouped.items():
        parser_sorted = sorted(
            records, key=lambda row: (row["src"], row["dst"], row["rn"])
        )
        for parser_rank, record in enumerate(parser_sorted):
            rn = int(record["rn"])
            shift = rn_shift_map.get(rn)
            sx = sy = sz = None
            rel_sx = rel_sy = rel_sz = None
            distance_cif = None
            if shift is not None:
                sx, sy, sz = shift
                if atom_image_shifts is not None:
                    src_shift = atom_image_shifts[int(record["src"])]
                    dst_shift = atom_image_shifts[int(record["dst"])]
                    rel_sx = sx + int(src_shift[0].item()) - int(dst_shift[0].item())
                    rel_sy = sy + int(src_shift[1].item()) - int(dst_shift[1].item())
                    rel_sz = sz + int(src_shift[2].item()) - int(dst_shift[2].item())
                    distance_cif = _edge_distance(
                        (
                            rel_sx,
                            rel_sy,
                            rel_sz,
                            int(record["src"]),
                            int(record["dst"]),
                        ),
                        cif_positions,
                        cif_box,
                    )
            detailed_rows.append(
                {
                    "matrix_name": matrix_name,
                    "key": key,
                    "file_order": int(record["file_order"]),
                    "parser_rank": parser_rank,
                    "src": int(record["src"]),
                    "dst": int(record["dst"]),
                    "rn": rn,
                    "raw_shift_x": sx,
                    "raw_shift_y": sy,
                    "raw_shift_z": sz,
                    "relabeled_shift_x": rel_sx,
                    "relabeled_shift_y": rel_sy,
                    "relabeled_shift_z": rel_sz,
                    "distance_cif": distance_cif,
                }
            )
        summary_rows.append(
            {"matrix_name": matrix_name, "key": key, "count": len(records)}
        )

    selected_rows = [
        row
        for row in detailed_rows
        if row["matrix_name"] == args.matrix_name and row["key"] == args.key
    ]
    if not selected_rows:
        available = sorted(
            {
                row["key"]
                for row in summary_rows
                if row["matrix_name"] == args.matrix_name
            }
        )
        raise KeyError(
            f"Key {args.key!r} not found in matrix {args.matrix_name}. "
            f"Available keys: {', '.join(available)}"
        )

    _write_csv(
        output_dir / "header_summary.csv",
        summary_rows,
        ["matrix_name", "key", "count"],
    )
    _write_csv(
        output_dir / f"{args.matrix_name}_{args.key.replace('-', '_')}_headers.csv",
        selected_rows,
        [
            "matrix_name",
            "key",
            "file_order",
            "parser_rank",
            "src",
            "dst",
            "rn",
            "raw_shift_x",
            "raw_shift_y",
            "raw_shift_z",
            "relabeled_shift_x",
            "relabeled_shift_y",
            "relabeled_shift_z",
            "distance_cif",
        ],
    )

    summary_payload = {
        "matrix_path": matrix_path,
        "info_path": info_path,
        "cif_path": cif_path if cif_path.exists() else None,
        "rn_shift_count": len(rn_shift_map),
        "matrix_name": args.matrix_name,
        "selected_key": args.key,
        "selected_key_count": len(selected_rows),
        "atom_shift_histogram": (
            {
                str(tuple(int(v) for v in shift.tolist())): int(
                    (atom_image_shifts == shift).all(dim=1).sum().item()
                )
                for shift in atom_image_shifts.unique(dim=0)
            }
            if atom_image_shifts is not None
            else None
        ),
        "counts_by_matrix": {
            matrix_name: int(
                sum(
                    row["count"]
                    for row in summary_rows
                    if row["matrix_name"] == matrix_name
                )
            )
            for matrix_name in ("hamiltonian", "overlap", "density")
        },
    }
    _write_json(output_dir / "header_summary.json", summary_payload)

    matrix_summary_rows = [
        row for row in summary_rows if row["matrix_name"] == args.matrix_name
    ]
    _plot_key_count_summary(
        output_dir / f"{args.matrix_name}_key_counts.png",
        matrix_summary_rows,
        top_keys=args.top_keys,
        matrix_name=args.matrix_name,
    )
    _plot_selected_key_rank_distance(
        output_dir
        / f"{args.matrix_name}_{args.key.replace('-', '_')}_rank_distance.png",
        selected_rows,
        matrix_name=args.matrix_name,
        key=args.key,
    )

    print("", flush=True)
    print("[SUMMARY] top keys", flush=True)
    for row in sorted(
        matrix_summary_rows, key=lambda row: (-int(row["count"]), row["key"])
    )[: args.top_keys]:
        print(f"  {row['key']:>8s}  count={row['count']}", flush=True)

    print("", flush=True)
    print(
        f"[DONE] wrote raw header diagnostics to {output_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
