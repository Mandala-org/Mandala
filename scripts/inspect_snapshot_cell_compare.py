from __future__ import annotations

import argparse
import sys
from pathlib import Path

import plotly.graph_objects as go
import torch
from ase.io import read as ase_read

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from data.openmx_info_parser import parse_info_out, recover_box


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare round-trip positions using the CIF box (A) and the box inferred "
            "from OpenMX cartesian+fractional coordinates (B)."
        )
    )
    parser.add_argument(
        "input_path",
        type=Path,
        help="Path to either the .out or .cif file in the snapshot directory.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Directory for the generated Plotly HTML output.",
    )
    return parser.parse_args()


def _box_edges(box: torch.Tensor) -> list[tuple[list[float], list[float]]]:
    origin = torch.zeros(3, dtype=box.dtype)
    v1, v2, v3 = box[0], box[1], box[2]
    corners = {
        "000": origin,
        "100": v1,
        "010": v2,
        "001": v3,
        "110": v1 + v2,
        "101": v1 + v3,
        "011": v2 + v3,
        "111": v1 + v2 + v3,
    }
    edges = [
        ("000", "100"),
        ("000", "010"),
        ("000", "001"),
        ("100", "110"),
        ("100", "101"),
        ("010", "110"),
        ("010", "011"),
        ("001", "101"),
        ("001", "011"),
        ("110", "111"),
        ("101", "111"),
        ("011", "111"),
    ]
    return [(corners[a].tolist(), corners[b].tolist()) for a, b in edges]


def _add_box(
    fig: go.Figure, box: torch.Tensor, *, name: str, color: str, dash: str
) -> None:
    for start, end in _box_edges(box):
        fig.add_trace(
            go.Scatter3d(
                x=[start[0], end[0]],
                y=[start[1], end[1]],
                z=[start[2], end[2]],
                mode="lines",
                line=dict(color=color, width=6, dash=dash),
                name=name,
                showlegend=False,
                hoverinfo="skip",
            )
        )


def _load_inputs(
    input_path: Path,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    list[str],
    str,
    str,
]:
    if input_path.suffix not in {".out", ".cif"}:
        raise ValueError(f"Expected a .out or .cif file, got {input_path}")

    if input_path.suffix == ".out":
        companion_path = input_path.with_suffix(".cif")
        if not companion_path.exists():
            raise FileNotFoundError(f"Matching CIF file not found: {companion_path}")
        info = parse_info_out(input_path, dtype=torch.float64)
        companion = ase_read(companion_path)
        pos_gt = torch.tensor(companion.get_positions(), dtype=torch.float64)
        box_A = torch.tensor(companion.cell.array, dtype=torch.float64)
        pos_out = info.positions.to(dtype=torch.float64)
        frac_out = info.frac.to(dtype=torch.float64)
        box_B = recover_box(frac_out, pos_out).to(dtype=torch.float64)
        symbols = companion.get_chemical_symbols()
        label = f"{input_path.name} / {companion_path.name}"
        ref_label = "CIF box (A)"
        other_label = "Inferred box (B) from .out"
    else:
        companion_path = input_path.with_suffix(".out")
        if not companion_path.exists():
            raise FileNotFoundError(
                f"Matching OpenMX .out file not found: {companion_path}"
            )
        cif = ase_read(input_path)
        info = parse_info_out(companion_path, dtype=torch.float64)
        pos_gt = torch.tensor(cif.get_positions(), dtype=torch.float64)
        box_A = torch.tensor(cif.cell.array, dtype=torch.float64)
        pos_out = info.positions.to(dtype=torch.float64)
        frac_out = info.frac.to(dtype=torch.float64)
        box_B = recover_box(frac_out, pos_out).to(dtype=torch.float64)
        symbols = cif.get_chemical_symbols()
        label = f"{input_path.name} / {companion_path.name}"
        ref_label = "CIF box (A)"
        other_label = "Inferred box (B) from .out"

    return (
        pos_gt,
        box_A,
        box_B,
        symbols,
        label,
        ref_label,
        other_label,
    )


def _round_trip_positions(
    positions: torch.Tensor,
    box: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    inv_box = torch.linalg.inv(box)
    frac = positions @ inv_box
    recon = frac @ box
    return frac, recon


def _print_diff(
    *,
    title: str,
    pos_rt: torch.Tensor,
    pos_gt: torch.Tensor,
    symbols: list[str],
) -> None:
    delta = pos_rt - pos_gt
    print(title)
    print("idx symbol  dx           dy           dz           |d|")
    for i, sym in enumerate(symbols):
        d = delta[i]
        print(
            f"{i:2d} {sym:2s} "
            f"{d[0]: .10f} {d[1]: .10f} {d[2]: .10f} "
            f"{float(torch.linalg.norm(d).item()): .10f}"
        )
    print()
    print("max abs component:", float(delta.abs().max().item()))
    print("max norm:", float(torch.linalg.norm(delta, dim=1).max().item()))


def main() -> None:
    args = _parse_args()
    input_path = args.input_path.expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    (
        pos_gt,
        box_A,
        box_B,
        symbols,
        label,
        ref_label,
        other_label,
    ) = _load_inputs(input_path)

    print(f"Reference pair: {label}")
    print()
    print(f"{ref_label}:")
    print(box_A)
    print()
    print(f"{other_label}:")
    print(box_B)
    print()
    print("Max abs box diff:", float(torch.max(torch.abs(box_A - box_B)).item()))
    print()
    frac_A, pos_A = _round_trip_positions(pos_gt, box_A)
    frac_B, pos_B = _round_trip_positions(pos_gt, box_B)
    _print_diff(title="pos_A - pos_gt", pos_rt=pos_A, pos_gt=pos_gt, symbols=symbols)
    print()
    _print_diff(title="pos_B - pos_gt", pos_rt=pos_B, pos_gt=pos_gt, symbols=symbols)
    print()
    print("max abs frac_A component:", float(frac_A.abs().max().item()))
    print("max abs frac_B component:", float(frac_B.abs().max().item()))

    fig = go.Figure()
    _add_box(fig, box_A, name="box_A", color="royalblue", dash="solid")
    _add_box(fig, box_B, name="box_B", color="crimson", dash="dash")

    for symbol in sorted(set(symbols)):
        idxs = [i for i, s in enumerate(symbols) if s == symbol]
        pts = pos_gt[idxs]
        fig.add_trace(
            go.Scatter3d(
                x=pts[:, 0].tolist(),
                y=pts[:, 1].tolist(),
                z=pts[:, 2].tolist(),
                mode="markers",
                marker=dict(size=5),
                name=symbol,
                text=[f"{symbol} #{i}" for i in idxs],
            )
        )

    fig.update_layout(
        title=f"{input_path.name}: box_A vs box_B",
        scene=dict(
            aspectmode="data",
            xaxis_title="x",
            yaxis_title="y",
            zaxis_title="z",
        ),
        legend_title_text="Elements / boxes",
        margin=dict(l=0, r=0, t=50, b=0),
    )

    out_dir = (
        args.out_dir
        if args.out_dir is not None
        else Path("eval_outputs") / input_path.parent.name
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{input_path.stem}_box_compare.html"
    fig.write_html(out_path, include_plotlyjs="cdn")
    print(f"Saved Plotly visualization to: {out_path}")


if __name__ == "__main__":
    main()
