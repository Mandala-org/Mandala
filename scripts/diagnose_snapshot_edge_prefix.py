#!/usr/bin/env python

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-codex")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import torch
from ase.io import read as ase_read
from e3nn.o3 import Irreps

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from core.block_irrep_mapper import BlockIrrepMapper  # noqa: E402
from core.orbital_irrep_config import OrbitalIrrepConfig  # noqa: E402
from data.graph_features import compute_graph_features  # noqa: E402
from data.openmx_info_parser import parse_info_out, recover_box  # noqa: E402
from data.openmx_parser import parse_openmx_scfout  # noqa: E402
from data.snapshot import Snapshot, _compute_atom_image_shifts  # noqa: E402
from net.common import Config  # noqa: E402


Edge5 = tuple[int, int, int, int, int]


def setup_argparse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Deep diagnostics for matrix-edge prefix ordering versus geometry-built "
            "graph ordering on a single OpenMX snapshot."
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
        default=Path("eval_outputs/snapshot_edge_prefix_diag"),
        help="Directory where diagnostic tables, plots, and HTML files are saved.",
    )
    parser.add_argument(
        "--convention",
        type=str,
        default="e3nn",
        choices=["openmx", "e3nn"],
        help="Matrix convention used for the final target snapshot.",
    )
    parser.add_argument(
        "--cutoff-radius",
        type=float,
        default=11.0,
        help="Cutoff used when rebuilding the geometry graph.",
    )
    parser.add_argument(
        "--l-max",
        type=int,
        default=5,
        help="Spherical harmonics l_max for graph feature construction.",
    )
    parser.add_argument(
        "--n-radial",
        type=int,
        default=8,
        help="Radial basis count for graph feature construction.",
    )
    parser.add_argument(
        "--matrix-name",
        type=str,
        default="hamiltonian",
        choices=["hamiltonian", "overlap", "density"],
        help="Target matrix used for detailed prefix comparison output.",
    )
    parser.add_argument(
        "--key",
        type=str,
        default="Zn-Se",
        help="Element-pair key for focused detailed diagnostics.",
    )
    parser.add_argument(
        "--focus-rank",
        type=int,
        default=None,
        help="Optional explicit focused rank within --key. Defaults to first mismatch.",
    )
    parser.add_argument(
        "--context",
        type=int,
        default=6,
        help="Number of rows around the focused rank used in text output and 3D view.",
    )
    parser.add_argument(
        "--top-keys",
        type=int,
        default=15,
        help="How many keys to show in console summaries and bar plots.",
    )
    parser.add_argument(
        "--max-key-rows",
        type=int,
        default=400,
        help="Maximum number of focused key rows written to CSV.",
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


def _edge_distance(
    edge: Edge5, positions: torch.Tensor, box: torch.Tensor | None
) -> float:
    sx, sy, sz, src, dst = edge
    disp = positions[dst] - positions[src]
    if box is not None:
        shift = torch.tensor(
            [sx, sy, sz], dtype=positions.dtype, device=positions.device
        )
        disp = disp + shift @ box
    return float(torch.linalg.norm(disp).item())


def _edge_displacement(
    edge: Edge5, positions: torch.Tensor, box: torch.Tensor | None
) -> torch.Tensor:
    sx, sy, sz, src, dst = edge
    disp = positions[dst] - positions[src]
    if box is not None:
        shift = torch.tensor(
            [sx, sy, sz], dtype=positions.dtype, device=positions.device
        )
        disp = disp + shift @ box
    return disp


def _edge_to_dict(edge: Edge5 | None) -> dict[str, int] | None:
    if edge is None:
        return None
    sx, sy, sz, src, dst = edge
    return {"sx": sx, "sy": sy, "sz": sz, "src": src, "dst": dst}


def _edge_to_str(edge: Edge5 | None) -> str:
    if edge is None:
        return "-"
    return f"({edge[0]}, {edge[1]}, {edge[2]}, {edge[3]}, {edge[4]})"


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, set):
        return sorted(value)
    raise TypeError(f"Unsupported JSON type: {type(value)!r}")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=_json_default)
    )


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _group_graph_edges_by_key(
    edge_index: torch.Tensor,
    edge_shift: torch.Tensor,
    atoms: tuple[str, ...],
) -> dict[str, list[Edge5]]:
    grouped: dict[str, list[Edge5]] = defaultdict(list)
    n_edges = int(edge_index.shape[1])
    for e in range(n_edges):
        src = int(edge_index[0, e].item())
        dst = int(edge_index[1, e].item())
        edge = (
            int(edge_shift[0, e].item()),
            int(edge_shift[1, e].item()),
            int(edge_shift[2, e].item()),
            src,
            dst,
        )
        key = f"{atoms[src]}-{atoms[dst]}"
        grouped[key].append(edge)
    return grouped


def _counter_subset(left: Counter[Edge5], right: Counter[Edge5]) -> bool:
    for edge, count in left.items():
        if right[edge] < count:
            return False
    return True


def _compare_edge_lists(
    *,
    key: str,
    target_edges: list[Edge5],
    graph_edges: list[Edge5],
    positions: torch.Tensor,
    box: torch.Tensor | None,
) -> dict[str, Any]:
    prefix_len = min(len(target_edges), len(graph_edges))
    mismatch_indices = [
        idx for idx in range(prefix_len) if target_edges[idx] != graph_edges[idx]
    ]
    first_mismatch_idx = mismatch_indices[0] if mismatch_indices else None
    target_counter = Counter(target_edges)
    graph_counter = Counter(graph_edges)
    graph_prefix_counter = Counter(graph_edges[: len(target_edges)])

    stats: dict[str, Any] = {
        "key": key,
        "target_len": len(target_edges),
        "graph_len": len(graph_edges),
        "prefix_exact": len(mismatch_indices) == 0
        and len(graph_edges) >= len(target_edges),
        "prefix_mismatch_count": len(mismatch_indices),
        "first_mismatch_idx": first_mismatch_idx,
        "target_subset_of_graph_multiset": _counter_subset(
            target_counter, graph_counter
        ),
        "target_equals_graph_prefix_multiset": (
            len(graph_edges) >= len(target_edges)
            and target_counter == graph_prefix_counter
        ),
        "target_multiset_cardinality": sum(target_counter.values()),
        "graph_multiset_cardinality": sum(graph_counter.values()),
    }

    if first_mismatch_idx is not None:
        target_edge = target_edges[first_mismatch_idx]
        graph_edge = graph_edges[first_mismatch_idx]
        stats["first_mismatch"] = {
            "target_edge": _edge_to_dict(target_edge),
            "graph_edge": _edge_to_dict(graph_edge),
            "target_distance": _edge_distance(target_edge, positions, box),
            "graph_distance": _edge_distance(graph_edge, positions, box),
        }
    else:
        stats["first_mismatch"] = None

    return stats


def _relabel_graph_edges_to_reference(
    grouped_edges: dict[str, list[Edge5]],
    atom_image_shifts: torch.Tensor,
) -> dict[str, list[Edge5]]:
    relabeled: dict[str, list[Edge5]] = {}
    for key, edges in grouped_edges.items():
        new_edges: list[Edge5] = []
        for sx, sy, sz, src, dst in edges:
            src_shift = atom_image_shifts[src]
            dst_shift = atom_image_shifts[dst]
            new_edges.append(
                (
                    sx - int(src_shift[0].item()) + int(dst_shift[0].item()),
                    sy - int(src_shift[1].item()) + int(dst_shift[1].item()),
                    sz - int(src_shift[2].item()) + int(dst_shift[2].item()),
                    src,
                    dst,
                )
            )
        relabeled[key] = new_edges
    return relabeled


def _graph_order_invariance_summary(
    graph_a: dict[str, list[Edge5]],
    graph_b_relabeled: dict[str, list[Edge5]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    all_keys = sorted(set(graph_a) | set(graph_b_relabeled))
    per_key: list[dict[str, Any]] = []
    mismatch_rows: list[dict[str, Any]] = []
    total_position_mismatches = 0

    for key in all_keys:
        edges_a = graph_a.get(key, [])
        edges_b = graph_b_relabeled.get(key, [])
        prefix_len = min(len(edges_a), len(edges_b))
        mismatch_indices = [
            idx for idx in range(prefix_len) if edges_a[idx] != edges_b[idx]
        ]
        total_position_mismatches += len(mismatch_indices)
        per_key.append(
            {
                "key": key,
                "len_a": len(edges_a),
                "len_b_relabeled": len(edges_b),
                "ordered_equal": edges_a == edges_b,
                "multiset_equal": Counter(edges_a) == Counter(edges_b),
                "mismatch_count": len(mismatch_indices),
                "first_mismatch_idx": mismatch_indices[0] if mismatch_indices else None,
            }
        )
        for idx in mismatch_indices[:64]:
            mismatch_rows.append(
                {
                    "key": key,
                    "rank": idx,
                    "edge_a": _edge_to_str(edges_a[idx]),
                    "edge_b_relabeled": _edge_to_str(edges_b[idx]),
                }
            )

    summary = {
        "keys_checked": len(all_keys),
        "keys_with_order_mismatch": sum(
            1 for row in per_key if not row["ordered_equal"]
        ),
        "keys_with_multiset_mismatch": sum(
            1 for row in per_key if not row["multiset_equal"]
        ),
        "total_position_mismatches": total_position_mismatches,
    }
    return summary, per_key + mismatch_rows[:0]  # keep stable shape for JSON callers


def _pair_multiplicity_histogram(
    grouped_edges: dict[str, list[Edge5]],
) -> dict[tuple[int, int], int]:
    counts: Counter[tuple[int, int]] = Counter()
    for edges in grouped_edges.values():
        for _sx, _sy, _sz, src, dst in edges:
            if src != dst:
                counts[(src, dst)] += 1
    return dict(counts)


def _build_snapshot_like_loader(
    matrix_path: Path,
    info_path: Path,
    *,
    convention: str,
    info_positions: torch.Tensor,
    info_box: torch.Tensor,
    cif_positions: torch.Tensor | None,
    cif_box: torch.Tensor | None,
    atom_image_shifts: torch.Tensor | None,
) -> tuple[Snapshot, Snapshot, Snapshot]:
    info = parse_info_out(info_path, dtype=torch.float64)
    orbital_cfg = OrbitalIrrepConfig.from_dict(info.orbital_set)
    raw_openmx = parse_openmx_scfout(
        matrix_path,
        info.elements,
        orbital_cfg,
        convention="openmx",
        symmetrize_density=True,
    )
    raw_openmx.matrix_path = matrix_path
    raw_openmx.info_path = info_path

    if (
        cif_positions is not None
        and cif_box is not None
        and atom_image_shifts is not None
    ):
        relabeled = raw_openmx.relabel_atom_images(atom_image_shifts)
        relabeled.positions = cif_positions
        relabeled.box = cif_box
    else:
        relabeled = raw_openmx
        relabeled.positions = info_positions
        relabeled.box = info_box

    relabeled.matrix_path = matrix_path
    relabeled.info_path = info_path
    converted = relabeled._change_basis(convention)
    converted.matrix_path = matrix_path
    converted.info_path = info_path
    canonical = converted.canonicalize_edges()
    canonical.matrix_path = matrix_path
    canonical.info_path = info_path
    return raw_openmx, converted, canonical


def _box_edges(box: torch.Tensor) -> list[tuple[list[float], list[float]]]:
    origin = torch.zeros(3, dtype=box.dtype, device=box.device)
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
    edge_names = [
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
    return [
        (corners[a].detach().cpu().tolist(), corners[b].detach().cpu().tolist())
        for a, b in edge_names
    ]


def _write_report(path: Path, lines: list[str]) -> None:
    path.write_text("\n".join(lines) + "\n")


def _plot_prefix_mismatch_counts(
    output_path: Path,
    per_key_rows: list[dict[str, Any]],
    *,
    top_keys: int,
) -> None:
    rows = sorted(
        [row for row in per_key_rows if row["prefix_mismatch_count"] > 0],
        key=lambda row: (-int(row["prefix_mismatch_count"]), row["key"]),
    )[:top_keys]
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(12, 6), constrained_layout=True)
    keys = [row["key"] for row in rows][::-1]
    counts = [row["prefix_mismatch_count"] for row in rows][::-1]
    ax.barh(keys, counts, color="#c0392b", alpha=0.85)
    ax.set_xlabel("Mismatch count within target prefix")
    ax.set_title("Per-key prefix mismatch counts")
    ax.grid(axis="x", alpha=0.2)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _plot_pair_multiplicity_histogram(
    output_path: Path,
    pair_mult: dict[tuple[int, int], int],
) -> None:
    if not pair_mult:
        return
    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    values = list(pair_mult.values())
    bins = range(min(values), max(values) + 2)
    ax.hist(values, bins=bins, align="left", color="#2c7fb8", alpha=0.85, rwidth=0.9)
    ax.set_xlabel("Periodic-image multiplicity per ordered atom pair")
    ax.set_ylabel("Count")
    ax.set_title("Graph multiplicity regime under cutoff")
    ax.grid(axis="y", alpha=0.2)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _plot_selected_key_rank_distances(
    output_path: Path,
    rows: list[dict[str, Any]],
    *,
    key: str,
    focus_rank: int | None,
) -> None:
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(14, 6), constrained_layout=True)
    ranks = [int(row["rank"]) for row in rows]
    target_dist = [row["target_distance"] for row in rows]
    graph_dist = [row["graph_distance"] for row in rows]
    ax.plot(ranks, target_dist, label="target", color="#1b9e77", linewidth=1.8)
    ax.plot(ranks, graph_dist, label="graph", color="#d95f02", linewidth=1.4)
    mismatch_ranks = [int(row["rank"]) for row in rows if not row["match"]]
    mismatch_target = [row["target_distance"] for row in rows if not row["match"]]
    ax.scatter(
        mismatch_ranks,
        mismatch_target,
        color="#b2182b",
        s=26,
        label="mismatch rank",
        zorder=3,
    )
    if focus_rank is not None:
        ax.axvline(focus_rank, color="black", linestyle="--", linewidth=1.0)
    ax.set_xlabel(f"Rank within key {key}")
    ax.set_ylabel("Distance (A)")
    ax.set_title(f"Target vs graph rank-distance ordering for {key}")
    ax.grid(alpha=0.2)
    ax.legend()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _plot_selected_key_zoom(
    output_path: Path,
    rows: list[dict[str, Any]],
    *,
    key: str,
    focus_rank: int | None,
    context: int,
) -> None:
    if not rows or focus_rank is None:
        return
    lo = max(0, focus_rank - context)
    hi = min(len(rows), focus_rank + context + 1)
    sub = rows[lo:hi]
    fig, ax = plt.subplots(figsize=(12, 6), constrained_layout=True)
    ranks = [int(row["rank"]) for row in sub]
    target_dist = [row["target_distance"] for row in sub]
    graph_dist = [row["graph_distance"] for row in sub]
    ax.plot(ranks, target_dist, marker="o", label="target", color="#1b9e77")
    ax.plot(ranks, graph_dist, marker="o", label="graph", color="#d95f02")
    for row in sub:
        if not row["match"]:
            ax.axvspan(
                int(row["rank"]) - 0.35,
                int(row["rank"]) + 0.35,
                color="#fddbc7",
                alpha=0.35,
            )
    ax.axvline(focus_rank, color="black", linestyle="--", linewidth=1.0)
    ax.set_xlabel(f"Rank within key {key}")
    ax.set_ylabel("Distance (A)")
    ax.set_title(f"Focused mismatch window for {key}")
    ax.grid(alpha=0.2)
    ax.legend()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _write_focus_scene_html(
    output_path: Path,
    *,
    positions: torch.Tensor,
    box: torch.Tensor,
    atoms: tuple[str, ...],
    atom_image_shifts: torch.Tensor | None,
    rows: list[dict[str, Any]],
    focus_rank: int | None,
    context: int,
) -> None:
    if not rows or focus_rank is None:
        return

    lo = max(0, focus_rank - context)
    hi = min(len(rows), focus_rank + context + 1)
    sub = rows[lo:hi]

    fig = go.Figure()
    for start, end in _box_edges(box):
        fig.add_trace(
            go.Scatter3d(
                x=[start[0], end[0]],
                y=[start[1], end[1]],
                z=[start[2], end[2]],
                mode="lines",
                line=dict(color="rgba(90,90,90,0.5)", width=4),
                hoverinfo="skip",
                showlegend=False,
            )
        )

    for symbol in sorted(set(atoms)):
        idxs = [idx for idx, atom in enumerate(atoms) if atom == symbol]
        pts = positions[idxs]
        fig.add_trace(
            go.Scatter3d(
                x=pts[:, 0].detach().cpu().tolist(),
                y=pts[:, 1].detach().cpu().tolist(),
                z=pts[:, 2].detach().cpu().tolist(),
                mode="markers+text",
                marker=dict(size=4, opacity=0.45),
                text=[f"{symbol}{idx}" for idx in idxs],
                textposition="top center",
                name=symbol,
            )
        )

    if atom_image_shifts is not None:
        shifted = torch.nonzero(
            (atom_image_shifts != 0).any(dim=1), as_tuple=False
        ).flatten()
        if shifted.numel() > 0:
            pts = positions[shifted]
            labels = [
                f"{atoms[int(idx)]}{int(idx)} shift={atom_image_shifts[int(idx)].tolist()}"
                for idx in shifted.tolist()
            ]
            fig.add_trace(
                go.Scatter3d(
                    x=pts[:, 0].detach().cpu().tolist(),
                    y=pts[:, 1].detach().cpu().tolist(),
                    z=pts[:, 2].detach().cpu().tolist(),
                    mode="markers+text",
                    marker=dict(size=7, color="#e41a1c", symbol="diamond"),
                    text=labels,
                    textposition="bottom center",
                    name="shifted atoms",
                )
            )

    target_x: list[float | None] = []
    target_y: list[float | None] = []
    target_z: list[float | None] = []
    graph_x: list[float | None] = []
    graph_y: list[float | None] = []
    graph_z: list[float | None] = []

    for row in sub:
        target_edge = row["target_edge_tuple"]
        graph_edge = row["graph_edge_tuple"]
        target_disp = _edge_displacement(target_edge, positions, box)
        graph_disp = _edge_displacement(graph_edge, positions, box)
        src = target_edge[3]
        src_pos = positions[src]
        target_end = src_pos + target_disp
        graph_end = positions[graph_edge[3]] + graph_disp
        target_x.extend([float(src_pos[0]), float(target_end[0]), None])
        target_y.extend([float(src_pos[1]), float(target_end[1]), None])
        target_z.extend([float(src_pos[2]), float(target_end[2]), None])
        graph_x.extend([float(positions[graph_edge[3], 0]), float(graph_end[0]), None])
        graph_y.extend([float(positions[graph_edge[3], 1]), float(graph_end[1]), None])
        graph_z.extend([float(positions[graph_edge[3], 2]), float(graph_end[2]), None])

    fig.add_trace(
        go.Scatter3d(
            x=target_x,
            y=target_y,
            z=target_z,
            mode="lines",
            line=dict(color="#1b9e77", width=8),
            name="target edges",
        )
    )
    fig.add_trace(
        go.Scatter3d(
            x=graph_x,
            y=graph_y,
            z=graph_z,
            mode="lines",
            line=dict(color="#d95f02", width=6, dash="dash"),
            name="graph edges",
        )
    )

    focus_row = rows[focus_rank]
    focus_target = focus_row["target_edge_tuple"]
    focus_graph = focus_row["graph_edge_tuple"]
    target_disp = _edge_displacement(focus_target, positions, box)
    graph_disp = _edge_displacement(focus_graph, positions, box)
    focus_points = [
        (
            positions[focus_target[3]],
            f"target src {atoms[focus_target[3]]}{focus_target[3]}",
            "#377eb8",
        ),
        (
            positions[focus_target[3]] + target_disp,
            f"target dst image {atoms[focus_target[4]]}{focus_target[4]}",
            "#1b9e77",
        ),
        (
            positions[focus_graph[3]] + graph_disp,
            f"graph dst image {atoms[focus_graph[4]]}{focus_graph[4]}",
            "#d95f02",
        ),
    ]
    for point, label, color in focus_points:
        fig.add_trace(
            go.Scatter3d(
                x=[float(point[0])],
                y=[float(point[1])],
                z=[float(point[2])],
                mode="markers+text",
                marker=dict(size=9, color=color),
                text=[label],
                textposition="top center",
                name=label,
            )
        )

    fig.update_layout(
        title="Focused edge mismatch neighborhood",
        scene=dict(
            aspectmode="data", xaxis_title="x", yaxis_title="y", zaxis_title="z"
        ),
        margin=dict(l=0, r=0, t=48, b=0),
    )
    fig.write_html(output_path, include_plotlyjs="cdn")


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
    if not info.positions.numel():
        raise RuntimeError(f"Info file does not contain positions: {info_path}")
    if not info.box.numel():
        raise RuntimeError(f"Info file does not contain recoverable box: {info_path}")

    info_positions = info.positions.to(dtype=torch.float64)
    info_box = info.box.to(dtype=torch.float64)
    info_box_recovered = recover_box(info.frac.to(dtype=torch.float64), info_positions)

    cif_path = info_path.with_suffix(".cif")
    cif_positions: torch.Tensor | None = None
    cif_box: torch.Tensor | None = None
    atoms_tuple: tuple[str, ...] = tuple(info.elements)
    atom_image_shifts: torch.Tensor | None = None
    max_atom_shift_residual: float | None = None

    if cif_path.exists():
        cif = ase_read(cif_path)
        cif_atoms = tuple(cif.get_chemical_symbols())
        if cif_atoms != atoms_tuple:
            raise RuntimeError(
                f"CIF atom order mismatch for {cif_path}: {cif_atoms[:8]} != {atoms_tuple[:8]}"
            )
        cif_positions = torch.tensor(cif.get_positions(), dtype=torch.float64)
        cif_box = torch.tensor(cif.cell.array, dtype=torch.float64)
        delta_frac = (cif_positions - info_positions) @ torch.linalg.inv(cif_box)
        atom_image_shifts = torch.round(delta_frac).to(dtype=torch.long)
        max_atom_shift_residual = float(
            torch.max(
                torch.abs(delta_frac - atom_image_shifts.to(delta_frac.dtype))
            ).item()
        )
        atom_image_shifts = _compute_atom_image_shifts(
            info_positions, cif_positions, cif_box
        )
    else:
        print(f"[WARN] No CIF found next to info file: {cif_path}", flush=True)

    raw_openmx, relabeled_precanonical, final_snapshot = _build_snapshot_like_loader(
        matrix_path,
        info_path,
        convention=args.convention,
        info_positions=info_positions,
        info_box=info_box,
        cif_positions=cif_positions,
        cif_box=cif_box,
        atom_image_shifts=atom_image_shifts,
    )

    cfg = Config(
        cutoff_radius=args.cutoff_radius,
        l_max=args.l_max,
        n_radial=args.n_radial,
        safety_checks=True,
        radial_embedding_scale="none",
    )
    mapper = BlockIrrepMapper(
        final_snapshot.hamiltonian.orbital_cfg,
        diagonal=False,
        device="cpu",
        dtype=torch.float32,
    )
    sh_irreps = Irreps.spherical_harmonics(cfg.l_max)

    edge_index_final, edge_shift_final, edge_type_idx_final, *_ = (
        compute_graph_features(
            positions=final_snapshot.positions,
            box=final_snapshot.box,
            atoms=final_snapshot.density.atoms,
            cfg=cfg,
            sh_irreps=sh_irreps,
            edge_type2idx=mapper.edge_type2idx,
        )
    )
    graph_final = _group_graph_edges_by_key(
        edge_index_final,
        edge_shift_final,
        tuple(final_snapshot.density.atoms),
    )

    graph_cif_openmx: dict[str, list[Edge5]] | None = None
    graph_out_openmx: dict[str, list[Edge5]] | None = None
    graph_out_relabeled_to_cif: dict[str, list[Edge5]] | None = None
    graph_invariance_summary: dict[str, Any] | None = None
    graph_invariance_rows: list[dict[str, Any]] = []

    if (
        cif_positions is not None
        and cif_box is not None
        and atom_image_shifts is not None
    ):
        edge_index_cif, edge_shift_cif, *_ = compute_graph_features(
            positions=cif_positions,
            box=cif_box,
            atoms=atoms_tuple,
            cfg=cfg,
            sh_irreps=sh_irreps,
            edge_type2idx=mapper.edge_type2idx,
        )
        edge_index_out, edge_shift_out, *_ = compute_graph_features(
            positions=info_positions,
            box=info_box_recovered,
            atoms=atoms_tuple,
            cfg=cfg,
            sh_irreps=sh_irreps,
            edge_type2idx=mapper.edge_type2idx,
        )
        graph_cif_openmx = _group_graph_edges_by_key(
            edge_index_cif, edge_shift_cif, atoms_tuple
        )
        graph_out_openmx = _group_graph_edges_by_key(
            edge_index_out, edge_shift_out, atoms_tuple
        )
        graph_out_relabeled_to_cif = _relabel_graph_edges_to_reference(
            graph_out_openmx, atom_image_shifts
        )

        graph_inv_rows: list[dict[str, Any]] = []
        keys = sorted(set(graph_cif_openmx) | set(graph_out_relabeled_to_cif))
        total_mismatches = 0
        keys_with_order_mismatch = 0
        keys_with_multiset_mismatch = 0
        for key in keys:
            edges_a = graph_cif_openmx.get(key, [])
            edges_b = graph_out_relabeled_to_cif.get(key, [])
            prefix_len = min(len(edges_a), len(edges_b))
            mismatch_indices = [
                idx for idx in range(prefix_len) if edges_a[idx] != edges_b[idx]
            ]
            ordered_equal = edges_a == edges_b
            multiset_equal = Counter(edges_a) == Counter(edges_b)
            if not ordered_equal:
                keys_with_order_mismatch += 1
            if not multiset_equal:
                keys_with_multiset_mismatch += 1
            total_mismatches += len(mismatch_indices)
            graph_inv_rows.append(
                {
                    "key": key,
                    "len_cif": len(edges_a),
                    "len_out_relabeled": len(edges_b),
                    "ordered_equal": ordered_equal,
                    "multiset_equal": multiset_equal,
                    "mismatch_count": len(mismatch_indices),
                    "first_mismatch_idx": (
                        mismatch_indices[0] if mismatch_indices else None
                    ),
                }
            )
        graph_invariance_summary = {
            "keys_checked": len(keys),
            "keys_with_order_mismatch": keys_with_order_mismatch,
            "keys_with_multiset_mismatch": keys_with_multiset_mismatch,
            "total_position_mismatches": total_mismatches,
        }
        graph_invariance_rows = graph_inv_rows

    matrix_objects = {
        "hamiltonian": final_snapshot.hamiltonian,
        "overlap": final_snapshot.overlap,
        "density": final_snapshot.density,
    }
    matrix_edge_identity = {
        "hamiltonian_vs_overlap": all(
            torch.equal(
                final_snapshot.hamiltonian.pair_edges[key],
                final_snapshot.overlap.pair_edges[key],
            )
            for key in final_snapshot.hamiltonian.pair_edges
        ),
        "hamiltonian_vs_density": all(
            torch.equal(
                final_snapshot.hamiltonian.pair_edges[key],
                final_snapshot.density.pair_edges[key],
            )
            for key in final_snapshot.hamiltonian.pair_edges
        ),
    }

    target_matrix = matrix_objects[args.matrix_name]
    per_key_rows: list[dict[str, Any]] = []
    mismatch_atom_counts: Counter[int] = Counter()
    for key, target_edges_t in target_matrix.pair_edges.items():
        target_edges = [tuple(map(int, row)) for row in target_edges_t.t().tolist()]
        graph_edges = graph_final.get(key, [])
        row = _compare_edge_lists(
            key=key,
            target_edges=target_edges,
            graph_edges=graph_edges,
            positions=final_snapshot.positions,
            box=final_snapshot.box,
        )
        per_key_rows.append(row)
        if row["first_mismatch"] is not None:
            mismatch = row["first_mismatch"]
            target_edge = mismatch["target_edge"]
            graph_edge = mismatch["graph_edge"]
            mismatch_atom_counts[target_edge["src"]] += 1
            mismatch_atom_counts[target_edge["dst"]] += 1
            mismatch_atom_counts[graph_edge["src"]] += 1
            mismatch_atom_counts[graph_edge["dst"]] += 1

    per_key_rows.sort(key=lambda row: row["key"])

    selected_key = args.key
    if selected_key not in target_matrix.pair_edges:
        available = ", ".join(sorted(target_matrix.pair_edges))
        raise KeyError(f"Key {selected_key!r} not found. Available keys: {available}")

    selected_target_edges = [
        tuple(map(int, row))
        for row in target_matrix.pair_edges[selected_key].t().tolist()
    ]
    selected_graph_edges = graph_final.get(selected_key, [])
    selected_summary = next(row for row in per_key_rows if row["key"] == selected_key)
    focus_rank = args.focus_rank
    if focus_rank is None:
        focus_rank = selected_summary["first_mismatch_idx"]
    if focus_rank is not None and (
        focus_rank < 0 or focus_rank >= len(selected_target_edges)
    ):
        raise ValueError(
            f"--focus-rank={focus_rank} is out of range for key {selected_key} "
            f"(len={len(selected_target_edges)})"
        )

    graph_rank_lookup = {edge: idx for idx, edge in enumerate(selected_graph_edges)}
    target_rank_lookup = {edge: idx for idx, edge in enumerate(selected_target_edges)}
    selected_rows: list[dict[str, Any]] = []
    for rank, target_edge in enumerate(selected_target_edges[: args.max_key_rows]):
        graph_edge = (
            selected_graph_edges[rank] if rank < len(selected_graph_edges) else None
        )
        target_dist = _edge_distance(
            target_edge, final_snapshot.positions, final_snapshot.box
        )
        graph_dist = (
            _edge_distance(graph_edge, final_snapshot.positions, final_snapshot.box)
            if graph_edge is not None
            else None
        )
        target_src, target_dst = target_edge[3], target_edge[4]
        row = {
            "rank": rank,
            "match": target_edge == graph_edge,
            "target_edge": _edge_to_str(target_edge),
            "graph_edge": _edge_to_str(graph_edge),
            "target_distance": target_dist,
            "graph_distance": graph_dist,
            "target_graph_rank": graph_rank_lookup.get(target_edge),
            "graph_target_rank": (
                target_rank_lookup.get(graph_edge) if graph_edge is not None else None
            ),
            "target_src_symbol": final_snapshot.density.atoms[target_src],
            "target_dst_symbol": final_snapshot.density.atoms[target_dst],
            "target_src_index": target_src,
            "target_dst_index": target_dst,
            "same_pair_ignoring_shift": (
                graph_edge is not None
                and target_edge[3] == graph_edge[3]
                and target_edge[4] == graph_edge[4]
            ),
            "target_edge_tuple": target_edge,
            "graph_edge_tuple": graph_edge,
        }
        selected_rows.append(row)

    selected_rows_json = [
        {
            key: value
            for key, value in row.items()
            if key not in {"target_edge_tuple", "graph_edge_tuple"}
        }
        for row in selected_rows
    ]

    atom_shift_rows: list[dict[str, Any]] = []
    if atom_image_shifts is not None:
        for idx, atom in enumerate(atoms_tuple):
            shift = atom_image_shifts[idx]
            atom_shift_rows.append(
                {
                    "atom_index": idx,
                    "symbol": atom,
                    "shift_x": int(shift[0].item()),
                    "shift_y": int(shift[1].item()),
                    "shift_z": int(shift[2].item()),
                    "is_shifted": bool((shift != 0).any().item()),
                }
            )

    pair_mult = _pair_multiplicity_histogram(graph_final)
    pair_mult_hist = Counter(pair_mult.values())

    summary_payload: dict[str, Any] = {
        "matrix_path": matrix_path,
        "info_path": info_path,
        "cif_path": cif_path if cif_path.exists() else None,
        "convention": args.convention,
        "cutoff_radius": args.cutoff_radius,
        "l_max": args.l_max,
        "n_radial": args.n_radial,
        "matrix_name": args.matrix_name,
        "selected_key": selected_key,
        "focus_rank": focus_rank,
        "geometry": {
            "atoms": len(atoms_tuple),
            "box_info": info_box.tolist(),
            "box_recovered_from_out": info_box_recovered.tolist(),
            "box_cif": cif_box.tolist() if cif_box is not None else None,
            "max_abs_box_diff_cif_vs_out": (
                float(torch.max(torch.abs(cif_box - info_box_recovered)).item())
                if cif_box is not None
                else None
            ),
            "max_atom_shift_residual": max_atom_shift_residual,
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
        },
        "matrix_edge_identity": matrix_edge_identity,
        "pair_multiplicity_histogram": {
            str(mult): int(count) for mult, count in sorted(pair_mult_hist.items())
        },
        "selected_key_summary": selected_summary,
        "per_key_summary": per_key_rows,
        "graph_order_invariance": graph_invariance_summary,
    }

    _write_json(output_dir / "summary.json", summary_payload)
    _write_csv(
        output_dir / "per_key_prefix_summary.csv",
        per_key_rows,
        [
            "key",
            "target_len",
            "graph_len",
            "prefix_exact",
            "prefix_mismatch_count",
            "first_mismatch_idx",
            "target_subset_of_graph_multiset",
            "target_equals_graph_prefix_multiset",
            "target_multiset_cardinality",
            "graph_multiset_cardinality",
            "first_mismatch",
        ],
    )
    _write_csv(
        output_dir / f"selected_key_{selected_key.replace('-', '_')}_alignment.csv",
        selected_rows_json,
        [
            "rank",
            "match",
            "target_edge",
            "graph_edge",
            "target_distance",
            "graph_distance",
            "target_graph_rank",
            "graph_target_rank",
            "target_src_symbol",
            "target_dst_symbol",
            "target_src_index",
            "target_dst_index",
            "same_pair_ignoring_shift",
        ],
    )
    if atom_shift_rows:
        _write_csv(
            output_dir / "atom_image_shifts.csv",
            atom_shift_rows,
            ["atom_index", "symbol", "shift_x", "shift_y", "shift_z", "is_shifted"],
        )
    if graph_invariance_rows:
        _write_csv(
            output_dir / "graph_cif_vs_out_order_invariance.csv",
            graph_invariance_rows,
            [
                "key",
                "len_cif",
                "len_out_relabeled",
                "ordered_equal",
                "multiset_equal",
                "mismatch_count",
                "first_mismatch_idx",
            ],
        )

    _plot_prefix_mismatch_counts(
        output_dir / "prefix_mismatch_counts.png",
        per_key_rows,
        top_keys=args.top_keys,
    )
    _plot_pair_multiplicity_histogram(
        output_dir / "pair_multiplicity_histogram.png",
        pair_mult,
    )
    _plot_selected_key_rank_distances(
        output_dir
        / f"selected_key_{selected_key.replace('-', '_')}_rank_vs_distance.png",
        selected_rows,
        key=selected_key,
        focus_rank=focus_rank,
    )
    _plot_selected_key_zoom(
        output_dir / f"selected_key_{selected_key.replace('-', '_')}_focus_window.png",
        selected_rows,
        key=selected_key,
        focus_rank=focus_rank,
        context=args.context,
    )
    if atom_image_shifts is not None:
        _write_focus_scene_html(
            output_path=output_dir
            / f"selected_key_{selected_key.replace('-', '_')}_focus_scene.html",
            positions=final_snapshot.positions,
            box=final_snapshot.box,
            atoms=tuple(final_snapshot.density.atoms),
            atom_image_shifts=atom_image_shifts,
            rows=selected_rows,
            focus_rank=focus_rank,
            context=args.context,
        )

    report_lines = [
        f"matrix: {matrix_path}",
        f"info:   {info_path}",
        f"cif:    {cif_path if cif_path.exists() else '-'}",
        "",
        f"selected matrix: {args.matrix_name}",
        f"selected key:    {selected_key}",
        f"focus rank:      {focus_rank}",
        "",
        f"matrix edges identical across hamiltonian/overlap: {matrix_edge_identity['hamiltonian_vs_overlap']}",
        f"matrix edges identical across hamiltonian/density: {matrix_edge_identity['hamiltonian_vs_density']}",
        "",
        f"pair multiplicity histogram: {dict(sorted(pair_mult_hist.items()))}",
        "",
        "selected key summary:",
        json.dumps(selected_summary, indent=2, sort_keys=True, default=_json_default),
    ]
    if graph_invariance_summary is not None:
        report_lines.extend(
            [
                "",
                "graph order invariance summary (CIF graph vs relabeled .out graph):",
                json.dumps(
                    graph_invariance_summary,
                    indent=2,
                    sort_keys=True,
                    default=_json_default,
                ),
            ]
        )
    _write_report(output_dir / "report.txt", report_lines)

    top_mismatch_keys = sorted(
        [row for row in per_key_rows if row["prefix_mismatch_count"] > 0],
        key=lambda row: (-int(row["prefix_mismatch_count"]), row["key"]),
    )[: args.top_keys]
    print("", flush=True)
    print("[SUMMARY] top mismatch keys", flush=True)
    for row in top_mismatch_keys:
        print(
            f"  {row['key']:>8s}  mismatches={row['prefix_mismatch_count']:4d}  "
            f"first_idx={row['first_mismatch_idx']}  "
            f"target_len={row['target_len']}  graph_len={row['graph_len']}",
            flush=True,
        )

    print("", flush=True)
    print(f"[SUMMARY] selected key {selected_key}", flush=True)
    print(
        f"  target_len={selected_summary['target_len']} "
        f"graph_len={selected_summary['graph_len']} "
        f"prefix_exact={selected_summary['prefix_exact']} "
        f"first_mismatch_idx={selected_summary['first_mismatch_idx']}",
        flush=True,
    )
    if focus_rank is not None and focus_rank < len(selected_rows):
        row = selected_rows[focus_rank]
        print(
            f"  focus target={row['target_edge']} dist={row['target_distance']:.6f}",
            flush=True,
        )
        if row["graph_edge_tuple"] is not None:
            print(
                f"  focus graph ={row['graph_edge']} dist={row['graph_distance']:.6f}",
                flush=True,
            )

    print("", flush=True)
    print(f"[DONE] wrote diagnostics to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
