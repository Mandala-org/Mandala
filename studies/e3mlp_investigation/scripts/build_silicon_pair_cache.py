from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
if str(SRC) not in sys.path:
    sys.path.append(str(SRC))

from data.factory import DatasetFactory  # noqa: E402
from net.common import Config  # noqa: E402
from studies.e3mlp_investigation.experiment_utils import (
    append_study_log,
    dump_json,
)  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build cached silicon pair-level data for no-GNN studies."
    )
    p.add_argument("--snapshot", type=str, default="2700K", choices=["2700K", "900K"])
    p.add_argument("--cutoff-radius", type=float, default=7.0)
    p.add_argument("--l-max", type=int, default=4)
    p.add_argument("--n-radial", type=int, default=64)
    p.add_argument(
        "--output-root",
        type=str,
        default="studies/e3mlp_investigation/cache/silicon_pairs",
    )
    p.add_argument("--append-log", action="store_true")
    return p.parse_args()


def pair_block_norms(blocks: torch.Tensor) -> torch.Tensor:
    return blocks.reshape(blocks.shape[0], -1).norm(dim=-1)


def save_distance_scatter(
    distances: torch.Tensor, values: torch.Tensor, title: str, path: Path
) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    hb = ax.hexbin(
        distances.cpu().numpy(),
        values.cpu().numpy(),
        gridsize=45,
        bins="log",
        cmap="viridis",
    )
    fig.colorbar(hb, ax=ax, label="count")
    ax.set_xlabel("distance")
    ax.set_ylabel("block Frobenius norm")
    ax.set_title(title)
    fig.savefig(path, bbox_inches="tight", dpi=180)
    plt.close(fig)


def save_distance_hist(distances: torch.Tensor, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(distances.cpu().numpy(), bins=60, color="#4477aa", alpha=0.9)
    ax.set_xlabel("distance")
    ax.set_ylabel("count")
    ax.set_title("Edge distance histogram")
    fig.savefig(path, bbox_inches="tight", dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_root) / args.snapshot
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = Config(
        cutoff_radius=args.cutoff_radius, l_max=args.l_max, n_radial=args.n_radial
    )
    fac = DatasetFactory(cfg)
    fac.add_snapshot(
        f"data/big/silicon/{args.snapshot}/Si_DM",
        f"data/big/silicon/{args.snapshot}/info.txt",
    )
    ds, _, mapper = fac.create()
    x, y = ds[0]

    cache_payload = {
        "inputs": x,
        "targets": y,
        "mapper": mapper,
        "config": {
            "cutoff_radius": args.cutoff_radius,
            "l_max": args.l_max,
            "n_radial": args.n_radial,
        },
    }
    torch.save(cache_payload, out_dir / "pair_cache.pt")

    edge_index = x["edge_index"]
    positions = x["positions"]
    edge_shift = x["edge_shift"]
    box = x.get("box")
    if box is not None:
        shift_float = edge_shift.T.to(dtype=positions.dtype)
        edge_vec = (
            positions[edge_index[1]] - positions[edge_index[0]] + shift_float @ box
        )
    else:
        edge_vec = positions[edge_index[1]] - positions[edge_index[0]]
    edge_dist = torch.linalg.norm(edge_vec, dim=-1)

    save_distance_hist(edge_dist, out_dir / "edge_distance_hist.png")

    for target_name in ["hamiltonian", "density", "overlap"]:
        target = y[target_name]
        for key, blocks in target.pair_blocks.items():
            norms = pair_block_norms(blocks)
            edges = target.pair_edges[key]
            edge_shift_key = edges[:3].T.to(dtype=positions.dtype)
            src = edges[3].long()
            dst = edges[4].long()
            if box is not None:
                disp = positions[dst] - positions[src] + edge_shift_key @ box
            else:
                disp = positions[dst] - positions[src]
            dist = torch.linalg.norm(disp, dim=-1)
            save_distance_scatter(
                dist,
                norms,
                f"{args.snapshot} {target_name} {key}: block norm vs distance",
                out_dir / f"{target_name}_{key.replace('/', '_')}_norm_vs_distance.png",
            )

    summary = {
        "snapshot": args.snapshot,
        "cutoff_radius": args.cutoff_radius,
        "l_max": args.l_max,
        "n_radial": args.n_radial,
        "num_atoms": int(x["node_type_idx"].shape[0]),
        "num_edges": int(x["edge_index"].shape[1]),
        "edge_types": list(mapper.edge_types),
        "targets": list(y.keys()),
    }
    dump_json(out_dir / "summary.json", summary)

    if args.append_log:
        title = f"2026-04-09 - silicon_pair_cache_{args.snapshot}"
        lines = [
            "Built silicon pair cache.",
            "",
            f"- output dir: [{args.snapshot}](/home/bartek/casus/mandala/{out_dir})",
            f"- cutoff radius: `{args.cutoff_radius}`",
            f"- l_max: `{args.l_max}`",
            f"- n_radial: `{args.n_radial}`",
            f"- atoms: `{summary['num_atoms']}`",
            f"- edges: `{summary['num_edges']}`",
            "",
            "Artifacts:",
            f"- [pair_cache.pt](/home/bartek/casus/mandala/{out_dir / 'pair_cache.pt'})",
            f"- [summary.json](/home/bartek/casus/mandala/{out_dir / 'summary.json'})",
            f"- [edge distance histogram](/home/bartek/casus/mandala/{out_dir / 'edge_distance_hist.png'})",
        ]
        append_study_log(Path("studies/e3mlp_investigation/STUDY_LOG.md"), title, lines)


if __name__ == "__main__":
    main()
