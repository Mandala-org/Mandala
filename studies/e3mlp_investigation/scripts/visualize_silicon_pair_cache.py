from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Visualize cached silicon pair targets.")
    p.add_argument("--snapshot", type=str, default="2700K", choices=["2700K", "900K"])
    p.add_argument(
        "--cache-root",
        type=str,
        default="studies/e3mlp_investigation/cache/silicon_pairs",
    )
    return p.parse_args()


def pair_distances(cache_payload, target_name: str, key: str) -> torch.Tensor:
    x = cache_payload["inputs"]
    positions = x["positions"]
    box = x.get("box")
    target = cache_payload["targets"][target_name]
    edges = target.pair_edges[key]
    shift = edges[:3].T.to(dtype=positions.dtype)
    src = edges[3].long()
    dst = edges[4].long()
    if box is not None:
        disp = positions[dst] - positions[src] + shift @ box
    else:
        disp = positions[dst] - positions[src]
    return torch.linalg.norm(disp, dim=-1)


def save_block_examples(cache_payload, snapshot: str, out_dir: Path) -> None:
    for target_name in ["hamiltonian", "density"]:
        target = cache_payload["targets"][target_name]
        for key, blocks in target.pair_blocks.items():
            dist = pair_distances(cache_payload, target_name, key)
            order = torch.argsort(dist)
            picks = {
                "nearest": order[0].item(),
                "median": order[len(order) // 2].item(),
                "farthest": order[-1].item(),
            }
            fig, axes = plt.subplots(1, 3, figsize=(12, 4))
            vmax = torch.quantile(blocks.abs().reshape(-1), 0.99).item()
            vmax = max(vmax, 1e-8)
            for ax, (label, idx) in zip(axes, picks.items()):
                im = ax.imshow(
                    blocks[idx].cpu().numpy(), cmap="coolwarm", vmin=-vmax, vmax=vmax
                )
                ax.set_title(f"{target_name} {label}\nd={dist[idx].item():.2f}")
            fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.8)
            fig.suptitle(f"{snapshot} {target_name} {key}: representative blocks")
            fig.savefig(
                out_dir
                / f"{target_name}_{key.replace('/', '_')}_representative_blocks.png",
                bbox_inches="tight",
                dpi=180,
            )
            plt.close(fig)

            norms = blocks.reshape(blocks.shape[0], -1).norm(dim=-1)
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.hist(norms.cpu().numpy(), bins=60, color="#cc6677", alpha=0.9)
            ax.set_xlabel("block Frobenius norm")
            ax.set_ylabel("count")
            ax.set_title(f"{snapshot} {target_name} {key}: block norm histogram")
            fig.savefig(
                out_dir / f"{target_name}_{key.replace('/', '_')}_norm_hist.png",
                bbox_inches="tight",
                dpi=180,
            )
            plt.close(fig)


def main() -> None:
    args = parse_args()
    cache_dir = Path(args.cache_root) / args.snapshot
    cache_payload = torch.load(
        cache_dir / "pair_cache.pt", map_location="cpu", weights_only=False
    )
    save_block_examples(cache_payload, args.snapshot, cache_dir)


if __name__ == "__main__":
    main()
