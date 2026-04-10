from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
if str(SRC) not in sys.path:
    sys.path.append(str(SRC))

from studies.e3mlp_investigation.experiment_utils import (  # noqa: E402
    configure_matplotlib,
    dump_json,
    dump_yaml,
    ensure_dir,
    save_df,
    save_plot,
)
from studies.e3mlp_investigation.irrep_presets import get_hidden_irreps  # noqa: E402
from studies.e3mlp_investigation.scripts.run_silicon_nognn_study import (  # noqa: E402
    SiliconNoGNNStudy,
    batched_forward,
    feature_irreps_from_cache,
    load_snapshot,
    prepare_pair_data,
)


@dataclass
class ProbeConfig:
    snapshots: list[str]
    target: str
    variants: list[str]
    hidden_irreps_preset: str
    aggregation: str
    architecture: str
    depth: int
    topk: int
    output_scale: float
    weight_init_scale: float
    residual_scale: float
    pre_norm: bool
    num_seeds: int
    edge_batch_size: int | None
    device: str


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Probe random-init output block scale vs silicon targets."
    )
    p.add_argument("--snapshots", type=str, default="2700K,900K")
    p.add_argument(
        "--target", type=str, default="hamiltonian", choices=["hamiltonian", "density"]
    )
    p.add_argument(
        "--variants",
        type=str,
        default="gate,gatemagnitudes,resgatemagnitudes,normact,film",
    )
    p.add_argument(
        "--hidden-irreps-preset",
        type=str,
        default="silicon",
        choices=["preliminary", "silicon", "silicon_doubled"],
    )
    p.add_argument(
        "--aggregation",
        type=str,
        default="mean",
        choices=["none", "sum", "mean", "attention", "topk_concat"],
    )
    p.add_argument(
        "--architecture",
        type=str,
        default="single",
        choices=["single", "message_then_predict"],
    )
    p.add_argument("--depth", type=int, default=4)
    p.add_argument("--topk", type=int, default=8)
    p.add_argument("--output-scale", type=float, default=0.5)
    p.add_argument("--weight-init-scale", type=float, default=0.5)
    p.add_argument("--residual-scale", type=float, default=0.05)
    p.add_argument("--pre-norm", action="store_true")
    p.add_argument("--num-seeds", type=int, default=1)
    p.add_argument("--edge-batch-size", type=int, default=1024)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--run-name", type=str, default=None)
    p.add_argument(
        "--output-root", type=str, default="studies/e3mlp_investigation/artifacts"
    )
    return p.parse_args()


def _parse_csv_str(s: str, cast):
    return [cast(item) for item in s.split(",") if item]


def split_masks(edges: torch.Tensor) -> dict[str, torch.Tensor]:
    sx, sy, sz, i, j = edges
    diag = (sx == 0) & (sy == 0) & (sz == 0) & (i == j)
    return {
        "diagonal": diag,
        "offdiagonal": ~diag,
        "all": torch.ones_like(diag, dtype=torch.bool),
    }


def block_norms(blocks: torch.Tensor) -> torch.Tensor:
    return blocks.reshape(blocks.shape[0], -1).norm(dim=-1)


def mean_abs_entry(blocks: torch.Tensor) -> torch.Tensor:
    return blocks.abs().mean(dim=(1, 2))


def evaluate_one(
    *,
    snapshot: str,
    target: str,
    variant: str,
    seed: int,
    cfg: ProbeConfig,
) -> tuple[list[dict], dict[str, torch.Tensor]]:
    torch.manual_seed(seed)

    inputs, targets, mapper = load_snapshot(snapshot)
    data = prepare_pair_data(inputs, targets, target, max_edges=None)
    raw_irreps = feature_irreps_from_cache(inputs["edge_length_emb"], inputs["edge_sh"])
    hidden_irreps = get_hidden_irreps(cfg.hidden_irreps_preset)
    output_irreps = mapper.get_pair_irreps(("Si", "Si"))

    model = SiliconNoGNNStudy(
        raw_irreps=raw_irreps,
        hidden_irreps=hidden_irreps,
        output_irreps=output_irreps,
        aggregation=cfg.aggregation,
        architecture=cfg.architecture,
        variant=variant,
        depth=cfg.depth,
        message_variant=variant,
        message_depth=max(1, min(2, cfg.depth)),
        predictor_variant=variant,
        predictor_depth=cfg.depth,
        topk=cfg.topk,
        output_scale=cfg.output_scale,
        weight_init_scale=cfg.weight_init_scale,
        residual_scale=cfg.residual_scale,
        pre_norm=cfg.pre_norm,
    ).to(cfg.device)
    model.eval()

    edge_features = data["edge_features"].to(cfg.device)
    edge_index = data["edge_index"].to(cfg.device)
    neighborhoods = [nbr.to(cfg.device) for nbr in data["neighborhoods"]]
    target_matrix = data["target"]
    target_blocks = target_matrix.pair_blocks["Si-Si"].cpu()

    with torch.no_grad():
        pred_vec = batched_forward(
            model,
            edge_features,
            edge_index,
            neighborhoods,
            edge_batch_size=cfg.edge_batch_size,
        )
        pred_blocks = mapper.vectors_to_blocks(("Si", "Si"), pred_vec.cpu())

    edges = target_matrix.pair_edges["Si-Si"].cpu()
    masks = split_masks(edges)
    target_frob = block_norms(target_blocks)
    pred_frob = block_norms(pred_blocks)
    target_abs = mean_abs_entry(target_blocks)
    pred_abs = mean_abs_entry(pred_blocks)

    rows: list[dict] = []
    for split, mask in masks.items():
        mask = mask.bool()
        if int(mask.sum().item()) == 0:
            continue
        t_f = target_frob[mask]
        p_f = pred_frob[mask]
        t_a = target_abs[mask]
        p_a = pred_abs[mask]
        rows.append(
            {
                "snapshot": snapshot,
                "target": target,
                "variant": variant,
                "seed": seed,
                "split": split,
                "num_blocks": int(mask.sum().item()),
                "target_block_fro_mean": float(t_f.mean().item()),
                "pred_block_fro_mean": float(p_f.mean().item()),
                "pred_to_target_fro_ratio": float(
                    p_f.mean().item() / max(t_f.mean().item(), 1e-12)
                ),
                "target_entry_abs_mean": float(t_a.mean().item()),
                "pred_entry_abs_mean": float(p_a.mean().item()),
                "pred_to_target_entry_abs_ratio": float(
                    p_a.mean().item() / max(t_a.mean().item(), 1e-12)
                ),
            }
        )

    return rows, {
        "target_frob": target_frob,
        "pred_frob": pred_frob,
        "diag_mask": masks["diagonal"],
    }


def make_plots(df: pd.DataFrame, out_dir: Path) -> list[Path]:
    paths: list[Path] = []
    plot_df = df[df["split"].isin(["diagonal", "offdiagonal"])].copy()
    if plot_df.empty:
        return paths

    long_df = pd.concat(
        [
            plot_df.rename(
                columns={
                    "target_block_fro_mean": "value",
                }
            ).assign(series="target_fro_mean"),
            plot_df.rename(
                columns={
                    "pred_block_fro_mean": "value",
                }
            ).assign(series="pred_fro_mean"),
        ],
        ignore_index=True,
    )
    fig, ax = plt.subplots(figsize=(12, 5))
    sns.barplot(
        data=long_df,
        x="variant",
        y="value",
        hue="series",
        ax=ax,
        errorbar=None,
    )
    ax.set_yscale("log")
    ax.set_title("Random-init block Frobenius magnitude vs target")
    save_plot(fig, out_dir / "block_fro_mean_bar.png")
    paths.append(out_dir / "block_fro_mean_bar.png")

    for split in ["diagonal", "offdiagonal"]:
        sub = plot_df[plot_df["split"] == split]
        if sub.empty:
            continue
        pivot = sub.pivot(
            index="variant", columns="snapshot", values="pred_to_target_fro_ratio"
        )
        fig, ax = plt.subplots(figsize=(7, 4))
        sns.heatmap(pivot, annot=True, fmt=".2e", cmap="mako", ax=ax)
        ax.set_title(f"Pred/target Frobenius ratio ({split})")
        path = out_dir / f"fro_ratio_heatmap_{split}.png"
        save_plot(fig, path)
        paths.append(path)

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.scatterplot(
        data=plot_df,
        x="target_block_fro_mean",
        y="pred_block_fro_mean",
        hue="variant",
        style="split",
        s=80,
        ax=ax,
    )
    lim_lo = min(
        plot_df["target_block_fro_mean"].min(), plot_df["pred_block_fro_mean"].min()
    )
    lim_hi = max(
        plot_df["target_block_fro_mean"].max(), plot_df["pred_block_fro_mean"].max()
    )
    ax.plot(
        [lim_lo, lim_hi], [lim_lo, lim_hi], linestyle="--", color="black", linewidth=1.0
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title("Random-init vs target block Frobenius scale")
    path = out_dir / "target_vs_pred_fro_scatter.png"
    save_plot(fig, path)
    paths.append(path)

    return paths


def main() -> None:
    args = parse_args()
    run_name = args.run_name or time.strftime("random_init_block_scale_%Y%m%d_%H%M%S")
    run_dir = ensure_dir(Path(args.output_root) / run_name)
    plots_dir = ensure_dir(run_dir / "plots")
    configure_matplotlib(Path("studies/e3mlp_investigation/cache/mplconfig"))

    cfg = ProbeConfig(
        snapshots=_parse_csv_str(args.snapshots, str),
        target=args.target,
        variants=_parse_csv_str(args.variants, str),
        hidden_irreps_preset=args.hidden_irreps_preset,
        aggregation=args.aggregation,
        architecture=args.architecture,
        depth=args.depth,
        topk=args.topk,
        output_scale=args.output_scale,
        weight_init_scale=args.weight_init_scale,
        residual_scale=args.residual_scale,
        pre_norm=args.pre_norm,
        num_seeds=args.num_seeds,
        edge_batch_size=args.edge_batch_size,
        device=args.device,
    )
    dump_yaml(run_dir / "config.yaml", cfg.__dict__)
    print(f"[random-init-probe] run_dir={run_dir}", flush=True)
    print(
        f"[random-init-probe] snapshots={cfg.snapshots} target={cfg.target} "
        f"variants={cfg.variants} device={cfg.device}",
        flush=True,
    )

    all_rows: list[dict] = []
    total = len(cfg.snapshots) * len(cfg.variants) * cfg.num_seeds
    done = 0
    for snapshot in cfg.snapshots:
        for variant in cfg.variants:
            for seed in range(cfg.num_seeds):
                print(
                    f"[random-init-probe] snapshot={snapshot} variant={variant} seed={seed}",
                    flush=True,
                )
                rows, _ = evaluate_one(
                    snapshot=snapshot,
                    target=cfg.target,
                    variant=variant,
                    seed=seed,
                    cfg=cfg,
                )
                all_rows.extend(rows)
                done += 1
                diag_row = next((r for r in rows if r["split"] == "diagonal"), None)
                off_row = next((r for r in rows if r["split"] == "offdiagonal"), None)
                if diag_row is not None and off_row is not None:
                    print(
                        f"[random-init-probe] {done}/{total} done "
                        f"diag_ratio={diag_row['pred_to_target_fro_ratio']:.3e} "
                        f"off_ratio={off_row['pred_to_target_fro_ratio']:.3e}",
                        flush=True,
                    )

    df = pd.DataFrame(all_rows)
    save_df(run_dir / "summary.csv", df)
    plot_paths = make_plots(df, plots_dir)
    dump_json(
        run_dir / "summary.json",
        {
            "run_name": run_name,
            "rows": df.to_dict(orient="records"),
            "plot_paths": [str(p) for p in plot_paths],
        },
    )
    print(f"[random-init-probe] finished; wrote results to {run_dir}", flush=True)


if __name__ == "__main__":
    main()
