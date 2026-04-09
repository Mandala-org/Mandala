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
import torch.nn as nn
import torch.nn.functional as F
from e3nn.o3 import Irreps
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
if str(SRC) not in sys.path:
    sys.path.append(str(SRC))

from core.sparse_math import trace_matmul_sparse_block_matrix  # noqa: E402
from data.block_matrix import BlockMatrix  # noqa: E402
from studies.e3mlp_investigation.experiment_utils import (  # noqa: E402
    append_study_log,
    configure_matplotlib,
    dump_json,
    dump_yaml,
    ensure_dir,
    save_df,
    save_plot,
)
from studies.e3mlp_investigation.irrep_presets import get_hidden_irreps  # noqa: E402
from studies.e3mlp_investigation.variants import build_variant  # noqa: E402


@dataclass
class StudyConfig:
    train_snapshots: list[str]
    eval_snapshots: list[str]
    target: str
    aggregation: str
    architecture: str
    variant: str
    depth: int
    hidden_irreps_preset: str
    num_steps: int
    lr: float
    loss_kind: str
    topk: int
    message_variant: str
    message_depth: int
    predictor_variant: str
    predictor_depth: int
    max_train_edges: int | None
    max_eval_edges: int | None
    device: str
    output_scale: float
    weight_init_scale: float
    residual_scale: float
    pre_norm: bool
    smoke_test: bool


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run a no-GNN silicon E3MLP study.")
    p.add_argument("--train-snapshots", type=str, default="2700K")
    p.add_argument("--eval-snapshots", type=str, default="900K")
    p.add_argument(
        "--target", type=str, default="hamiltonian", choices=["hamiltonian", "density"]
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
    p.add_argument(
        "--variant",
        type=str,
        default="gatemagnitudes",
        choices=[
            "normact",
            "gate",
            "gatemagnitudes",
            "film",
            "resnormact",
            "resgatemagnitudes",
            "bilinear",
        ],
    )
    p.add_argument("--depth", type=int, default=4)
    p.add_argument(
        "--hidden-irreps-preset",
        type=str,
        default="preliminary",
        choices=["preliminary", "silicon", "silicon_doubled"],
    )
    p.add_argument("--num-steps", type=int, default=80)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument(
        "--loss-kind", type=str, default="mse", choices=["mse", "mae", "huber"]
    )
    p.add_argument("--topk", type=int, default=8)
    p.add_argument("--message-variant", type=str, default="gatemagnitudes")
    p.add_argument("--message-depth", type=int, default=2)
    p.add_argument("--predictor-variant", type=str, default="gatemagnitudes")
    p.add_argument("--predictor-depth", type=int, default=4)
    p.add_argument("--max-train-edges", type=int, default=None)
    p.add_argument("--max-eval-edges", type=int, default=None)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--output-scale", type=float, default=1.0)
    p.add_argument("--weight-init-scale", type=float, default=1.0)
    p.add_argument("--residual-scale", type=float, default=0.1)
    p.add_argument("--pre-norm", action="store_true")
    p.add_argument("--run-name", type=str, default=None)
    p.add_argument(
        "--output-root", type=str, default="studies/e3mlp_investigation/artifacts"
    )
    p.add_argument("--smoke-test", action="store_true")
    p.add_argument("--append-log", action="store_true")
    return p.parse_args()


def _parse_csv_str(s: str, cast):
    return [cast(item) for item in s.split(",") if item]


def loss_fn(name: str, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    name = name.lower()
    if name == "mse":
        return F.mse_loss(pred, target)
    if name == "mae":
        return F.l1_loss(pred, target)
    if name == "huber":
        return F.huber_loss(pred, target, delta=1.0)
    raise ValueError(f"Unsupported loss kind '{name}'")


def relative_vector_error(
    pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8
) -> torch.Tensor:
    diff = (pred - target).norm(dim=-1)
    denom = target.norm(dim=-1).clamp_min(eps)
    return (diff / denom).mean()


def scale_irreps(irreps: Irreps, factor: int) -> Irreps:
    if factor == 1:
        return Irreps(irreps)
    return Irreps([(mul * factor, ir) for mul, ir in Irreps(irreps)])


def feature_irreps_from_cache(
    edge_length_emb: torch.Tensor, edge_sh: torch.Tensor
) -> Irreps:
    scalar_dim = edge_length_emb.shape[-1]
    sh_irreps = Irreps.spherical_harmonics(4)
    return Irreps(f"{scalar_dim}x0e") + sh_irreps


def edge_distances(x: dict[str, torch.Tensor]) -> torch.Tensor:
    edge_index = x["edge_index"]
    positions = x["positions"]
    edge_shift = x["edge_shift"]
    box = x.get("box")
    shift_float = edge_shift.T.to(dtype=positions.dtype)
    if box is not None:
        disp = positions[edge_index[1]] - positions[edge_index[0]] + shift_float @ box
    else:
        disp = positions[edge_index[1]] - positions[edge_index[0]]
    return torch.linalg.norm(disp, dim=-1)


def build_neighborhoods(
    edge_index: torch.Tensor,
    distances: torch.Tensor,
    num_nodes: int,
) -> list[torch.Tensor]:
    neighborhoods: list[list[tuple[float, int]]] = [[] for _ in range(num_nodes)]
    src = edge_index[0].tolist()
    dst = edge_index[1].tolist()
    for idx, (i, j) in enumerate(zip(src, dst)):
        d = float(distances[idx].item())
        neighborhoods[i].append((d, idx))
        neighborhoods[j].append((d, idx))
    out: list[torch.Tensor] = []
    for items in neighborhoods:
        items.sort(key=lambda t: t[0])
        out.append(torch.tensor([idx for _, idx in items], dtype=torch.long))
    return out


class NeighborhoodAggregator(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        aggregation: str,
        topk: int,
        attention_hidden: int = 64,
    ) -> None:
        super().__init__()
        self.feature_dim = feature_dim
        self.aggregation = aggregation
        self.topk = topk
        self.attention_hidden = attention_hidden
        if aggregation == "attention":
            self.scorer = nn.Sequential(
                nn.Linear(feature_dim, attention_hidden),
                nn.SiLU(),
                nn.Linear(attention_hidden, 1),
            )
        else:
            self.scorer = None

    @property
    def output_dim(self) -> int:
        if self.aggregation == "topk_concat":
            return self.topk * self.feature_dim
        return self.feature_dim

    def forward(
        self,
        edge_features: torch.Tensor,
        neighborhoods: list[torch.Tensor],
    ) -> torch.Tensor:
        node_ctx: list[torch.Tensor] = []
        for nbrs in neighborhoods:
            if nbrs.numel() == 0:
                node_ctx.append(
                    torch.zeros(
                        self.output_dim,
                        device=edge_features.device,
                        dtype=edge_features.dtype,
                    )
                )
                continue
            idx = nbrs[: self.topk].to(edge_features.device)
            feats = edge_features.index_select(0, idx)
            if self.aggregation == "none":
                node_ctx.append(feats[0])
            elif self.aggregation == "sum":
                node_ctx.append(feats.sum(dim=0))
            elif self.aggregation == "mean":
                node_ctx.append(feats.mean(dim=0))
            elif self.aggregation == "attention":
                scores = self.scorer(feats).squeeze(-1)
                weights = torch.softmax(scores, dim=0)
                node_ctx.append((weights.unsqueeze(-1) * feats).sum(dim=0))
            elif self.aggregation == "topk_concat":
                flat = feats.reshape(-1)
                if feats.shape[0] < self.topk:
                    pad = torch.zeros(
                        (self.topk - feats.shape[0]) * self.feature_dim,
                        device=edge_features.device,
                        dtype=edge_features.dtype,
                    )
                    flat = torch.cat([flat, pad], dim=0)
                node_ctx.append(flat)
            else:
                raise ValueError(f"Unsupported aggregation '{self.aggregation}'")
        return torch.stack(node_ctx, dim=0)


def assemble_edge_inputs(
    edge_features: torch.Tensor,
    node_ctx: torch.Tensor,
    edge_index: torch.Tensor,
    aggregation: str,
) -> torch.Tensor:
    if aggregation == "none":
        return edge_features
    src_ctx = node_ctx.index_select(0, edge_index[0].to(node_ctx.device))
    dst_ctx = node_ctx.index_select(0, edge_index[1].to(node_ctx.device))
    return torch.cat([src_ctx, dst_ctx, edge_features], dim=-1)


def build_block_matrix(
    template: BlockMatrix,
    pair_key: str,
    blocks: torch.Tensor,
) -> BlockMatrix:
    pair_blocks = {pair_key: blocks}
    pair_edges = {pair_key: template.pair_edges[pair_key]}
    lookup = {
        (int(sx), int(sy), int(sz), int(i), int(j)): (pair_key, idx)
        for idx, (sx, sy, sz, i, j) in enumerate(pair_edges[pair_key].t().tolist())
    }
    return BlockMatrix(
        atoms=template.atoms,
        atom_counts=template.atom_counts,
        pair_blocks=pair_blocks,
        pair_edges=pair_edges,
        lookup=lookup,
        orbital_cfg=template.orbital_cfg,
        basis=template.basis,
    )


def make_per_irrep_mae_plot(
    irreps: Irreps,
    target: torch.Tensor,
    pred: torch.Tensor,
    out_dir: Path,
    title: str,
) -> Path:
    rows: list[dict] = []
    cursor = 0
    for mul, ir in irreps:
        size = mul * ir.dim
        t = target[:, cursor : cursor + size].reshape(target.shape[0], mul, ir.dim)
        p = pred[:, cursor : cursor + size].reshape(pred.shape[0], mul, ir.dim)
        mae = (p - t).abs().mean(dim=(0, 2))
        for idx in range(mul):
            rows.append(
                {"irrep": f"{mul}x{ir}", "copy": idx, "mae": float(mae[idx].item())}
            )
        cursor += size
    df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=df, x="irrep", y="mae", ax=ax, errorbar=None)
    ax.set_yscale("log")
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=90)
    path = out_dir / "per_irrep_mae.png"
    save_plot(fig, path)
    return path


def make_distance_error_plot(
    distances: torch.Tensor,
    per_edge_metric: torch.Tensor,
    out_dir: Path,
    title: str,
    *,
    ylabel: str,
    filename: str,
) -> Path:
    fig, ax = plt.subplots(figsize=(7, 5))
    hb = ax.hexbin(
        distances.cpu().numpy(),
        per_edge_metric.cpu().numpy(),
        gridsize=45,
        bins="log",
        cmap="viridis",
    )
    fig.colorbar(hb, ax=ax, label="count")
    ax.set_xlabel("distance")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    path = out_dir / filename
    save_plot(fig, path)
    return path


def make_sample_block_plot(
    target_blocks: torch.Tensor,
    pred_blocks: torch.Tensor,
    out_dir: Path,
    title: str,
) -> Path:
    n = min(3, target_blocks.shape[0])
    if n == 0:
        return out_dir / "sample_blocks.png"
    norms = target_blocks.reshape(target_blocks.shape[0], -1).norm(dim=-1)
    order = torch.argsort(norms)
    picks = torch.stack([order[0], order[len(order) // 2], order[-1]]).unique()
    fig, axes = plt.subplots(len(picks), 3, figsize=(9, 3 * len(picks)))
    if len(picks) == 1:
        axes = [axes]
    labels = ["target", "pred", "abs err"]
    for row, edge_idx in enumerate(picks.tolist()):
        mats = [
            target_blocks[edge_idx],
            pred_blocks[edge_idx],
            (pred_blocks[edge_idx] - target_blocks[edge_idx]).abs(),
        ]
        vmax = max(mat.abs().max().item() for mat in mats[:2])
        for col, (mat, label) in enumerate(zip(mats, labels)):
            ax = axes[row, col]
            im = ax.imshow(
                mat.cpu().numpy(),
                cmap="coolwarm" if col < 2 else "magma",
                aspect="auto",
                vmin=-vmax if col < 2 else None,
                vmax=vmax if col < 2 else None,
            )
            ax.set_title(f"edge {edge_idx} {label}")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(title)
    path = out_dir / "sample_blocks.png"
    save_plot(fig, path)
    return path


def loss_summary(
    loss_kind: str, pred: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    return loss_fn(loss_kind, pred, target)


class SiliconNoGNNStudy(nn.Module):
    def __init__(
        self,
        *,
        raw_irreps: Irreps,
        hidden_irreps: Irreps,
        output_irreps: Irreps,
        aggregation: str,
        architecture: str,
        variant: str,
        depth: int,
        message_variant: str,
        message_depth: int,
        predictor_variant: str,
        predictor_depth: int,
        topk: int,
        output_scale: float = 1.0,
        weight_init_scale: float = 1.0,
        residual_scale: float = 0.1,
        pre_norm: bool = False,
    ) -> None:
        super().__init__()
        self.raw_irreps = raw_irreps
        self.hidden_irreps = hidden_irreps
        self.output_irreps = output_irreps
        self.aggregation = aggregation
        self.architecture = architecture
        self.topk = topk
        self.raw_aggregator = NeighborhoodAggregator(raw_irreps.dim, aggregation, topk)
        self.raw_feature_dim = raw_irreps.dim
        self.message_encoder = None
        self.predictor = None
        self.message_aggregator = None

        if architecture == "single":
            predictor_input_irreps = (
                raw_irreps
                if aggregation == "none"
                else (
                    scale_irreps(raw_irreps, topk)
                    if aggregation == "topk_concat"
                    else scale_irreps(raw_irreps, 3)
                )
            )
            if aggregation == "topk_concat":
                predictor_input_irreps = scale_irreps(raw_irreps, topk * 2) + raw_irreps
            self.predictor = build_variant(
                variant,
                predictor_input_irreps,
                hidden_irreps,
                output_irreps,
                num_layers=depth,
                output_scale=output_scale,
                weight_init_scale=weight_init_scale,
                residual_scale=residual_scale,
                pre_norm=pre_norm,
            )
        elif architecture == "message_then_predict":
            self.message_encoder = build_variant(
                message_variant,
                raw_irreps,
                hidden_irreps,
                hidden_irreps,
                num_layers=message_depth,
                output_scale=output_scale,
                weight_init_scale=weight_init_scale,
                residual_scale=residual_scale,
                pre_norm=pre_norm,
            )
            self.message_aggregator = NeighborhoodAggregator(
                hidden_irreps.dim, aggregation, topk
            )
            encoded_irreps = hidden_irreps
            predictor_input_irreps = (
                encoded_irreps
                if aggregation == "none"
                else (
                    scale_irreps(encoded_irreps, topk)
                    if aggregation == "topk_concat"
                    else scale_irreps(encoded_irreps, 3)
                )
            )
            if aggregation == "topk_concat":
                predictor_input_irreps = (
                    scale_irreps(encoded_irreps, topk * 2) + encoded_irreps
                )
            self.predictor = build_variant(
                predictor_variant,
                predictor_input_irreps,
                hidden_irreps,
                output_irreps,
                num_layers=predictor_depth,
                output_scale=output_scale,
                weight_init_scale=weight_init_scale,
                residual_scale=residual_scale,
                pre_norm=pre_norm,
            )
        else:
            raise ValueError(f"Unsupported architecture '{architecture}'")

    def forward(
        self,
        edge_features: torch.Tensor,
        edge_index: torch.Tensor,
        neighborhoods: list[torch.Tensor],
    ) -> torch.Tensor:
        if self.architecture == "single":
            node_ctx = self.raw_aggregator(edge_features, neighborhoods)
            model_in = assemble_edge_inputs(
                edge_features, node_ctx, edge_index, self.aggregation
            )
            return self.predictor(model_in)

        encoded = self.message_encoder(edge_features)
        node_ctx = self.message_aggregator(encoded, neighborhoods)
        model_in = assemble_edge_inputs(encoded, node_ctx, edge_index, self.aggregation)
        return self.predictor(model_in)


def load_snapshot(
    snapshot_name: str,
) -> tuple[dict[str, torch.Tensor], dict[str, BlockMatrix], object]:
    path = (
        Path("studies/e3mlp_investigation/cache/silicon_pairs")
        / snapshot_name
        / "pair_cache.pt"
    )
    payload = torch.load(path, map_location="cpu")
    return payload["inputs"], payload["targets"], payload["mapper"]


def prepare_pair_data(
    inputs: dict[str, torch.Tensor],
    targets: dict[str, BlockMatrix],
    target_name: str,
    *,
    max_edges: int | None,
) -> dict[str, torch.Tensor | BlockMatrix | list[torch.Tensor]]:
    edge_dist = edge_distances(inputs)
    num_nodes = int(inputs["node_type_idx"].shape[0])
    neighborhoods = build_neighborhoods(inputs["edge_index"], edge_dist, num_nodes)
    edge_features = torch.cat([inputs["edge_length_emb"], inputs["edge_sh"]], dim=-1)
    target = targets[target_name]
    gt_density = targets["density"]
    gt_hamiltonian = targets["hamiltonian"]
    if max_edges is not None and max_edges < edge_features.shape[0]:
        edge_features = edge_features[:max_edges]
        edge_dist = edge_dist[:max_edges]
        neighborhoods = [nbrs[nbrs < max_edges] for nbrs in neighborhoods]
        edge_index = inputs["edge_index"][:, :max_edges]

        def _truncate_matrix(mat: BlockMatrix) -> BlockMatrix:
            pair_blocks = {k: v[:max_edges] for k, v in mat.pair_blocks.items()}
            pair_edges = {k: v[:, :max_edges] for k, v in mat.pair_edges.items()}
            lookup = {
                (int(sx), int(sy), int(sz), int(i), int(j)): (key, idx)
                for key, edges in pair_edges.items()
                for idx, (sx, sy, sz, i, j) in enumerate(edges.t().tolist())
            }
            return BlockMatrix(
                atoms=mat.atoms,
                atom_counts=mat.atom_counts,
                pair_blocks=pair_blocks,
                pair_edges=pair_edges,
                lookup=lookup,
                orbital_cfg=mat.orbital_cfg,
                basis=mat.basis,
            )

        target = _truncate_matrix(target)
        gt_density = _truncate_matrix(gt_density)
        gt_hamiltonian = _truncate_matrix(gt_hamiltonian)
    else:
        edge_index = inputs["edge_index"]
    return {
        "edge_features": edge_features,
        "edge_dist": edge_dist,
        "edge_index": edge_index,
        "neighborhoods": neighborhoods,
        "target": target,
        "gt_density": gt_density,
        "gt_hamiltonian": gt_hamiltonian,
    }


def evaluate(
    *,
    model: SiliconNoGNNStudy,
    edge_features: torch.Tensor,
    edge_index: torch.Tensor,
    neighborhoods: list[torch.Tensor],
    target_matrix: BlockMatrix,
    gt_density: BlockMatrix,
    gt_hamiltonian: BlockMatrix,
    target_vec: torch.Tensor,
    mapper,
    target_name: str,
    loss_kind: str,
    eval_distances: torch.Tensor,
    out_dir: Path,
) -> dict[str, float | Path]:
    with torch.no_grad():
        pred_vec = model(edge_features, edge_index, neighborhoods)
        pred_loss = loss_summary(loss_kind, pred_vec, target_vec)
        pred_rel = relative_vector_error(pred_vec, target_vec)
        pred_vec_cpu = pred_vec.detach().cpu()
        target_vec_cpu = target_vec.detach().cpu()
        pred_blocks = mapper.vectors_to_blocks(("Si", "Si"), pred_vec_cpu)
        pred_matrix = build_block_matrix(target_matrix, "Si-Si", pred_blocks)
        if target_name == "hamiltonian":
            energy_pred = trace_matmul_sparse_block_matrix(pred_matrix, gt_density)
            energy_true = trace_matmul_sparse_block_matrix(gt_hamiltonian, gt_density)
        else:
            energy_pred = trace_matmul_sparse_block_matrix(gt_hamiltonian, pred_matrix)
            energy_true = trace_matmul_sparse_block_matrix(gt_hamiltonian, gt_density)

        block_mae = (
            (pred_blocks - target_matrix.pair_blocks["Si-Si"]).abs().mean().item()
        )
        block_mse = F.mse_loss(pred_blocks, target_matrix.pair_blocks["Si-Si"]).item()
        block_rel = relative_vector_error(
            pred_blocks.reshape(pred_blocks.shape[0], -1),
            target_matrix.pair_blocks["Si-Si"].reshape(
                target_matrix.pair_blocks["Si-Si"].shape[0], -1
            ),
        ).item()
        per_edge_mae = (
            (pred_blocks - target_matrix.pair_blocks["Si-Si"]).abs().mean(dim=(1, 2))
        )
        per_edge_rel = (pred_blocks - target_matrix.pair_blocks["Si-Si"]).reshape(
            pred_blocks.shape[0], -1
        ).norm(dim=-1) / target_matrix.pair_blocks["Si-Si"].reshape(
            target_matrix.pair_blocks["Si-Si"].shape[0], -1
        ).norm(
            dim=-1
        ).clamp_min(
            1e-8
        )
        distances = eval_distances.to(per_edge_mae.device)

    plot_paths = []
    plot_paths.append(
        make_per_irrep_mae_plot(
            mapper.get_pair_irreps(("Si", "Si")),
            target_vec_cpu,
            pred_vec_cpu,
            out_dir,
            f"{target_name} per-irrep MAE",
        )
    )
    plot_paths.append(
        make_distance_error_plot(
            distances,
            per_edge_mae.detach().cpu(),
            out_dir,
            f"{target_name} block MAE vs distance",
            ylabel="block MAE",
            filename="distance_mae_hexbin.png",
        )
    )
    plot_paths.append(
        make_distance_error_plot(
            distances,
            per_edge_rel.detach().cpu(),
            out_dir,
            f"{target_name} block relative error vs distance",
            ylabel="block relative error",
            filename="distance_relative_error_hexbin.png",
        )
    )
    plot_paths.append(
        make_sample_block_plot(
            target_matrix.pair_blocks["Si-Si"],
            pred_blocks,
            out_dir,
            f"{target_name} sample blocks",
        )
    )

    return {
        "eval_loss": float(pred_loss.item()),
        "eval_rel": float(pred_rel.item()),
        "block_mae": float(block_mae),
        "block_mse": float(block_mse),
        "block_rel": float(block_rel),
        "energy_pred": float(energy_pred.item()),
        "energy_true": float(energy_true.item()),
        "energy_mae": float(abs(energy_pred - energy_true).item()),
        "energy_rel": float(
            abs(energy_pred - energy_true).item() / max(abs(energy_true.item()), 1e-8)
        ),
        "plot_paths": plot_paths,
    }


def main() -> None:
    args = parse_args()
    run_name = args.run_name or time.strftime("silicon_nognn_%Y%m%d_%H%M%S")
    run_dir = ensure_dir(Path(args.output_root) / run_name)
    plots_dir = ensure_dir(run_dir / "plots")
    configure_matplotlib(Path("studies/e3mlp_investigation/cache/mplconfig"))

    cfg = StudyConfig(
        train_snapshots=_parse_csv_str(args.train_snapshots, str),
        eval_snapshots=_parse_csv_str(args.eval_snapshots, str),
        target=args.target,
        aggregation=args.aggregation,
        architecture=args.architecture,
        variant=args.variant,
        depth=args.depth,
        hidden_irreps_preset=args.hidden_irreps_preset,
        num_steps=4 if args.smoke_test else args.num_steps,
        lr=args.lr,
        loss_kind=args.loss_kind,
        topk=args.topk,
        message_variant=args.message_variant,
        message_depth=args.message_depth,
        predictor_variant=args.predictor_variant,
        predictor_depth=args.predictor_depth,
        max_train_edges=args.max_train_edges,
        max_eval_edges=args.max_eval_edges,
        device=args.device,
        output_scale=args.output_scale,
        weight_init_scale=args.weight_init_scale,
        residual_scale=args.residual_scale,
        pre_norm=args.pre_norm,
        smoke_test=args.smoke_test,
    )
    dump_yaml(run_dir / "config.yaml", cfg.__dict__)
    print(f"[silicon] run_dir={run_dir}", flush=True)
    print(
        f"[silicon] train_snapshots={cfg.train_snapshots} eval_snapshots={cfg.eval_snapshots} "
        f"target={cfg.target} aggregation={cfg.aggregation} architecture={cfg.architecture}",
        flush=True,
    )
    print(
        f"[silicon] variant={cfg.variant} depth={cfg.depth} loss={cfg.loss_kind} "
        f"output_scale={cfg.output_scale} weight_init_scale={cfg.weight_init_scale} "
        f"residual_scale={cfg.residual_scale} pre_norm={cfg.pre_norm}",
        flush=True,
    )

    train_payloads = [load_snapshot(s) for s in cfg.train_snapshots]
    eval_payloads = [load_snapshot(s) for s in cfg.eval_snapshots]
    train_inputs, train_targets, mapper = train_payloads[0]
    eval_inputs, eval_targets, _ = eval_payloads[0]

    if cfg.target not in train_targets:
        raise ValueError(f"Target '{cfg.target}' missing from train cache")
    if cfg.target not in eval_targets:
        raise ValueError(f"Target '{cfg.target}' missing from eval cache")

    train_data = prepare_pair_data(
        train_inputs, train_targets, cfg.target, max_edges=cfg.max_train_edges
    )
    eval_data = prepare_pair_data(
        eval_inputs, eval_targets, cfg.target, max_edges=cfg.max_eval_edges
    )

    raw_irreps = feature_irreps_from_cache(
        train_inputs["edge_length_emb"], train_inputs["edge_sh"]
    )
    hidden_irreps = get_hidden_irreps(cfg.hidden_irreps_preset)
    output_irreps = mapper.get_pair_irreps(("Si", "Si"))

    model = SiliconNoGNNStudy(
        raw_irreps=raw_irreps,
        hidden_irreps=hidden_irreps,
        output_irreps=output_irreps,
        aggregation=cfg.aggregation,
        architecture=cfg.architecture,
        variant=cfg.variant,
        depth=cfg.depth,
        message_variant=cfg.message_variant,
        message_depth=cfg.message_depth,
        predictor_variant=cfg.predictor_variant,
        predictor_depth=cfg.predictor_depth,
        topk=cfg.topk,
        output_scale=cfg.output_scale,
        weight_init_scale=cfg.weight_init_scale,
        residual_scale=cfg.residual_scale,
        pre_norm=cfg.pre_norm,
    ).to(cfg.device)

    train_edge_features = train_data["edge_features"].to(cfg.device)
    train_edge_index = train_data["edge_index"].to(cfg.device)
    train_neighborhoods = [nbr.to(cfg.device) for nbr in train_data["neighborhoods"]]
    train_target_matrix = train_data["target"]
    eval_edge_features = eval_data["edge_features"].to(cfg.device)
    eval_edge_index = eval_data["edge_index"].to(cfg.device)
    eval_neighborhoods = [nbr.to(cfg.device) for nbr in eval_data["neighborhoods"]]
    eval_target_matrix = eval_data["target"]
    train_target_vec = (
        train_target_matrix.to_vectors(mapper).pair_vectors["Si-Si"].to(cfg.device)
    )
    eval_target_vec = (
        eval_target_matrix.to_vectors(mapper).pair_vectors["Si-Si"].to(cfg.device)
    )
    # train_distances = train_data["edge_dist"].to(cfg.device) # not used
    eval_distances = eval_data["edge_dist"].to(cfg.device)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    rows: list[dict] = []
    t0 = time.perf_counter()
    step_iter = tqdm(
        range(cfg.num_steps),
        desc=f"silicon {cfg.target} {cfg.aggregation}",
        dynamic_ncols=True,
    )
    for step in step_iter:
        print(f"[silicon] step={step + 1}/{cfg.num_steps} start", flush=True)
        optimizer.zero_grad(set_to_none=True)
        pred_vec = model(train_edge_features, train_edge_index, train_neighborhoods)
        loss = loss_summary(cfg.loss_kind, pred_vec, train_target_vec)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            eval_pred_vec = model(
                eval_edge_features, eval_edge_index, eval_neighborhoods
            )
            eval_loss = loss_summary(cfg.loss_kind, eval_pred_vec, eval_target_vec)
            eval_rel = relative_vector_error(eval_pred_vec, eval_target_vec)
        rows.append(
            {
                "step": step,
                "target": cfg.target,
                "aggregation": cfg.aggregation,
                "architecture": cfg.architecture,
                "variant": cfg.variant,
                "depth": cfg.depth,
                "loss_kind": cfg.loss_kind,
                "train_loss": float(loss.item()),
                "eval_loss": float(eval_loss.item()),
                "eval_rel": float(eval_rel.item()),
            }
        )
        step_iter.set_postfix(
            train=f"{loss.item():.3e}",
            eval=f"{eval_loss.item():.3e}",
            rel=f"{eval_rel.item():.3e}",
        )
        if (
            step == 0
            or (step + 1) == cfg.num_steps
            or (step + 1) % max(1, cfg.num_steps // 4) == 0
        ):
            print(
                f"[silicon] progress step={step + 1}/{cfg.num_steps} "
                f"train={loss.item():.4e} eval={eval_loss.item():.4e} rel={eval_rel.item():.4e}",
                flush=True,
            )

    elapsed = time.perf_counter() - t0
    df = pd.DataFrame(rows)
    save_df(run_dir / "metrics.csv", df)
    plot_paths = []
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.lineplot(data=df, x="step", y="train_loss", ax=ax, label="train")
    sns.lineplot(data=df, x="step", y="eval_loss", ax=ax, label="eval")
    ax.set_yscale("log")
    ax.set_title("No-GNN silicon loss curves")
    path = plots_dir / "loss_curves.png"
    save_plot(fig, path)
    plot_paths.append(path)
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.lineplot(data=df, x="step", y="eval_rel", ax=ax, label="eval relative error")
    ax.set_yscale("log")
    ax.set_title("No-GNN silicon relative error curve")
    path = plots_dir / "relative_error_curve.png"
    save_plot(fig, path)
    plot_paths.append(path)

    with torch.no_grad():
        final_eval = evaluate(
            model=model,
            edge_features=eval_edge_features,
            edge_index=eval_edge_index,
            neighborhoods=eval_neighborhoods,
            target_matrix=eval_target_matrix,
            gt_density=eval_data["gt_density"],
            gt_hamiltonian=eval_data["gt_hamiltonian"],
            target_vec=eval_target_vec,
            mapper=mapper,
            target_name=cfg.target,
            loss_kind=cfg.loss_kind,
            eval_distances=eval_distances,
            out_dir=plots_dir,
        )
        plot_paths.extend(final_eval["plot_paths"])

    summary = pd.DataFrame(
        [
            {
                "target": cfg.target,
                "aggregation": cfg.aggregation,
                "architecture": cfg.architecture,
                "variant": cfg.variant,
                "depth": cfg.depth,
                "loss_kind": cfg.loss_kind,
                "train_snapshots": ",".join(cfg.train_snapshots),
                "eval_snapshots": ",".join(cfg.eval_snapshots),
                "final_train_loss": float(df.iloc[-1]["train_loss"]),
                "final_eval_loss": float(df.iloc[-1]["eval_loss"]),
                "final_eval_rel": float(df.iloc[-1]["eval_rel"]),
                "final_eval_block_mae": final_eval["block_mae"],
                "final_eval_block_mse": final_eval["block_mse"],
                "final_eval_block_rel": final_eval["block_rel"],
                "final_energy_mae": final_eval["energy_mae"],
                "final_energy_rel": final_eval["energy_rel"],
                "elapsed_sec": elapsed,
            }
        ]
    )
    save_df(run_dir / "summary.csv", summary)
    dump_json(
        run_dir / "summary.json",
        {
            "run_name": run_name,
            "hidden_irreps": str(hidden_irreps),
            "raw_irreps": str(raw_irreps),
            "output_irreps": str(output_irreps),
            "plot_paths": [str(p) for p in plot_paths],
            "summary": summary.to_dict(orient="records"),
        },
    )

    if args.append_log:
        title = time.strftime("%Y-%m-%d %H:%M") + f" - {run_name}"
        lines = [
            "Executed silicon no-GNN study.",
            "",
            f"- run dir: [{run_dir.name}](/home/bartek/casus/mandala/{run_dir})",
            f"- train snapshots: `{cfg.train_snapshots}`",
            f"- eval snapshots: `{cfg.eval_snapshots}`",
            f"- target: `{cfg.target}`",
            f"- aggregation: `{cfg.aggregation}`",
            f"- architecture: `{cfg.architecture}`",
            f"- variant: `{cfg.variant}` depth={cfg.depth}",
            f"- hidden irreps preset: `{cfg.hidden_irreps_preset}`",
            "",
            "Top provisional rows:",
            (
                f"- final eval loss={summary.iloc[0]['final_eval_loss']:.4e} "
                f"block_mae={summary.iloc[0]['final_eval_block_mae']:.4e} "
                f"energy_mae={summary.iloc[0]['final_energy_mae']:.4e}"
            ),
            "",
            "Artifacts:",
            f"- [summary.csv](/home/bartek/casus/mandala/{run_dir / 'summary.csv'})",
            f"- [loss curves](/home/bartek/casus/mandala/{plots_dir / 'loss_curves.png'})",
            f"- [per-irrep MAE](/home/bartek/casus/mandala/{plots_dir / 'per_irrep_mae.png'})",
            f"- [distance MAE](/home/bartek/casus/mandala/{plots_dir / 'distance_mae_hexbin.png'})",
            f"- [sample blocks](/home/bartek/casus/mandala/{plots_dir / 'sample_blocks.png'})",
            "",
            "Status:",
            "- provisional local result; larger runs may override it",
        ]
        append_study_log(Path("studies/e3mlp_investigation/STUDY_LOG.md"), title, lines)

    print(f"[silicon] finished in {elapsed:.1f}s", flush=True)


if __name__ == "__main__":
    main()
