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
import torch.nn.functional as F
from e3nn.o3 import FullyConnectedTensorProduct, Irrep, Irreps, Linear
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

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
    teacher_kind: str
    variants: list[str]
    depths: list[int]
    loss_kinds: list[str]
    num_train: int
    num_val: int
    batch_size: int
    num_steps: int
    lr: float
    num_seeds: int
    hidden_irreps_preset: str
    device: str
    output_scale: float
    weight_init_scale: float
    residual_scale: float
    pre_norm: bool
    smoke_test: bool


class Teacher(torch.nn.Module):
    def __init__(self, irreps: Irreps, teacher_kind: str) -> None:
        super().__init__()
        self.irreps = Irreps(irreps)
        self.teacher_kind = teacher_kind
        self.linear = Linear(self.irreps, self.irreps, biases=False)
        self.linear.weight.data.mul_(0.6)

        reachable: set[Irrep] = set()
        for _, ir1 in self.irreps:
            for _, ir2 in self.irreps:
                for ir_out in ir1 * ir2:
                    reachable.add(ir_out)
        interaction_irreps = Irreps(
            [(1, ir) for ir in sorted(reachable, key=lambda ir: (ir.l, ir.p))]
        )
        self.tp = FullyConnectedTensorProduct(
            self.irreps,
            self.irreps,
            interaction_irreps,
            internal_weights=True,
            shared_weights=True,
        )
        self.tp.weight.data.mul_(0.25)
        self.proj = Linear(interaction_irreps, self.irreps, biases=False)
        self.proj.weight.data.mul_(0.25)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        linear = self.linear(x)
        quad = self.proj(self.tp(x, x))
        if self.teacher_kind == "linear":
            return linear
        if self.teacher_kind == "quadratic":
            return quad
        if self.teacher_kind == "mixed":
            return linear + quad
        raise ValueError(f"Unknown teacher kind '{self.teacher_kind}'")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run a synthetic teacher E3MLP study.")
    p.add_argument(
        "--teacher-kind",
        type=str,
        default="mixed",
        choices=["linear", "quadratic", "mixed"],
    )
    p.add_argument(
        "--variants",
        type=str,
        default="normact,gatemagnitudes,resnormact,film,bilinear",
    )
    p.add_argument("--depths", type=str, default="2,4,6")
    p.add_argument("--loss-kinds", type=str, default="mse,mae,huber")
    p.add_argument("--num-train", type=int, default=256)
    p.add_argument("--num-val", type=int, default=64)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-steps", type=int, default=30)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--num-seeds", type=int, default=2)
    p.add_argument("--hidden-irreps-preset", type=str, default="preliminary")
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


def make_dataset(
    irreps: Irreps, teacher_kind: str, n: int, seed: int
) -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(seed)
    teacher = Teacher(irreps, teacher_kind)
    x = irreps.randn(n, -1)
    with torch.no_grad():
        y = teacher(x)
    return x, y


def train_one(
    *,
    irreps: Irreps,
    variant: str,
    depth: int,
    loss_kind: str,
    teacher_kind: str,
    num_train: int,
    num_val: int,
    batch_size: int,
    num_steps: int,
    lr: float,
    seed: int,
    device: str,
    output_scale: float,
    weight_init_scale: float,
    residual_scale: float,
    pre_norm: bool,
):
    print(
        f"[synthetic] seed={seed} variant={variant} depth={depth} loss={loss_kind} device={device}",
        flush=True,
    )
    train_x, train_y = make_dataset(irreps, teacher_kind, num_train, seed)
    val_x, val_y = make_dataset(irreps, teacher_kind, num_val, seed + 10_000)
    train_x = train_x.to(device)
    train_y = train_y.to(device)
    val_x = val_x.to(device)
    val_y = val_y.to(device)

    model = build_variant(
        variant,
        irreps,
        irreps,
        irreps,
        num_layers=depth,
        output_scale=output_scale,
        weight_init_scale=weight_init_scale,
        residual_scale=residual_scale,
        pre_norm=pre_norm,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    rows: list[dict] = []

    step_iter = tqdm(
        range(num_steps),
        desc=f"synthetic {variant} d={depth} {loss_kind}",
        leave=False,
        dynamic_ncols=True,
    )
    for step in step_iter:
        idx = torch.randint(0, num_train, (batch_size,))
        x = train_x[idx]
        y = train_y[idx]
        optimizer.zero_grad(set_to_none=True)
        pred = model(x)
        loss = loss_fn(loss_kind, pred, y)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            val_pred = model(val_x)
            val_loss = loss_fn(loss_kind, val_pred, val_y)
            val_mae = F.l1_loss(val_pred, val_y)
            val_mse = F.mse_loss(val_pred, val_y)
            val_rel = relative_vector_error(val_pred, val_y)
        rows.append(
            {
                "variant": variant,
                "depth": depth,
                "loss_kind": loss_kind,
                "teacher_kind": teacher_kind,
                "seed": seed,
                "step": step,
                "train_loss": float(loss.item()),
                "val_loss": float(val_loss.item()),
                "val_mae": float(val_mae.item()),
                "val_mse": float(val_mse.item()),
                "val_rel": float(val_rel.item()),
            }
        )
        if (
            step == 0
            or (step + 1) == num_steps
            or (step + 1) % max(1, num_steps // 4) == 0
        ):
            print(
                f"[synthetic] progress seed={seed} variant={variant} depth={depth} loss={loss_kind} "
                f"step={step + 1}/{num_steps} train={loss.item():.4e} val={val_loss.item():.4e} rel={val_rel.item():.4e}",
                flush=True,
            )
    with torch.no_grad():
        final_pred = model(val_x)
    return rows, val_x, val_y, final_pred


def make_plots(df: pd.DataFrame, out_dir: Path) -> list[Path]:
    paths: list[Path] = []
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.lineplot(
        data=df,
        x="step",
        y="val_loss",
        hue="variant",
        style="depth",
        ax=ax,
        errorbar=None,
    )
    ax.set_yscale("log")
    ax.set_title("Synthetic teacher validation loss")
    path = out_dir / "val_loss_curves.png"
    save_plot(fig, path)
    paths.append(path)

    final = (
        df.sort_values("step")
        .groupby(
            ["variant", "depth", "loss_kind", "teacher_kind", "seed"], as_index=False
        )
        .tail(1)
    )
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.barplot(data=final, x="variant", y="val_mae", hue="depth", ax=ax)
    ax.set_yscale("log")
    ax.set_title("Final validation MAE by variant")
    path = out_dir / "final_val_mae.png"
    save_plot(fig, path)
    paths.append(path)
    return paths


def make_per_irrep_error_plot(
    irreps: Irreps,
    target: torch.Tensor,
    pred: torch.Tensor,
    out_dir: Path,
) -> Path:
    rows: list[dict] = []
    cursor = 0
    for mul, ir in irreps:
        size = mul * ir.dim
        t = target[:, cursor : cursor + size].reshape(target.shape[0], mul, ir.dim)
        p = pred[:, cursor : cursor + size].reshape(pred.shape[0], mul, ir.dim)
        mae = (p - t).abs().mean(dim=(0, 2))
        for idx in range(mul):
            rows.append({"irrep": f"{ir}", "copy": idx, "mae": float(mae[idx].item())})
        cursor += size
    df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(9, 4))
    sns.barplot(data=df, x="irrep", y="mae", ax=ax, estimator="mean", errorbar=None)
    ax.set_yscale("log")
    ax.set_title("Per-irrep validation MAE")
    path = out_dir / "per_irrep_val_mae.png"
    save_plot(fig, path)
    return path


def make_norm_scatter_plot(
    target: torch.Tensor, pred: torch.Tensor, out_dir: Path
) -> Path:
    target_norm = target.norm(dim=-1).cpu().numpy()
    pred_norm = pred.norm(dim=-1).cpu().numpy()
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(target_norm, pred_norm, s=12, alpha=0.7)
    lim_lo = min(target_norm.min(), pred_norm.min())
    lim_hi = max(target_norm.max(), pred_norm.max())
    ax.plot(
        [lim_lo, lim_hi], [lim_lo, lim_hi], linestyle="--", color="black", linewidth=1.0
    )
    ax.set_xlabel("Target norm")
    ax.set_ylabel("Prediction norm")
    ax.set_title("Target vs prediction norm")
    path = out_dir / "target_vs_prediction_norm.png"
    save_plot(fig, path)
    return path


def main() -> None:
    args = parse_args()
    run_name = args.run_name or time.strftime("synthetic_teacher_%Y%m%d_%H%M%S")
    run_dir = ensure_dir(Path(args.output_root) / run_name)
    plots_dir = ensure_dir(run_dir / "plots")
    configure_matplotlib(Path("studies/e3mlp_investigation/cache/mplconfig"))

    cfg = StudyConfig(
        teacher_kind=args.teacher_kind,
        variants=_parse_csv_str(args.variants, str),
        depths=_parse_csv_str(args.depths, int),
        loss_kinds=_parse_csv_str(args.loss_kinds, str),
        num_train=64 if args.smoke_test else args.num_train,
        num_val=32 if args.smoke_test else args.num_val,
        batch_size=min(args.batch_size, 16) if args.smoke_test else args.batch_size,
        num_steps=4 if args.smoke_test else args.num_steps,
        lr=args.lr,
        num_seeds=1 if args.smoke_test else args.num_seeds,
        hidden_irreps_preset=args.hidden_irreps_preset,
        device=args.device,
        output_scale=args.output_scale,
        weight_init_scale=args.weight_init_scale,
        residual_scale=args.residual_scale,
        pre_norm=args.pre_norm,
        smoke_test=args.smoke_test,
    )
    irreps = get_hidden_irreps(cfg.hidden_irreps_preset)
    dump_yaml(run_dir / "config.yaml", cfg.__dict__)
    print(f"[synthetic] run_dir={run_dir}", flush=True)
    print(
        f"[synthetic] teacher_kind={cfg.teacher_kind} device={cfg.device}", flush=True
    )
    print(f"[synthetic] hidden_irreps={irreps}", flush=True)
    print(
        f"[synthetic] variants={cfg.variants} depths={cfg.depths} losses={cfg.loss_kinds} "
        f"output_scale={cfg.output_scale} weight_init_scale={cfg.weight_init_scale} "
        f"residual_scale={cfg.residual_scale} pre_norm={cfg.pre_norm}",
        flush=True,
    )
    total_configs = (
        len(cfg.variants) * len(cfg.depths) * len(cfg.loss_kinds) * cfg.num_seeds
    )
    completed = 0

    rows: list[dict] = []
    sample_payload = None
    for variant in cfg.variants:
        print(f"[synthetic] variant sweep start: {variant}", flush=True)
        for depth in tqdm(
            cfg.depths,
            desc=f"[synthetic] depths {variant}",
            leave=False,
            dynamic_ncols=True,
        ):
            for loss_kind in tqdm(
                cfg.loss_kinds,
                desc=f"[synthetic] losses d={depth}",
                leave=False,
                dynamic_ncols=True,
            ):
                for seed in tqdm(
                    range(cfg.num_seeds),
                    desc=f"[synthetic] seeds d={depth} {loss_kind}",
                    leave=False,
                    dynamic_ncols=True,
                ):
                    run_rows, val_x, val_y, final_pred = train_one(
                        irreps=irreps,
                        variant=variant,
                        depth=depth,
                        loss_kind=loss_kind,
                        teacher_kind=cfg.teacher_kind,
                        num_train=cfg.num_train,
                        num_val=cfg.num_val,
                        batch_size=cfg.batch_size,
                        num_steps=cfg.num_steps,
                        lr=cfg.lr,
                        seed=seed,
                        device=cfg.device,
                        output_scale=cfg.output_scale,
                        weight_init_scale=cfg.weight_init_scale,
                        residual_scale=cfg.residual_scale,
                        pre_norm=cfg.pre_norm,
                    )
                    rows.extend(run_rows)
                    completed += 1
                    current = pd.DataFrame(run_rows).iloc[-1]
                    print(
                        f"[synthetic] config {completed}/{total_configs} done "
                        f"variant={variant} depth={depth} loss={loss_kind} seed={seed} "
                        f"val_mae={current['val_mae']:.4e} val_rel={current['val_rel']:.4e}",
                        flush=True,
                    )
                    if sample_payload is None:
                        sample_payload = (
                            variant,
                            depth,
                            loss_kind,
                            val_y[:8],
                            final_pred[:8],
                        )
        print(f"[synthetic] variant sweep done: {variant}", flush=True)
    df = pd.DataFrame(rows)
    save_df(run_dir / "metrics.csv", df)
    plot_paths = make_plots(df, plots_dir)

    if sample_payload is not None:
        variant, depth, loss_kind, target_sample, pred_sample = sample_payload
        fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
        axes[0].imshow(target_sample.cpu().numpy(), aspect="auto", cmap="coolwarm")
        axes[0].set_title(
            f"Target sample coefficients ({variant}, depth={depth}, loss={loss_kind})"
        )
        axes[1].imshow(
            pred_sample.detach().cpu().numpy(), aspect="auto", cmap="coolwarm"
        )
        axes[1].set_title("Prediction sample coefficients")
        path = plots_dir / "sample_target_vs_prediction.png"
        save_plot(fig, path)
        plot_paths.append(path)
        plot_paths.append(
            make_per_irrep_error_plot(irreps, target_sample, pred_sample, plots_dir)
        )
        plot_paths.append(make_norm_scatter_plot(target_sample, pred_sample, plots_dir))

    final = (
        df.sort_values("step")
        .groupby(
            ["variant", "depth", "loss_kind", "teacher_kind", "seed"], as_index=False
        )
        .tail(1)
    )
    summary = (
        final.groupby(["variant", "depth", "loss_kind"], as_index=False)
        .agg(
            val_loss=("val_loss", "mean"),
            val_mae=("val_mae", "mean"),
            val_mse=("val_mse", "mean"),
            val_rel=("val_rel", "mean"),
        )
        .sort_values(["val_rel", "val_mae", "val_mse"])
    )
    save_df(run_dir / "summary.csv", summary)
    print(f"[synthetic] wrote summary and plots to {run_dir}", flush=True)
    dump_json(
        run_dir / "summary.json",
        {
            "run_name": run_name,
            "teacher_kind": cfg.teacher_kind,
            "hidden_irreps": str(irreps),
            "top_rows": summary.head(10).to_dict(orient="records"),
            "plots": [str(p) for p in plot_paths],
        },
    )

    if args.append_log:
        title = time.strftime("%Y-%m-%d %H:%M") + f" - {run_name}"
        lines = [
            "Executed synthetic teacher study.",
            "",
            f"- run dir: [{run_dir.name}](/home/bartek/casus/mandala/{run_dir})",
            f"- teacher kind: `{cfg.teacher_kind}`",
            f"- hidden irreps preset: `{cfg.hidden_irreps_preset}`",
            f"- hidden irreps: `{irreps}`",
            f"- smoke test: `{cfg.smoke_test}`",
            "",
            "Top provisional rows:",
        ]
        for row in summary.head(5).to_dict(orient="records"):
            lines.append(
                f"- `{row['variant']}` depth={row['depth']} loss={row['loss_kind']} "
                f"val_mae={row['val_mae']:.4e} val_mse={row['val_mse']:.4e}"
            )
        lines.extend(
            [
                "",
                "Artifacts:",
                f"- [summary.csv](/home/bartek/casus/mandala/{run_dir / 'summary.csv'})",
                f"- [validation loss curves](/home/bartek/casus/mandala/{plots_dir / 'val_loss_curves.png'})",
                f"- [sample target vs prediction](/home/bartek/casus/mandala/{plots_dir / 'sample_target_vs_prediction.png'})",
                f"- [per-irrep validation MAE](/home/bartek/casus/mandala/{plots_dir / 'per_irrep_val_mae.png'})",
                "",
                "Status:",
                "- provisional local result; larger runs may override it",
            ]
        )
        append_study_log(Path("studies/e3mlp_investigation/STUDY_LOG.md"), title, lines)


if __name__ == "__main__":
    main()
