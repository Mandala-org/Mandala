from __future__ import annotations

import argparse
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
import torch.nn.functional as F
from e3nn.o3 import Irreps
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
    variants: list[str]
    depths: list[int]
    output_scales: list[float]
    weight_init_scales: list[float]
    residual_scales: list[float]
    num_steps: int
    num_seeds: int
    batch_size: int
    lr: float
    hidden_irreps_preset: str
    device: str
    pre_norm: bool
    smoke_test: bool


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run a fast E3MLP stability micro-study.")
    p.add_argument(
        "--variants",
        type=str,
        default="normact,gatemagnitudes,resnormact,resgatemagnitudes,film,bilinear",
    )
    p.add_argument("--depths", type=str, default="4,6,10")
    p.add_argument("--output-scales", type=str, default="0.5,1.0,1.5")
    p.add_argument("--weight-init-scales", type=str, default="0.5,1.0,1.5")
    p.add_argument("--residual-scales", type=str, default="0.1,0.25")
    p.add_argument("--num-steps", type=int, default=12)
    p.add_argument("--num-seeds", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument(
        "--hidden-irreps-preset",
        type=str,
        default="preliminary",
        choices=["preliminary", "silicon", "silicon_doubled"],
    )
    p.add_argument("--device", type=str, default="cpu")
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


def _record_layer_stats(
    activations: dict[str, list[float]],
    x: torch.Tensor,
    irreps: Irreps,
    prefix: str,
) -> None:
    cursor = 0
    for mul, ir in irreps:
        size = mul * ir.dim
        block = x[..., cursor : cursor + size].reshape(x.shape[0], mul, ir.dim)
        mags = torch.linalg.norm(block, dim=-1).mean().item()
        activations[f"{prefix}_{mul}x{ir}"] = activations.get(
            f"{prefix}_{mul}x{ir}", []
        )
        activations[f"{prefix}_{mul}x{ir}"].append(mags)
        cursor += size


def relative_vector_error(
    pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8
) -> torch.Tensor:
    diff = (pred - target).norm(dim=-1)
    denom = target.norm(dim=-1).clamp_min(eps)
    return (diff / denom).mean()


def train_once(
    *,
    variant: str,
    depth: int,
    output_scale: float,
    weight_init_scale: float,
    residual_scale: float,
    irreps: Irreps,
    num_steps: int,
    batch_size: int,
    lr: float,
    device: str,
    seed: int,
    pre_norm: bool,
):
    print(
        f"[stability] seed={seed} variant={variant} depth={depth} "
        f"output_scale={output_scale} weight_init_scale={weight_init_scale} "
        f"residual_scale={residual_scale} device={device}",
        flush=True,
    )
    torch.manual_seed(seed)
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
    target_linear = torch.randn(irreps.dim, irreps.dim, device=device) / math.sqrt(
        irreps.dim
    )

    step_rows: list[dict] = []
    per_irrep_rows: list[dict] = []
    t0 = time.perf_counter()
    step_iter = tqdm(
        range(num_steps),
        desc=f"stability {variant} d={depth} os={output_scale}",
        leave=False,
        dynamic_ncols=True,
    )
    for step in step_iter:
        x = irreps.randn(batch_size, -1).to(device)
        y_target = x @ target_linear.T
        optimizer.zero_grad(set_to_none=True)

        hidden_stats: dict[str, list[float]] = {}
        y, intermediates = model.forward_with_intermediates(x)
        for idx, tensor in enumerate(intermediates):
            if tensor.shape[-1] == irreps.dim:
                _record_layer_stats(
                    hidden_stats, tensor.detach(), irreps, f"layer{idx:02d}"
                )

        loss = F.mse_loss(y, y_target)
        loss.backward()

        grad_sq = 0.0
        max_grad = 0.0
        weight_sq = 0.0
        for p in model.parameters():
            if p.grad is not None:
                g = p.grad.detach().norm().item()
                grad_sq += g * g
                max_grad = max(max_grad, g)
            w = p.detach().norm().item()
            weight_sq += w * w

        optimizer.step()

        output_norm = y.detach().norm(dim=-1).mean().item()
        input_norm = x.detach().norm(dim=-1).mean().item()
        rel_err = relative_vector_error(y.detach(), y_target.detach()).item()
        nan_flag = bool(torch.isnan(y).any().item() or torch.isnan(loss).item())
        step_rows.append(
            {
                "variant": variant,
                "depth": depth,
                "output_scale": output_scale,
                "weight_init_scale": weight_init_scale,
                "residual_scale": residual_scale,
                "seed": seed,
                "step": step,
                "loss": float(loss.item()),
                "input_norm": input_norm,
                "output_norm": output_norm,
                "output_to_input_ratio": output_norm / max(input_norm, 1e-12),
                "relative_error": rel_err,
                "grad_norm": grad_sq**0.5,
                "max_grad": max_grad,
                "weight_norm": weight_sq**0.5,
                "has_nan": nan_flag,
            }
        )
        if (
            step == 0
            or (step + 1) == num_steps
            or (step + 1) % max(1, num_steps // 4) == 0
        ):
            print(
                f"[stability] progress seed={seed} variant={variant} depth={depth} "
                f"step={step + 1}/{num_steps} loss={loss.item():.4e} "
                f"ratio={output_norm / max(input_norm, 1e-12):.3f} rel={rel_err:.4e}",
                flush=True,
            )

        for key, values in hidden_stats.items():
            per_irrep_rows.append(
                {
                    "variant": variant,
                    "depth": depth,
                    "output_scale": output_scale,
                    "weight_init_scale": weight_init_scale,
                    "residual_scale": residual_scale,
                    "seed": seed,
                    "step": step,
                    "irrep_key": key,
                    "mean_activation": float(sum(values) / len(values)),
                }
            )

        if nan_flag:
            print(
                f"[stability] early stop on NaN at step={step} variant={variant} depth={depth}",
                flush=True,
            )
            break

    elapsed = time.perf_counter() - t0
    return step_rows, per_irrep_rows, elapsed


def make_summary_plots(
    df_steps: pd.DataFrame, df_irreps: pd.DataFrame, out_dir: Path
) -> list[Path]:
    paths: list[Path] = []

    if not df_steps.empty:
        last = (
            df_steps.sort_values("step")
            .groupby(
                [
                    "variant",
                    "depth",
                    "output_scale",
                    "weight_init_scale",
                    "residual_scale",
                    "seed",
                ],
                as_index=False,
            )
            .tail(1)
        )

        heat = last.groupby(["variant", "depth", "output_scale"], as_index=False)[
            "output_to_input_ratio"
        ].mean()
        for variant, group in heat.groupby("variant"):
            pivot = group.pivot(
                index="depth", columns="output_scale", values="output_to_input_ratio"
            )
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.heatmap(pivot, annot=True, fmt=".2f", cmap="viridis", ax=ax)
            ax.set_title(f"{variant}: output/input norm ratio")
            path = out_dir / f"heatmap_norm_ratio_{variant}.png"
            save_plot(fig, path)
            paths.append(path)

        fig, ax = plt.subplots(figsize=(8, 5))
        sns.lineplot(
            data=df_steps,
            x="step",
            y="loss",
            hue="variant",
            style="depth",
            estimator="mean",
            errorbar=None,
            ax=ax,
        )
        ax.set_yscale("log")
        ax.set_title("Training loss over steps")
        path = out_dir / "loss_curves.png"
        save_plot(fig, path)
        paths.append(path)

        fig, ax = plt.subplots(figsize=(8, 5))
        sns.lineplot(
            data=df_steps,
            x="step",
            y="grad_norm",
            hue="variant",
            style="depth",
            estimator="mean",
            errorbar=None,
            ax=ax,
        )
        ax.set_yscale("log")
        ax.set_title("Gradient norms over steps")
        path = out_dir / "gradient_curves.png"
        save_plot(fig, path)
        paths.append(path)

    if not df_irreps.empty:
        summary = df_irreps.groupby(["variant", "depth", "irrep_key"], as_index=False)[
            "mean_activation"
        ].mean()
        for variant, group in summary.groupby("variant"):
            fig, ax = plt.subplots(figsize=(10, 5))
            sns.barplot(
                data=group, x="irrep_key", y="mean_activation", hue="depth", ax=ax
            )
            ax.tick_params(axis="x", rotation=90)
            ax.set_title(f"{variant}: mean activation by irrep")
            path = out_dir / f"per_irrep_activation_{variant}.png"
            save_plot(fig, path)
            paths.append(path)

    return paths


def main() -> None:
    args = parse_args()
    run_name = args.run_name or time.strftime("stability_microstudy_%Y%m%d_%H%M%S")
    output_root = ensure_dir(Path(args.output_root))
    run_dir = ensure_dir(output_root / run_name)
    plots_dir = ensure_dir(run_dir / "plots")
    configure_matplotlib(Path("studies/e3mlp_investigation/cache/mplconfig"))

    cfg = StudyConfig(
        variants=_parse_csv_str(args.variants, str),
        depths=_parse_csv_str(args.depths, int),
        output_scales=_parse_csv_str(args.output_scales, float),
        weight_init_scales=_parse_csv_str(args.weight_init_scales, float),
        residual_scales=_parse_csv_str(args.residual_scales, float),
        num_steps=3 if args.smoke_test else args.num_steps,
        num_seeds=1 if args.smoke_test else args.num_seeds,
        batch_size=args.batch_size,
        lr=args.lr,
        hidden_irreps_preset=args.hidden_irreps_preset,
        device=args.device,
        pre_norm=args.pre_norm,
        smoke_test=args.smoke_test,
    )
    dump_yaml(run_dir / "config.yaml", cfg.__dict__)
    print(f"[stability] run_dir={run_dir}", flush=True)

    irreps = get_hidden_irreps(cfg.hidden_irreps_preset)
    print(f"[stability] hidden_irreps={irreps}", flush=True)
    print(
        f"[stability] variants={cfg.variants} depths={cfg.depths} output_scales={cfg.output_scales} "
        f"weight_init_scales={cfg.weight_init_scales} residual_scales={cfg.residual_scales} pre_norm={cfg.pre_norm}",
        flush=True,
    )
    total_configs = (
        len(cfg.variants)
        * len(cfg.depths)
        * len(cfg.output_scales)
        * len(cfg.weight_init_scales)
        * len(cfg.residual_scales)
        * cfg.num_seeds
    )
    completed = 0
    all_steps: list[dict] = []
    all_irreps: list[dict] = []
    timing_rows: list[dict] = []

    for variant in cfg.variants:
        print(f"[stability] variant sweep start: {variant}", flush=True)
        for depth in tqdm(
            cfg.depths,
            desc=f"[stability] depths {variant}",
            leave=False,
            dynamic_ncols=True,
        ):
            for output_scale in tqdm(
                cfg.output_scales,
                desc=f"[stability] output scales d={depth}",
                leave=False,
                dynamic_ncols=True,
            ):
                for weight_init_scale in tqdm(
                    cfg.weight_init_scales,
                    desc=f"[stability] init scales d={depth} os={output_scale}",
                    leave=False,
                    dynamic_ncols=True,
                ):
                    for residual_scale in tqdm(
                        cfg.residual_scales,
                        desc=f"[stability] residual scales d={depth} os={output_scale}",
                        leave=False,
                        dynamic_ncols=True,
                    ):
                        for seed in tqdm(
                            range(cfg.num_seeds),
                            desc=f"[stability] seeds d={depth} os={output_scale}",
                            leave=False,
                            dynamic_ncols=True,
                        ):
                            step_rows, per_irrep_rows, elapsed = train_once(
                                variant=variant,
                                depth=depth,
                                output_scale=output_scale,
                                weight_init_scale=weight_init_scale,
                                residual_scale=residual_scale,
                                irreps=irreps,
                                num_steps=cfg.num_steps,
                                batch_size=cfg.batch_size,
                                lr=cfg.lr,
                                device=cfg.device,
                                seed=seed,
                                pre_norm=cfg.pre_norm,
                            )
                            all_steps.extend(step_rows)
                            all_irreps.extend(per_irrep_rows)
                            timing_rows.append(
                                {
                                    "variant": variant,
                                    "depth": depth,
                                    "output_scale": output_scale,
                                    "weight_init_scale": weight_init_scale,
                                    "residual_scale": residual_scale,
                                    "seed": seed,
                                    "elapsed_sec": elapsed,
                                }
                            )
                            completed += 1
                            if step_rows:
                                current = step_rows[-1]
                                print(
                                    f"[stability] config {completed}/{total_configs} done "
                                    f"variant={variant} depth={depth} os={output_scale} "
                                    f"init={weight_init_scale} res={residual_scale} seed={seed} "
                                    f"loss={current['loss']:.4e} ratio={current['output_to_input_ratio']:.3f} "
                                    f"rel={current['relative_error']:.4e}",
                                    flush=True,
                                )
        print(f"[stability] variant sweep done: {variant}", flush=True)

    df_steps = pd.DataFrame(all_steps)
    df_irreps = pd.DataFrame(all_irreps)
    df_timing = pd.DataFrame(timing_rows)
    save_df(run_dir / "metrics_steps.csv", df_steps)
    save_df(run_dir / "metrics_irreps.csv", df_irreps)
    save_df(run_dir / "timings.csv", df_timing)

    plot_paths = make_summary_plots(df_steps, df_irreps, plots_dir)

    if not df_steps.empty:
        last = (
            df_steps.sort_values("step")
            .groupby(
                [
                    "variant",
                    "depth",
                    "output_scale",
                    "weight_init_scale",
                    "residual_scale",
                    "seed",
                ],
                as_index=False,
            )
            .tail(1)
        )
        summary = (
            last.groupby(
                [
                    "variant",
                    "depth",
                    "output_scale",
                    "weight_init_scale",
                    "residual_scale",
                ],
                as_index=False,
            )
            .agg(
                final_loss=("loss", "mean"),
                final_ratio=("output_to_input_ratio", "mean"),
                final_relative_error=("relative_error", "mean"),
                final_grad_norm=("grad_norm", "mean"),
                has_nan=("has_nan", "max"),
            )
            .sort_values(
                ["has_nan", "final_relative_error", "final_loss", "final_grad_norm"]
            )
        )
    else:
        summary = pd.DataFrame()
    save_df(run_dir / "summary.csv", summary)
    print(f"[stability] wrote summary and plots to {run_dir}", flush=True)
    dump_json(
        run_dir / "summary.json",
        {
            "run_name": run_name,
            "hidden_irreps": str(irreps),
            "num_configs": int(
                len(cfg.variants)
                * len(cfg.depths)
                * len(cfg.output_scales)
                * len(cfg.weight_init_scales)
                * len(cfg.residual_scales)
                * cfg.num_seeds
            ),
            "plot_paths": [str(p) for p in plot_paths],
            "top_rows": summary.head(10).to_dict(orient="records"),
        },
    )

    if args.append_log:
        title = time.strftime("%Y-%m-%d %H:%M") + f" - {run_name}"
        top_rows = (
            summary.head(5).to_dict(orient="records") if not summary.empty else []
        )
        lines = [
            "Executed stability micro-study.",
            "",
            f"- run dir: [{run_dir.name}](/home/bartek/casus/mandala/{run_dir})",
            f"- hidden irreps preset: `{cfg.hidden_irreps_preset}`",
            f"- hidden irreps: `{irreps}`",
            f"- smoke test: `{cfg.smoke_test}`",
            "",
            "Top provisional rows:",
        ]
        for row in top_rows:
            lines.append(
                f"- `{row['variant']}` depth={row['depth']} output_scale={row['output_scale']} "
                f"weight_init_scale={row['weight_init_scale']} residual_scale={row['residual_scale']} "
                f"loss={row['final_loss']:.4e} ratio={row['final_ratio']:.3f} grad={row['final_grad_norm']:.3e} "
                f"nan={bool(row['has_nan'])}"
            )
        lines.extend(
            [
                "",
                "Artifacts:",
                f"- [summary.csv](/home/bartek/casus/mandala/{run_dir / 'summary.csv'})",
                f"- [loss curves](/home/bartek/casus/mandala/{plots_dir / 'loss_curves.png'})",
                f"- [gradient curves](/home/bartek/casus/mandala/{plots_dir / 'gradient_curves.png'})",
                "",
                "Status:",
                "- provisional local result; larger runs may override it",
            ]
        )
        append_study_log(
            Path("studies/e3mlp_investigation/STUDY_LOG.md"),
            title,
            lines,
        )


if __name__ == "__main__":
    main()
