from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from studies.e3mlp_investigation.experiment_utils import (  # noqa: E402
    configure_matplotlib,
    ensure_dir,
    save_plot,
)


def load_csv(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    return pd.read_csv(path)


def make_synth_heatmap(df: pd.DataFrame, teacher: str, out_dir: Path) -> list[Path]:
    paths: list[Path] = []
    pivot = (
        df.groupby(["variant", "depth"], as_index=False)["val_mae"]
        .min()
        .pivot(index="variant", columns="depth", values="val_mae")
    )
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.heatmap(pivot, annot=True, fmt=".3f", cmap="mako_r", ax=ax)
    ax.set_title(f"{teacher}: best validation MAE by variant and depth")
    path = out_dir / f"synthetic_heatmap_{teacher}_val_mae.png"
    save_plot(fig, path)
    paths.append(path)

    top = df.nsmallest(10, "val_mae").copy()
    top["label"] = top.apply(
        lambda row: f"{row['variant']} d={int(row['depth'])} {row['loss_kind']}", axis=1
    )
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=top, x="val_mae", y="label", hue="loss_kind", dodge=False, ax=ax)
    ax.set_xscale("log")
    ax.set_title(f"{teacher}: top configurations")
    path = out_dir / f"synthetic_top10_{teacher}.png"
    save_plot(fig, path)
    paths.append(path)
    return paths


def make_stability_summary(df: pd.DataFrame, out_dir: Path) -> list[Path]:
    paths: list[Path] = []
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.scatterplot(
        data=df,
        x="final_ratio",
        y="final_loss",
        hue="variant",
        style="depth",
        ax=ax,
        alpha=0.8,
    )
    ax.set_yscale("log")
    ax.axvline(1.0, linestyle="--", linewidth=1.0, color="black")
    ax.set_title("Stability tradeoff: loss vs output/input ratio")
    path = out_dir / "stability_loss_vs_ratio_scatter.png"
    save_plot(fig, path)
    paths.append(path)
    return paths


def make_silicon_summary(df: pd.DataFrame, out_dir: Path) -> list[Path]:
    paths: list[Path] = []
    if "final_energy_mae" in df.columns:
        fig, ax = plt.subplots(figsize=(8, 5))
        labels = (
            df["target"]
            + " | "
            + df["aggregation"]
            + " | "
            + df["architecture"]
            + " | "
            + df["variant"]
        )
        plot_df = df.assign(label=labels)
        sns.barplot(data=plot_df, x="final_energy_mae", y="label", ax=ax)
        ax.set_xscale("log")
        ax.set_title("Available silicon energy MAE results")
        path = out_dir / "silicon_energy_mae_bar.png"
        save_plot(fig, path)
        paths.append(path)
    return paths


def main() -> None:
    configure_matplotlib(Path("studies/e3mlp_investigation/cache/mplconfig"))
    artifacts = Path("studies/e3mlp_investigation/artifacts")
    out_dir = ensure_dir(artifacts / "summary_overview")

    synth_runs = {
        "linear": artifacts / "batch1_synth_linear_20260409_170313" / "summary.csv",
        "mixed": artifacts / "batch1_synth_mixed_20260409_170325" / "summary.csv",
        "quadratic": artifacts
        / "batch1_synth_quadratic_20260409_170335"
        / "summary.csv",
    }
    for teacher, path in synth_runs.items():
        df = load_csv(path)
        if df is not None and not df.empty:
            make_synth_heatmap(df, teacher, out_dir)

    stab = load_csv(
        artifacts / "batch1_stability_advanced_20260409_170318" / "summary.csv"
    )
    if stab is not None and not stab.empty:
        make_stability_summary(stab, out_dir)

    silicon_frames = []
    for path in [
        artifacts / "smoke_silicon_nognn_01" / "summary.csv",
        artifacts / "smoke_silicon_nognn_attention_01" / "summary.csv",
        artifacts / "smoke_silicon_nognn_density_01" / "summary.csv",
    ]:
        df = load_csv(path)
        if df is not None and not df.empty:
            silicon_frames.append(df)
    if silicon_frames:
        make_silicon_summary(pd.concat(silicon_frames, ignore_index=True), out_dir)

    print(f"[visual-summary] wrote plots to {out_dir}", flush=True)


if __name__ == "__main__":
    main()
