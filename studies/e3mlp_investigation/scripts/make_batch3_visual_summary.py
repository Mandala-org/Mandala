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


def maybe_read(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    return pd.read_csv(path)


def save_synth_heatmap(df: pd.DataFrame, out_path: Path, title: str) -> None:
    pivot = (
        df.groupby(["variant", "depth"], as_index=False)["val_mae"]
        .min()
        .pivot(index="variant", columns="depth", values="val_mae")
    )
    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.heatmap(pivot, annot=True, fmt=".3f", cmap="mako_r", ax=ax)
    ax.set_title(title)
    save_plot(fig, out_path)


def main() -> None:
    configure_matplotlib(Path("studies/e3mlp_investigation/cache/mplconfig"))
    artifacts = Path("studies/e3mlp_investigation/artifacts")
    out_dir = ensure_dir(artifacts / "summary_overview")

    silicon_rows = []
    for run in artifacts.glob("batch3_silicon_*"):
        summary = maybe_read(run / "summary.csv")
        if summary is None or summary.empty:
            continue
        row = summary.iloc[0].to_dict()
        row["run"] = run.name
        silicon_rows.append(row)

    if silicon_rows:
        sdf = pd.DataFrame(silicon_rows)
        sdf["label"] = sdf["run"].str.replace(
            "batch3_silicon_hamiltonian_mean_", "", regex=False
        )

        fig, ax = plt.subplots(figsize=(12, 8))
        plot_df = sdf.sort_values("final_eval_loss")
        sns.barplot(
            data=plot_df,
            x="final_eval_loss",
            y="label",
            hue="variant",
            dodge=False,
            ax=ax,
        )
        ax.set_xscale("log")
        ax.set_title("Batch 3 silicon: final eval loss")
        save_plot(fig, out_dir / "batch3_silicon_eval_loss_bar.png")

        fig, ax = plt.subplots(figsize=(12, 8))
        plot_df = sdf.sort_values("final_energy_mae")
        sns.barplot(
            data=plot_df,
            x="final_energy_mae",
            y="label",
            hue="variant",
            dodge=False,
            ax=ax,
        )
        ax.set_xscale("log")
        ax.set_title("Batch 3 silicon: final energy MAE")
        save_plot(fig, out_dir / "batch3_silicon_energy_mae_bar.png")

        fig, ax = plt.subplots(figsize=(8, 6))
        sns.scatterplot(
            data=sdf,
            x="final_eval_loss",
            y="final_energy_mae",
            hue="variant",
            style="variant",
            s=80,
            ax=ax,
        )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title("Batch 3 silicon: block loss vs energy MAE")
        save_plot(fig, out_dir / "batch3_silicon_loss_vs_energy_scatter.png")

        gate_df = sdf[sdf["variant"] == "gatemagnitudes"].copy()
        if not gate_df.empty:

            def lr_bucket(run: str) -> str:
                if "lr100x" in run:
                    return "100x"
                if "lr30x" in run:
                    return "30x"
                if "lr10x" in run:
                    return "10x"
                if "lr3x" in run:
                    return "3x"
                return "baseline"

            gate_df["lr_bucket"] = gate_df["run"].map(lr_bucket)
            order = ["baseline", "3x", "10x", "30x", "100x"]
            gate_df["lr_bucket"] = pd.Categorical(
                gate_df["lr_bucket"], categories=order, ordered=True
            )
            gate_df = gate_df.sort_values("lr_bucket")

            melted = gate_df.melt(
                id_vars=["lr_bucket"],
                value_vars=["final_eval_loss", "final_energy_mae"],
                var_name="metric",
                value_name="value",
            )
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.lineplot(
                data=melted, x="lr_bucket", y="value", hue="metric", marker="o", ax=ax
            )
            ax.set_yscale("log")
            ax.set_title("Gatemagnitudes silicon LR sweep")
            save_plot(fig, out_dir / "batch3_silicon_gatemagnitudes_lr_sweep.png")

    mixed = maybe_read(
        artifacts / "batch3_synth_mixed_prenorm_huber_01" / "summary.csv"
    )
    if mixed is not None and not mixed.empty:
        save_synth_heatmap(
            mixed,
            out_dir / "batch3_synth_mixed_prenorm_heatmap.png",
            "Batch 3 mixed synthetic: best val MAE",
        )

    quadratic = maybe_read(
        artifacts / "batch3_synth_quadratic_prenorm_huber_01" / "summary.csv"
    )
    if quadratic is not None and not quadratic.empty:
        save_synth_heatmap(
            quadratic,
            out_dir / "batch3_synth_quadratic_prenorm_heatmap.png",
            "Batch 3 quadratic synthetic: best val MAE",
        )

    print(f"[batch3-visual-summary] wrote plots to {out_dir}", flush=True)


if __name__ == "__main__":
    main()
