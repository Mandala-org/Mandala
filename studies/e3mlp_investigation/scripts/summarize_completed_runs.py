from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from studies.e3mlp_investigation.experiment_utils import (
    ensure_dir,
    save_plot,
)  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Summarize completed E3MLP study runs.")
    p.add_argument(
        "--artifacts-root",
        type=str,
        default="studies/e3mlp_investigation/artifacts",
    )
    p.add_argument(
        "--output-dir",
        type=str,
        default="studies/e3mlp_investigation/artifacts/summary_overview",
    )
    return p.parse_args()


def _load_summary_rows(root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    synth_rows = []
    stab_rows = []
    silicon_rows = []
    for path in root.iterdir():
        if not path.is_dir():
            continue
        summary = path / "summary.csv"
        if not summary.exists():
            continue
        df = pd.read_csv(summary)
        df = df.copy()
        df["run_name"] = path.name
        if "teacher_kind" in df.columns:
            synth_rows.append(df)
        elif "output_scale" in df.columns:
            stab_rows.append(df)
        elif "aggregation" in df.columns and "architecture" in df.columns:
            silicon_rows.append(df)
    synth = pd.concat(synth_rows, ignore_index=True) if synth_rows else pd.DataFrame()
    stab = pd.concat(stab_rows, ignore_index=True) if stab_rows else pd.DataFrame()
    silicon = (
        pd.concat(silicon_rows, ignore_index=True) if silicon_rows else pd.DataFrame()
    )
    return synth, stab, silicon


def _plot_synth_summary(df: pd.DataFrame, out_dir: Path) -> list[Path]:
    paths: list[Path] = []
    if df.empty:
        return paths
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=df, x="variant", y="val_mae", hue="depth", ax=ax, errorbar=None)
    ax.set_yscale("log")
    ax.set_title("Synthetic runs: final validation MAE")
    ax.tick_params(axis="x", rotation=45)
    path = out_dir / "synthetic_final_val_mae.png"
    save_plot(fig, path)
    paths.append(path)

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(
        data=df, x="variant", y="val_mse", hue="loss_kind", ax=ax, errorbar=None
    )
    ax.set_yscale("log")
    ax.set_title("Synthetic runs: final validation MSE")
    ax.tick_params(axis="x", rotation=45)
    path = out_dir / "synthetic_final_val_mse.png"
    save_plot(fig, path)
    paths.append(path)
    return paths


def _plot_stability_summary(df: pd.DataFrame, out_dir: Path) -> list[Path]:
    paths: list[Path] = []
    if df.empty:
        return paths
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=df, x="variant", y="final_loss", hue="depth", ax=ax, errorbar=None)
    ax.set_yscale("log")
    ax.set_title("Stability runs: final loss")
    ax.tick_params(axis="x", rotation=45)
    path = out_dir / "stability_final_loss.png"
    save_plot(fig, path)
    paths.append(path)

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(
        data=df, x="variant", y="final_ratio", hue="depth", ax=ax, errorbar=None
    )
    ax.set_title("Stability runs: output/input norm ratio")
    ax.tick_params(axis="x", rotation=45)
    path = out_dir / "stability_final_ratio.png"
    save_plot(fig, path)
    paths.append(path)
    return paths


def _plot_silicon_summary(df: pd.DataFrame, out_dir: Path) -> list[Path]:
    paths: list[Path] = []
    if df.empty:
        return paths
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(
        data=df,
        x="aggregation",
        y="final_eval_block_mae",
        hue="variant",
        ax=ax,
        errorbar=None,
    )
    ax.set_yscale("log")
    ax.set_title("Silicon no-GNN runs: final block MAE")
    ax.tick_params(axis="x", rotation=45)
    path = out_dir / "silicon_final_block_mae.png"
    save_plot(fig, path)
    paths.append(path)

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(
        data=df,
        x="aggregation",
        y="final_energy_mae",
        hue="target",
        ax=ax,
        errorbar=None,
    )
    ax.set_yscale("log")
    ax.set_title("Silicon no-GNN runs: final energy MAE")
    ax.tick_params(axis="x", rotation=45)
    path = out_dir / "silicon_final_energy_mae.png"
    save_plot(fig, path)
    paths.append(path)
    return paths


def main() -> None:
    args = parse_args()
    root = Path(args.artifacts_root)
    out_dir = ensure_dir(Path(args.output_dir))
    synth, stab, silicon = _load_summary_rows(root)
    print(
        f"[summary] loaded synth_rows={len(synth)} stab_rows={len(stab)} silicon_rows={len(silicon)}",
        flush=True,
    )

    paths = []
    paths.extend(_plot_synth_summary(synth, out_dir))
    paths.extend(_plot_stability_summary(stab, out_dir))
    paths.extend(_plot_silicon_summary(silicon, out_dir))

    combined = []
    if not synth.empty:
        combined.append(synth.assign(kind="synthetic"))
    if not stab.empty:
        combined.append(stab.assign(kind="stability"))
    if not silicon.empty:
        combined.append(silicon.assign(kind="silicon"))
    if combined:
        pd.concat(combined, ignore_index=True).to_csv(
            out_dir / "combined_summary.csv", index=False
        )

    pd.DataFrame(
        [
            {"kind": "synthetic", "rows": len(synth)},
            {"kind": "stability", "rows": len(stab)},
            {"kind": "silicon", "rows": len(silicon)},
        ]
    ).to_csv(out_dir / "summary_counts.csv", index=False)
    print(f"[summary] wrote {len(paths)} plot(s) to {out_dir}", flush=True)


if __name__ == "__main__":
    main()
