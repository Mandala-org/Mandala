from __future__ import annotations

import csv
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
FIGURES = ROOT / "figures"
RESULTS = FIGURES / "results"
ZN_ASSET_ROOT = ROOT.parent / "eval_outputs" / "paper_ZnCuSnSeS_best_val094"

CONTROL = "#315b7d"
TREATMENT = "#2f8f6b"
ACCENT_1 = "#5a8fbd"
ACCENT_2 = "#d07b3f"
ACCENT_3 = "#8d5aab"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _values_by_setting(
    rows: Iterable[dict[str, str]], metric: str, order: list[str]
) -> dict[str, np.ndarray]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        value = row[metric]
        if value:
            grouped[row["setting"]].append(float(value))
    missing = [setting for setting in order if setting not in grouped]
    if missing:
        raise ValueError(f"Missing {metric} values for settings: {missing}")
    return {setting: np.asarray(grouped[setting], dtype=float) for setting in order}


def _boxplot(
    ax,
    values: dict[str, np.ndarray],
    order: list[str],
    labels: list[str],
    colors: list[str],
    *,
    ylabel: str,
    title: str,
    xlabel: str | None = None,
) -> None:
    positions = np.arange(len(order), dtype=float)
    boxes = ax.boxplot(
        [values[setting] for setting in order],
        positions=positions,
        widths=0.58,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "#111111", "linewidth": 1.9},
        whiskerprops={"color": "#444444"},
        capprops={"color": "#444444"},
    )
    for box, color in zip(boxes["boxes"], colors):
        box.set_facecolor(color)
        box.set_alpha(0.8)

    ax.set_xticks(positions, labels)
    ax.set_ylabel(ylabel)
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    ax.set_title(title, pad=10)
    ax.grid(axis="y", alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)


def _save(fig, name: str) -> None:
    fig.savefig(FIGURES / name, dpi=260)
    plt.close(fig)


def plot_envelope_ablation() -> None:
    rows = read_csv(DATA / "zncusnses_envelope_ablation.csv")
    order = ["Envelope off", "Multiply prediction"]
    values = _values_by_setting(rows, "hamiltonian_mae", order)
    fig, ax = plt.subplots(figsize=(7.2, 4.8), constrained_layout=True)
    _boxplot(
        ax,
        values,
        order,
        ["Envelope\noff", "Multiply\nprediction"],
        [CONTROL, TREATMENT],
        ylabel="Hamiltonian MAE (eV)",
        title=r"ZnCuSnSeS: envelope factorization",
    )
    _save(fig, "result_envelope_ablation.png")


def plot_silicon_aggregation_ablation() -> None:
    rows = read_csv(DATA / "silicon_node_aggregation_ablation.csv")
    order = ["Average", "Attention"]
    values = _values_by_setting(rows, "hamiltonian_mae", order)
    fig, ax = plt.subplots(figsize=(7.2, 4.8), constrained_layout=True)
    _boxplot(
        ax,
        values,
        order,
        ["Average", "Attention"],
        [CONTROL, TREATMENT],
        ylabel="Hamiltonian MAE (eV)",
        title="Perturbed silicon: node-message aggregation",
    )
    _save(fig, "result_silicon_node_aggregation_ablation.png")


def plot_siox_energy_guidance_ablation() -> None:
    rows = read_csv(DATA / "siox_mature_energy_guidance_ablation.csv")
    order = [
        "No energy guidance",
        r"Energy coefficient $3\times10^{-5}$",
        r"Energy coefficient $10^{-4}$",
        r"Energy coefficient $10^{-3}$",
    ]
    labels = ["0", r"$3\times10^{-5}$", r"$10^{-4}$", r"$10^{-3}$"]
    colors = [CONTROL, ACCENT_1, TREATMENT, ACCENT_3]
    hamiltonian = _values_by_setting(rows, "hamiltonian_mae", order)
    energy = {
        setting: values / 150.0
        for setting, values in _values_by_setting(rows, "observable_mae", order).items()
    }
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.6), constrained_layout=True)
    _boxplot(
        axes[0],
        hamiltonian,
        order,
        labels,
        colors,
        ylabel="Hamiltonian MAE (eV)",
        xlabel="Energy-loss coefficient",
        title=r"SiO$_2$: matrix accuracy",
    )
    _boxplot(
        axes[1],
        energy,
        order,
        labels,
        colors,
        ylabel="Band-energy MAE (eV/atom)",
        xlabel="Energy-loss coefficient",
        title=r"SiO$_2$: band-energy accuracy",
    )
    _save(fig, "result_siox_energy_guidance_ablation.png")


def plot_zncusnses_spectral_ablation() -> None:
    rows = read_csv(DATA / "zncusnses_mature_spectral_ablation.csv")
    order = ["No spectral guidance", r"Spectral coefficient $10^{-3}$"]
    labels = ["No spectral\nguidance", "Spectral coefficient\n$10^{-3}$"]
    hamiltonian = _values_by_setting(rows, "hamiltonian_mae", order)
    spectral = _values_by_setting(rows, "observable_mae", order)
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.6), constrained_layout=True)
    _boxplot(
        axes[0],
        hamiltonian,
        order,
        labels,
        [CONTROL, TREATMENT],
        ylabel="Hamiltonian MAE (eV)",
        title=r"ZnCuSnSeS: Hamiltonian",
    )
    _boxplot(
        axes[1],
        spectral,
        order,
        labels,
        [CONTROL, TREATMENT],
        ylabel="Spectral MAE (eV)",
        title=r"ZnCuSnSeS: eigenvalues",
    )
    _save(fig, "result_zncusnses_spectral_guidance_ablation.png")


def refresh_zncusnses_result_assets() -> None:
    assets = {
        "hamiltonian_first6_clim_0p01.png": "paper_zncusnses_hamiltonian_first6.png",
        "hamiltonian_correlation.png": "paper_zncusnses_hamiltonian_correlation.png",
    }
    for source_name, target_name in assets.items():
        source = ZN_ASSET_ROOT / source_name
        if not source.is_file():
            raise FileNotFoundError(
                f"Required ZnCuSnSeS result asset is missing: {source}"
            )
        shutil.copy2(source, RESULTS / target_name)


if __name__ == "__main__":
    FIGURES.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    plot_envelope_ablation()
    plot_silicon_aggregation_ablation()
    plot_siox_energy_guidance_ablation()
    plot_zncusnses_spectral_ablation()
    refresh_zncusnses_result_assets()
    print("Generated ablation figures and refreshed ZnCuSnSeS result assets.")
