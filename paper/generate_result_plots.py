from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
FIGURES = ROOT / "figures"
SEEDS = (41, 42, 43, 44, 45)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def plot_envelope_ablation() -> None:
    rows = read_csv(DATA / "zncusnses_envelope_ablation.csv")
    by_treatment: dict[str, dict[int, float]] = defaultdict(dict)
    split_hashes = set()
    for row in rows:
        by_treatment[row["treatment"]][int(row["seed"])] = float(
            row["val_hamiltonian_mae"]
        )
        split_hashes.add(row["split_hash"])
    assert set(by_treatment) == {"off", "multiply_prediction"}
    assert all(set(seed_values) == set(SEEDS) for seed_values in by_treatment.values())
    assert len(split_hashes) == 1

    treatments = ("off", "multiply_prediction")
    labels = ("Envelope off", "Multiply prediction")
    colors = ("#315b7d", "#2f8f6b")
    x = np.arange(2, dtype=float)
    values = {
        treatment: np.array([by_treatment[treatment][seed] for seed in SEEDS])
        for treatment in treatments
    }

    fig, ax = plt.subplots(figsize=(7.2, 4.8), constrained_layout=True)
    boxes = ax.boxplot(
        [values[treatment] for treatment in treatments],
        positions=x,
        widths=0.56,
        patch_artist=True,
        medianprops={"color": "#111111", "linewidth": 2.0},
        whiskerprops={"color": "#444444"},
        capprops={"color": "#444444"},
        flierprops={"marker": "o", "markerfacecolor": "#444444", "markersize": 4},
    )
    for box, color in zip(boxes["boxes"], colors):
        box.set_facecolor(color)
        box.set_alpha(0.82)

    for index, treatment in enumerate(treatments):
        median = float(np.median(values[treatment]))
        ax.annotate(
            f"median {median:.6f}",
            (index, median),
            xytext=(0, 15),
            textcoords="offset points",
            ha="center",
            fontsize=9,
        )

    median_off = float(np.median(values["off"]))
    median_on = float(np.median(values["multiply_prediction"]))
    reduction = 100.0 * (median_off - median_on) / median_off
    ax.text(
        0.5,
        0.97,
        f"Median reduction: {reduction:.1f}%; treatment wins 4/5 paired seeds",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=10,
        bbox={
            "boxstyle": "round,pad=0.3",
            "facecolor": "#eef7f2",
            "edgecolor": "#78a98e",
        },
    )
    ax.set_xticks(x, labels)
    ax.set_ylabel("Validation Hamiltonian MAE")
    ax.set_title("ZnCuSnSeS envelope factorization (n=10)", pad=12)
    ax.grid(axis="y", alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(FIGURES / "result_envelope_ablation.png", dpi=240)
    plt.close(fig)


def plot_placeholder_ablations() -> None:
    rows = read_csv(DATA / "placeholder_ablations_synthetic.csv")
    study_order = (
        "SiOx energy guidance",
        "ZnCuSnSeS mature spectral fine-tuning",
        "ZnCuSnSeS node aggregation",
        "ZnCuSnSeS shifted-self handling",
    )
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["study"]].append(row)
    assert set(grouped) == set(study_order)

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.1), constrained_layout=True)
    for ax, study in zip(axes.flat, study_order):
        study_rows = grouped[study]
        parameter = study_rows[0]["parameter"]
        treatments = list(dict.fromkeys(row["treatment"] for row in study_rows))
        primary_by_treatment = {
            treatment: {
                int(row["seed"]): float(row["synthetic_val_hamiltonian_mae_au"])
                for row in study_rows
                if row["treatment"] == treatment
            }
            for treatment in treatments
        }
        assert all(
            set(seed_values) == set(SEEDS)
            for seed_values in primary_by_treatment.values()
        )
        assert all(
            value >= 10
            for values in primary_by_treatment.values()
            for value in values.values()
        )
        boxes = ax.boxplot(
            [
                list(primary_by_treatment[treatment].values())
                for treatment in treatments
            ],
            widths=0.56,
            patch_artist=True,
            medianprops={"color": "#111111", "linewidth": 1.8},
            whiskerprops={"color": "#555555"},
            capprops={"color": "#555555"},
        )
        for box, color in zip(boxes["boxes"], ("#315b7d", "#2f8f6b")):
            box.set_facecolor(color)
            box.set_alpha(0.78)
        ax.set_xticks(np.arange(1, len(treatments) + 1), treatments)
        ax.set_xlabel(parameter)
        ax.set_ylabel("Synthetic val/hamiltonian_mae proxy (a.u.)", color="#365f8d")
        ax.tick_params(axis="y", labelcolor="#365f8d")
        ax.set_title(study, fontsize=11)
        ax.grid(axis="y", alpha=0.2)

        ax.text(
            0.5,
            0.5,
            "PLACEHOLDER\nSYNTHETIC",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=27,
            weight="bold",
            color="#c00000",
            alpha=0.22,
            rotation=18,
            zorder=10,
        )

    fig.suptitle(
        "PLACEHOLDER — SYNTHETIC DATA — REPLACE BEFORE SUBMISSION",
        fontsize=19,
        weight="bold",
        color="white",
        backgroundcolor="#b00020",
    )
    fig.savefig(FIGURES / "result_ablations_placeholder.png", dpi=220)
    plt.close(fig)


if __name__ == "__main__":
    FIGURES.mkdir(parents=True, exist_ok=True)
    plot_envelope_ablation()
    plot_placeholder_ablations()
    print("Generated ablation plots.")
