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


def bootstrap_median_ci(values: np.ndarray, *, seed: int = 7) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    samples = rng.choice(values, size=(20_000, len(values)), replace=True)
    medians = np.median(samples, axis=1)
    return tuple(np.quantile(medians, [0.025, 0.975]))


def plot_real_envelope() -> None:
    rows = read_csv(DATA / "zncusnses_envelope_real.csv")
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
    for seed_index, _seed in enumerate(SEEDS):
        pair = [values[treatment][seed_index] for treatment in treatments]
        ax.plot(x, pair, color="#aab2b8", linewidth=1.1, zorder=1)
        ax.scatter(
            x, pair, color=colors, s=42, edgecolor="white", linewidth=0.7, zorder=3
        )

    for index, treatment in enumerate(treatments):
        treatment_values = values[treatment]
        median = float(np.median(treatment_values))
        ci_low, ci_high = bootstrap_median_ci(treatment_values, seed=17 + index)
        ax.errorbar(
            index,
            median,
            yerr=[[median - ci_low], [ci_high - median]],
            fmt="_",
            markersize=24,
            markeredgewidth=3,
            color="#111111",
            capsize=5,
            linewidth=1.6,
            zorder=4,
        )
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
    ax.set_title("ZnCuSnSeS envelope factorization (real W&B results)", pad=12)
    ax.grid(axis="y", alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(FIGURES / "result_envelope_real.png", dpi=240)
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
        x = np.arange(len(treatments), dtype=float)
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
        for seed in SEEDS:
            pair = [primary_by_treatment[treatment][seed] for treatment in treatments]
            ax.plot(x, pair, color="#a8adb2", linewidth=0.9, alpha=0.8)
            ax.scatter(x, pair, color="#365f8d", s=24, zorder=3)
        medians = [
            np.median(list(primary_by_treatment[treatment].values()))
            for treatment in treatments
        ]
        ax.scatter(
            x, medians, marker="_", s=380, linewidths=2.8, color="black", zorder=4
        )
        ax.set_xticks(x, treatments)
        ax.set_xlabel(parameter)
        ax.set_ylabel("Synthetic val/hamiltonian_mae proxy (a.u.)", color="#365f8d")
        ax.tick_params(axis="y", labelcolor="#365f8d")
        ax.set_title(study, fontsize=11)
        ax.grid(axis="y", alpha=0.2)

        secondary_metric = next(
            (row["secondary_metric"] for row in study_rows if row["secondary_metric"]),
            None,
        )
        if secondary_metric is not None:
            secondary_ax = ax.twinx()
            secondary_by_treatment = {
                treatment: [
                    float(row["synthetic_secondary_au"])
                    for row in study_rows
                    if row["treatment"] == treatment
                ]
                for treatment in treatments
            }
            for treatment_index, treatment in enumerate(treatments):
                jitter = np.linspace(-0.055, 0.055, len(SEEDS))
                secondary_ax.scatter(
                    treatment_index + jitter,
                    secondary_by_treatment[treatment],
                    marker="x",
                    color="#b45f06",
                    s=28,
                    linewidth=1.2,
                )
            secondary_ax.set_ylabel(
                f"Synthetic {secondary_metric} proxy (a.u.)", color="#b45f06"
            )
            secondary_ax.tick_params(axis="y", labelcolor="#b45f06")

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
    plot_real_envelope()
    plot_placeholder_ablations()
    print("Generated real and placeholder result plots.")
