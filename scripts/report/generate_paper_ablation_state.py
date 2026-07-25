from __future__ import annotations

import argparse
import hashlib
import math
import statistics
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np


@dataclass(frozen=True)
class MetricSpec:
    label: str
    suffixes: tuple[str, ...]


@dataclass(frozen=True)
class AblationSpec:
    key: str
    title: str
    projects: tuple[str, ...]
    settings: tuple[str, ...]
    baseline: str
    setting_labels: dict[str, str]
    description: str
    aim: str
    metrics: tuple[MetricSpec, ...]
    expected_seeds: tuple[int, ...] = (41, 42, 43, 44, 45)
    notes: str = ""


@dataclass
class RunRecord:
    project: str
    run_id: str
    name: str
    state: str
    setting: str
    seed: int
    config: dict[str, Any]
    summary: dict[str, Any]
    created_at: str

    @property
    def epoch(self) -> int | None:
        value = self.summary.get("epoch")
        return int(value) if value is not None else None

    @property
    def runtime(self) -> float:
        value = self.summary.get("_runtime")
        return float(value) if value is not None else 0.0


HAMILTONIAN = MetricSpec("Hamiltonian MAE", ("hamiltonian_mae",))
SPECTRAL = MetricSpec("Spectral MAE (eV)", ("spectral_mae_ev",))
ENERGY = MetricSpec(
    "Energy MAE",
    ("energy_mae", "energy_mae_study", "energy_mae_gt_density"),
)
NUM_ELECTRONS = MetricSpec(
    "Electron-count MAE",
    ("num_electrons_mae", "num_electrons_mae_study"),
)
DENSITY = MetricSpec("Density MAE", ("density_mae",))
OVERLAP = MetricSpec("Overlap MAE", ("overlap_mae",))


SPECS: tuple[AblationSpec, ...] = (
    AblationSpec(
        key="zncusnses_envelope_12h",
        title="ZnCuSnSeS envelope factorization, 12 h",
        projects=("paper_zncusnses_envelope_12h",),
        settings=("off", "multiply_prediction"),
        baseline="off",
        setting_labels={
            "off": "Envelope off",
            "multiply_prediction": "Multiply prediction",
        },
        description=(
            "Fresh models were trained with or without multiplication of predicted "
            "Hamiltonian blocks by the fitted pair-specific radial envelope."
        ),
        aim=(
            "Test whether an explicit distance-decay prior reduces the burden on the "
            "equivariant network and improves Hamiltonian accuracy."
        ),
        metrics=(HAMILTONIAN,),
    ),
    AblationSpec(
        key="zncusnses_envelope_47h",
        title="ZnCuSnSeS envelope factorization, 47 h",
        projects=("paper_zncusnses_envelope_47h",),
        settings=("off", "multiply_prediction"),
        baseline="off",
        setting_labels={
            "off": "Envelope off",
            "multiply_prediction": "Multiply prediction",
        },
        description=(
            "The envelope comparison was repeated with a much longer training budget "
            "to determine whether its benefit persists after extensive optimization."
        ),
        aim=(
            "Separate a genuine asymptotic modeling advantage from a short-budget "
            "optimization-speed advantage."
        ),
        metrics=(HAMILTONIAN,),
    ),
    AblationSpec(
        key="zncusnses_pair_radial_mlp",
        title="ZnCuSnSeS pair-conditioned radial MLP",
        projects=("paper_zncusnses_pair_radial_mlp_12h",),
        settings=("disabled", "enabled"),
        baseline="disabled",
        setting_labels={"disabled": "Shared radial MLP", "enabled": "Pair-conditioned"},
        description=(
            "The radial MLP was either shared conventionally or explicitly conditioned "
            "on the ordered atom-pair type."
        ),
        aim=(
            "Test whether chemically distinct radial responses require pair-conditioned "
            "radial processing beyond pair embeddings elsewhere in the network."
        ),
        metrics=(HAMILTONIAN,),
    ),
    AblationSpec(
        key="zncusnses_node_aggregation",
        title="ZnCuSnSeS node-message aggregation",
        projects=("paper_zncusnses_node_aggregation_12h",),
        settings=("sum", "attention"),
        baseline="sum",
        setting_labels={"sum": "Sum", "attention": "Attention"},
        description=(
            "Fresh models used either direct summation or learned attention to aggregate "
            "incoming equivariant node-update messages."
        ),
        aim=(
            "Measure whether adaptive neighbor weighting improves accuracy in the "
            "chemically heterogeneous ZnCuSnSeS environment."
        ),
        metrics=(HAMILTONIAN,),
    ),
    AblationSpec(
        key="zncusnses_shifted_self",
        title="ZnCuSnSeS shifted-self treatment",
        projects=("paper_zncusnses_shifted_self_12h",),
        settings=("disabled", "enabled"),
        baseline="disabled",
        setting_labels={
            "disabled": "Shared self handling",
            "enabled": "Separated shifted self",
        },
        description=(
            "Periodic shifted self-edges were either handled together with ordinary "
            "self interactions or separated explicitly."
        ),
        aim=(
            "Test whether distinguishing on-site and periodic-image self interactions "
            "provides a useful inductive bias."
        ),
        metrics=(HAMILTONIAN,),
    ),
    AblationSpec(
        key="silicon_edge_sh_square",
        title="Perturbed-Si spherical-harmonic tensor-square edge features",
        projects=("paper_silicon_edge_sh_square_12h",),
        settings=("disabled", "enabled"),
        baseline="disabled",
        setting_labels={"disabled": "SH square off", "enabled": "SH square on"},
        description=(
            "The rich edge encoder was trained with or without tensor-square features "
            "derived from spherical harmonics."
        ),
        aim=(
            "Test whether higher-order angular edge information improves Hamiltonian "
            "prediction beyond the base spherical-harmonic representation."
        ),
        metrics=(HAMILTONIAN,),
    ),
    AblationSpec(
        key="silicon_node_aggregation",
        title="Perturbed-Si node-message aggregation",
        projects=("paper_silicon_node_aggregation_12h",),
        settings=("average", "attention"),
        baseline="average",
        setting_labels={"average": "Average", "attention": "Attention"},
        description=(
            "Fresh models used degree-normalized averaging or learned attention for "
            "node-message aggregation."
        ),
        aim=(
            "Test whether attention improves over a simple coordination-normalized "
            "aggregator on the structurally simpler silicon system."
        ),
        metrics=(HAMILTONIAN,),
        notes=(
            "The ten scientific runs completed training but are marked failed because "
            "the dataset had no held-out test split. Validation metrics are valid. Ten "
            "later repair attempts failed immediately on existing run directories and "
            "are excluded."
        ),
    ),
    AblationSpec(
        key="silicon_spectral",
        title="Perturbed-Si spectral guidance from scratch",
        projects=("paper_silicon_spectral_guidance_12h",),
        settings=("0", "0.003"),
        baseline="0",
        setting_labels={"0": "No spectral loss", "0.003": "Spectral coefficient 3e-3"},
        description=(
            "Fresh models were trained with Hamiltonian supervision alone or with an "
            "additional differentiable eigenvalue loss around the Fermi level."
        ),
        aim=(
            "Test whether direct spectral supervision improves eigenvalues and whether "
            "that improvement trades off against elementwise Hamiltonian accuracy."
        ),
        metrics=(HAMILTONIAN, SPECTRAL),
    ),
    AblationSpec(
        key="siox_energy",
        title="SiOx energy guidance from scratch",
        projects=("paper_siox_energy_guidance_12h",),
        settings=("0", "0.003"),
        baseline="0",
        setting_labels={"0": "No energy loss", "0.003": "Energy coefficient 3e-3"},
        description=(
            "Fresh SiOx models were trained with matrix loss alone or with an additional "
            "total-energy guidance term."
        ),
        aim=(
            "Test whether a global physical observable improves energy consistency and "
            "Hamiltonian prediction."
        ),
        metrics=(HAMILTONIAN, ENERGY),
    ),
    AblationSpec(
        key="siox_mature_energy",
        title="Mature SiOx energy-guidance strength",
        projects=(
            "paper_siox_mature_energy_guidance_12h",
            "paper_siox_mature_energy_guidance_3em5_12h",
        ),
        settings=("0", "3e-05", "0.0001", "0.001"),
        baseline="0",
        setting_labels={
            "0": "No energy loss",
            "3e-05": "Energy coefficient 3e-5",
            "0.0001": "Energy coefficient 1e-4",
            "0.001": "Energy coefficient 1e-3",
        },
        description=(
            "A mature SiOx checkpoint was fine-tuned with several energy-loss strengths, "
            "including a later softer-guidance extension."
        ),
        aim=(
            "Identify whether weak energy guidance can improve physical consistency "
            "without degrading an already accurate Hamiltonian model."
        ),
        metrics=(HAMILTONIAN, ENERGY),
        notes=(
            "Three 3e-5 runs completed locally, including test evaluation, but W&B upload "
            "failed and their final summaries are not synchronized. They are counted as "
            "locally complete but excluded from formal aggregate metrics until synced."
        ),
    ),
    AblationSpec(
        key="zncusnses_mature_spectral",
        title="Mature ZnCuSnSeS full-network spectral fine-tuning",
        projects=("paper_zncusnses_mature_full_network_spectral_12h",),
        settings=("0", "0.001"),
        baseline="0",
        setting_labels={"0": "No spectral loss", "0.001": "Spectral coefficient 1e-3"},
        description=(
            "A mature ZnCuSnSeS checkpoint was fine-tuned across the full network with "
            "and without spectral guidance."
        ),
        aim=(
            "Test whether spectral supervision is more effective after the matrix model "
            "has already reached a useful accuracy regime."
        ),
        metrics=(HAMILTONIAN, SPECTRAL),
    ),
    AblationSpec(
        key="zncusnses_big_mature_spectral",
        title="Large-data mature ZnCuSnSeS spectral fine-tuning",
        projects=("paper_zncusnses_big_mature_spectral_24h",),
        settings=("0", "0.0003"),
        baseline="0",
        setting_labels={"0": "No spectral loss", "0.0003": "Spectral coefficient 3e-4"},
        description=(
            "The enlarged ZnCuSnSeS model and dataset were fine-tuned for 24 hours with "
            "a softer spectral coefficient or a matched control objective."
        ),
        aim=(
            "Test whether weak spectral guidance improves spectral accuracy at scale "
            "without sacrificing the very low Hamiltonian MAE of the large model."
        ),
        metrics=(HAMILTONIAN, SPECTRAL),
        notes=(
            "Control seed 43 completed locally but its W&B uploader failed; seed 44 did "
            "not produce usable output. Several remaining runs are still active."
        ),
    ),
    AblationSpec(
        key="silicon_mature_energy",
        title="Mature perturbed-Si energy guidance, 47 h",
        projects=("paper_silicon_sleek75_energy_guidance_47h",),
        settings=("0", "0.001"),
        baseline="0",
        setting_labels={"0": "No energy loss", "0.001": "Energy coefficient 1e-3"},
        description=(
            "A less-mature perturbed-silicon checkpoint is being fine-tuned with and "
            "without energy guidance while predicting Hamiltonian, density, and overlap."
        ),
        aim=(
            "Test whether observable guidance can improve energy and electron-count "
            "consistency in a model that still has room to improve."
        ),
        metrics=(HAMILTONIAN, ENERGY, NUM_ELECTRONS, DENSITY, OVERLAP),
        notes=(
            "Only control runs have completed so far; no treatment comparison is yet "
            "available."
        ),
    ),
)


LOCAL_COMPLETE_UNSYNCED = {
    "zn5j7msl",
    "ghr9q7la",
    "7qi2ac6v",
    "movyhh8o",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate the live paper ablation state report from W&B."
    )
    parser.add_argument("--entity", default="b-brzoza")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("paper/ablation_studies_state.md"),
    )
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    return parser.parse_args()


def _canonical_setting(value: Any) -> str:
    if isinstance(value, bool):
        return "enabled" if value else "disabled"
    if isinstance(value, (int, float)):
        return f"{float(value):g}"
    text = str(value).strip()
    try:
        return f"{float(text):g}"
    except ValueError:
        return text


def _config_value(config: dict[str, Any], name: str) -> Any:
    return config.get(name, config.get(name.replace("_", "-")))


def _run_record(project: str, run: Any) -> RunRecord | None:
    config = dict(run.config)
    setting = _config_value(config, "ablation_setting")
    seed = _config_value(config, "seed")
    if setting is None or seed is None:
        return None
    return RunRecord(
        project=project,
        run_id=str(run.id),
        name=str(run.name),
        state=str(run.state),
        setting=_canonical_setting(setting),
        seed=int(seed),
        config=config,
        summary=dict(run.summary),
        created_at=str(run.created_at),
    )


def _is_formally_usable(spec: AblationSpec, record: RunRecord) -> bool:
    if record.state == "finished":
        return True
    if (
        spec.key == "silicon_node_aggregation"
        and record.state == "failed"
        and record.epoch is not None
        and record.summary.get("val/hamiltonian_mae") is not None
    ):
        return True
    return False


def _record_priority(spec: AblationSpec, record: RunRecord) -> tuple[int, float, str]:
    if record.state == "finished":
        priority = 4
    elif _is_formally_usable(spec, record):
        priority = 3
    elif record.state == "running":
        priority = 2
    elif record.run_id in LOCAL_COMPLETE_UNSYNCED:
        priority = 1
    else:
        priority = 0
    return priority, record.runtime, record.created_at


def _deduplicate(
    spec: AblationSpec,
    records: Iterable[RunRecord],
) -> dict[tuple[str, int], RunRecord]:
    selected: dict[tuple[str, int], RunRecord] = {}
    for record in records:
        key = (record.setting, record.seed)
        current = selected.get(key)
        if current is None or _record_priority(spec, record) > _record_priority(
            spec, current
        ):
            selected[key] = record
    return selected


def _metric_value(record: RunRecord, key: str) -> float | None:
    value = record.summary.get(key)
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _choose_metric_key(
    metric: MetricSpec,
    records: list[RunRecord],
    settings: tuple[str, ...],
) -> str | None:
    present_settings = {
        record.setting for record in records if record.setting in settings
    }
    if not present_settings:
        return None
    for split in ("test", "val"):
        for suffix in metric.suffixes:
            key = f"{split}/{suffix}"
            represented = {
                record.setting
                for record in records
                if _metric_value(record, key) is not None
            }
            if present_settings.issubset(represented):
                return key
    return None


def _stable_rng(*parts: str) -> np.random.Generator:
    digest = hashlib.sha256("|".join(parts).encode()).digest()
    seed = int.from_bytes(digest[:8], byteorder="little", signed=False)
    return np.random.default_rng(seed)


def _bootstrap_ci(
    values: list[float],
    statistic: Callable[[np.ndarray], float],
    *,
    samples: int,
    rng: np.random.Generator,
) -> tuple[float, float]:
    array = np.asarray(values, dtype=float)
    if len(array) == 1:
        value = float(statistic(array))
        return value, value
    indices = rng.integers(0, len(array), size=(samples, len(array)))
    stats = np.asarray([statistic(array[index]) for index in indices], dtype=float)
    stats = stats[np.isfinite(stats)]
    if len(stats) == 0:
        return math.nan, math.nan
    return float(np.quantile(stats, 0.025)), float(np.quantile(stats, 0.975))


def _t_critical_95(df: int) -> float:
    table = {
        1: 12.706,
        2: 4.303,
        3: 3.182,
        4: 2.776,
        5: 2.571,
        6: 2.447,
        7: 2.365,
        8: 2.306,
        9: 2.262,
        10: 2.228,
    }
    return table.get(df, 1.96 if df > 30 else 2.131)


def _mean_ci_half_width(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    return (
        _t_critical_95(len(values) - 1)
        * statistics.stdev(values)
        / math.sqrt(len(values))
    )


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values)
    ranks = np.empty(len(values), dtype=float)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + 1 + end)
        start = end
    return ranks


def _rank_biserial(improvements: np.ndarray) -> float:
    nonzero = improvements[np.abs(improvements) > 1e-15]
    if len(nonzero) == 0:
        return 0.0
    ranks = _average_ranks(np.abs(nonzero))
    positive = float(ranks[nonzero > 0].sum())
    negative = float(ranks[nonzero < 0].sum())
    total = positive + negative
    return (positive - negative) / total if total else 0.0


def _format_number(value: float) -> str:
    if not math.isfinite(value):
        return "NA"
    magnitude = abs(value)
    if magnitude == 0:
        return "0"
    if magnitude < 1e-3 or magnitude >= 1e3:
        return f"{value:.3e}"
    return f"{value:.6f}".rstrip("0").rstrip(".")


def _format_ci(low: float, high: float) -> str:
    return f"[{_format_number(low)}, {_format_number(high)}]"


def _summarize_values(
    values: list[float],
    *,
    bootstrap_samples: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
    mean = statistics.fmean(values)
    std = statistics.stdev(values) if len(values) > 1 else 0.0
    median = statistics.median(values)
    median_low, median_high = _bootstrap_ci(
        values,
        lambda x: float(np.median(x)),
        samples=bootstrap_samples,
        rng=rng,
    )
    return {
        "n": len(values),
        "mean": mean,
        "std": std,
        "mean_ci_half": _mean_ci_half_width(values),
        "median": median,
        "median_low": median_low,
        "median_high": median_high,
        "min": min(values),
    }


def _effect_summary(
    baseline: dict[int, float],
    treatment: dict[int, float],
    *,
    bootstrap_samples: int,
    rng: np.random.Generator,
) -> dict[str, Any] | None:
    seeds = sorted(set(baseline) & set(treatment))
    if not seeds:
        return None
    deltas = [treatment[seed] - baseline[seed] for seed in seeds]
    improvements = [baseline[seed] - treatment[seed] for seed in seeds]
    relative = [
        100.0 * (baseline[seed] - treatment[seed]) / abs(baseline[seed])
        for seed in seeds
        if baseline[seed] != 0
    ]
    rank_value = _rank_biserial(np.asarray(improvements))
    rank_low, rank_high = _bootstrap_ci(
        improvements,
        _rank_biserial,
        samples=bootstrap_samples,
        rng=rng,
    )
    return {
        "n": len(seeds),
        "delta_mean": statistics.fmean(deltas),
        "delta_ci_half": _mean_ci_half_width(deltas),
        "relative_mean": statistics.fmean(relative) if relative else math.nan,
        "relative_ci_half": _mean_ci_half_width(relative) if relative else math.nan,
        "rank_biserial": rank_value,
        "rank_low": rank_low,
        "rank_high": rank_high,
        "improved": sum(value > 0 for value in improvements),
    }


def _status_cell(counter: Counter[str]) -> str:
    order = ("finished", "running", "failed", "crashed", "killed")
    parts = [f"{name}={counter[name]}" for name in order if counter[name]]
    return ", ".join(parts) if parts else "none"


def _setting_label(spec: AblationSpec, setting: str) -> str:
    return spec.setting_labels.get(setting, setting)


def _render_analysis(
    spec: AblationSpec,
    raw_records: list[RunRecord],
    *,
    entity: str,
    bootstrap_samples: int,
) -> tuple[list[str], dict[str, Any]]:
    selected = _deduplicate(spec, raw_records)
    usable = [
        record
        for record in selected.values()
        if _is_formally_usable(spec, record) and record.setting in spec.settings
    ]
    raw_states = Counter(record.state for record in raw_records)
    started = sum(
        (setting, seed) in selected
        for setting in spec.settings
        for seed in spec.expected_seeds
    )
    usable_count = len(usable)
    running = sum(
        record.state == "running"
        for record in selected.values()
        if record.setting in spec.settings
    )
    local_unsynced = sum(
        record.run_id in LOCAL_COMPLETE_UNSYNCED for record in raw_records
    )

    lines = [f"## {spec.title}", ""]
    project_links = ", ".join(
        f"[{project}](https://wandb.ai/{entity}/{project})" for project in spec.projects
    )
    lines.extend(
        [
            f"**W&B:** {project_links}",
            "",
            f"**What was tested.** {spec.description}",
            "",
            f"**Aim.** {spec.aim}",
            "",
            (
                f"**Execution status.** Expected {len(spec.settings) * len(spec.expected_seeds)} "
                f"scientific runs; {started} unique setting/seed combinations have started, "
                f"{usable_count} currently provide formally usable final metrics, and "
                f"{running} are running. Raw W&B records: {_status_cell(raw_states)}."
            ),
        ]
    )
    if local_unsynced:
        lines.append(
            f"{local_unsynced} additional runs completed locally but have unsynchronized "
            "final W&B summaries."
        )
    if spec.notes:
        lines.extend(["", f"**Execution note.** {spec.notes}"])

    metric_results: dict[str, Any] = {}
    aggregate_rows: list[str] = []
    effect_rows: list[str] = []
    narrative_parts: list[str] = []
    for metric in spec.metrics:
        key = _choose_metric_key(metric, usable, spec.settings)
        if key is None:
            narrative_parts.append(
                f"{metric.label} is not yet available consistently across the settings."
            )
            continue
        by_setting: dict[str, dict[int, float]] = {}
        for setting in spec.settings:
            seed_values: dict[int, float] = {}
            for record in usable:
                if record.setting != setting:
                    continue
                value = _metric_value(record, key)
                if value is not None:
                    seed_values[record.seed] = value
            by_setting[setting] = seed_values

        summaries: dict[str, dict[str, Any]] = {}
        for setting, seed_values in by_setting.items():
            values = list(seed_values.values())
            if not values:
                continue
            summary = _summarize_values(
                values,
                bootstrap_samples=bootstrap_samples,
                rng=_stable_rng(spec.key, metric.label, setting, "aggregate"),
            )
            summaries[setting] = summary
            aggregate_rows.append(
                "| "
                + " | ".join(
                    [
                        f"{metric.label} (`{key}`)",
                        key.split("/", 1)[0],
                        _setting_label(spec, setting),
                        str(summary["n"]),
                        (
                            f"{_format_number(summary['mean'])} ± "
                            f"{_format_number(summary['mean_ci_half'])}"
                        ),
                        _format_number(summary["std"]),
                        (
                            f"{_format_number(summary['median'])} "
                            f"{_format_ci(summary['median_low'], summary['median_high'])}"
                        ),
                        _format_number(summary["min"]),
                    ]
                )
                + " |"
            )

        effects: dict[str, Any] = {}
        baseline_values = by_setting.get(spec.baseline, {})
        for setting in spec.settings:
            if setting == spec.baseline:
                continue
            effect = _effect_summary(
                baseline_values,
                by_setting.get(setting, {}),
                bootstrap_samples=bootstrap_samples,
                rng=_stable_rng(spec.key, metric.label, setting, "effect"),
            )
            if effect is None:
                continue
            effects[setting] = effect
            effect_rows.append(
                "| "
                + " | ".join(
                    [
                        f"{metric.label} (`{key}`)",
                        (
                            f"{_setting_label(spec, setting)} vs "
                            f"{_setting_label(spec, spec.baseline)}"
                        ),
                        str(effect["n"]),
                        (
                            f"{_format_number(effect['delta_mean'])} ± "
                            f"{_format_number(effect['delta_ci_half'])}"
                        ),
                        (
                            f"{_format_number(effect['relative_mean'])}% ± "
                            f"{_format_number(effect['relative_ci_half'])} pp"
                        ),
                        (
                            f"{_format_number(effect['rank_biserial'])} "
                            f"{_format_ci(effect['rank_low'], effect['rank_high'])}"
                        ),
                        f"{effect['improved']}/{effect['n']}",
                    ]
                )
                + " |"
            )

        if summaries:
            best_setting = min(
                summaries, key=lambda setting: summaries[setting]["mean"]
            )
            best_summary = summaries[best_setting]
            sentence = (
                f"For {metric.label}, the lowest current mean is "
                f"**{_setting_label(spec, best_setting)}** at "
                f"{_format_number(best_summary['mean'])} ± "
                f"{_format_number(best_summary['mean_ci_half'])} (95% CI half-width; "
                f"{key.split('/', 1)[0]}, "
                f"n={best_summary['n']})."
            )
            treatment_effects = [
                (setting, effect)
                for setting, effect in effects.items()
                if effect["n"] >= 2
            ]
            if treatment_effects:
                strongest_setting, strongest = max(
                    treatment_effects,
                    key=lambda item: abs(item[1]["relative_mean"]),
                )
                direction = (
                    "improvement" if strongest["relative_mean"] > 0 else "degradation"
                )
                sentence += (
                    f" The largest paired comparison currently indicates a "
                    f"{_format_number(abs(strongest['relative_mean']))}% {direction} "
                    f"for {_setting_label(spec, strongest_setting)} "
                    f"(paired n={strongest['n']})."
                )
            narrative_parts.append(sentence)
        metric_results[metric.label] = {
            "key": key,
            "summaries": summaries,
            "effects": effects,
        }

    lines.extend(
        [
            "",
            "### Metric estimates",
            "",
            (
                "| Metric | Split | Setting | n | Mean ± 95% CI half-width | "
                "SD across seeds | Median [bootstrap 95% CI] | Best observed value |"
            ),
            "|---|---:|---|---:|---:|---:|---:|---:|",
        ]
    )
    lines.extend(
        aggregate_rows or ["| No consistent metrics yet | - | - | 0 | - | - | - | - |"]
    )
    lines.extend(
        [
            "",
            "### Paired effect strength",
            "",
            (
                "| Metric | Comparison | Paired n | Treatment − baseline mean "
                "± 95% CI half-width | Relative improvement ± 95% CI half-width | "
                "Paired rank-biserial r [bootstrap 95% CI] | Seeds improved |"
            ),
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    lines.extend(
        effect_rows or ["| No complete paired comparison yet | - | 0 | - | - | - | - |"]
    )
    lines.extend(
        [
            "",
            "### Interpretation",
            "",
            (
                " ".join(narrative_parts)
                if narrative_parts
                else "No quantitative interpretation is possible yet."
            ),
            "",
        ]
    )
    return lines, {
        "expected": len(spec.settings) * len(spec.expected_seeds),
        "started": started,
        "usable": usable_count,
        "running": running,
        "raw_states": raw_states,
        "local_unsynced": local_unsynced,
        "metrics": metric_results,
    }


def _render_methodology(timestamp: str) -> list[str]:
    return [
        "# Paper ablation studies: live state report",
        "",
        f"_Generated from W&B on {timestamp}._",
        "",
        "## Statistical conventions",
        "",
        "- Held-out `test/*` metrics are used when the compared settings all provide "
        "that metric; otherwise the corresponding `val/*` metric is used consistently.",
        "- `Hamiltonian MAE` is always requested. Energy, spectral, density, overlap, "
        "and electron-count metrics are included only where they address the ablation.",
        "- Setting means are reported with a Student-t 95% confidence-interval "
        "half-width. Sample SD is shown separately as seed-to-seed spread; medians use "
        "a nonparametric bootstrap 95% confidence interval.",
        "- Treatment effects use matched seeds. Negative `treatment − baseline` is better "
        "for these error metrics; positive relative improvement is better.",
        "- Effect strength is the paired rank-biserial correlation. `+1` means every "
        "paired seed improved under treatment, `−1` means every paired seed worsened. "
        "Its uncertainty is a paired bootstrap 95% interval.",
        "- Mean paired-difference and relative-improvement error bars are 95% Student-t "
        "confidence-interval half-widths. With only five seeds these intervals are "
        "necessarily wide; incomplete comparisons are explicitly labeled provisional.",
        "- Failed runs are excluded except the perturbed-Si node-aggregation runs, which "
        "completed training and failed only because no held-out test split existed. "
        "Uploader-crashed runs with stale W&B summaries are not used quantitatively.",
        "",
        "## Updating this report",
        "",
        "Run:",
        "",
        "```bash",
        "MPLCONFIGDIR=/tmp/matplotlib-paper-report \\",
        "  mandala-venv/bin/python -u scripts/report/generate_paper_ablation_state.py",
        "```",
        "",
    ]


def main() -> None:
    args = _parse_args()
    import wandb

    api = wandb.Api(timeout=120)
    project_cache: dict[str, list[RunRecord]] = {}
    project_names = sorted(
        project.name
        for project in api.projects(entity=args.entity)
        if project.name.startswith("paper_")
    )
    needed_projects = sorted({project for spec in SPECS for project in spec.projects})
    for project in needed_projects:
        records = []
        for run in api.runs(f"{args.entity}/{project}"):
            record = _run_record(project, run)
            if record is not None:
                records.append(record)
        project_cache[project] = records

    timestamp = datetime.now().astimezone().isoformat(timespec="seconds")
    lines = _render_methodology(timestamp)
    overview_rows: list[str] = []
    rendered_sections: list[str] = []
    represented_projects: set[str] = set()
    for spec in SPECS:
        raw_records = [
            record
            for project in spec.projects
            for record in project_cache.get(project, [])
        ]
        represented_projects.update(spec.projects)
        section, status = _render_analysis(
            spec,
            raw_records,
            entity=args.entity,
            bootstrap_samples=args.bootstrap_samples,
        )
        rendered_sections.extend(section)
        overview_rows.append(
            "| "
            + " | ".join(
                [
                    f"[{spec.title}](#{spec.title.lower().replace(' ', '-').replace(',', '')})",
                    str(status["expected"]),
                    str(status["started"]),
                    str(status["usable"]),
                    str(status["running"]),
                    _status_cell(status["raw_states"]),
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "## Execution overview",
            "",
            (
                "| Ablation | Expected | Unique combinations started | "
                "Usable final metrics | Running | Raw W&B states |"
            ),
            "|---|---:|---:|---:|---:|---|",
            *overview_rows,
            "",
        ]
    )
    empty_projects = [
        project
        for project in project_names
        if project not in represented_projects
        and not list(api.runs(f"{args.entity}/{project}"))
    ]
    if empty_projects:
        lines.extend(
            [
                "## Empty or superseded paper projects",
                "",
                (
                    "The following `paper_` projects currently contain no run records. "
                    "They are not treated as evidence and may represent superseded or "
                    "not-yet-started sweep definitions:"
                ),
                "",
                *[f"- `{project}`" for project in empty_projects],
                "",
            ]
        )
    lines.extend(rendered_sections)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
