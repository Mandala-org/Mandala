"""Download the completed paper-ablation summaries used by the figure script."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
ENTITY = "b-brzoza"


@dataclass(frozen=True)
class Study:
    output_name: str
    projects: tuple[str, ...]
    setting_key: str
    setting_labels: dict[str, str]
    observable_metric: str | None = None
    observable_label: str | None = None
    allow_validation_fallback: bool = False


STUDIES = (
    Study(
        output_name="zncusnses_envelope_ablation.csv",
        projects=("paper_zncusnses_envelope_12h",),
        setting_key="hamiltonian_envelope_mode",
        setting_labels={
            "off": "Envelope off",
            "multiply_prediction": "Multiply prediction",
        },
    ),
    Study(
        output_name="zncusnses_pair_radial_mlp_ablation.csv",
        projects=("paper_zncusnses_pair_radial_mlp_12h",),
        setting_key="pair_conditioned_radial_mlp",
        setting_labels={
            "false": "Shared radial MLP",
            "true": "Pair-conditioned radial MLP",
        },
    ),
    Study(
        output_name="silicon_node_aggregation_ablation.csv",
        projects=("paper_silicon_node_aggregation_12h",),
        setting_key="node_update_message_agg",
        setting_labels={"average": "Average", "attention": "Attention"},
        allow_validation_fallback=True,
    ),
    Study(
        output_name="siox_mature_energy_guidance_ablation.csv",
        projects=(
            "paper_siox_mature_energy_guidance_12h",
            "paper_siox_mature_energy_guidance_3em5_12h",
        ),
        setting_key="loss_coef_observables",
        setting_labels={
            "0": "No energy guidance",
            "3e-05": r"Energy coefficient $3\times10^{-5}$",
            "0.0001": r"Energy coefficient $10^{-4}$",
            "0.001": r"Energy coefficient $10^{-3}$",
        },
        observable_metric="energy_mae_gt_density",
        observable_label="Band-energy MAE",
    ),
    Study(
        output_name="zncusnses_mature_spectral_ablation.csv",
        projects=("paper_zncusnses_mature_full_network_spectral_12h",),
        setting_key="spectral_loss_coef",
        setting_labels={
            "0": "No spectral guidance",
            "0.001": r"Spectral coefficient $10^{-3}$",
        },
        observable_metric="spectral_mae_ev",
        observable_label="Spectral MAE",
    ),
)


def _config_value(config: dict[str, Any], name: str) -> Any:
    return config.get(name, config.get(name.replace("_", "-")))


def _key(value: Any) -> str:
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, (int, float)):
        return f"{value:g}"
    return str(value).lower()


def _metric(
    summary: dict[str, Any], name: str, *, allow_validation_fallback: bool
) -> tuple[float | None, str | None]:
    splits = ("test", "val") if allow_validation_fallback else ("test",)
    for split in splits:
        value = summary.get(f"{split}/{name}")
        if value is not None:
            return float(value), split
    return None, None


def _priority(record: dict[str, Any]) -> tuple[int, int, str]:
    complete = int(record["observable_mae"] is not None)
    state = {"finished": 2, "failed": 1, "crashed": 1}.get(record["state"], 0)
    return complete, state, str(record["created_at"])


def _download_study(api, study: Study) -> list[dict[str, Any]]:
    selected: dict[tuple[str, int], dict[str, Any]] = {}
    for project in study.projects:
        for run in api.runs(f"{ENTITY}/{project}"):
            if run.state in {"running", "pending", "queued"}:
                continue
            config = dict(run.config)
            summary = dict(run.summary)
            setting_key = _key(_config_value(config, study.setting_key))
            if setting_key not in study.setting_labels:
                continue
            seed = _config_value(config, "seed")
            if seed is None:
                continue
            hamiltonian_mae, hamiltonian_split = _metric(
                summary,
                "hamiltonian_mae",
                allow_validation_fallback=study.allow_validation_fallback,
            )
            if hamiltonian_mae is None:
                continue
            observable_mae = observable_split = None
            if study.observable_metric is not None:
                observable_mae, observable_split = _metric(
                    summary,
                    study.observable_metric,
                    allow_validation_fallback=study.allow_validation_fallback,
                )
            record = {
                "study": study.output_name.removesuffix(".csv"),
                "source_project": project,
                "run_id": run.id,
                "run_name": run.name,
                "run_url": run.url,
                "state": run.state,
                "created_at": run.created_at,
                "seed": int(seed),
                "setting_key": setting_key,
                "setting": study.setting_labels[setting_key],
                "hamiltonian_mae": hamiltonian_mae,
                "hamiltonian_split": hamiltonian_split,
                "observable_name": study.observable_label or "",
                "observable_mae": observable_mae,
                "observable_split": observable_split or "",
            }
            key = (setting_key, int(seed))
            existing = selected.get(key)
            if existing is None or _priority(record) > _priority(existing):
                selected[key] = record
    return sorted(selected.values(), key=lambda row: (row["setting_key"], row["seed"]))


def main() -> None:
    import wandb

    DATA.mkdir(parents=True, exist_ok=True)
    api = wandb.Api()
    fields = (
        "study",
        "source_project",
        "run_id",
        "run_name",
        "run_url",
        "state",
        "created_at",
        "seed",
        "setting_key",
        "setting",
        "hamiltonian_mae",
        "hamiltonian_split",
        "observable_name",
        "observable_mae",
        "observable_split",
    )
    for study in STUDIES:
        records = _download_study(api, study)
        path = DATA / study.output_name
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(fieldnames=fields, f=handle, lineterminator="\n")
            writer.writeheader()
            writer.writerows(records)
        print(f"{path}: {len(records)} usable run summaries")


if __name__ == "__main__":
    main()
