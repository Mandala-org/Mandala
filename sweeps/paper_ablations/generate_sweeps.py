from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parent
SEEDS = [41, 42, 43, 44, 45]
CHECKPOINT_ROOT = Path("/bigdata/casus/wdm/hamiltonian_learning/models/checkpoints")


def fixed(value: Any) -> dict[str, Any]:
    return {"value": value}


def varied(values: list[Any]) -> dict[str, Any]:
    return {"values": values}


def dataset_parameters(kind: str) -> dict[str, dict[str, Any]]:
    if kind == "zncusnses":
        return {
            "dataset-kind": fixed("ZnCuSnSeS"),
            "data-path": fixed(
                "/bigdata/casus/wdm/hamiltonian_learning/data/ZnCuSnSeS"
            ),
            "snapshot-cache-dir": fixed(
                "/bigdata/casus/wdm/hamiltonian_learning/data/ZnCuSnSeS/snapshot_cache"
            ),
            "dataset-device": fixed("cpu"),
            "scales": fixed([1]),
        }
    if kind == "siox":
        return {
            "dataset-kind": fixed("siox"),
            "data-path": fixed("/bigdata/casus/wdm/hamiltonian_learning/data/SiOx_new"),
            "snapshot-cache-dir": fixed(
                "/bigdata/casus/wdm/hamiltonian_learning/data/SiOx_new/snapshot_cache"
            ),
            "dataset-device": fixed("cuda"),
            "cutoff-radius": fixed(10.0),
            "l-max": fixed(4),
            "hidden-irreps": fixed(
                "128x0e+16x0o+8x1e+64x1o+24x2e+8x2o+8x3e+24x3o+16x4e"
            ),
            "lr": fixed(0.0133),
        }
    if kind == "silicon":
        return {
            "dataset-kind": fixed("silicon_scales"),
            "data-path": fixed(
                "/bigdata/casus/wdm/hamiltonian_learning/data/perturbed_snapshots_Si"
            ),
            "snapshot-cache-dir": fixed(
                "/bigdata/casus/wdm/hamiltonian_learning/data/"
                "perturbed_snapshots_Si/snapshot_cache"
            ),
            "dataset-device": fixed("cpu"),
            "scales": fixed([1]),
            "num-train-per-scale": fixed(70),
            "num-val-per-scale": fixed(13),
            "num-test-per-scale": fixed(13),
            "cutoff-radius": fixed(8.0),
            "l-max": fixed(4),
            "hidden-irreps": fixed(
                "128x0e+16x0o+8x1e+64x1o+24x2e+8x2o+8x3e+24x3o+16x4e"
            ),
            "lr": fixed(0.003),
            "lr-scheduler-patience": fixed(120),
        }
    raise ValueError(f"Unknown dataset kind: {kind}")


def sweep(
    *,
    slug: str,
    dataset_kind: str,
    treatment_parameter: str,
    treatment_values: list[Any],
    extra_parameters: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    project = f"paper_{slug}"
    params = deepcopy(dataset_parameters(dataset_kind))
    params.update(
        {
            "wandb-project": fixed(project),
            "wandb-group": fixed(project),
            "wandb-tags": fixed(["paper_ablation", "fresh_initialization"]),
            "checkpoint-dir": fixed(str(CHECKPOINT_ROOT / project)),
            "experiment-id": fixed("paper_round1b_20260718"),
            "ablation-name": fixed(project),
            "ablation-setting-from": fixed(treatment_parameter),
            treatment_parameter.replace("_", "-"): varied(treatment_values),
            "seed": varied(SEEDS),
        }
    )
    if extra_parameters:
        params.update(extra_parameters)
    return {
        "program": "scripts/wandb_run.py",
        "method": "grid",
        "metric": {"name": "val/hamiltonian_mae", "goal": "minimize"},
        "project": project,
        "name": project,
        "run_cap": 10,
        "command": ["${env}", "python", "-u", "${program}", "${args}"],
        "parameters": params,
    }


def finetune_sweep(
    *,
    slug: str,
    dataset_kind: str,
    treatment_parameter: str,
    treatment_values: list[Any],
    resume_from_checkpoint: str,
    max_wall_clock_hours: float,
    extra_parameters: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    project = f"paper_{slug}"
    params = deepcopy(dataset_parameters(dataset_kind))
    params.update(
        {
            "wandb-project": fixed(project),
            "wandb-group": fixed(project),
            "wandb-tags": fixed(["paper_ablation", "checkpoint_finetuning"]),
            "checkpoint-dir": fixed(str(CHECKPOINT_ROOT / project)),
            "resume-from-checkpoint": fixed(resume_from_checkpoint),
            "resume-mode": fixed("best"),
            "compatibility": fixed(True),
            "max-wall-clock-hours": fixed(max_wall_clock_hours),
            "data-split-seed": fixed(42),
            "experiment-id": fixed("paper_round2_20260719"),
            "ablation-name": fixed(project),
            "ablation-setting-from": fixed(treatment_parameter),
            treatment_parameter.replace("_", "-"): varied(treatment_values),
            "seed": varied(SEEDS),
        }
    )
    if extra_parameters:
        params.update(extra_parameters)
    return {
        "program": "scripts/wandb_run.py",
        "method": "grid",
        "metric": {"name": "val/hamiltonian_mae", "goal": "minimize"},
        "project": project,
        "name": project,
        "run_cap": len(treatment_values) * len(SEEDS),
        "command": ["${env}", "python", "-u", "${program}", "${args}"],
        "parameters": params,
    }


def build_sweeps() -> dict[str, dict[str, Any]]:
    zn_envelope = fixed(
        "eval_outputs/zncusnses_radial_fit_study/slater_soft_cutoff_envelope.json"
    )
    siox_envelope = fixed(
        "eval_outputs/SiOx_0_50372000_radial_fit_study/"
        "slater_exp_quad_soft_wall_envelope.json"
    )
    zn_mature_checkpoint = (
        "/bigdata/casus/wdm/hamiltonian_learning/models/checkpoints/"
        "ZnCuSnSeS_hamiltonian_envelope_stageA_selected_47h5/"
        "dandy-sweep-34-restart47h5/best_model.pt"
    )
    silicon_sleek75_checkpoint = (
        "/bigdata/casus/wdm/hamiltonian_learning/models/checkpoints/"
        "silicon_perturbed_scales_hamiltonian_compat_11h5/"
        "sleek-sweep-75/best_model.pt"
    )
    siox_mature_checkpoint = (
        "/bigdata/casus/wdm/hamiltonian_learning/models/checkpoints/"
        "SiOx_hamiltonian_envelope_selected_95h8_mae/"
        "dutiful-sweep-12-restart47h9-pat60-restart95h8-mae-lr0p0027/"
        "best_model.pt"
    )
    return {
        "paper_zncusnses_envelope_12h.yaml": sweep(
            slug="zncusnses_envelope_12h",
            dataset_kind="zncusnses",
            treatment_parameter="hamiltonian_envelope_mode",
            treatment_values=["off", "multiply_prediction"],
            extra_parameters={"hamiltonian-envelope-path": zn_envelope},
        ),
        "paper_zncusnses_pair_radial_mlp_12h.yaml": sweep(
            slug="zncusnses_pair_radial_mlp_12h",
            dataset_kind="zncusnses",
            treatment_parameter="pair_conditioned_radial_mlp",
            treatment_values=[False, True],
            extra_parameters={
                "hamiltonian-envelope-path": zn_envelope,
                "hamiltonian-envelope-mode": fixed("multiply_prediction"),
            },
        ),
        "paper_silicon_edge_sh_square_12h.yaml": sweep(
            slug="silicon_edge_sh_square_12h",
            dataset_kind="silicon",
            treatment_parameter="edge_encoder_use_sh_tensor_square",
            treatment_values=[False, True],
        ),
        "paper_zncusnses_node_aggregation_12h.yaml": sweep(
            slug="zncusnses_node_aggregation_12h",
            dataset_kind="zncusnses",
            treatment_parameter="node_update_message_agg",
            treatment_values=["sum", "attention"],
            extra_parameters={
                "hamiltonian-envelope-path": zn_envelope,
                "hamiltonian-envelope-mode": fixed("multiply_prediction"),
            },
        ),
        "paper_zncusnses_shifted_self_12h.yaml": sweep(
            slug="zncusnses_shifted_self_12h",
            dataset_kind="zncusnses",
            treatment_parameter="separate_shifted_self",
            treatment_values=[False, True],
            extra_parameters={
                "hamiltonian-envelope-path": zn_envelope,
                "hamiltonian-envelope-mode": fixed("multiply_prediction"),
            },
        ),
        "paper_siox_envelope_12h.yaml": sweep(
            slug="siox_envelope_12h",
            dataset_kind="siox",
            treatment_parameter="hamiltonian_envelope_mode",
            treatment_values=["off", "multiply_prediction"],
            extra_parameters={"hamiltonian-envelope-path": siox_envelope},
        ),
        "paper_silicon_spectral_guidance_12h.yaml": sweep(
            slug="silicon_spectral_guidance_12h",
            dataset_kind="silicon",
            treatment_parameter="spectral_loss_coef",
            treatment_values=[0.0, 0.003],
            extra_parameters={"spectral-loss-enabled": fixed(True)},
        ),
        "paper_zncusnses_spectral_guidance_12h.yaml": sweep(
            slug="zncusnses_spectral_guidance_12h",
            dataset_kind="zncusnses",
            treatment_parameter="spectral_loss_coef",
            treatment_values=[0.0, 0.003],
            extra_parameters={
                "spectral-loss-enabled": fixed(True),
                "hamiltonian-envelope-path": zn_envelope,
                "hamiltonian-envelope-mode": fixed("multiply_prediction"),
            },
        ),
        "paper_siox_energy_guidance_12h.yaml": sweep(
            slug="siox_energy_guidance_12h",
            dataset_kind="siox",
            treatment_parameter="loss_coef_observables",
            treatment_values=[0.0, 0.003],
            extra_parameters={
                "enable-energy": fixed(True),
                "train-on-energy": fixed(True),
                "train-on-num-electrons": fixed(False),
                "train-observables-on-gt": fixed(True),
                "allow-zero-observable-loss-control": fixed(True),
            },
        ),
        "paper_zncusnses_envelope_47h.yaml": sweep(
            slug="zncusnses_envelope_47h",
            dataset_kind="zncusnses",
            treatment_parameter="hamiltonian_envelope_mode",
            treatment_values=["off", "multiply_prediction"],
            extra_parameters={
                "hamiltonian-envelope-path": zn_envelope,
                "max-wall-clock-hours": fixed(47.0),
                "max-epochs": fixed(30000),
                "experiment-id": fixed("paper_round2_20260719"),
            },
        ),
        "paper_zncusnses_mature_full_network_spectral_12h.yaml": finetune_sweep(
            slug="zncusnses_mature_full_network_spectral_12h",
            dataset_kind="zncusnses",
            treatment_parameter="spectral_loss_coef",
            treatment_values=[0.0, 1.0e-3],
            resume_from_checkpoint=zn_mature_checkpoint,
            max_wall_clock_hours=12.0,
            extra_parameters={
                "scales": fixed([1]),
                "lr": fixed(1.0e-5),
                "max-epochs": fixed(30000),
                "accumulate-grad-batches": fixed(4),
                "hamiltonian-envelope-path": zn_envelope,
                "hamiltonian-envelope-mode": fixed("multiply_prediction"),
                "spectral-loss-enabled": fixed(True),
                "train-on-spectral": fixed(True),
                "spectral-loss-kind": fixed("huber"),
                "spectral-loss-kmesh": fixed("2x2x2"),
                "spectral-loss-window-ev": fixed(10.0),
                "spectral-loss-taper-ev": fixed(2.0),
                "spectral-loss-huber-delta-ev": fixed(0.1),
                "spectral-loss-overlap-psd-cleanup": fixed(True),
                "spectral-loss-overlap-jitter": fixed(False),
                "checkpoint-monitor": fixed("val/hamiltonian_mae"),
            },
        ),
        "paper_silicon_sleek75_energy_guidance_47h.yaml": finetune_sweep(
            slug="silicon_sleek75_energy_guidance_47h",
            dataset_kind="silicon",
            treatment_parameter="loss_coef_observables",
            treatment_values=[0.0, 1.0e-3],
            resume_from_checkpoint=silicon_sleek75_checkpoint,
            max_wall_clock_hours=47.0,
            extra_parameters={
                "scales": fixed([1]),
                "lr": fixed(3.0e-4),
                "max-epochs": fixed(30000),
                "hidden-base-dim": fixed(32),
                "hidden-irreps": fixed(
                    "128x0e+128x0o+64x1e+64x1o+32x2e+32x2o+" "16x3e+16x3o+16x4e"
                ),
                "use-self-connection": fixed(True),
                "node-update-attention-scalar-dim": fixed(128),
                "internal-e3mlp-variant": fixed("film"),
                "head-e3mlp-variant": fixed("film"),
                "neck-depth": fixed(2),
                "head-e3mlp-layers": fixed(2),
                "head-diag-output-scale": fixed(2.0),
                "e3mlp-film-hidden-dim": fixed(128),
                "e3mlp-residual-scale": fixed(1.0),
                "radial-layers": fixed([128, 128]),
                "matrix-targets": fixed(["hamiltonian", "density", "overlap"]),
                "loss-l1-fraction": fixed(1.0),
                "observable-loss-kind": fixed("mae"),
                "enable-energy": fixed(True),
                "enable-num-electrons": fixed(True),
                "train-on-energy": fixed(True),
                "train-on-num-electrons": fixed(False),
                "train-observables-on-gt": fixed(False),
                "allow-zero-observable-loss-control": fixed(True),
                "lr-scheduler-target": fixed("val/loss_total"),
                "lr-scheduler-patience": fixed(120),
                "revert-monitor": fixed("val/loss_total"),
                "revert-decay-rate": fixed(0.8),
                "checkpoint-monitor": fixed("val/hamiltonian_mae"),
            },
        ),
        "paper_siox_mature_energy_guidance_12h.yaml": finetune_sweep(
            slug="siox_mature_energy_guidance_12h",
            dataset_kind="siox",
            treatment_parameter="loss_coef_observables",
            treatment_values=[0.0, 1.0e-4, 1.0e-3],
            resume_from_checkpoint=siox_mature_checkpoint,
            max_wall_clock_hours=12.0,
            extra_parameters={
                "num-train": fixed(120),
                "num-val": fixed(20),
                "num-test": fixed(25),
                "lr": fixed(5.0e-5),
                "max-epochs": fixed(30000),
                "hidden-irreps": fixed(
                    "128x0e+64x0o+32x1e+32x1o+24x2e+24x2o+" "16x3e+16x3o+8x4e"
                ),
                "neck-depth": fixed(1),
                "matrix-targets": fixed(["hamiltonian"]),
                "loss-l1-fraction": fixed(1.0),
                "observable-loss-kind": fixed("mae"),
                "enable-energy": fixed(True),
                "enable-num-electrons": fixed(False),
                "train-on-energy": fixed(True),
                "train-on-num-electrons": fixed(False),
                "train-observables-on-gt": fixed(True),
                "allow-zero-observable-loss-control": fixed(True),
                "hamiltonian-envelope-path": siox_envelope,
                "hamiltonian-envelope-mode": fixed("multiply_prediction"),
                "lr-scheduler-target": fixed("val/loss_total"),
                "checkpoint-monitor": fixed("val/hamiltonian_mae"),
            },
        ),
    }


def main() -> None:
    for filename, payload in build_sweeps().items():
        path = ROOT / filename
        path.write_text(
            yaml.safe_dump(payload, sort_keys=False, width=1000),
            encoding="utf-8",
        )
        print(path)


if __name__ == "__main__":
    main()
