"""Download the exact W&B configurations used by the paper result models."""

from __future__ import annotations

import argparse
from pathlib import Path

import wandb
import yaml


PAPER_RUNS = {
    "zncu2sn_ses2.yaml": (
        "b-brzoza/"
        "mandala-ZnCuSnSeS-hamiltonian-envelope-stageA-selected-47h5/"
        "lk9fvoxv"
    ),
    "siox.yaml": ("b-brzoza/mandala-SiOx-hamiltonian-mae-stage2-47h25/ebg4g3h3"),
    "silicon_perturbed.yaml": (
        "b-brzoza/mandala-silicon-hdo-energy-stage2-47h25/92sqde98"
    ),
}


def _export_run(api: wandb.Api, run_path: str, output_path: Path) -> None:
    run = api.run(run_path)

    # W&B stores ``cfg`` as a Python repr of the same settings exposed as
    # individual structured keys.  Keeping the structured keys avoids a large,
    # redundant, non-YAML value while preserving every resolved setting.
    excluded_keys = {"cfg", "run_name", "wandb_project"}
    resolved_config = {}
    for key, value in sorted(run.config.items()):
        if key in excluded_keys:
            continue
        if value is not None and (
            key == "save_dir" or key.endswith("_path") or key.endswith("_dir")
        ):
            value = "..."
        resolved_config[key] = value
    payload = {"configuration": resolved_config}

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        yaml.safe_dump(
            payload,
            sort_keys=False,
            allow_unicode=False,
            width=1000,
        ),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("paper/data/evaluated_model_configs"),
    )
    args = parser.parse_args()

    api = wandb.Api()
    for filename, run_path in PAPER_RUNS.items():
        output_path = args.output_dir / filename
        _export_run(api, run_path, output_path)
        print(f"Downloaded {run_path} -> {output_path}")


if __name__ == "__main__":
    main()
