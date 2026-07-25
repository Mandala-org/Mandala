"""Export raw W&B records used by the manually curated paper ablation report.

This deliberately does not calculate statistics or write Markdown.  The paper
report is edited by hand so its scientific decisions and interpretation remain
explicitly reviewable.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PAPER_PROJECTS = (
    "paper_zncusnses_envelope_12h",
    "paper_zncusnses_envelope_47h",
    "paper_zncusnses_pair_radial_mlp_12h",
    "paper_zncusnses_node_aggregation_12h",
    "paper_zncusnses_shifted_self_12h",
    "paper_silicon_edge_sh_square_12h",
    "paper_silicon_node_aggregation_12h",
    "paper_silicon_spectral_guidance_12h",
    "paper_siox_energy_guidance_12h",
    "paper_siox_mature_energy_guidance_12h",
    "paper_siox_mature_energy_guidance_3em5_12h",
    "paper_zncusnses_mature_full_network_spectral_12h",
    "paper_zncusnses_big_mature_spectral_24h",
    "paper_silicon_mature_energy_guidance_47h",
    "paper_additional_seeds_siox_mature_energy_guidance_12h",
    "paper_additional_seeds_zncusnses_mature_full_network_spectral_12h",
    "paper_additional_seeds_zncusnses_envelope_12h",
    "paper_additional_seeds_zncusnses_pair_radial_mlp_12h",
    "paper_additional_seeds_silicon_node_aggregation_12h",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export raw numeric W&B ablation records as JSON; does not edit Markdown."
    )
    parser.add_argument("--entity", default="b-brzoza")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("paper/data/ablation_studies_raw.json"),
    )
    parser.add_argument("--project", action="append", choices=PAPER_PROJECTS)
    return parser.parse_args()


def _json_safe(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return str(value)


def _numeric_summary(summary: dict[str, Any]) -> dict[str, float | int]:
    return {
        str(name): value
        for name, value in summary.items()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    }


def main() -> None:
    args = _parse_args()
    import wandb

    api = wandb.Api()
    projects = args.project or PAPER_PROJECTS
    payload: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "entity": args.entity,
        "projects": {},
    }
    for project in projects:
        records = []
        for run in api.runs(f"{args.entity}/{project}"):
            records.append(
                {
                    "id": run.id,
                    "name": run.name,
                    "state": run.state,
                    "created_at": run.created_at,
                    "config": _json_safe(dict(run.config)),
                    "summary_numeric": _numeric_summary(dict(run.summary)),
                }
            )
        payload["projects"][project] = records

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"Wrote raw ablation records to {args.output}")


if __name__ == "__main__":
    main()
