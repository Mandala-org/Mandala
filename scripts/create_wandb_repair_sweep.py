from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml


def _parse_yaml_value(raw: str) -> Any:
    return yaml.safe_load(raw)


def _parse_override(raw: str) -> tuple[str, Any]:
    if "=" not in raw:
        raise argparse.ArgumentTypeError(
            f"Expected PARAMETER=YAML_VALUE, received {raw!r}"
        )
    key, value = raw.split("=", 1)
    key = key.strip()
    if not key:
        raise argparse.ArgumentTypeError("Parameter name cannot be empty")
    return key, _parse_yaml_value(value)


def _grid_cardinality(parameters: dict[str, Any]) -> int:
    cardinality = 1
    for name, spec in parameters.items():
        if not isinstance(spec, dict):
            raise ValueError(f"Parameter {name!r} must contain a mapping")
        if "values" not in spec:
            continue
        values = spec["values"]
        if not isinstance(values, list) or not values:
            raise ValueError(f"Parameter {name!r} has an invalid values list")
        cardinality *= len(values)
    return cardinality


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a fresh W&B sweep containing only selected repair runs."
    )
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument(
        "--value",
        action="append",
        default=[],
        type=_parse_override,
        metavar="PARAMETER=YAML_VALUE",
        help="Set a fixed W&B parameter value. May be repeated.",
    )
    parser.add_argument(
        "--values",
        action="append",
        default=[],
        type=_parse_override,
        metavar="PARAMETER=YAML_LIST",
        help="Set a grid parameter values list. May be repeated.",
    )
    args = parser.parse_args()

    payload = yaml.safe_load(args.source.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"Sweep YAML must contain a mapping: {args.source}")
    parameters = payload.get("parameters")
    if not isinstance(parameters, dict):
        raise ValueError(f"Sweep YAML has no parameters mapping: {args.source}")

    payload["name"] = args.name
    for key, value in args.value:
        parameters[key] = {"value": value}
    for key, values in args.values:
        if not isinstance(values, list) or not values:
            raise ValueError(f"--values {key} requires a non-empty YAML list")
        parameters[key] = {"values": values}

    cardinality = _grid_cardinality(parameters)
    payload["run_cap"] = cardinality
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        yaml.safe_dump(payload, sort_keys=False, width=120),
    )
    print(f"Wrote repair sweep: {args.output}")
    print(f"Sweep name: {args.name}")
    print(f"Grid cardinality/run_cap: {cardinality}")


if __name__ == "__main__":
    main()
