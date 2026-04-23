from __future__ import annotations

import argparse
import ast
import sys
import typing
from pathlib import Path
from types import NoneType, UnionType
from typing import get_type_hints, Union

import torch
from omegaconf import OmegaConf

# Add project root to the Python path
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

from net.common import Config  # noqa: E402
from scripts.train import run_training  # noqa: E402


def setup_argparse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one Mandala training job.")
    default_config = Config()

    parser.add_argument(
        *_arg_names("dataset_kind"),
        type=str,
        default="silicon",
        choices=["silicon", "siox"],
    )
    parser.add_argument(*_arg_names("data_path"), type=str, required=True)
    parser.add_argument(*_arg_names("min_temp"), type=int, default=300)
    parser.add_argument(*_arg_names("max_temp"), type=int, default=3000)
    parser.add_argument(*_arg_names("temp_step"), type=int, default=300)
    parser.add_argument(*_arg_names("n_snapshots_per_temp"), type=int, default=50)
    parser.add_argument(*_arg_names("val_temp"), type=int, default=1500)
    parser.add_argument(*_arg_names("val_n_snapshots"), type=int, default=None)
    parser.add_argument(*_arg_names("num_train"), type=int, default=None)
    parser.add_argument(*_arg_names("num_val"), type=int, default=None)
    parser.add_argument(*_arg_names("val_fraction"), type=float, default=0.2)
    parser.add_argument(*_arg_names("precision"), type=str, default="32-true")
    parser.add_argument(
        *_arg_names("checkpoint_dir"),
        type=str,
        default="checkpoints/main",
    )
    parser.add_argument(*_arg_names("resume_from_checkpoint"), type=str, default=None)
    parser.add_argument(
        *_arg_names("resume_mode"),
        type=str,
        default="latest",
        choices=["latest", "best", "final"],
    )
    parser.add_argument(*_arg_names("generate_video"), type=str_to_bool, default=True)
    parser.add_argument(*_arg_names("log_artifacts"), type=str_to_bool, default=True)
    parser.add_argument(
        *_arg_names("wandb_mode"),
        type=str,
        default=None,
        choices=["online", "offline", "disabled"],
    )
    parser.add_argument(*_arg_names("sweep_yaml"), type=str, default=None)
    parser.add_argument(*_arg_names("convention"), type=str, default="e3nn")

    config_fields = get_type_hints(Config)
    for name, field_type in config_fields.items():
        default_value = getattr(default_config, name)
        if field_type is bool:
            parser.add_argument(
                *_arg_names(name), type=str_to_bool, default=default_value
            )
            continue

        origin = typing.get_origin(field_type)
        if origin in {Union, typing.Union, UnionType}:
            union_args = typing.get_args(field_type)
            non_none_args = [
                arg
                for arg in union_args
                if arg is not type(None) and arg is not NoneType
            ]
            arg_type = non_none_args[0] if len(non_none_args) == 1 else str
        else:
            arg_type = field_type

        if arg_type is torch.device or arg_type is torch.dtype:
            arg_type = str

        is_sequence = False
        try:
            if (
                "Sequence" in str(field_type)
                or "list" in str(field_type)
                or (origin and issubclass(origin, typing.Sequence))
            ):
                is_sequence = True
        except TypeError:
            pass

        if is_sequence:
            parser.add_argument(
                *_arg_names(name), type=str_to_list, default=default_value
            )
        else:
            if not callable(arg_type):
                arg_type = str
            parser.add_argument(*_arg_names(name), type=arg_type, default=default_value)

    return parser.parse_args()


def main() -> None:
    args = setup_argparse()
    print("=== wandb_run.py starting ===")
    print(f"dataset_kind={args.dataset_kind}")
    print(f"data_path={args.data_path}")
    print(f"sweep_yaml={args.sweep_yaml}")
    print(f"checkpoint_dir={args.checkpoint_dir}")
    print(f"wandb_mode={args.wandb_mode}")
    print(f"resume_from_checkpoint={args.resume_from_checkpoint}")
    parsed_yaml = None
    if args.sweep_yaml is not None:
        print(f"--- Loading sweep YAML: {args.sweep_yaml} ---")
        parsed_yaml = OmegaConf.to_container(
            OmegaConf.load(args.sweep_yaml), resolve=True
        )
        if not isinstance(parsed_yaml, dict):
            raise ValueError(
                f"Sweep YAML must contain a top-level mapping: {args.sweep_yaml}"
            )
        print(f"--- Loaded sweep YAML with keys: {sorted(parsed_yaml.keys())} ---")
    run_training(args, parsed_yaml=parsed_yaml)


def str_to_bool(value):
    if isinstance(value, bool):
        return value
    lowered = value.lower()
    if lowered in ("yes", "true", "t", "y", "1"):
        return True
    if lowered in ("no", "false", "f", "n", "0"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def str_to_list(value):
    if isinstance(value, list):
        return value
    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        if isinstance(value, str):
            raw = value.strip()
            if not raw:
                return []
            if "," in raw:
                return [part.strip() for part in raw.split(",") if part.strip()]
            return [raw]
        raise argparse.ArgumentTypeError(f"Failed to parse {value!r} as a list.")


def _arg_names(name: str) -> tuple[str, ...]:
    primary = f"--{name}"
    alias = f"--{name.replace('_', '-')}"
    if alias == primary:
        return (primary,)
    return (primary, alias)


if __name__ == "__main__":
    main()
