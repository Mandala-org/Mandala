from __future__ import annotations

import argparse
import ast
import re
import sys
import typing
from pathlib import Path
from types import NoneType, UnionType
from typing import get_type_hints, Union
from urllib.parse import urlparse

import torch
from omegaconf import OmegaConf

# Add project root to the Python path
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

from net.common import Config  # noqa: E402
from scripts.train import run_training  # noqa: E402


def setup_argparse(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one Mandala training job.")
    default_config = Config()

    parser.add_argument(
        *_arg_names("dataset_kind"),
        type=str,
        default="ZnCuSnSeS",
        choices=[
            "silicon",
            "silicon_scales",
            "siox",
            "deeph_e3",
            "ZnCuSnSeS_small",
            "ZnCuSnSeS",
        ],
    )
    parser.add_argument(*_arg_names("data_path"), type=str, default=None)
    parser.add_argument(*_arg_names("scales"), type=str_to_list, default=None)
    parser.add_argument(*_arg_names("num_train_per_scale"), type=int, default=70)
    parser.add_argument(*_arg_names("num_val_per_scale"), type=int, default=15)
    parser.add_argument(*_arg_names("num_test_per_scale"), type=int, default=15)
    parser.add_argument(*_arg_names("min_temp"), type=int, default=300)
    parser.add_argument(*_arg_names("max_temp"), type=int, default=3000)
    parser.add_argument(*_arg_names("temp_step"), type=int, default=300)
    parser.add_argument(*_arg_names("n_snapshots_per_temp"), type=int, default=50)
    parser.add_argument(*_arg_names("val_temp"), type=int, default=1500)
    parser.add_argument(*_arg_names("val_n_snapshots"), type=int, default=None)
    parser.add_argument(*_arg_names("num_train"), type=int, default=120)
    parser.add_argument(*_arg_names("num_val"), type=int, default=20)
    parser.add_argument(*_arg_names("num_test"), type=int, default=25)
    parser.add_argument(*_arg_names("val_fraction"), type=float, default=0.2)
    parser.add_argument(*_arg_names("precision"), type=str, default="32-true")
    parser.add_argument(
        *_arg_names("checkpoint_dir"),
        type=str,
        default=(
            "/bigdata/casus/wdm/hamiltonian_learning/models/checkpoints/paper_runs"
        ),
    )
    parser.add_argument(*_arg_names("resume_from_checkpoint"), type=str, default=None)
    parser.add_argument(*_arg_names("resume_from_wandb"), type=str, default=None)
    parser.add_argument(
        *_arg_names("fork_run"),
        type=str_to_bool,
        default=False,
        help=(
            "Resume training state from a checkpoint or W&B run, but continue as a "
            "fresh run with a newly assigned run name unless one is set explicitly."
        ),
    )
    parser.add_argument(
        *_arg_names("resume_mode"),
        type=str,
        default="latest",
        choices=["latest", "best", "final"],
    )
    parser.add_argument(*_arg_names("generate_video"), type=str_to_bool, default=False)
    parser.add_argument(*_arg_names("log_artifacts"), type=str_to_bool, default=True)
    parser.add_argument(
        *_arg_names("wandb_mode"),
        type=str,
        default="online",
        choices=["online", "offline", "disabled"],
    )
    parser.add_argument(*_arg_names("wandb_group"), type=str, default=None)
    parser.add_argument(*_arg_names("wandb_tags"), type=str_to_list, default=None)
    parser.add_argument(*_arg_names("sweep_yaml"), type=str, default=None)
    parser.add_argument(*_arg_names("convention"), type=str, default="e3nn")
    parser.add_argument(*_arg_names("max_wall_clock_hours"), type=float, default=12.0)
    parser.add_argument(
        *_arg_names("evaluate_test_after_fit"),
        type=str_to_bool,
        default=True,
        help="Evaluate the held-out test split exactly once after fitting.",
    )

    config_fields = get_type_hints(Config)
    for name, field_type in config_fields.items():
        default_value = getattr(default_config, name)
        if name == "run_name":
            default_value = None
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

    explicit_args = _collect_explicit_arg_dests(
        parser, sys.argv[1:] if argv is None else argv
    )
    args = parser.parse_args(argv)
    setattr(args, "_explicit_args", explicit_args)
    return args


def main() -> None:
    args = setup_argparse()
    if args.resume_from_wandb is not None and bool(
        getattr(args, "compatibility", False)
    ):
        raise ValueError(
            "--resume-from-wandb and --compatibility cannot be used together. "
            "resume_from_wandb always means continuing the same run identity."
        )
    if args.resume_from_wandb is not None:
        _apply_wandb_resume_metadata(args)
    _resolve_ablation_metadata(args)
    _normalize_wall_clock_args(args)
    _validate_required_args(args)
    print("=== wandb_run.py starting ===")
    print(f"dataset_kind={args.dataset_kind}")
    print(f"data_path={args.data_path}")
    print(f"sweep_yaml={args.sweep_yaml}")
    print(f"checkpoint_dir={args.checkpoint_dir}")
    print(f"wandb_mode={args.wandb_mode}")
    print(f"resume_from_checkpoint={args.resume_from_checkpoint}")
    print(f"resume_from_wandb={args.resume_from_wandb}")
    print(f"fork_run={args.fork_run}")
    print(f"compatibility={getattr(args, 'compatibility', False)}")
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
        if "parameters" in parsed_yaml:
            raise ValueError(
                "--sweep-yaml expects a resolved training config YAML, not a W&B sweep "
                "specification. This file contains a top-level 'parameters' section. "
                "Launch it with 'wandb sweep <yaml>' and then run a W&B agent, or "
                "materialize one concrete config first."
            )
        print(f"--- Loaded sweep YAML with keys: {sorted(parsed_yaml.keys())} ---")
    run_training(args, parsed_yaml=parsed_yaml)


def _resolve_ablation_metadata(args: argparse.Namespace) -> None:
    source = getattr(args, "ablation_setting_from", None)
    if source in (None, ""):
        return

    source_attr = str(source).strip().replace("-", "_")
    if not hasattr(args, source_attr):
        raise ValueError(
            f"ablation_setting_from={source!r} does not name a known configuration option"
        )
    value = getattr(args, source_attr)
    if value is None:
        raise ValueError(
            f"ablation_setting_from={source!r} resolved to None; a concrete value is required"
        )

    setting = _ablation_value_label(value)
    args.ablation_setting = setting
    if getattr(args, "run_name", None) not in (None, "", "mandala-run"):
        return

    ablation_name = getattr(args, "ablation_name", None)
    experiment_id = getattr(args, "experiment_id", None)
    if not ablation_name or not str(ablation_name).startswith("paper_"):
        raise ValueError(
            "Automatic ablation run naming requires ablation_name to start with 'paper_'"
        )
    if not experiment_id or not str(experiment_id).startswith("paper_"):
        raise ValueError(
            "Automatic ablation run naming requires experiment_id to start with 'paper_'"
        )
    args.run_name = f"{_slug(str(ablation_name))}_{_slug(setting)}_seed{int(args.seed)}"


def _ablation_value_label(value: object) -> str:
    if isinstance(value, bool):
        return "enabled" if value else "disabled"
    if isinstance(value, float):
        return f"{value:.12g}"
    return str(value)


def _slug(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_").lower()
    if not slug:
        raise ValueError(f"Cannot construct a run-name component from {value!r}")
    return slug


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


def _normalize_wall_clock_args(args: argparse.Namespace) -> None:
    explicit_args = set(getattr(args, "_explicit_args", set()))
    explicit_seconds = "max_wall_clock_seconds" in explicit_args
    explicit_hours = "max_wall_clock_hours" in explicit_args
    hours = getattr(args, "max_wall_clock_hours", None)
    if explicit_seconds and explicit_hours:
        raise ValueError(
            "--max-wall-clock-seconds and --max-wall-clock-hours are mutually exclusive"
        )
    if hours is None:
        return
    if hours <= 0:
        raise ValueError("--max-wall-clock-hours must be positive")
    args.max_wall_clock_seconds = float(hours) * 3600.0


def _apply_wandb_resume_metadata(args: argparse.Namespace) -> None:
    if args.resume_from_checkpoint is not None:
        raise ValueError(
            "--resume-from-wandb and --resume-from-checkpoint are mutually exclusive"
        )
    resolved = _resolve_wandb_resume(args.resume_from_wandb, args.resume_mode)
    explicit_args = set(getattr(args, "_explicit_args", set()))
    for key, value in resolved["config"].items():
        if key in explicit_args:
            continue
        if hasattr(args, key):
            setattr(args, key, value)
    args.resume_from_checkpoint = resolved["checkpoint_path"]
    inherit_source_run_name = not bool(getattr(args, "fork_run", False))
    if (
        inherit_source_run_name
        and "run_name" not in explicit_args
        and getattr(args, "run_name", None) in (None, "", "mandala-run")
    ):
        args.run_name = resolved["run_name"]
    if "checkpoint_dir" not in explicit_args:
        args.checkpoint_dir = resolved["checkpoint_dir"]
    if "wandb_project" not in explicit_args and getattr(
        args, "wandb_project", None
    ) in (None, ""):
        args.wandb_project = resolved["project"]
    if (
        bool(getattr(args, "fork_run", False))
        and "run_name" not in explicit_args
        and getattr(args, "run_name", None) in ("", "mandala-run")
    ):
        args.run_name = None


def _resolve_wandb_resume(run_url: str, resume_mode: str) -> dict[str, str]:
    entity, project, run_id = _parse_wandb_run_url(run_url)
    try:
        import wandb
    except Exception as exc:  # pragma: no cover - depends on environment
        raise RuntimeError(
            "wandb is required for --resume-from-wandb but could not be imported"
        ) from exc

    api = wandb.Api()
    run = api.run(f"{entity}/{project}/{run_id}")
    summary = getattr(run, "summary", {})
    summary_get = summary.get if hasattr(summary, "get") else dict(summary).get
    key_by_mode = {
        "latest": "checkpoint/latest_path",
        "best": "checkpoint/best_path",
        "final": "checkpoint/final_path",
    }
    summary_key = key_by_mode[resume_mode]
    checkpoint_path = summary_get(summary_key)
    if not checkpoint_path:
        raise ValueError(
            f"W&B run {entity}/{project}/{run_id} does not expose {summary_key!r} in its summary"
        )
    checkpoint_path = str(checkpoint_path)
    checkpoint_parent = Path(checkpoint_path).expanduser().parent
    run_name = checkpoint_parent.name or getattr(run, "name", None) or run_id
    checkpoint_dir = str(checkpoint_parent.parent)
    config = _normalize_wandb_config(getattr(run, "config", {}))
    return {
        "entity": entity,
        "project": project,
        "run_id": run_id,
        "run_name": str(run_name),
        "checkpoint_path": checkpoint_path,
        "checkpoint_dir": checkpoint_dir,
        "config": config,
    }


def _parse_wandb_run_url(run_url: str) -> tuple[str, str, str]:
    parsed = urlparse(run_url)
    if not parsed.scheme or not parsed.netloc:
        raise ValueError(f"Invalid W&B run URL: {run_url!r}")
    parts = [part for part in parsed.path.split("/") if part]
    if len(parts) >= 4 and parts[2] == "runs":
        entity, project, _, run_id = parts[:4]
    elif len(parts) >= 6 and parts[2] == "sweeps" and parts[4] == "runs":
        entity, project = parts[:2]
        run_id = parts[5]
    else:
        raise ValueError(
            "Expected W&B run URL like "
            "'https://wandb.ai/<entity>/<project>/runs/<run_id>' "
            "or "
            "'https://wandb.ai/<entity>/<project>/sweeps/<sweep_id>/runs/<run_id>'"
        )
    if not entity or not project or not run_id:
        raise ValueError(f"Invalid W&B run URL: {run_url!r}")
    return entity, project, run_id


def _normalize_wandb_config(config: object) -> dict[str, object]:
    if config is None:
        return {}
    if not isinstance(config, dict):
        try:
            config = dict(config)
        except Exception:
            return {}
    normalized: dict[str, object] = {}
    for key, value in config.items():
        if not isinstance(key, str) or key.startswith("_"):
            continue
        norm_key = key.replace("-", "_")
        normalized[norm_key] = _unwrap_wandb_config_value(value)
    return normalized


def _unwrap_wandb_config_value(value: object) -> object:
    if isinstance(value, dict) and "value" in value and len(value) == 1:
        return value["value"]
    return value


def _collect_explicit_arg_dests(
    parser: argparse.ArgumentParser, argv: list[str]
) -> set[str]:
    explicit: set[str] = set()
    option_map = parser._option_string_actions  # type: ignore[attr-defined]
    idx = 0
    while idx < len(argv):
        token = argv[idx]
        if token == "--":
            break
        if not token.startswith("-"):
            idx += 1
            continue
        option = token.split("=", 1)[0]
        action = option_map.get(option)
        if action is not None:
            explicit.add(action.dest)
        idx += 1
    return explicit


def _validate_required_args(args: argparse.Namespace) -> None:
    if getattr(args, "data_path", None) in (None, ""):
        raise ValueError(
            "data_path must be provided explicitly or available in the resumed W&B run config"
        )


if __name__ == "__main__":
    main()
