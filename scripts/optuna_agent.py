from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path
from typing import Any

import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf

# Add project root to the Python path
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

from scripts.train import run_training  # noqa: E402


class OptunaPruningCallback(pl.Callback):
    def __init__(self, trial: Any, monitor: str):
        super().__init__()
        self.trial = trial
        self.monitor = monitor

    def on_validation_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        if trainer.sanity_checking:
            return
        metric = trainer.callback_metrics.get(self.monitor)
        if metric is None:
            return
        if torch.is_tensor(metric):
            metric_value = float(metric.detach().cpu().item())
        else:
            metric_value = float(metric)
        if not math.isfinite(metric_value):
            return
        self.trial.report(metric_value, step=int(trainer.current_epoch))
        if self.trial.should_prune():
            import optuna

            raise optuna.TrialPruned(
                f"Trial pruned at epoch {trainer.current_epoch} on {self.monitor}={metric_value:.6g}"
            )


def setup_argparse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one Optuna agent worker.")
    parser.add_argument("--study-yaml", required=True, type=str)
    parser.add_argument("--storage", required=True, type=str)
    parser.add_argument("--study-name", default=None, type=str)
    parser.add_argument("--count", default=None, type=int)
    parser.add_argument("--timeout", default=None, type=int)
    parser.add_argument("--wandb-mode", default="offline", type=str)
    parser.add_argument("--checkpoint-dir", default="checkpoints/optuna", type=str)
    parser.add_argument("--heartbeat-interval", default=60, type=int)
    parser.add_argument("--grace-period", default=300, type=int)
    parser.add_argument("--agent-label", default=None, type=str)
    return parser.parse_args()


def main() -> None:
    args = setup_argparse()
    print("=== optuna_agent.py starting ===")
    print(f"study_yaml={args.study_yaml}")
    print(f"storage={args.storage}")
    print(f"study_name={args.study_name}")
    print(f"count={args.count}")
    print(f"timeout={args.timeout}")
    print(f"wandb_mode={args.wandb_mode}")
    print(f"checkpoint_dir={args.checkpoint_dir}")
    print(f"heartbeat_interval={args.heartbeat_interval}")
    print(f"grace_period={args.grace_period}")
    print(f"agent_label={args.agent_label}")
    run_optuna_agent(args)


def run_optuna_agent(args: argparse.Namespace) -> None:
    try:
        import optuna
    except ImportError as exc:
        raise ImportError(
            "Optuna agent requested, but the `optuna` package is not installed."
        ) from exc

    study_cfg = _load_yaml(args.study_yaml)
    print(f"--- Loaded study YAML: {args.study_yaml} ---")
    print(f"--- Study YAML keys: {sorted(study_cfg.keys())} ---")
    storage = _build_storage(
        optuna, args.storage, args.heartbeat_interval, args.grace_period
    )
    study_name = (
        args.study_name or study_cfg.get("study_name") or Path(args.study_yaml).stem
    )
    print(f"--- Connecting to Optuna study: {study_name} ---")
    print(f"--- Storage URL: {args.storage} ---")
    study = _get_or_create_study(optuna, study_name, storage, study_cfg)
    metric_name = _get_metric_name(study_cfg)
    count = args.count if args.count is not None else study_cfg.get("n_trials")
    print(f"--- Objective metric: {metric_name} ---")
    print(f"--- Trial count for this worker: {count} (None means indefinite) ---")

    def objective(trial: Any) -> float:
        print(f"=== Starting Optuna trial {trial.number} ===")
        run_args = _build_run_args_from_trial(
            trial,
            study_cfg,
            checkpoint_dir=args.checkpoint_dir,
            wandb_mode=args.wandb_mode,
            study_name=study_name,
            agent_label=args.agent_label,
        )
        metric_log = run_training(
            run_args,
            parsed_yaml=study_cfg,
            extra_callbacks=[OptunaPruningCallback(trial, metric_name)],
            objective_metric=metric_name,
        )
        print(
            f"=== Finished Optuna trial {trial.number}: {metric_name}={metric_log.get(metric_name)} ==="
        )
        for key, value in metric_log.items():
            trial.set_user_attr(key, value)
        return float(metric_log[metric_name])

    study.optimize(
        objective,
        n_trials=count,
        timeout=args.timeout,
        gc_after_trial=bool(study_cfg.get("gc_after_trial", True)),
    )


def _load_yaml(path: str) -> dict[str, Any]:
    parsed = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    if not isinstance(parsed, dict):
        raise ValueError(f"Study YAML must contain a top-level mapping: {path}")
    return parsed


def _build_storage(optuna: Any, url: str, heartbeat_interval: int, grace_period: int):
    print(
        f"--- Building Optuna storage: heartbeat_interval={heartbeat_interval}, grace_period={grace_period} ---"
    )
    try:
        return optuna.storages.RDBStorage(
            url=url,
            engine_kwargs={"pool_pre_ping": True},
            heartbeat_interval=heartbeat_interval,
            grace_period=grace_period,
        )
    except TypeError:
        return optuna.storages.RDBStorage(
            url=url,
            engine_kwargs={"pool_pre_ping": True},
        )


def _get_or_create_study(
    optuna: Any,
    study_name: str,
    storage: Any,
    study_cfg: dict[str, Any],
):
    goal = _get_metric_goal(study_cfg)
    sampler = _build_sampler(optuna, study_cfg)
    pruner = _build_pruner(optuna, study_cfg)
    for attempt in range(10):
        try:
            return optuna.create_study(
                study_name=study_name,
                storage=storage,
                direction=goal,
                sampler=sampler,
                pruner=pruner,
                load_if_exists=True,
            )
        except Exception as exc:  # noqa: BLE001
            message = str(exc).lower()
            if any(
                token in message
                for token in (
                    "already exists",
                    "duplicate key value violates unique constraint",
                    "ix_studies_study_name",
                )
            ):
                print(
                    f"--- Study {study_name} already exists or raced during creation; loading existing study ---"
                )
                return optuna.load_study(study_name=study_name, storage=storage)
            if attempt == 9:
                raise
            sleep_s = min(2.0 * (attempt + 1), 10.0)
            print(
                f"--- Study creation attempt {attempt + 1} failed: {exc!r}; retrying in {sleep_s:.1f}s ---"
            )
            time.sleep(sleep_s)


def _build_sampler(optuna: Any, study_cfg: dict[str, Any]):
    method = str(study_cfg.get("method", "bayes")).lower()
    seed = int(study_cfg.get("sampler_seed", 42))
    if method in {"bayes", "tpe"}:
        return optuna.samplers.TPESampler(seed=seed)
    if method == "random":
        return optuna.samplers.RandomSampler(seed=seed)
    if method == "grid":
        search_space: dict[str, list[Any]] = {}
        for raw_name, spec in study_cfg.get("parameters", {}).items():
            if not isinstance(spec, dict) or "values" not in spec:
                raise ValueError(
                    "Grid Optuna studies require every searched parameter to define `values`."
                )
            search_space[_canonical_name(raw_name)] = list(spec["values"])
        return optuna.samplers.GridSampler(search_space)
    raise ValueError(f"Unsupported Optuna method: {method!r}")


def _build_pruner(optuna: Any, study_cfg: dict[str, Any]):
    early_terminate = study_cfg.get("early_terminate")
    if early_terminate is None:
        return None
    if not isinstance(early_terminate, dict):
        raise ValueError("early_terminate must be a mapping.")
    prune_type = str(early_terminate.get("type", "")).lower()
    if prune_type == "hyperband":
        return optuna.pruners.HyperbandPruner(
            min_resource=int(early_terminate.get("min_iter", 1)),
            max_resource=early_terminate.get("max_iter", "auto"),
            reduction_factor=int(early_terminate.get("eta", 3)),
        )
    raise ValueError(f"Unsupported Optuna pruner type: {prune_type!r}")


def _get_metric_name(study_cfg: dict[str, Any]) -> str:
    metric = study_cfg.get("metric")
    if not isinstance(metric, dict) or "name" not in metric:
        raise ValueError("Study YAML must define metric.name.")
    return str(metric["name"])


def _get_metric_goal(study_cfg: dict[str, Any]) -> str:
    metric = study_cfg.get("metric")
    if not isinstance(metric, dict):
        raise ValueError("Study YAML must define metric.goal.")
    goal = str(metric.get("goal", "minimize")).lower()
    if goal not in {"minimize", "maximize"}:
        raise ValueError(f"Unsupported metric.goal: {goal!r}")
    return goal


def _build_run_args_from_trial(
    trial: Any,
    study_cfg: dict[str, Any],
    *,
    checkpoint_dir: str,
    wandb_mode: str,
    study_name: str,
    agent_label: str | None,
) -> argparse.Namespace:
    run_args: dict[str, Any] = {
        "checkpoint_dir": checkpoint_dir,
        "resume_from_checkpoint": None,
        "resume_mode": "latest",
        "wandb_mode": wandb_mode,
        "precision": "32-true",
        "generate_video": True,
        "log_artifacts": True,
    }
    parameters = study_cfg.get("parameters", {})
    if not isinstance(parameters, dict):
        raise ValueError("Study YAML parameters must be a mapping.")
    for raw_name, spec in parameters.items():
        run_args[_canonical_name(raw_name)] = _sample_parameter(trial, raw_name, spec)
    run_name = f"{study_name}-trial-{trial.number:05d}"
    if agent_label:
        run_name = f"{run_name}-{agent_label}"
    run_args["run_name"] = run_name
    print(
        f"--- Trial {trial.number} resolved run args: run_name={run_name}, data_path={run_args.get('data_path')}, dataset_kind={run_args.get('dataset_kind', 'silicon')}, checkpoint_dir={checkpoint_dir} ---"
    )
    return argparse.Namespace(**run_args)


def _sample_parameter(trial: Any, name: str, spec: Any) -> Any:
    param_name = _canonical_name(name)
    if not isinstance(spec, dict):
        return _normalize_trial_value(name, spec)
    if "value" in spec:
        return _normalize_trial_value(name, spec["value"])
    if "values" in spec:
        values = list(spec["values"])
        if not values:
            raise ValueError(f"Optuna parameter {name!r} has no candidate values.")
        return trial.suggest_categorical(param_name, values)
    if "distribution" in spec:
        distribution = str(spec["distribution"]).lower()
        if distribution == "categorical":
            return trial.suggest_categorical(param_name, list(spec["choices"]))
        if distribution == "int":
            return trial.suggest_int(
                param_name,
                int(spec["low"]),
                int(spec["high"]),
                step=int(spec.get("step", 1)),
                log=bool(spec.get("log", False)),
            )
        if distribution == "float":
            step = spec.get("step")
            return trial.suggest_float(
                param_name,
                float(spec["low"]),
                float(spec["high"]),
                step=None if step is None else float(step),
                log=bool(spec.get("log", False)),
            )
        raise ValueError(f"Unsupported distribution for {name!r}: {distribution!r}")
    if "min" in spec and "max" in spec:
        step = spec.get("step")
        log = bool(spec.get("log", False))
        if spec.get("type") == "int" or (
            isinstance(spec["min"], int)
            and isinstance(spec["max"], int)
            and (step is None or isinstance(step, int))
        ):
            return trial.suggest_int(
                param_name,
                int(spec["min"]),
                int(spec["max"]),
                step=int(step if step is not None else 1),
                log=log,
            )
        return trial.suggest_float(
            param_name,
            float(spec["min"]),
            float(spec["max"]),
            step=None if step is None else float(step),
            log=log,
        )
    raise ValueError(
        f"Unsupported parameter spec for {name!r}. Expected value, values, distribution, or min/max."
    )


def _normalize_trial_value(name: str, value: Any) -> Any:
    if _canonical_name(name) in {"matrix_targets", "radial_layers"}:
        if isinstance(value, str):
            raw = value.strip()
            if not raw:
                return []
            if raw.startswith("[") or raw.startswith("("):
                try:
                    parsed = OmegaConf.to_container(OmegaConf.create(raw), resolve=True)
                except Exception:
                    parsed = None
                else:
                    if isinstance(parsed, list):
                        return parsed
            if "," in raw:
                return [part.strip() for part in raw.split(",") if part.strip()]
            return [raw]
        if isinstance(value, tuple):
            return list(value)
    return value


def _canonical_name(name: str) -> str:
    return name.replace("-", "_")


if __name__ == "__main__":
    main()
