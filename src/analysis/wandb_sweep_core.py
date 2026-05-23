from __future__ import annotations

import ast
import hashlib
import json
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import numpy as np

try:
    import wandb
except Exception as exc:  # pragma: no cover - informative failure
    raise SystemExit(
        "wandb is required for W&B sweep analysis. Install the project dependencies first."
    ) from exc


MISSING = object()


@dataclass(frozen=True)
class RunRecord:
    run_id: str
    name: str
    state: str
    score: float | None
    config: dict[str, Any]


@dataclass(frozen=True)
class SweepAnalysisResult:
    sweep_ref: str
    sweep_path: str
    rank_metric: str
    rank_goal: str
    objective_metric: str | None
    runs_loaded: int
    skipped_runs: int
    state_counts: dict[str, int]
    records: list[RunRecord]
    ranked_records: list[RunRecord]
    top_runs: list[RunRecord]
    ci_runs: list[RunRecord]
    variable_specs: list[dict[str, Any]]


def analyze_sweep(
    sweep_ref: str,
    *,
    rank_metric: str | None = None,
    rank_goal: str = "auto",
    top_k: int = 5,
    top_k_ci: int | None = None,
) -> SweepAnalysisResult:
    sweep = load_sweep(sweep_ref)
    sweep_path = sweep_path_from_ref(sweep_ref)
    runs = list(sweep.runs)
    if not runs:
        raise SystemExit(f"Sweep {sweep_path} has no runs.")

    sweep_config = require_mapping(getattr(sweep, "config", {}), "sweep config")
    sweep_params = require_mapping(
        sweep_config.get("parameters", {}), "sweep config.parameters"
    )
    objective = require_mapping(sweep_config.get("metric", {}), "sweep config.metric")
    objective_metric = objective.get("name")
    resolved_rank_metric = rank_metric or objective_metric
    if not resolved_rank_metric:
        raise SystemExit(
            "Could not infer a ranking metric from the sweep metadata. "
            "Pass --rank-metric explicitly."
        )
    if "goal" not in objective:
        raise SystemExit(
            "Sweep metric metadata is missing goal; expected 'minimize' or 'maximize'."
        )

    resolved_rank_goal = resolve_rank_goal(
        resolved_rank_metric, rank_goal, objective["goal"]
    )
    records, skipped_runs = collect_run_records(runs, resolved_rank_metric)
    ranked_records = rank_records(records, resolved_rank_goal)
    highlight_count = min(top_k, len(ranked_records))
    ci_count = min(top_k_ci if top_k_ci is not None else top_k, len(ranked_records))
    variable_specs = find_swept_variables(records, sweep_params)
    state_counts = dict(Counter(str(getattr(run, "state", "")) for run in runs))

    return SweepAnalysisResult(
        sweep_ref=sweep_ref,
        sweep_path=sweep_path,
        rank_metric=resolved_rank_metric,
        rank_goal=resolved_rank_goal,
        objective_metric=objective_metric,
        runs_loaded=len(runs),
        skipped_runs=skipped_runs,
        state_counts=state_counts,
        records=records,
        ranked_records=ranked_records,
        top_runs=ranked_records[:highlight_count],
        ci_runs=ranked_records[:ci_count],
        variable_specs=variable_specs,
    )


def load_sweep(ref: str):
    entity, project, sweep_id = parse_sweep_ref(ref)
    api = wandb.Api()
    return api.sweep(f"{entity}/{project}/{sweep_id}")


def parse_sweep_ref(ref: str) -> tuple[str, str, str]:
    if "://" in ref:
        parsed = urlparse(ref)
        parts = [part for part in parsed.path.split("/") if part]
    else:
        parts = [part for part in ref.split("/") if part]

    if len(parts) >= 4 and parts[2] == "sweeps":
        entity, project, _, sweep_id = parts[:4]
    elif len(parts) == 3:
        entity, project, sweep_id = parts
    else:
        raise ValueError(
            "Expected a W&B sweep URL like "
            "'https://wandb.ai/<entity>/<project>/sweeps/<sweep_id>'"
        )
    if not entity or not project or not sweep_id:
        raise ValueError(f"Invalid W&B sweep reference: {ref!r}")
    return entity, project, sweep_id


def sweep_path_from_ref(ref: str) -> str:
    entity, project, sweep_id = parse_sweep_ref(ref)
    return f"{entity}/{project}/{sweep_id}"


def run_url_for_record(sweep_path: str, record: RunRecord) -> str:
    entity, project, _sweep_id = sweep_path.split("/", 2)
    return f"https://wandb.ai/{entity}/{project}/runs/{record.run_id}"


def collect_run_records(
    runs: list[Any], rank_metric: str
) -> tuple[list[RunRecord], int]:
    records: list[RunRecord] = []
    skipped_runs = 0
    for run in runs:
        config = normalize_run_config(
            require_mapping(
                getattr(run, "config", {}), f"run {getattr(run, 'id', '')} config"
            )
        )
        summary = require_mapping(
            getattr(run, "summary", {}), f"run {getattr(run, 'id', '')} summary"
        )
        score = None
        if rank_metric in summary:
            try:
                score = float(summary[rank_metric])
            except Exception as exc:
                raise SystemExit(
                    f"Expected run {getattr(run, 'id', '')} summary[{rank_metric!r}] to be numeric."
                ) from exc
            if not math.isfinite(score):
                score = None
                skipped_runs += 1
        else:
            skipped_runs += 1
        records.append(
            RunRecord(
                run_id=str(getattr(run, "id", "")),
                name=str(getattr(run, "name", "")),
                state=str(getattr(run, "state", "")),
                score=score,
                config=config,
            )
        )
    return records, skipped_runs


def rank_records(records: list[RunRecord], rank_goal: str) -> list[RunRecord]:
    ranked = [
        record
        for record in records
        if record.score is not None and math.isfinite(record.score)
    ]
    if not ranked:
        return []
    ranked.sort(
        key=lambda record: record.score if record.score is not None else math.inf
    )
    if rank_goal == "maximize":
        ranked.reverse()
    return ranked


def best_score_for_records(
    records: list[RunRecord],
    rank_goal: str,
) -> float | None:
    scores = [
        record.score
        for record in records
        if record.score is not None and math.isfinite(record.score)
    ]
    if not scores:
        return None
    if rank_goal == "maximize":
        return max(scores)
    return min(scores)


def score_within_orders_of_magnitude(
    score: float | None,
    best_score: float | None,
    rank_goal: str,
    *,
    max_orders_worse: float = 3.0,
) -> bool:
    if score is None or best_score is None:
        return False
    if not math.isfinite(score) or not math.isfinite(best_score):
        return False
    if score <= 0 or best_score <= 0:
        return False
    factor = 10.0**max_orders_worse
    if rank_goal == "maximize":
        return score >= best_score / factor
    return score <= best_score * factor


def filter_records_within_orders_of_magnitude(
    records: list[RunRecord],
    *,
    rank_goal: str,
    max_orders_worse: float = 3.0,
) -> list[RunRecord]:
    best_score = best_score_for_records(records, rank_goal)
    if best_score is None:
        return []
    return [
        record
        for record in records
        if score_within_orders_of_magnitude(
            record.score,
            best_score,
            rank_goal,
            max_orders_worse=max_orders_worse,
        )
    ]


def resolve_rank_goal(rank_metric: str, rank_goal: str, sweep_goal: str) -> str:
    if rank_goal != "auto":
        return rank_goal
    if sweep_goal not in {"minimize", "maximize"}:
        raise SystemExit(
            f"Sweep metric goal must be 'minimize' or 'maximize', got {sweep_goal!r}."
        )
    return sweep_goal


def find_swept_variables(
    records: list[RunRecord], sweep_params: dict[str, Any]
) -> list[dict[str, Any]]:
    all_keys = set(sweep_params)
    for record in records:
        all_keys.update(record.config)
    all_keys = {key for key in all_keys if is_analysis_variable_key(key)}

    variable_specs: list[dict[str, Any]] = []
    for key in sorted(all_keys):
        values = [record.config.get(key, MISSING) for record in records]
        present = [value for value in values if value is not MISSING]
        if len(present) < 2:
            continue

        unique = {canonical_value(value) for value in present}
        if len(unique) <= 1:
            continue

        numeric_values = [coerce_numeric(value) for value in present]
        is_numeric = all(value is not None for value in numeric_values)
        display_name = display_name_for_key(key)
        param_meta = require_mapping(
            sweep_params.get(key, {}), f"sweep parameter {key!r}"
        )
        distribution = param_meta.get("distribution")
        variable_specs.append(
            {
                "key": key,
                "display_name": display_name,
                "kind": "numeric" if is_numeric else "categorical",
                "distribution": distribution,
                "values": present,
                "unique_values": [
                    display_value(value) for value in unique_in_order(present)
                ],
                "numeric_values": [
                    value for value in numeric_values if value is not None
                ],
            }
        )
    return variable_specs


def build_rank_run_styles(
    ranked_records: list[RunRecord], highlight_count: int
) -> dict[str, dict[str, Any]]:
    if not ranked_records:
        return {}

    highlight_count = max(0, min(highlight_count, len(ranked_records)))
    top_records = ranked_records[:highlight_count]
    rest_records = ranked_records[highlight_count:]

    top_colors = gradient_colors("#d62728", "#f1c40f", len(top_records))
    rest_colors = gradient_colors("#f1c40f", "#1f77b4", len(rest_records))

    styles: dict[str, dict[str, Any]] = {}
    for record, color in zip(top_records, top_colors):
        styles[record.run_id] = {
            "color": color,
            "size": 52,
            "alpha": 0.98,
        }
    for record, color in zip(rest_records, rest_colors):
        styles[record.run_id] = {
            "color": color,
            "size": 14,
            "alpha": 0.38,
        }
    return styles


def gradient_colors(start_hex: str, end_hex: str, count: int) -> list[str]:
    if count <= 0:
        return []
    start = np.asarray(hex_to_rgb(start_hex), dtype=float)
    end = np.asarray(hex_to_rgb(end_hex), dtype=float)
    if count == 1:
        return [rgb_to_hex(start)]
    colors = []
    for t in np.linspace(0.0, 1.0, count):
        rgb = start * (1.0 - t) + end * t
        colors.append(rgb_to_hex(rgb))
    return colors


def hex_to_rgb(value: str) -> tuple[float, float, float]:
    value = value.lstrip("#")
    return tuple(int(value[i : i + 2], 16) / 255.0 for i in (0, 2, 4))


def rgb_to_hex(rgb: np.ndarray | tuple[float, float, float]) -> str:
    arr = np.asarray(rgb, dtype=float)
    return "#{:02x}{:02x}{:02x}".format(
        int(np.clip(arr[0], 0.0, 1.0) * 255.0),
        int(np.clip(arr[1], 0.0, 1.0) * 255.0),
        int(np.clip(arr[2], 0.0, 1.0) * 255.0),
    )


def categorical_jitter(run_id: str, key: str, width: float = 0.12) -> float:
    seed_bytes = hashlib.sha1(f"{run_id}:{key}".encode("utf-8")).digest()[:8]
    seed = int.from_bytes(seed_bytes, "big", signed=False)
    rng = np.random.default_rng(seed)
    return float(rng.uniform(-width, width))


def numeric_jitter(
    base_x: float,
    run_id: str,
    key: str,
    values: np.ndarray,
    *,
    use_log: bool,
    width_fraction: float = 0.03,
) -> float:
    arr = np.asarray(values, dtype=float)
    if arr.size < 2:
        return base_x

    seed_bytes = hashlib.sha1(f"{run_id}:{key}:numeric".encode("utf-8")).digest()[:8]
    seed = int.from_bytes(seed_bytes, "big", signed=False)
    rng = np.random.default_rng(seed)
    centered = float(rng.uniform(-1.0, 1.0))

    if use_log:
        factor = 10 ** (centered * 0.03)
        return max(base_x * factor, np.finfo(float).tiny)

    span = float(np.max(arr) - np.min(arr))
    scale = max(span, abs(base_x), 1.0)
    width = max(scale * width_fraction, 1e-6)
    return base_x + centered * width


def top_run_marker_x(
    base_x: float,
    idx: int,
    n_top_runs: int,
    values: np.ndarray,
    *,
    use_log: bool,
    categorical: bool = False,
) -> float:
    if n_top_runs <= 1:
        return base_x

    centered = idx - (n_top_runs - 1) / 2.0
    if categorical:
        return base_x + centered * 0.08

    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return base_x

    if use_log:
        factor = 1.0 + centered * 0.025
        return max(base_x * factor, np.finfo(float).tiny)

    lo = float(np.min(arr))
    hi = float(np.max(arr))
    span = hi - lo
    if span <= 0:
        span = max(abs(lo), 1.0)
    step = max(span * 0.01, 1e-6)
    return base_x + centered * step


def numeric_bins(
    values: np.ndarray, base_bins: int, use_log: bool
) -> tuple[np.ndarray, bool]:
    if values.size == 0:
        return np.linspace(0.0, 1.0, 2), False

    if use_log and np.any(values <= 0):
        use_log = False

    if use_log:
        positive = values[values > 0]
        lo = float(np.min(positive))
        hi = float(np.max(positive))
        if lo == hi:
            lo /= 1.5
            hi *= 1.5
        bin_count = max(8, min(30, base_bins))
        bins = np.logspace(np.log10(lo), np.log10(hi), bin_count)
        return bins, True

    lo = float(np.min(values))
    hi = float(np.max(values))
    if lo == hi:
        pad = 0.5 if lo == 0 else max(abs(lo) * 0.1, 1e-6)
        return np.linspace(lo - pad, hi + pad, max(4, base_bins)), False
    bins = np.histogram_bin_edges(values, bins=min(base_bins, max(4, values.size)))
    if len(bins) < 2:
        bins = np.linspace(lo, hi, max(4, base_bins))
    return bins, False


def compute_numeric_ci_band_from_points(
    points: list[tuple[RunRecord, float, float]],
    ci_run_ids: set[str],
    k_fold: int,
    use_log_x: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    ci_points = [(x, y) for record, x, y in points if record.run_id in ci_run_ids]
    return compute_numeric_ci_band_from_xy(ci_points, k_fold, use_log_x)


def compute_numeric_ci_band_from_xy(
    points: list[tuple[float, float]],
    k_fold: int,
    use_log_x: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    if len(points) < 2:
        return None

    xs = np.asarray([x for x, _ in points], dtype=float)
    ys = np.asarray([y for _, y in points], dtype=float)
    valid = np.isfinite(xs) & np.isfinite(ys) & (ys > 0)
    if use_log_x:
        valid &= xs > 0
    xs = xs[valid]
    ys = ys[valid]
    if xs.size < 2:
        return None

    x_min = float(np.min(xs))
    x_max = float(np.max(xs))
    if x_min == x_max:
        return None

    if use_log_x:
        x_grid = np.logspace(np.log10(x_min), np.log10(x_max), 200)
    else:
        x_grid = np.linspace(x_min, x_max, 200)

    n = xs.size
    fold_count = max(2, min(int(k_fold), n))
    indices = np.arange(n)
    rng = np.random.default_rng(0)
    rng.shuffle(indices)
    folds = np.array_split(indices, fold_count)

    fold_preds: list[np.ndarray] = []
    for fold in folds:
        train_idx = np.setdiff1d(indices, fold, assume_unique=False)
        if train_idx.size < 1:
            continue
        fit = fit_log_linear_regression(xs[train_idx], ys[train_idx])
        if fit is None:
            continue
        slope, intercept = fit
        fold_preds.append(predict_log_linear_regression(x_grid, slope, intercept))

    if not fold_preds:
        return None

    center_fit = fit_log_linear_regression(xs, ys)
    if center_fit is None:
        return None
    slope, intercept = center_fit
    y_center = predict_log_linear_regression(x_grid, slope, intercept)
    y_stack = np.vstack(fold_preds)
    y_low = np.min(y_stack, axis=0)
    y_high = np.max(y_stack, axis=0)
    return x_grid, y_center, y_low, y_high


def fit_log_linear_regression(
    xs: np.ndarray, ys: np.ndarray
) -> tuple[float, float] | None:
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    mask = np.isfinite(xs) & np.isfinite(ys) & (ys > 0)
    xs = xs[mask]
    ys = ys[mask]
    if xs.size == 0:
        return None
    if xs.size == 1 or np.allclose(xs, xs[0]):
        return 0.0, float(np.log(ys).mean())
    slope, intercept = np.polyfit(xs, np.log(ys), 1)
    return float(slope), float(intercept)


def predict_log_linear_regression(
    xs: np.ndarray, slope: float, intercept: float
) -> np.ndarray:
    xs = np.asarray(xs, dtype=float)
    return np.exp(intercept + slope * xs)


def normalize_mapping(mapping: Any) -> dict[str, Any]:
    data = require_mapping(mapping, "mapping")
    normalized: dict[str, Any] = {}
    for key, value in data.items():
        if not isinstance(key, str):
            raise SystemExit("Expected mapping keys to be strings.")
        normalized[key.replace("-", "_")] = unwrap_wandb_value(value)
    return normalized


def normalize_run_config(mapping: Any) -> dict[str, Any]:
    normalized = normalize_mapping(mapping)
    return {
        key: normalize_run_config_value(key, value) for key, value in normalized.items()
    }


def normalize_run_config_value(key: str, value: Any) -> Any:
    if isinstance(value, str):
        parsed = parse_string_literal(value)
        if parsed is not None:
            value = parsed

    if is_sequence_like_key(key):
        if isinstance(value, str):
            return [value.strip()]
        if isinstance(value, (list, tuple)):
            return [normalize_sequence_item(item) for item in value]
        if isinstance(value, dict):
            return {k: normalize_sequence_item(v) for k, v in value.items()}
        return [value]

    if isinstance(value, list):
        return [normalize_run_config_value(key, item) for item in value]
    if isinstance(value, tuple):
        return tuple(normalize_run_config_value(key, item) for item in value)
    if isinstance(value, dict):
        return {
            k: normalize_run_config_value(f"{key}.{k}", v) for k, v in value.items()
        }
    return value


def normalize_sequence_item(value: Any) -> Any:
    if isinstance(value, str):
        parsed = parse_string_literal(value)
        if parsed is not None:
            value = parsed
        else:
            return value.strip()
    if isinstance(value, list):
        return [normalize_sequence_item(item) for item in value]
    if isinstance(value, tuple):
        return tuple(normalize_sequence_item(item) for item in value)
    if isinstance(value, dict):
        return {k: normalize_sequence_item(v) for k, v in value.items()}
    return value


def is_sequence_like_key(key: str) -> bool:
    normalized = key.replace("-", "_")
    return normalized in {"matrix_targets", "radial_layers", "scales"} or (
        normalized.endswith("_targets") or normalized.endswith("_layers")
    )


def is_analysis_variable_key(key: str) -> bool:
    if key in {
        "cfg",
        "run_name",
        "wandb_project",
        "checkpoint_dir",
        "save_dir",
        "snapshot_cache_dir",
        "data_path",
        "dataset_kind",
        "dataset_device",
        "precision",
        "device",
        "gpus",
        "num_workers",
        "seed",
        "resume_mode",
        "wandb_mode",
        "benchmark",
        "verbosity",
        "bench_verbosity",
        "tune",
    }:
        return False
    if key.startswith("log_") or key.startswith("wandb_"):
        return False
    return True


def parse_string_literal(value: str) -> Any | None:
    stripped = value.strip()
    if not stripped:
        return value
    if stripped[0] in "[{(" and stripped[-1] in "]})":
        try:
            return ast.literal_eval(stripped)
        except Exception:
            return value
    if "," in stripped and " " not in stripped:
        parts = [part.strip() for part in stripped.split(",") if part.strip()]
        if len(parts) > 1:
            return parts
    return None


def require_mapping(mapping: Any, context: str) -> dict[str, Any]:
    if isinstance(mapping, dict):
        return mapping
    try:
        result = dict(mapping)
    except Exception as exc:
        raise SystemExit(f"Expected {context} to be a mapping.") from exc
    return result


def unwrap_wandb_value(value: Any) -> Any:
    if isinstance(value, dict) and set(value) == {"value"}:
        return value["value"]
    return value


def coerce_numeric(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float, np.number)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except Exception:
            return None
    return None


def canonical_value(value: Any) -> Any:
    numeric = coerce_numeric(value)
    if numeric is not None:
        return ("num", numeric)
    if isinstance(value, bool):
        return ("bool", bool(value))
    if isinstance(value, (list, tuple)):
        return ("seq", tuple(canonical_value(item) for item in value))
    if isinstance(value, dict):
        return (
            "dict",
            tuple(sorted((str(k), canonical_value(v)) for k, v in value.items())),
        )
    return ("str", str(value))


def display_value(value: Any) -> str:
    numeric = coerce_numeric(value)
    if numeric is not None:
        return f"{numeric:g}"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (list, tuple, dict)):
        return json.dumps(value, sort_keys=True, default=json_default)
    return str(value)


def shorten_label(text: str, max_chars: int) -> str:
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    if max_chars <= 4:
        return text[:max_chars]
    head = max(1, max_chars - 1)
    return text[: head - 1].rstrip() + "…"


def display_name_for_key(name: str) -> str:
    return name.replace("_", "-")


def format_categorical_value(key: str, value: Any, max_chars: int) -> str:
    text = display_value(value)
    if key == "hidden_irreps":
        text = truncate_hidden_irreps_value(text)
    return shorten_label(text, max_chars)


def truncate_hidden_irreps_value(value: str) -> str:
    parts = value.split("+")
    if len(parts) < 3:
        return value
    return "+".join(parts[:2]) + "+"


def unique_in_order(values: list[Any]) -> list[Any]:
    seen = set()
    ordered: list[Any] = []
    for value in values:
        marker = canonical_value(value)
        if marker in seen:
            continue
        seen.add(marker)
        ordered.append(value)
    return ordered


def slugify(text: str) -> str:
    cleaned = []
    for char in text.lower():
        if char.isalnum():
            cleaned.append(char)
        else:
            cleaned.append("-")
    slug = "".join(cleaned).strip("-")
    while "--" in slug:
        slug = slug.replace("--", "-")
    return slug or "wandb-sweep"


def json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    return str(value)


def summarize_variable_spec(spec: dict[str, Any]) -> str:
    display_name = spec["display_name"]
    kind = spec["kind"]
    distribution = spec["distribution"]
    unique_values = spec["unique_values"]
    n_unique = len(unique_values)
    n_present = len(spec["values"])
    if kind == "numeric":
        numeric_values = np.asarray(spec["numeric_values"], dtype=float)
        lo = float(np.min(numeric_values))
        hi = float(np.max(numeric_values))
        distribution_text = (
            f" The sweep metadata marks it as `{distribution}`."
            if distribution is not None
            else ""
        )
        examples = ", ".join(unique_values[:4])
        if n_unique > 4:
            examples += ", ..."
        return (
            f"`{display_name}` is a numeric parameter that appears in `{n_present}` runs "
            f"with `{n_unique}` distinct values spanning `{lo:g}` to `{hi:g}`."
            f"{distribution_text} Example values: {examples}."
        )
    examples = ", ".join(unique_values[:6])
    if n_unique > 6:
        examples += ", ..."
    distribution_text = (
        f" The sweep metadata marks it as `{distribution}`."
        if distribution is not None
        else ""
    )
    return (
        f"`{display_name}` is a categorical parameter that appears in `{n_present}` runs "
        f"with `{n_unique}` distinct values."
        f"{distribution_text} Example values: {examples}."
    )


def build_summary_payload(result: SweepAnalysisResult) -> dict[str, Any]:
    return {
        "sweep_path": result.sweep_path,
        "rank_metric": result.rank_metric,
        "rank_goal": result.rank_goal,
        "objective_metric": result.objective_metric,
        "runs_loaded": result.runs_loaded,
        "skipped_runs": result.skipped_runs,
        "state_counts": result.state_counts,
        "top_runs": [
            {
                "run_id": rec.run_id,
                "name": rec.name,
                "state": rec.state,
                "score": rec.score,
                "url": run_url_for_record(result.sweep_path, rec),
                "config": rec.config,
            }
            for rec in result.top_runs
        ],
        "non_constant_variables": [
            {
                "name": spec["display_name"],
                "normalized_name": spec["key"],
                "distribution": spec["distribution"],
                "kind": spec["kind"],
                "unique_values": spec["unique_values"],
            }
            for spec in result.variable_specs
        ],
    }


def build_markdown_report(result: SweepAnalysisResult) -> str:
    lines = [
        f"# Sweep summary: {result.sweep_path}",
        "",
        f"- Ranking metric: `{result.rank_metric}`",
        f"- Ranking goal: `{result.rank_goal}`",
        f"- Runs loaded: `{result.runs_loaded}`",
        f"- Non-constant swept variables: `{len(result.variable_specs)}`",
        "",
        "## Variables",
        "",
    ]
    for spec in result.variable_specs:
        lines.append(f"### `{spec['display_name']}`")
        lines.append(summarize_variable_spec(spec))
        lines.append("")
        for idx, rec in enumerate(result.top_runs, start=1):
            value = rec.config.get(spec["key"], MISSING)
            lines.append(
                f"- {idx}. `{rec.name}` (`{rec.run_id}`): "
                f"`{display_value(value) if value is not MISSING else 'missing'}`"
            )
        lines.append("")
    return "\n".join(lines) + "\n"


def print_analysis_summary(result: SweepAnalysisResult) -> None:
    print(f"Sweep: {result.sweep_path}")
    print(f"Runs loaded: {result.runs_loaded}")
    print(f"Runs available for plots: {len(result.records)}")
    print(
        f"Runs skipped for ranking without {result.rank_metric!r}: {result.skipped_runs}"
    )
    print(f"Ranking metric: {result.rank_metric} ({result.rank_goal})")
    print(f"Top runs used for overlays: {len(result.top_runs)}")
    print(f"Non-constant swept variables: {len(result.variable_specs)}")
    print()
    print("Run states:")
    for state, count in sorted(result.state_counts.items()):
        print(f"  {state}: {count}")
    print()
    print("Non-constant swept variables:")
    for spec in result.variable_specs:
        distribution = spec["distribution"]
        suffix = f", distribution={distribution}" if distribution is not None else ""
        print(f"  - {spec['display_name']} ({spec['kind']}{suffix})")
    print()
    print(f"Top runs by {result.rank_metric}:")
    for idx, record in enumerate(result.top_runs, start=1):
        print(f"  {idx}. {record.name} ({record.run_id}) -> {record.score}")
    print()


def default_output_dir(sweep_path: str) -> Path:
    return Path("wandb_sweep_analysis") / slugify(sweep_path)
