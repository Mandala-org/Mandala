from __future__ import annotations

import uuid
from typing import Any

_PLACEHOLDER_RUN_NAMES = {"", "run", "mandala-run"}


def sanitize_run_name(name: str) -> str:
    cleaned = str(name).strip()
    if not cleaned:
        return ""
    return cleaned.replace("/", "_").replace("\\", "_")


def resolve_run_name(requested_run_name: str | None, logger: Any | None) -> str:
    """Resolve a single canonical run name for both logging and checkpoints."""
    if requested_run_name:
        resolved = sanitize_run_name(requested_run_name)
        if logger is not None:
            _sync_logger_run_name(logger, resolved)
        return resolved

    if logger is not None:
        experiment = getattr(logger, "experiment", None)
        if experiment is not None:
            for attr in ("name", "id"):
                candidate = getattr(experiment, attr, None)
                if not candidate:
                    continue
                resolved = sanitize_run_name(str(candidate))
                if resolved and resolved not in _PLACEHOLDER_RUN_NAMES:
                    _sync_logger_run_name(logger, resolved)
                    return resolved

    return f"run_{uuid.uuid4().hex[:10]}"


def _sync_logger_run_name(logger: Any, resolved_run_name: str) -> None:
    experiment = getattr(logger, "experiment", None)
    if experiment is None:
        return
    try:
        if getattr(experiment, "name", None) != resolved_run_name:
            experiment.name = resolved_run_name
    except Exception:
        pass
