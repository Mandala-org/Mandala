from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import torch


SPECTRAL_FERMI_CACHE_VERSION = "v1"


def spectral_fermi_cache_key(matrix_path: Path | str, info_path: Path | str) -> str:
    matrix = str(Path(matrix_path).expanduser().resolve())
    info = str(Path(info_path).expanduser().resolve())
    return f"{matrix}::{info}"


def spectral_fermi_cache_hash(matrix_path: Path | str, info_path: Path | str) -> str:
    key = spectral_fermi_cache_key(matrix_path, info_path)
    return hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]


def _normalize_cache_payload(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise TypeError(
            f"Unexpected spectral Fermi cache payload type: {type(payload)!r}"
        )
    records = payload.get("records")
    if not isinstance(records, dict):
        raise ValueError("Spectral Fermi cache payload must contain a 'records' dict.")
    failures = payload.get("failures", [])
    if failures is None:
        failures = []
    if not isinstance(failures, list):
        raise ValueError(
            "Spectral Fermi cache payload field 'failures' must be a list."
        )
    version = str(payload.get("version", "") or "")
    if version and version != SPECTRAL_FERMI_CACHE_VERSION:
        raise ValueError(
            f"Unsupported spectral Fermi cache version {version!r}; "
            f"expected {SPECTRAL_FERMI_CACHE_VERSION!r}."
        )
    return {
        "version": version or SPECTRAL_FERMI_CACHE_VERSION,
        "records": records,
        "failures": failures,
        "settings": payload.get("settings", {}),
    }


def load_spectral_fermi_cache(path: str | Path) -> dict[str, Any]:
    cache_path = Path(path).expanduser().resolve()
    if cache_path.is_dir():
        merged: dict[str, Any] = {
            "version": SPECTRAL_FERMI_CACHE_VERSION,
            "records": {},
            "failures": [],
            "settings": {},
        }
        shard_paths = sorted(cache_path.glob("*.pt"))
        if not shard_paths:
            raise FileNotFoundError(
                f"No spectral Fermi cache shard files found in {cache_path}"
            )
        for shard_path in shard_paths:
            shard = _normalize_cache_payload(
                torch.load(shard_path, map_location="cpu", weights_only=False)
            )
            overlap = set(merged["records"]).intersection(shard["records"])
            if overlap:
                raise ValueError(
                    f"Duplicate spectral Fermi cache keys found while loading {cache_path}: "
                    f"{sorted(list(overlap))[:3]}"
                )
            merged["records"].update(shard["records"])
            merged["failures"].extend(shard["failures"])
        return merged
    if not cache_path.exists():
        raise FileNotFoundError(f"Spectral Fermi cache not found: {cache_path}")
    return _normalize_cache_payload(
        torch.load(cache_path, map_location="cpu", weights_only=False)
    )


def lookup_spectral_fermi_record(
    cache_payload: dict[str, Any],
    matrix_path: Path | str,
    info_path: Path | str,
) -> dict[str, Any] | None:
    key = spectral_fermi_cache_key(matrix_path, info_path)
    records = cache_payload.get("records", {})
    record = records.get(key)
    if record is None:
        return None
    if not isinstance(record, dict):
        raise TypeError(f"Unexpected spectral Fermi cache record type for key {key!r}")
    return record
