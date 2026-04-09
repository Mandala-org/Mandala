from __future__ import annotations

import json
import os
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import pandas as pd
import yaml


def configure_matplotlib(cache_dir: Path) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def dump_yaml(path: Path, obj) -> None:
    with path.open("w") as f:
        yaml.safe_dump(obj, f, sort_keys=False)


def dump_json(path: Path, obj) -> None:
    with path.open("w") as f:
        json.dump(obj, f, indent=2)


def save_df(path: Path, df: pd.DataFrame) -> None:
    df.to_csv(path, index=False)


def append_study_log(log_path: Path, title: str, lines: Iterable[str]) -> None:
    with log_path.open("a") as f:
        f.write("\n\n")
        f.write(f"## {title}\n\n")
        for line in lines:
            f.write(f"{line}\n")


def serialize_config(obj):
    if is_dataclass(obj):
        return asdict(obj)
    if isinstance(obj, dict):
        return obj
    raise TypeError(f"Cannot serialize config of type {type(obj)!r}")


def save_plot(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, bbox_inches="tight", dpi=180)
    plt.close(fig)
