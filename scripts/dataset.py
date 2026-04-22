from __future__ import annotations

import dataclasses
import glob
import math
import random
import sys
from pathlib import Path
from typing import Any

# Add project root to the Python path
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

from data.factory import DatasetFactory  # noqa: E402
from net.common import Config  # noqa: E402


def discover_silicon_snapshot_pairs(root: Path) -> list[tuple[Path, Path]]:
    pairs: list[tuple[Path, Path]] = []
    for matrix_path in sorted(root.rglob("Si_DM")):
        if not matrix_path.is_file():
            continue
        info_path = matrix_path.parent / "info.dat"
        if not info_path.exists():
            continue
        pairs.append((matrix_path.resolve(), info_path.resolve()))
    return pairs


def discover_siox_snapshot_pairs(root: Path) -> list[tuple[Path, Path]]:
    pairs: list[tuple[Path, Path]] = []
    for sample_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        matrix_path = sample_dir / "HS.out"
        if not matrix_path.exists():
            continue
        info_path = _resolve_siox_info_path(sample_dir)
        if info_path is None:
            continue
        pairs.append((matrix_path.resolve(), info_path.resolve()))
    return pairs


def _resolve_siox_info_path(sample_dir: Path) -> Path | None:
    preferred = sample_dir / "SiO2.out"
    if preferred.exists():
        return preferred
    candidates = sorted(
        path
        for path in sample_dir.glob("*.out")
        if path.name not in {"HS.out", "log.out"}
    )
    if candidates:
        return candidates[0]
    fallback = sample_dir / "log.out"
    if fallback.exists():
        return fallback
    return None


def _create_datasets_from_pairs(
    train_pairs: list[tuple[Path, Path]],
    val_pairs: list[tuple[Path, Path]],
    cfg: Config,
    *,
    convention: str = "e3nn",
):
    cfg_ds = dataclasses.replace(cfg)
    if not cfg.apply_cutoff_to_targets:
        cfg_ds.cutoff_radius = None
    fac = DatasetFactory(cfg_ds, convention=convention)
    for matrix_path, info_path in train_pairs:
        fac.add_snapshot(matrix_path, info_path, purpose="train")
    for matrix_path, info_path in val_pairs:
        fac.add_snapshot(matrix_path, info_path, purpose="val")
    return fac.create()


def build_silicon_datasets(
    *,
    data_path: str | Path,
    cfg: Config,
    min_temp: int = 300,
    max_temp: int = 3000,
    temp_step: int = 300,
    n_snapshots_per_temp: int = 50,
    val_temp: int = 1500,
    val_n_snapshots: int | None = None,
    num_train: int | None = None,
    num_val: int | None = None,
    seed: int = 42,
    convention: str = "e3nn",
):
    data_root = Path(data_path)
    rng = random.Random(seed)

    if num_train is not None or num_val is not None:
        if num_train is None or num_val is None:
            raise ValueError("Global split mode requires both num_train and num_val.")
        if num_train <= 0:
            raise ValueError("num_train must be > 0")
        if num_val < 0:
            raise ValueError("num_val must be >= 0")
        all_pairs = discover_silicon_snapshot_pairs(data_root)
        rng.shuffle(all_pairs)
        if len(all_pairs) < num_train + num_val:
            raise ValueError(
                f"Requested train+val={num_train + num_val} but found only {len(all_pairs)} snapshots under {data_root}"
            )
        train_pairs = all_pairs[:num_train]
        val_pairs = all_pairs[num_train : num_train + num_val]
        return _create_datasets_from_pairs(
            train_pairs, val_pairs, cfg, convention=convention
        )

    all_temps = range(min_temp, max_temp + 1, temp_step)
    train_temps = [
        temp
        for temp in all_temps
        if temp != val_temp or min_temp == max_temp == val_temp
    ]

    train_pairs: list[tuple[Path, Path]] = []
    for temp in train_temps:
        temp_path = data_root / f"{temp}K"
        snapshot_paths = sorted(glob.glob(str(temp_path / "*/Si_DM")))
        num_to_sample = min(len(snapshot_paths), n_snapshots_per_temp)
        selected_paths = rng.sample(snapshot_paths, num_to_sample)
        for matrix_path in selected_paths:
            info_path = Path(matrix_path).parent / "info.dat"
            if info_path.exists():
                train_pairs.append((Path(matrix_path), info_path))

    val_pairs: list[tuple[Path, Path]] = []
    val_path = data_root / f"{val_temp}K"
    val_snapshot_paths = sorted(glob.glob(str(val_path / "*/Si_DM")))
    num_val_to_sample = min(len(val_snapshot_paths), n_snapshots_per_temp)
    if val_n_snapshots is not None:
        num_val_to_sample = min(num_val_to_sample, val_n_snapshots)
    selected_val_paths = rng.sample(val_snapshot_paths, num_val_to_sample)
    for matrix_path in selected_val_paths:
        info_path = Path(matrix_path).parent / "info.dat"
        if info_path.exists():
            val_pairs.append((Path(matrix_path), info_path))

    return _create_datasets_from_pairs(
        train_pairs, val_pairs, cfg, convention=convention
    )


def build_siox_datasets(
    *,
    data_path: str | Path,
    cfg: Config,
    num_train: int | None = None,
    num_val: int | None = None,
    val_fraction: float = 0.2,
    seed: int = 42,
    convention: str = "e3nn",
):
    if val_fraction < 0.0 or val_fraction >= 1.0:
        raise ValueError("val_fraction must be in [0, 1).")
    all_pairs = discover_siox_snapshot_pairs(Path(data_path))
    rng = random.Random(seed)
    rng.shuffle(all_pairs)
    if not all_pairs:
        raise ValueError(f"No SiOx snapshots found under {data_path}")

    if num_train is not None or num_val is not None:
        if num_train is None or num_val is None:
            raise ValueError(
                "SiOx global split mode requires both num_train and num_val."
            )
        if num_train <= 0:
            raise ValueError("num_train must be > 0")
        if num_val < 0:
            raise ValueError("num_val must be >= 0")
    else:
        inferred_num_val = int(math.floor(len(all_pairs) * val_fraction))
        if len(all_pairs) > 1:
            inferred_num_val = max(1, inferred_num_val)
        num_val = inferred_num_val
        num_train = len(all_pairs) - num_val

    assert num_train is not None and num_val is not None
    if len(all_pairs) < num_train + num_val:
        raise ValueError(
            f"Requested train+val={num_train + num_val} but found only {len(all_pairs)} SiOx snapshots under {data_path}"
        )
    train_pairs = all_pairs[:num_train]
    val_pairs = all_pairs[num_train : num_train + num_val]
    return _create_datasets_from_pairs(
        train_pairs, val_pairs, cfg, convention=convention
    )


def build_datasets_from_yaml(
    parsed_yaml: dict[str, Any],
    cfg: Config,
    *,
    overrides: dict[str, Any] | None = None,
    convention: str = "e3nn",
):
    if overrides is None:
        overrides = {}
    parameters = parsed_yaml.get("parameters", {})
    if not isinstance(parameters, dict):
        raise ValueError("Expected parsed_yaml['parameters'] to be a mapping.")

    dataset_kind = _get_dataset_value(
        parameters, overrides, "dataset_kind", default="silicon"
    )
    if dataset_kind == "silicon":
        return build_silicon_datasets(
            data_path=_require_dataset_value(parameters, overrides, "data_path"),
            cfg=cfg,
            min_temp=int(
                _get_dataset_value(parameters, overrides, "min_temp", default=300)
            ),
            max_temp=int(
                _get_dataset_value(parameters, overrides, "max_temp", default=3000)
            ),
            temp_step=int(
                _get_dataset_value(parameters, overrides, "temp_step", default=300)
            ),
            n_snapshots_per_temp=int(
                _get_dataset_value(
                    parameters, overrides, "n_snapshots_per_temp", default=50
                )
            ),
            val_temp=int(
                _get_dataset_value(parameters, overrides, "val_temp", default=1500)
            ),
            val_n_snapshots=_optional_int(
                _get_dataset_value(
                    parameters, overrides, "val_n_snapshots", default=None
                )
            ),
            num_train=_optional_int(
                _get_dataset_value(parameters, overrides, "num_train", default=None)
            ),
            num_val=_optional_int(
                _get_dataset_value(parameters, overrides, "num_val", default=None)
            ),
            seed=int(
                _get_dataset_value(parameters, overrides, "seed", default=cfg.seed)
            ),
            convention=convention,
        )
    if dataset_kind == "siox":
        return build_siox_datasets(
            data_path=_require_dataset_value(parameters, overrides, "data_path"),
            cfg=cfg,
            num_train=_optional_int(
                _get_dataset_value(parameters, overrides, "num_train", default=None)
            ),
            num_val=_optional_int(
                _get_dataset_value(parameters, overrides, "num_val", default=None)
            ),
            val_fraction=float(
                _get_dataset_value(parameters, overrides, "val_fraction", default=0.2)
            ),
            seed=int(
                _get_dataset_value(parameters, overrides, "seed", default=cfg.seed)
            ),
            convention=convention,
        )
    raise ValueError(f"Unsupported dataset_kind in YAML: {dataset_kind!r}")


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


def _require_dataset_value(
    parameters: dict[str, Any], overrides: dict[str, Any], name: str
) -> Any:
    value = _get_dataset_value(parameters, overrides, name, default=None)
    if value is None:
        raise ValueError(f"Missing required dataset parameter: {name}")
    return value


def _get_dataset_value(
    parameters: dict[str, Any],
    overrides: dict[str, Any],
    name: str,
    *,
    default: Any,
) -> Any:
    alias = name.replace("_", "-")
    if name in overrides and overrides[name] is not None:
        return overrides[name]
    if alias in overrides and overrides[alias] is not None:
        return overrides[alias]
    for key in (name, alias):
        if key in parameters:
            spec = parameters[key]
            if isinstance(spec, dict):
                if "value" in spec:
                    return spec["value"]
                if "values" in spec and len(spec["values"]) == 1:
                    return spec["values"][0]
            elif spec is not None:
                return spec
    return default
