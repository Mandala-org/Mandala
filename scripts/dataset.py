from __future__ import annotations

import dataclasses
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
    print(f"--- Discovering silicon snapshots under {root} ---")
    pairs = discover_single_snapshot_pairs(root, label="silicon")
    print(f"--- Found {len(pairs)} silicon snapshot pairs ---")
    return pairs


def discover_single_snapshot_pairs(
    root: Path,
    *,
    label: str,
) -> list[tuple[Path, Path]]:
    print(f"--- Discovering {label} snapshots under {root} ---")
    pairs: list[tuple[Path, Path]] = []

    if root.is_dir():
        matrix_path = root / "HS.out"
        if matrix_path.exists():
            info_path = _resolve_single_snapshot_info_path(root)
            if info_path is not None:
                pairs.append((matrix_path.resolve(), info_path.resolve()))

    for sample_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        matrix_path = sample_dir / "HS.out"
        if not matrix_path.exists():
            continue
        info_path = _resolve_single_snapshot_info_path(sample_dir)
        if info_path is None:
            continue
        pairs.append((matrix_path.resolve(), info_path.resolve()))
    print(f"--- Found {len(pairs)} {label} snapshot pairs ---")
    return pairs


def discover_scale_snapshot_pairs(
    root: Path,
    *,
    scales: list[int],
    label: str,
) -> dict[int, list[tuple[Path, Path]]]:
    print(f"--- Discovering {label} snapshots under {root} for scales={scales} ---")
    pairs_by_scale: dict[int, list[tuple[Path, Path]]] = {}
    for scale in scales:
        scale_dir = root / f"scale_{scale}"
        if not scale_dir.is_dir():
            raise ValueError(f"Missing scale directory: {scale_dir}")
        pairs: list[tuple[Path, Path]] = []
        for sample_dir in sorted(path for path in scale_dir.iterdir() if path.is_dir()):
            matrix_path = sample_dir / "HS.out"
            if not matrix_path.exists():
                continue
            info_path = _resolve_single_snapshot_info_path(sample_dir)
            if info_path is None:
                continue
            pairs.append((matrix_path.resolve(), info_path.resolve()))
        print(f"--- Found {len(pairs)} {label} snapshot pairs in scale_{scale} ---")
        pairs_by_scale[scale] = pairs
    return pairs_by_scale


def discover_siox_snapshot_pairs(root: Path) -> list[tuple[Path, Path]]:
    print(f"--- Discovering SiOx snapshots under {root} ---")
    return discover_single_snapshot_pairs(root, label="SiOx")


def _resolve_single_snapshot_info_path(sample_dir: Path) -> Path | None:
    for name in ("Si.out", "SiO2.out", "ZnCuSeS.out", "info.dat", "info.txt"):
        candidate = sample_dir / name
        if candidate.exists():
            return candidate
    return None


def _create_datasets_from_pairs(
    train_pairs: list[tuple[Path, Path]],
    val_pairs: list[tuple[Path, Path]],
    cfg: Config,
    *,
    convention: str = "e3nn",
):
    print(
        f"--- Creating datasets from pairs: train={len(train_pairs)}, val={len(val_pairs)}, convention={convention} ---"
    )
    if train_pairs:
        print(f"train[0]={train_pairs[0][0]} | {train_pairs[0][1]}")
    if val_pairs:
        print(f"val[0]={val_pairs[0][0]} | {val_pairs[0][1]}")
    fac = DatasetFactory(dataclasses.replace(cfg), convention=convention)
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
    print(
        f"--- Silicon dataset builder: data_path={data_root}, seed={seed}, num_train={num_train}, num_val={num_val}, min_temp={min_temp}, max_temp={max_temp}, val_temp={val_temp} ---"
    )

    if num_train is not None or num_val is not None:
        if num_train is None or num_val is None:
            raise ValueError("Global split mode requires both num_train and num_val.")
        if num_train <= 0:
            raise ValueError("num_train must be > 0")
        if num_val < 0:
            raise ValueError("num_val must be >= 0")
        if min_temp == max_temp == val_temp:
            temp_path = data_root / f"{val_temp}K"
            all_pairs = discover_single_snapshot_pairs(temp_path, label="silicon")
            print(
                f"--- Silicon single-temp split selected: temp={val_temp}K, discovered={len(all_pairs)} ---"
            )
            rng.shuffle(all_pairs)
            if len(all_pairs) < num_train + num_val:
                raise ValueError(
                    f"Requested train+val={num_train + num_val} but found only {len(all_pairs)} snapshots under {data_root}"
                )
            train_pairs = all_pairs[:num_train]
            val_pairs = all_pairs[num_train : num_train + num_val]
            print(
                f"--- Silicon single-temp global split selected: train={len(train_pairs)}, val={len(val_pairs)} ---"
            )
            return _create_datasets_from_pairs(
                train_pairs, val_pairs, cfg, convention=convention
            )

        all_temps = list(range(min_temp, max_temp + 1, temp_step))
        if not all_temps:
            raise ValueError(
                f"No temperatures selected by min_temp={min_temp}, max_temp={max_temp}, temp_step={temp_step}"
            )
        if num_train % len(all_temps) != 0 or num_val % len(all_temps) != 0:
            raise ValueError(
                "Balanced silicon split requires num_train and num_val to be divisible "
                f"by the number of temperatures ({len(all_temps)}). Got num_train={num_train}, num_val={num_val}."
            )

        train_per_temp = num_train // len(all_temps)
        val_per_temp = num_val // len(all_temps)
        train_pairs = []
        val_pairs = []
        print(
            "--- Silicon balanced split selected: "
            f"temps={len(all_temps)}, train_per_temp={train_per_temp}, val_per_temp={val_per_temp} ---"
        )
        for temp in all_temps:
            temp_path = data_root / f"{temp}K"
            snapshot_pairs = discover_single_snapshot_pairs(temp_path, label="silicon")
            snapshot_paths = [matrix_path for matrix_path, _ in snapshot_pairs]
            if len(snapshot_paths) < train_per_temp + val_per_temp:
                raise ValueError(
                    f"Requested train+val per temp={train_per_temp + val_per_temp} "
                    f"but found only {len(snapshot_paths)} snapshots under {temp_path}"
                )
            shuffled_paths = list(snapshot_paths)
            rng.shuffle(shuffled_paths)
            selected_train = shuffled_paths[:train_per_temp]
            selected_val = shuffled_paths[
                train_per_temp : train_per_temp + val_per_temp
            ]
            print(
                f"--- Silicon temp {temp}K: discovered={len(snapshot_paths)}, "
                f"train={len(selected_train)}, val={len(selected_val)} ---"
            )
            pair_map = {
                matrix_path: info_path for matrix_path, info_path in snapshot_pairs
            }
            for matrix_path in selected_train:
                train_pairs.append((Path(matrix_path), pair_map[Path(matrix_path)]))
            for matrix_path in selected_val:
                val_pairs.append((Path(matrix_path), pair_map[Path(matrix_path)]))

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
        snapshot_pairs = discover_single_snapshot_pairs(temp_path, label="silicon")
        snapshot_paths = [matrix_path for matrix_path, _ in snapshot_pairs]
        num_to_sample = min(len(snapshot_paths), n_snapshots_per_temp)
        print(
            f"--- Silicon temp {temp}K: discovered={len(snapshot_paths)}, sampled={num_to_sample} ---"
        )
        selected_paths = rng.sample(snapshot_paths, num_to_sample)
        pair_map = {matrix_path: info_path for matrix_path, info_path in snapshot_pairs}
        for matrix_path in selected_paths:
            train_pairs.append((Path(matrix_path), pair_map[Path(matrix_path)]))

    val_pairs: list[tuple[Path, Path]] = []
    val_path = data_root / f"{val_temp}K"
    val_snapshot_pairs = discover_single_snapshot_pairs(val_path, label="silicon")
    val_snapshot_paths = [matrix_path for matrix_path, _ in val_snapshot_pairs]
    num_val_to_sample = min(len(val_snapshot_paths), n_snapshots_per_temp)
    if val_n_snapshots is not None:
        num_val_to_sample = min(num_val_to_sample, val_n_snapshots)
    print(
        f"--- Silicon val temp {val_temp}K: discovered={len(val_snapshot_paths)}, sampled={num_val_to_sample} ---"
    )
    selected_val_paths = rng.sample(val_snapshot_paths, num_val_to_sample)
    val_pair_map = {
        matrix_path: info_path for matrix_path, info_path in val_snapshot_pairs
    }
    for matrix_path in selected_val_paths:
        val_pairs.append((Path(matrix_path), val_pair_map[Path(matrix_path)]))

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
    print(
        f"--- SiOx dataset builder: data_path={data_path}, seed={seed}, num_train={num_train}, num_val={num_val}, val_fraction={val_fraction} ---"
    )
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
    print(
        f"--- SiOx split selected: train={len(train_pairs)}, val={len(val_pairs)} ---"
    )
    return _create_datasets_from_pairs(
        train_pairs, val_pairs, cfg, convention=convention
    )


def build_zncusnses_small_datasets(
    *,
    data_path: str | Path,
    cfg: Config,
    num_train: int | None = None,
    num_val: int | None = None,
    val_fraction: float = 0.2,
    seed: int = 42,
    convention: str = "e3nn",
):
    print(
        f"--- ZnCuSnSeS_small dataset builder: data_path={data_path}, seed={seed}, num_train={num_train}, num_val={num_val}, val_fraction={val_fraction} ---"
    )
    if val_fraction < 0.0 or val_fraction >= 1.0:
        raise ValueError("val_fraction must be in [0, 1).")
    all_pairs = discover_single_snapshot_pairs(Path(data_path), label="ZnCuSnSeS_small")
    rng = random.Random(seed)
    rng.shuffle(all_pairs)
    if not all_pairs:
        raise ValueError(f"No ZnCuSnSeS_small snapshots found under {data_path}")

    if num_train is not None or num_val is not None:
        if num_train is None or num_val is None:
            raise ValueError(
                "ZnCuSnSeS_small global split mode requires both num_train and num_val."
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
            f"Requested train+val={num_train + num_val} but found only {len(all_pairs)} ZnCuSnSeS_small snapshots under {data_path}"
        )
    train_pairs = all_pairs[:num_train]
    val_pairs = all_pairs[num_train : num_train + num_val]
    print(
        f"--- ZnCuSnSeS_small split selected: train={len(train_pairs)}, val={len(val_pairs)} ---"
    )
    return _create_datasets_from_pairs(
        train_pairs, val_pairs, cfg, convention=convention
    )


def build_zncusnses_datasets(
    *,
    data_path: str | Path,
    cfg: Config,
    scales: list[int],
    num_train_per_scale: int,
    num_val_per_scale: int,
    seed: int = 42,
    convention: str = "e3nn",
):
    print(
        f"--- ZnCuSnSeS dataset builder: data_path={data_path}, seed={seed}, scales={scales}, num_train_per_scale={num_train_per_scale}, num_val_per_scale={num_val_per_scale} ---"
    )
    if not scales:
        raise ValueError("scales must contain at least one scale id")
    if num_train_per_scale <= 0:
        raise ValueError("num_train_per_scale must be > 0")
    if num_val_per_scale < 0:
        raise ValueError("num_val_per_scale must be >= 0")

    pairs_by_scale = discover_scale_snapshot_pairs(
        Path(data_path), scales=scales, label="ZnCuSnSeS"
    )
    rng = random.Random(seed)
    train_pairs: list[tuple[Path, Path]] = []
    val_pairs: list[tuple[Path, Path]] = []

    for scale in scales:
        scale_pairs = list(pairs_by_scale[scale])
        rng.shuffle(scale_pairs)
        required = num_train_per_scale + num_val_per_scale
        if len(scale_pairs) < required:
            raise ValueError(
                f"Requested train+val={required} per scale but found only {len(scale_pairs)} snapshots under scale_{scale}"
            )
        selected_train = scale_pairs[:num_train_per_scale]
        selected_val = scale_pairs[
            num_train_per_scale : num_train_per_scale + num_val_per_scale
        ]
        print(
            f"--- ZnCuSnSeS scale {scale}: discovered={len(scale_pairs)}, train={len(selected_train)}, val={len(selected_val)} ---"
        )
        train_pairs.extend(selected_train)
        val_pairs.extend(selected_val)

    print(
        f"--- ZnCuSnSeS split selected: train={len(train_pairs)}, val={len(val_pairs)} ---"
    )
    return _create_datasets_from_pairs(
        train_pairs, val_pairs, cfg, convention=convention
    )


def build_silicon_scales_datasets(
    *,
    data_path: str | Path,
    cfg: Config,
    scales: list[int],
    num_train_per_scale: int,
    num_val_per_scale: int,
    seed: int = 42,
    convention: str = "e3nn",
):
    print(
        f"--- Silicon-scales dataset builder: data_path={data_path}, seed={seed}, scales={scales}, num_train_per_scale={num_train_per_scale}, num_val_per_scale={num_val_per_scale} ---"
    )
    allow_incomplete_dataset = bool(getattr(cfg, "allow_incomplete_dataset", False))
    if not scales:
        raise ValueError("scales must contain at least one scale id")
    if num_train_per_scale <= 0:
        raise ValueError("num_train_per_scale must be > 0")
    if num_val_per_scale < 0:
        raise ValueError("num_val_per_scale must be >= 0")

    pairs_by_scale = discover_scale_snapshot_pairs(
        Path(data_path), scales=scales, label="silicon_scales"
    )
    rng = random.Random(seed)
    train_pairs: list[tuple[Path, Path]] = []
    val_pairs: list[tuple[Path, Path]] = []

    for scale in scales:
        scale_pairs = list(pairs_by_scale[scale])
        rng.shuffle(scale_pairs)
        required = num_train_per_scale + num_val_per_scale
        if len(scale_pairs) < required:
            if not allow_incomplete_dataset:
                raise ValueError(
                    f"Requested train+val={required} per scale but found only {len(scale_pairs)} snapshots under scale_{scale}"
                )
            print(
                f"--- Silicon_scales scale {scale}: discovered={len(scale_pairs)} is below requested {required}; using all available snapshots ---"
            )
        train_stop = min(num_train_per_scale, len(scale_pairs))
        val_stop = min(required, len(scale_pairs))
        selected_train = scale_pairs[:train_stop]
        selected_val = scale_pairs[train_stop:val_stop]
        print(
            f"--- Silicon_scales scale {scale}: discovered={len(scale_pairs)}, train={len(selected_train)}, val={len(selected_val)} ---"
        )
        train_pairs.extend(selected_train)
        val_pairs.extend(selected_val)

    print(
        f"--- Silicon_scales split selected: train={len(train_pairs)}, val={len(val_pairs)} ---"
    )
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
    print("--- Building datasets from parsed YAML ---")
    parameters = parsed_yaml.get("parameters", {})
    if not isinstance(parameters, dict):
        raise ValueError("Expected parsed_yaml['parameters'] to be a mapping.")

    dataset_kind = _get_dataset_value(
        parameters, overrides, "dataset_kind", default="silicon"
    )
    print(f"--- YAML dataset_kind resolved to {dataset_kind} ---")
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
    if dataset_kind == "ZnCuSnSeS_small":
        return build_zncusnses_small_datasets(
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
    if dataset_kind == "ZnCuSnSeS":
        return build_zncusnses_datasets(
            data_path=_require_dataset_value(parameters, overrides, "data_path"),
            cfg=cfg,
            scales=[
                int(x)
                for x in _get_dataset_value(parameters, overrides, "scales", default=[])
            ],
            num_train_per_scale=int(
                _get_dataset_value(
                    parameters, overrides, "num_train_per_scale", default=40
                )
            ),
            num_val_per_scale=int(
                _get_dataset_value(
                    parameters, overrides, "num_val_per_scale", default=10
                )
            ),
            seed=int(
                _get_dataset_value(parameters, overrides, "seed", default=cfg.seed)
            ),
            convention=convention,
        )
    if dataset_kind == "silicon_scales":
        return build_silicon_scales_datasets(
            data_path=_require_dataset_value(parameters, overrides, "data_path"),
            cfg=cfg,
            scales=[
                int(x)
                for x in _get_dataset_value(parameters, overrides, "scales", default=[])
            ],
            num_train_per_scale=int(
                _get_dataset_value(
                    parameters, overrides, "num_train_per_scale", default=40
                )
            ),
            num_val_per_scale=int(
                _get_dataset_value(
                    parameters, overrides, "num_val_per_scale", default=10
                )
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
