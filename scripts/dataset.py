from __future__ import annotations

import dataclasses
import math
import hashlib
import json
import numpy as np
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


def discover_deeph_e3_snapshot_pairs(root: Path) -> list[tuple[Path, Path]]:
    """Discover DeepH-E3 snapshots recursively below ``root``."""
    root = root.expanduser().resolve()
    print(f"--- Discovering DeepH-E3 snapshots under {root} ---")
    if not root.is_dir():
        raise FileNotFoundError(f"DeepH-E3 dataset root not found: {root}")
    pairs = []
    for matrix_path in sorted(root.rglob("hamiltonians.h5")):
        info_path = matrix_path.parent / "info.json"
        if not info_path.is_file():
            raise FileNotFoundError(
                f"Found {matrix_path} without required sibling info.json"
            )
        pairs.append((matrix_path.resolve(), info_path.resolve()))
    if not pairs:
        raise ValueError(f"No DeepH-E3 hamiltonians.h5 snapshots found under {root}")
    print(f"--- Found {len(pairs)} DeepH-E3 snapshot pairs ---")
    return pairs


def _resolve_single_snapshot_info_path(sample_dir: Path) -> Path | None:
    for name in ("Si.out", "SiO2.out", "ZnCuSeS.out", "info.dat", "info.txt"):
        candidate = sample_dir / name
        if candidate.exists():
            return candidate
    return None


def _create_datasets_from_pairs(
    train_pairs: list[tuple[Path, Path]],
    val_pairs: list[tuple[Path, Path]],
    test_pairs: list[tuple[Path, Path]] | None,
    cfg: Config,
    *,
    convention: str = "e3nn",
):
    print(
        f"--- Creating datasets from pairs: train={len(train_pairs)}, val={len(val_pairs)}, "
        f"test={len(test_pairs or [])}, convention={convention} ---"
    )
    if train_pairs:
        print(f"train[0]={train_pairs[0][0]} | {train_pairs[0][1]}")
    if val_pairs:
        print(f"val[0]={val_pairs[0][0]} | {val_pairs[0][1]}")
    if test_pairs:
        print(f"test[0]={test_pairs[0][0]} | {test_pairs[0][1]}")
    split_sets = [set(train_pairs), set(val_pairs), set(test_pairs or [])]
    if (
        split_sets[0] & split_sets[1]
        or split_sets[0] & split_sets[2]
        or split_sets[1] & split_sets[2]
    ):
        raise ValueError(
            "Dataset split leakage detected: train/val/test pairs overlap."
        )
    fac = DatasetFactory(dataclasses.replace(cfg), convention=convention)
    for matrix_path, info_path in train_pairs:
        fac.add_snapshot(matrix_path, info_path, purpose="train")
    for matrix_path, info_path in val_pairs:
        fac.add_snapshot(matrix_path, info_path, purpose="val")
    for matrix_path, info_path in test_pairs or []:
        fac.add_snapshot(matrix_path, info_path, purpose="test")
    return fac.create(include_test=bool(test_pairs))


def _split_shuffled_pairs(
    all_pairs: list[tuple[Path, Path]],
    *,
    num_train: int,
    num_val: int,
    num_test: int,
) -> tuple[list[tuple[Path, Path]], list[tuple[Path, Path]], list[tuple[Path, Path]]]:
    if num_train <= 0 or num_val < 0 or num_test < 0:
        raise ValueError("num_train must be > 0 and num_val/num_test must be >= 0")
    required = num_train + num_val + num_test
    if len(all_pairs) < required:
        raise ValueError(
            f"Requested train+val+test={required} but found only {len(all_pairs)} snapshots"
        )
    train_pairs = all_pairs[:num_train]
    val_stop = num_train + num_val
    val_pairs = all_pairs[num_train:val_stop]
    test_pairs = all_pairs[val_stop : val_stop + num_test]
    return train_pairs, val_pairs, test_pairs


def build_datasets_from_manifest(manifest_path, cfg, *, convention="e3nn"):
    """Load explicit OpenMX split pairs without discovering or reshuffling data."""

    document = json.loads(Path(manifest_path).read_text())
    if document.get("schema_version") != 1:
        raise ValueError("Unsupported dataset manifest version")
    pairs = {}
    for split in ("train", "val", "test"):
        pairs[split] = [
            (Path(row["matrix_path"]), Path(row["info_path"]))
            for row in document["splits"][split]
        ]
        for matrix, info in pairs[split]:
            if not matrix.is_absolute() or not info.is_absolute():
                raise ValueError("Manifest paths must be absolute")
            if not matrix.is_file() or not info.is_file():
                raise FileNotFoundError(f"Missing manifest input: {matrix}, {info}")
    if not pairs["train"] or not pairs["val"]:
        raise ValueError("Manifest needs nonempty training and validation splits")
    return _create_datasets_from_pairs(
        pairs["train"], pairs["val"], pairs["test"], cfg, convention=convention
    )


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
    num_test: int = 0,
    data_split_seed: int = 42,
    convention: str = "e3nn",
):
    data_root = Path(data_path)
    rng = random.Random(data_split_seed)
    print(
        f"--- Silicon dataset builder: data_path={data_root}, data_split_seed={data_split_seed}, "
        f"num_train={num_train}, num_val={num_val}, num_test={num_test}, "
        f"min_temp={min_temp}, max_temp={max_temp}, val_temp={val_temp} ---"
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
            train_pairs, val_pairs, test_pairs = _split_shuffled_pairs(
                all_pairs, num_train=num_train, num_val=num_val, num_test=num_test
            )
            print(
                f"--- Silicon single-temp global split selected: train={len(train_pairs)}, "
                f"val={len(val_pairs)}, test={len(test_pairs)} ---"
            )
            return _create_datasets_from_pairs(
                train_pairs, val_pairs, test_pairs, cfg, convention=convention
            )

        all_temps = list(range(min_temp, max_temp + 1, temp_step))
        if not all_temps:
            raise ValueError(
                f"No temperatures selected by min_temp={min_temp}, max_temp={max_temp}, temp_step={temp_step}"
            )
        if (
            num_train % len(all_temps) != 0
            or num_val % len(all_temps) != 0
            or num_test % len(all_temps) != 0
        ):
            raise ValueError(
                "Balanced silicon split requires num_train and num_val to be divisible "
                f"by the number of temperatures ({len(all_temps)}). Got "
                f"num_train={num_train}, num_val={num_val}, num_test={num_test}."
            )

        train_per_temp = num_train // len(all_temps)
        val_per_temp = num_val // len(all_temps)
        test_per_temp = num_test // len(all_temps)
        train_pairs = []
        val_pairs = []
        test_pairs = []
        print(
            "--- Silicon balanced split selected: "
            f"temps={len(all_temps)}, train_per_temp={train_per_temp}, "
            f"val_per_temp={val_per_temp}, test_per_temp={test_per_temp} ---"
        )
        for temp in all_temps:
            temp_path = data_root / f"{temp}K"
            snapshot_pairs = discover_single_snapshot_pairs(temp_path, label="silicon")
            snapshot_paths = [matrix_path for matrix_path, _ in snapshot_pairs]
            if len(snapshot_paths) < train_per_temp + val_per_temp + test_per_temp:
                raise ValueError(
                    f"Requested train+val+test per temp={train_per_temp + val_per_temp + test_per_temp} "
                    f"but found only {len(snapshot_paths)} snapshots under {temp_path}"
                )
            shuffled_paths = list(snapshot_paths)
            rng.shuffle(shuffled_paths)
            selected_train = shuffled_paths[:train_per_temp]
            selected_val = shuffled_paths[
                train_per_temp : train_per_temp + val_per_temp
            ]
            selected_test = shuffled_paths[
                train_per_temp
                + val_per_temp : train_per_temp
                + val_per_temp
                + test_per_temp
            ]
            print(
                f"--- Silicon temp {temp}K: discovered={len(snapshot_paths)}, "
                f"train={len(selected_train)}, val={len(selected_val)}, test={len(selected_test)} ---"
            )
            pair_map = {
                matrix_path: info_path for matrix_path, info_path in snapshot_pairs
            }
            for matrix_path in selected_train:
                train_pairs.append((Path(matrix_path), pair_map[Path(matrix_path)]))
            for matrix_path in selected_val:
                val_pairs.append((Path(matrix_path), pair_map[Path(matrix_path)]))
            for matrix_path in selected_test:
                test_pairs.append((Path(matrix_path), pair_map[Path(matrix_path)]))

        return _create_datasets_from_pairs(
            train_pairs, val_pairs, test_pairs, cfg, convention=convention
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
        train_pairs, val_pairs, [], cfg, convention=convention
    )


def build_siox_datasets(
    *,
    data_path: str | Path,
    cfg: Config,
    num_train: int | None = None,
    num_val: int | None = None,
    num_test: int = 0,
    val_fraction: float = 0.2,
    data_split_seed: int = 42,
    convention: str = "e3nn",
):
    print(
        f"--- SiOx dataset builder: data_path={data_path}, data_split_seed={data_split_seed}, "
        f"num_train={num_train}, num_val={num_val}, num_test={num_test}, val_fraction={val_fraction} ---"
    )
    if val_fraction < 0.0 or val_fraction >= 1.0:
        raise ValueError("val_fraction must be in [0, 1).")
    all_pairs = discover_siox_snapshot_pairs(Path(data_path))
    rng = random.Random(data_split_seed)
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
        num_train = len(all_pairs) - num_val - num_test

    assert num_train is not None and num_val is not None
    train_pairs, val_pairs, test_pairs = _split_shuffled_pairs(
        all_pairs, num_train=num_train, num_val=num_val, num_test=num_test
    )
    print(
        f"--- SiOx split selected: train={len(train_pairs)}, val={len(val_pairs)}, test={len(test_pairs)} ---"
    )
    return _create_datasets_from_pairs(
        train_pairs, val_pairs, test_pairs, cfg, convention=convention
    )


def build_deeph_e3_datasets(
    *,
    data_path: str | Path,
    cfg: Config,
    num_train: int,
    num_val: int,
    num_test: int = 0,
    data_split_seed: int = 42,
    convention: str = "e3nn",
):
    """Build deterministic shuffled splits from a DeepH-E3 dataset tree."""
    if convention not in {"openmx", "e3nn"}:
        raise ValueError(
            "DeepH-E3 datasets support only convention='openmx' or 'e3nn'."
        )
    print(
        "--- DeepH-E3 units: source Hamiltonian/Fermi=eV -> internal=Hartree; "
        "positions/lattice/edge lengths=Angstrom; logged hamiltonian MAE=eV, "
        "MSE=eV^2; matrix optimization losses remain in internal units ---"
    )
    all_pairs = discover_deeph_e3_snapshot_pairs(Path(data_path))
    random.Random(data_split_seed).shuffle(all_pairs)
    train_pairs, val_pairs, test_pairs = _split_shuffled_pairs(
        all_pairs,
        num_train=num_train,
        num_val=num_val,
        num_test=num_test,
    )
    print(
        "--- DeepH-E3 split selected: "
        f"train={len(train_pairs)}, val={len(val_pairs)}, test={len(test_pairs)}, "
        f"seed={data_split_seed} ---"
    )
    return _create_datasets_from_pairs(
        train_pairs, val_pairs, test_pairs, cfg, convention=convention
    )


def build_zncusnses_small_datasets(
    *,
    data_path: str | Path,
    cfg: Config,
    num_train: int | None = None,
    num_val: int | None = None,
    num_test: int = 0,
    val_fraction: float = 0.2,
    data_split_seed: int = 42,
    convention: str = "e3nn",
):
    print(
        f"--- ZnCuSnSeS_small dataset builder: data_path={data_path}, data_split_seed={data_split_seed}, "
        f"num_train={num_train}, num_val={num_val}, num_test={num_test}, val_fraction={val_fraction} ---"
    )
    if val_fraction < 0.0 or val_fraction >= 1.0:
        raise ValueError("val_fraction must be in [0, 1).")
    all_pairs = discover_single_snapshot_pairs(Path(data_path), label="ZnCuSnSeS_small")
    rng = random.Random(data_split_seed)
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
        num_train = len(all_pairs) - num_val - num_test

    assert num_train is not None and num_val is not None
    train_pairs, val_pairs, test_pairs = _split_shuffled_pairs(
        all_pairs, num_train=num_train, num_val=num_val, num_test=num_test
    )
    print(
        f"--- ZnCuSnSeS_small split selected: train={len(train_pairs)}, val={len(val_pairs)}, test={len(test_pairs)} ---"
    )
    return _create_datasets_from_pairs(
        train_pairs, val_pairs, test_pairs, cfg, convention=convention
    )


def build_zncusnses_datasets(
    *,
    data_path: str | Path,
    cfg: Config,
    scales: list[int],
    num_train_per_scale: int,
    num_val_per_scale: int,
    num_test_per_scale: int = 0,
    data_split_seed: int = 42,
    convention: str = "e3nn",
):
    print(
        f"--- ZnCuSnSeS dataset builder: data_path={data_path}, data_split_seed={data_split_seed}, "
        f"scales={scales}, num_train_per_scale={num_train_per_scale}, "
        f"num_val_per_scale={num_val_per_scale}, num_test_per_scale={num_test_per_scale} ---"
    )
    if not scales:
        raise ValueError("scales must contain at least one scale id")
    if num_train_per_scale <= 0:
        raise ValueError("num_train_per_scale must be > 0")
    if num_val_per_scale < 0 or num_test_per_scale < 0:
        raise ValueError("num_val_per_scale and num_test_per_scale must be >= 0")

    pairs_by_scale = discover_scale_snapshot_pairs(
        Path(data_path), scales=scales, label="ZnCuSnSeS"
    )
    rng = random.Random(data_split_seed)
    train_pairs: list[tuple[Path, Path]] = []
    val_pairs: list[tuple[Path, Path]] = []
    test_pairs: list[tuple[Path, Path]] = []

    for scale in scales:
        scale_pairs = list(pairs_by_scale[scale])
        rng.shuffle(scale_pairs)
        required = num_train_per_scale + num_val_per_scale + num_test_per_scale
        if len(scale_pairs) < required:
            raise ValueError(
                f"Requested train+val+test={required} per scale but found only {len(scale_pairs)} snapshots under scale_{scale}"
            )
        selected_train = scale_pairs[:num_train_per_scale]
        selected_val = scale_pairs[
            num_train_per_scale : num_train_per_scale + num_val_per_scale
        ]
        selected_test = scale_pairs[num_train_per_scale + num_val_per_scale : required]
        print(
            f"--- ZnCuSnSeS scale {scale}: discovered={len(scale_pairs)}, train={len(selected_train)}, "
            f"val={len(selected_val)}, test={len(selected_test)} ---"
        )
        train_pairs.extend(selected_train)
        val_pairs.extend(selected_val)
        test_pairs.extend(selected_test)

    print(
        f"--- ZnCuSnSeS split selected: train={len(train_pairs)}, val={len(val_pairs)}, test={len(test_pairs)} ---"
    )
    return _create_datasets_from_pairs(
        train_pairs, val_pairs, test_pairs, cfg, convention=convention
    )


def build_silicon_scales_datasets(
    *,
    data_path: str | Path,
    cfg: Config,
    scales: list[int],
    num_train_per_scale: int,
    num_val_per_scale: int,
    num_test_per_scale: int = 0,
    data_split_seed: int = 42,
    convention: str = "e3nn",
):
    print(
        f"--- Silicon-scales dataset builder: data_path={data_path}, data_split_seed={data_split_seed}, "
        f"scales={scales}, num_train_per_scale={num_train_per_scale}, "
        f"num_val_per_scale={num_val_per_scale}, num_test_per_scale={num_test_per_scale} ---"
    )
    allow_incomplete_dataset = bool(getattr(cfg, "allow_incomplete_dataset", False))
    if not scales:
        raise ValueError("scales must contain at least one scale id")
    if num_train_per_scale <= 0:
        raise ValueError("num_train_per_scale must be > 0")
    if num_val_per_scale < 0 or num_test_per_scale < 0:
        raise ValueError("num_val_per_scale and num_test_per_scale must be >= 0")

    pairs_by_scale = discover_scale_snapshot_pairs(
        Path(data_path), scales=scales, label="silicon_scales"
    )
    rng = random.Random(data_split_seed)
    train_pairs: list[tuple[Path, Path]] = []
    val_pairs: list[tuple[Path, Path]] = []
    test_pairs: list[tuple[Path, Path]] = []

    for scale in scales:
        scale_pairs = list(pairs_by_scale[scale])
        rng.shuffle(scale_pairs)
        required = num_train_per_scale + num_val_per_scale + num_test_per_scale
        if len(scale_pairs) < required:
            if not allow_incomplete_dataset:
                raise ValueError(
                    f"Requested train+val+test={required} per scale but found only {len(scale_pairs)} snapshots under scale_{scale}"
                )
            print(
                f"--- Silicon_scales scale {scale}: discovered={len(scale_pairs)} is below requested {required}; using all available snapshots ---"
            )
        train_stop = min(num_train_per_scale, len(scale_pairs))
        val_stop = min(train_stop + num_val_per_scale, len(scale_pairs))
        selected_train = scale_pairs[:train_stop]
        selected_val = scale_pairs[train_stop:val_stop]
        test_stop = min(required, len(scale_pairs))
        selected_test = scale_pairs[val_stop:test_stop]
        print(
            f"--- Silicon_scales scale {scale}: discovered={len(scale_pairs)}, train={len(selected_train)}, "
            f"val={len(selected_val)}, test={len(selected_test)} ---"
        )
        train_pairs.extend(selected_train)
        val_pairs.extend(selected_val)
        test_pairs.extend(selected_test)

    print(
        f"--- Silicon_scales split selected: train={len(train_pairs)}, val={len(val_pairs)}, test={len(test_pairs)} ---"
    )
    return _create_datasets_from_pairs(
        train_pairs, val_pairs, test_pairs, cfg, convention=convention
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
            num_test=int(
                _get_dataset_value(parameters, overrides, "num_test", default=0)
            ),
            data_split_seed=int(
                _get_dataset_value(
                    parameters,
                    overrides,
                    "data_split_seed",
                    default=cfg.data_split_seed,
                )
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
            num_test=int(
                _get_dataset_value(parameters, overrides, "num_test", default=0)
            ),
            val_fraction=float(
                _get_dataset_value(parameters, overrides, "val_fraction", default=0.2)
            ),
            data_split_seed=int(
                _get_dataset_value(
                    parameters,
                    overrides,
                    "data_split_seed",
                    default=cfg.data_split_seed,
                )
            ),
            convention=convention,
        )
    if dataset_kind == "deeph_e3":
        return build_deeph_e3_datasets(
            data_path=_require_dataset_value(parameters, overrides, "data_path"),
            cfg=cfg,
            num_train=int(
                _get_dataset_value(parameters, overrides, "num_train", default=120)
            ),
            num_val=int(
                _get_dataset_value(parameters, overrides, "num_val", default=20)
            ),
            num_test=int(
                _get_dataset_value(parameters, overrides, "num_test", default=0)
            ),
            data_split_seed=int(
                _get_dataset_value(
                    parameters,
                    overrides,
                    "data_split_seed",
                    default=cfg.data_split_seed,
                )
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
            num_test=int(
                _get_dataset_value(parameters, overrides, "num_test", default=0)
            ),
            val_fraction=float(
                _get_dataset_value(parameters, overrides, "val_fraction", default=0.2)
            ),
            data_split_seed=int(
                _get_dataset_value(
                    parameters,
                    overrides,
                    "data_split_seed",
                    default=cfg.data_split_seed,
                )
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
            num_test_per_scale=int(
                _get_dataset_value(
                    parameters, overrides, "num_test_per_scale", default=0
                )
            ),
            data_split_seed=int(
                _get_dataset_value(
                    parameters,
                    overrides,
                    "data_split_seed",
                    default=cfg.data_split_seed,
                )
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
            num_test_per_scale=int(
                _get_dataset_value(
                    parameters, overrides, "num_test_per_scale", default=0
                )
            ),
            data_split_seed=int(
                _get_dataset_value(
                    parameters,
                    overrides,
                    "data_split_seed",
                    default=cfg.data_split_seed,
                )
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


def validate_training_options(document):
    training = document.get("training", {})
    allowed = {
        "seed",
        "data_split_seed",
        "dtype",
        "batch_size",
        "max_epochs",
        "min_lr",
        "grad_clip_val",
        "grad_clip_algorithm",
        "accumulate_grad_batches",
        "split",
    }
    unknown = set(training) - allowed
    if unknown:
        raise ValueError(f"Unknown training options: {sorted(unknown)}")
    if training.get("grad_clip_algorithm", "norm") not in {"norm", "value"}:
        raise ValueError("grad_clip_algorithm must be norm or value")
    for key in (
        "seed",
        "data_split_seed",
        "batch_size",
        "max_epochs",
        "accumulate_grad_batches",
    ):
        if key in training:
            value = training[key]
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < (0 if "seed" in key else 1)
            ):
                raise ValueError(
                    f"training.{key} must be an integer in its valid positive range"
                )
    for key in ("min_lr", "grad_clip_val"):
        if training.get(key) is not None:
            value = training[key]
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not np.isfinite(value)
                or value <= 0
            ):
                raise ValueError(f"training.{key} must be finite and positive")
    settings = training.get("split", {})
    allowed_split = {
        f"{name}_{kind}"
        for name in ("train", "val", "test")
        for kind in ("size", "ratio")
    }
    if set(settings) - allowed_split:
        raise ValueError(
            f"Unknown split options: {sorted(set(settings) - allowed_split)}"
        )
    for key, value in settings.items():
        if key.endswith("_size"):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{key} must be a nonnegative integer")
        elif (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not np.isfinite(value)
            or not 0 <= value <= 1
        ):
            raise ValueError(f"{key} must be a finite ratio in [0,1]")


def dataset_paths(document):
    validate_training_options(document)
    data, training = document["data"], document.get("training", {})
    graph_source = data.get("graph_source", "stored")
    if graph_source not in {"stored", "radius"}:
        raise ValueError("graph_source must be stored or radius")
    if graph_source == "radius":
        radius = data.get("graph_radius")
        if (
            isinstance(radius, bool)
            or not isinstance(radius, (int, float))
            or not np.isfinite(radius)
            or radius <= 0
        ):
            raise ValueError("Radius graph requires a finite positive graph_radius")
    if "splits" in data:
        splits = {
            key: [Path(p) for p in data["splits"].get(key, [])]
            for key in ("train", "val", "test")
        }
    else:
        root = Path(data["root"])
        paths = sorted(
            {p.parent for p in root.rglob("HS.out")}
            | {
                p.parent
                for p in root.rglob("info.json")
                if graph_source == "radius"
                or (
                    p.parent
                    / data.get(
                        "support_file",
                        data.get("matrix_files", {}).get(
                            "hamiltonian", "hamiltonians.h5"
                        ),
                    )
                ).is_file()
            }
        )
        if not paths:
            raise ValueError(f"No operator snapshots under {root}")
        order = np.random.RandomState(
            training.get("data_split_seed", training.get("seed", 42))
        ).permutation(len(paths))
        settings = training.get("split", {})
        ntrain = settings.get(
            "train_size", int(settings.get("train_ratio", 0.67) * len(paths))
        )
        nval = settings.get(
            "val_size", int(settings.get("val_ratio", 0.33) * len(paths))
        )
        ntest = settings.get(
            "test_size", int(settings.get("test_ratio", 0.0) * len(paths))
        )
        if min(ntrain, nval, ntest) < 0 or ntrain + nval + ntest > len(paths):
            raise ValueError("Requested split sizes exceed available snapshots")
        splits = {
            "train": [paths[i] for i in order[:ntrain]],
            "val": [paths[i] for i in order[ntrain : ntrain + nval]],
            "test": [paths[i] for i in order[ntrain + nval : ntrain + nval + ntest]],
        }
    flat = [p.resolve() for values in splits.values() for p in values]
    if len(flat) != len(set(flat)):
        raise ValueError("Dataset splits overlap or contain duplicate snapshots")
    if not splits["train"] or not splits["val"]:
        raise ValueError("Both training and validation splits must be nonempty")
    return splits


def input_manifest(splits):
    result = {}
    for split, paths in splits.items():
        result[split] = []
        for path in paths:
            hashes = {}
            for source in sorted(path.iterdir()):
                if (
                    source.suffix not in {".dat", ".json", ".h5", ".out", ".cif"}
                    or not source.is_file()
                ):
                    continue
                digest = hashlib.sha256()
                with source.open("rb") as handle:
                    for chunk in iter(lambda: handle.read(1024**2), b""):
                        digest.update(chunk)
                hashes[source.name] = digest.hexdigest()
            result[split].append({"path": str(path.resolve()), "sha256": hashes})
    return result
