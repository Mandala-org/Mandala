#!/usr/bin/env python

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from itertools import combinations_with_replacement
from pathlib import Path
from typing import Callable

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-codex")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares
import torch
from tqdm.auto import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from data.snapshot import Snapshot  # noqa: E402


EPS = 1.0e-18


@dataclass(frozen=True)
class CurveFamily:
    name: str
    param_names: tuple[str, ...]
    init_fn: Callable[[np.ndarray, np.ndarray], np.ndarray]
    predict_fn: Callable[[np.ndarray, np.ndarray], np.ndarray]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fit candidate radial decay families to ZnCuSnSeS Hamiltonian block "
            "magnitudes and produce overlay plots."
        )
    )
    parser.add_argument(
        "--snapshot-path",
        type=Path,
        default=Path("data/small/ZnCu2Sn_SeS_2_scale_1_010"),
        help="Snapshot directory containing HS.out and the OpenMX info file.",
    )
    parser.add_argument(
        "--matrix-path",
        type=Path,
        default=None,
        help="Optional explicit matrix path. Overrides --snapshot-path discovery.",
    )
    parser.add_argument(
        "--info-path",
        type=Path,
        default=None,
        help="Optional explicit info path. Overrides --snapshot-path discovery.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("eval_outputs/zncusnses_radial_fit_study"),
        help="Directory for plots and fit summaries.",
    )
    parser.add_argument(
        "--convention",
        type=str,
        default="e3nn",
        help="Snapshot convention used during loading.",
    )
    parser.add_argument(
        "--cutoff-radius",
        type=float,
        default=11.0,
        help="Target cutoff radius to mirror benchmark preprocessing.",
    )
    parser.add_argument(
        "--apply-cutoff",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Apply the target cutoff before plotting and fitting.",
    )
    parser.add_argument(
        "--symmetrize-targets",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Symmetrize targets the same way dataset preprocessing does.",
    )
    parser.add_argument(
        "--min-points-per-pair",
        type=int,
        default=6,
        help="Skip fitting families for pair groups with fewer points than this.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help=(
            "Optional cache directory for the processed snapshot and extracted pair "
            "payload. Defaults to <output-dir>/cache."
        ),
    )
    return parser.parse_args()


def _discover_info_file(snapshot_dir: Path) -> Path | None:
    preferred = [
        "ZnCuSeS.out",
        "info.dat",
        "info.txt",
        "SiO2.out",
    ]
    for name in preferred:
        candidate = snapshot_dir / name
        if candidate.exists():
            return candidate
    for candidate in sorted(snapshot_dir.glob("*.out")):
        if candidate.name in {"HS.out", "log.out"}:
            continue
        return candidate
    return None


def _discover_snapshot_paths(
    snapshot_path: Path,
    matrix_path: Path | None,
    info_path: Path | None,
) -> tuple[Path, Path]:
    if matrix_path is not None and info_path is not None:
        return matrix_path, info_path
    if matrix_path is not None or info_path is not None:
        raise ValueError("Provide both --matrix-path and --info-path together.")
    if snapshot_path.is_file():
        matrix_candidate = snapshot_path
        info_candidate = _discover_info_file(snapshot_path.parent)
        if info_candidate is None:
            raise FileNotFoundError(
                f"Could not infer info file next to matrix file: {snapshot_path}"
            )
        return matrix_candidate, info_candidate
    if not snapshot_path.is_dir():
        raise FileNotFoundError(f"Snapshot path does not exist: {snapshot_path}")
    matrix_candidate = snapshot_path / "HS.out"
    if not matrix_candidate.exists():
        raise FileNotFoundError(f"Matrix file not found: {matrix_candidate}")
    info_candidate = _discover_info_file(snapshot_path)
    if info_candidate is None:
        raise FileNotFoundError(
            f"Could not infer info file under snapshot directory: {snapshot_path}"
        )
    return matrix_candidate, info_candidate


def _load_processed_snapshot(
    matrix_path: Path,
    info_path: Path,
    *,
    convention: str,
    apply_cutoff: bool,
    cutoff_radius: float,
    symmetrize_targets: bool,
) -> Snapshot:
    snap = Snapshot.from_openmx(
        matrix_path=matrix_path,
        info_path=info_path,
        convention=convention,
    )
    if apply_cutoff:
        snap = snap.filter_by_distance(cutoff_radius)
    if symmetrize_targets:
        snap = snap.symmetrize_matrices(
            hamiltonian=True,
            overlap=True,
            density=True,
        )
    return snap


def _cache_signature(
    matrix_path: Path,
    info_path: Path,
    *,
    convention: str,
    apply_cutoff: bool,
    cutoff_radius: float,
    symmetrize_targets: bool,
) -> str:
    payload = {
        "matrix_path": str(matrix_path.resolve()),
        "matrix_mtime_ns": matrix_path.stat().st_mtime_ns,
        "matrix_size": matrix_path.stat().st_size,
        "info_path": str(info_path.resolve()),
        "info_mtime_ns": info_path.stat().st_mtime_ns,
        "info_size": info_path.stat().st_size,
        "convention": convention,
        "apply_cutoff": bool(apply_cutoff),
        "cutoff_radius": float(cutoff_radius),
        "symmetrize_targets": bool(symmetrize_targets),
    }
    blob = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


def _load_cached_snapshot(
    cache_file: Path,
) -> Snapshot | None:
    if not cache_file.exists():
        return None
    try:
        return Snapshot.load(cache_file, device="cpu")
    except Exception as exc:
        print(
            f"[CACHE] Failed to load snapshot cache {cache_file}: {exc!r}", flush=True
        )
        return None


def _save_cached_snapshot(snapshot: Snapshot, cache_file: Path) -> None:
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    snapshot.save(cache_file)


def _load_cached_pair_data(
    cache_file: Path,
) -> tuple[dict[str, dict[str, np.ndarray]], list[str]] | None:
    if not cache_file.exists():
        return None
    try:
        payload = torch.load(cache_file, map_location="cpu", weights_only=False)
        order = list(payload["order"])
        pair_data = {
            key: {
                "distance": np.asarray(value["distance"], dtype=np.float64),
                "magnitude": np.asarray(value["magnitude"], dtype=np.float64),
            }
            for key, value in payload["pair_data"].items()
        }
        return pair_data, order
    except Exception as exc:
        print(
            f"[CACHE] Failed to load pair-data cache {cache_file}: {exc!r}", flush=True
        )
        return None


def _save_cached_pair_data(
    pair_data: dict[str, dict[str, np.ndarray]],
    order: list[str],
    cache_file: Path,
) -> None:
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "order": order,
        "pair_data": {
            key: {
                "distance": value["distance"],
                "magnitude": value["magnitude"],
            }
            for key, value in pair_data.items()
        },
    }
    torch.save(payload, cache_file)


def _unordered_pair_key(a: str, b: str, order_index: dict[str, int]) -> str:
    if order_index[a] <= order_index[b]:
        return f"{a}-{b}"
    return f"{b}-{a}"


def _block_sum_squares(blocks: torch.Tensor) -> torch.Tensor:
    return torch.sum(blocks * blocks, dim=(1, 2))


def _edge_distances(
    edges: torch.Tensor,
    positions: torch.Tensor,
    box: torch.Tensor | None,
) -> torch.Tensor:
    src = edges[3].to(dtype=torch.long)
    dst = edges[4].to(dtype=torch.long)
    disp = positions[dst] - positions[src]
    if box is not None:
        shift = edges[:3].T.to(device=positions.device, dtype=positions.dtype)
        disp = disp + shift @ box
    return torch.linalg.norm(disp, dim=-1)


def _collect_pair_data(
    snap: Snapshot,
) -> tuple[dict[str, dict[str, np.ndarray]], list[str]]:
    matrix = snap.hamiltonian
    positions = snap.positions
    box = snap.box
    if positions is None:
        raise RuntimeError("Snapshot positions are missing.")

    element_order = list(dict.fromkeys(matrix.atoms))
    order_index = {el: idx for idx, el in enumerate(element_order)}
    subplot_order = [
        f"{a}-{b}" for a, b in combinations_with_replacement(element_order, 2)
    ]
    pair_payload: dict[str, dict[str, list[float]]] = {
        key: {"distance": [], "magnitude": []} for key in subplot_order
    }

    for directed_key, blocks in tqdm(
        matrix.pair_blocks.items(),
        total=len(matrix.pair_blocks),
        desc="Collecting pair payload",
    ):
        edges = matrix.pair_edges[directed_key]
        values = _block_sum_squares(blocks).detach().cpu()
        dists = _edge_distances(edges, positions, box).detach().cpu()
        el_a, el_b = directed_key.split("-")
        pair_key = _unordered_pair_key(el_a, el_b, order_index)
        payload = pair_payload[pair_key]
        for idx in range(values.shape[0]):
            payload["distance"].append(float(dists[idx].item()))
            payload["magnitude"].append(float(values[idx].item()))

    out: dict[str, dict[str, np.ndarray]] = {}
    for pair_key, payload in pair_payload.items():
        out[pair_key] = {
            "distance": np.asarray(payload["distance"], dtype=np.float64),
            "magnitude": np.asarray(payload["magnitude"], dtype=np.float64),
        }
    return out, subplot_order


def _exp_init(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    amp = max(float(np.max(y)), EPS)
    return np.array([math.log(amp), math.log(0.5)], dtype=np.float64)


def _exp_predict(theta: np.ndarray, x: np.ndarray) -> np.ndarray:
    amp = math.exp(theta[0])
    rate = math.exp(theta[1])
    return amp * np.exp(-rate * x)


def _exp_quad_init(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    amp = max(float(np.max(y)), EPS)
    return np.array([math.log(amp), math.log(0.2), math.log(0.02)], dtype=np.float64)


def _exp_quad_predict(theta: np.ndarray, x: np.ndarray) -> np.ndarray:
    amp = math.exp(theta[0])
    b = math.exp(theta[1])
    c = math.exp(theta[2])
    return amp * np.exp(-b * x - c * x * x)


def _soft_cutoff_init(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    amp = max(float(np.max(y)), EPS)
    r_mid = float(np.median(x)) if x.size else 5.0
    return np.array([math.log(amp), r_mid, math.log(0.5)], dtype=np.float64)


def _soft_cutoff_predict(theta: np.ndarray, x: np.ndarray) -> np.ndarray:
    amp = math.exp(theta[0])
    r0 = theta[1]
    tau = math.exp(theta[2])
    z = np.clip((x - r0) / tau, -60.0, 60.0)
    return amp / (1.0 + np.exp(z))


def _soft_cutoff_exp_init(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    amp = max(float(np.max(y)), EPS)
    r_mid = float(np.median(x)) if x.size else 5.0
    return np.array(
        [math.log(amp), math.log(0.2), r_mid, math.log(0.5)], dtype=np.float64
    )


def _soft_cutoff_exp_predict(theta: np.ndarray, x: np.ndarray) -> np.ndarray:
    amp = math.exp(theta[0])
    b = math.exp(theta[1])
    r0 = theta[2]
    tau = math.exp(theta[3])
    z = np.clip((x - r0) / tau, -60.0, 60.0)
    return amp * np.exp(-b * x) / (1.0 + np.exp(z))


def _slater_like_init(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    amp = max(float(np.max(y)), EPS)
    return np.array([math.log(amp), math.log(0.25), 0.0], dtype=np.float64)


def _slater_like_predict(theta: np.ndarray, x: np.ndarray) -> np.ndarray:
    amp = math.exp(theta[0])
    rate = math.exp(theta[1])
    nu = theta[2]
    return amp * np.power(1.0 + x, nu) * np.exp(-rate * x)


def _stretched_exp_init(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    amp = max(float(np.max(y)), EPS)
    scale = max(float(np.median(x)) if x.size else 5.0, 1.0e-3)
    return np.array([math.log(amp), math.log(scale), math.log(1.0)], dtype=np.float64)


def _stretched_exp_predict(theta: np.ndarray, x: np.ndarray) -> np.ndarray:
    amp = math.exp(theta[0])
    scale = math.exp(theta[1])
    power = math.exp(theta[2])
    return amp * np.exp(-np.power(x / max(scale, 1.0e-12), power))


def _bi_exp_init(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    amp = max(float(np.max(y)), EPS)
    return np.array(
        [
            math.log(0.7 * amp),
            math.log(0.3 * amp),
            math.log(0.2),
            math.log(1.0),
        ],
        dtype=np.float64,
    )


def _bi_exp_predict(theta: np.ndarray, x: np.ndarray) -> np.ndarray:
    amp1 = math.exp(theta[0])
    amp2 = math.exp(theta[1])
    rate1 = math.exp(theta[2])
    rate2 = math.exp(theta[3])
    return amp1 * np.exp(-rate1 * x) + amp2 * np.exp(-rate2 * x)


def _slater_soft_cutoff_init(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    amp = max(float(np.max(y)), EPS)
    r_mid = float(np.quantile(x, 0.8)) if x.size else 5.0
    return np.array(
        [math.log(amp), math.log(0.2), 0.0, r_mid, math.log(0.4)],
        dtype=np.float64,
    )


def _slater_soft_cutoff_predict(theta: np.ndarray, x: np.ndarray) -> np.ndarray:
    amp = math.exp(theta[0])
    rate = math.exp(theta[1])
    nu = theta[2]
    r0 = theta[3]
    tau = math.exp(theta[4])
    z = np.clip((x - r0) / tau, -60.0, 60.0)
    return amp * np.power(1.0 + x, nu) * np.exp(-rate * x) / (1.0 + np.exp(z))


def _bi_exp_soft_cutoff_init(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    amp = max(float(np.max(y)), EPS)
    r_mid = float(np.quantile(x, 0.8)) if x.size else 5.0
    return np.array(
        [
            math.log(0.7 * amp),
            math.log(0.3 * amp),
            math.log(0.2),
            math.log(1.0),
            r_mid,
            math.log(0.4),
        ],
        dtype=np.float64,
    )


def _bi_exp_soft_cutoff_predict(theta: np.ndarray, x: np.ndarray) -> np.ndarray:
    amp1 = math.exp(theta[0])
    amp2 = math.exp(theta[1])
    rate1 = math.exp(theta[2])
    rate2 = math.exp(theta[3])
    r0 = theta[4]
    tau = math.exp(theta[5])
    z = np.clip((x - r0) / tau, -60.0, 60.0)
    core = amp1 * np.exp(-rate1 * x) + amp2 * np.exp(-rate2 * x)
    return core / (1.0 + np.exp(z))


CURVE_FAMILIES: tuple[CurveFamily, ...] = (
    CurveFamily(
        name="exp",
        param_names=("log_amp", "log_rate"),
        init_fn=_exp_init,
        predict_fn=_exp_predict,
    ),
    CurveFamily(
        name="exp_quad",
        param_names=("log_amp", "log_b", "log_c"),
        init_fn=_exp_quad_init,
        predict_fn=_exp_quad_predict,
    ),
    CurveFamily(
        name="soft_cutoff",
        param_names=("log_amp", "r0", "log_tau"),
        init_fn=_soft_cutoff_init,
        predict_fn=_soft_cutoff_predict,
    ),
    CurveFamily(
        name="soft_cutoff_exp",
        param_names=("log_amp", "log_rate", "r0", "log_tau"),
        init_fn=_soft_cutoff_exp_init,
        predict_fn=_soft_cutoff_exp_predict,
    ),
    CurveFamily(
        name="slater_like",
        param_names=("log_amp", "log_rate", "nu"),
        init_fn=_slater_like_init,
        predict_fn=_slater_like_predict,
    ),
    CurveFamily(
        name="stretched_exp",
        param_names=("log_amp", "log_scale", "log_power"),
        init_fn=_stretched_exp_init,
        predict_fn=_stretched_exp_predict,
    ),
    CurveFamily(
        name="bi_exp",
        param_names=("log_amp1", "log_amp2", "log_rate1", "log_rate2"),
        init_fn=_bi_exp_init,
        predict_fn=_bi_exp_predict,
    ),
    CurveFamily(
        name="slater_soft_cutoff",
        param_names=("log_amp", "log_rate", "nu", "r0", "log_tau"),
        init_fn=_slater_soft_cutoff_init,
        predict_fn=_slater_soft_cutoff_predict,
    ),
    CurveFamily(
        name="bi_exp_soft_cutoff",
        param_names=("log_amp1", "log_amp2", "log_rate1", "log_rate2", "r0", "log_tau"),
        init_fn=_bi_exp_soft_cutoff_init,
        predict_fn=_bi_exp_soft_cutoff_predict,
    ),
)


def _fit_family(
    family: CurveFamily,
    x: np.ndarray,
    y: np.ndarray,
) -> tuple[np.ndarray, dict[str, float]]:
    x = np.asarray(x, dtype=np.float64)
    y = np.maximum(np.asarray(y, dtype=np.float64), EPS)

    def residuals(theta: np.ndarray) -> np.ndarray:
        pred = np.maximum(family.predict_fn(theta, x), EPS)
        return np.log10(pred) - np.log10(y)

    theta0 = family.init_fn(x, y)
    result = least_squares(
        residuals,
        theta0,
        method="trf",
        max_nfev=20000,
    )
    theta = result.x
    pred = np.maximum(family.predict_fn(theta, x), EPS)
    log_resid = np.log10(pred) - np.log10(y)
    raw_resid = pred - y
    metrics = {
        "log_rmse": float(np.sqrt(np.mean(log_resid**2))),
        "log_mae": float(np.mean(np.abs(log_resid))),
        "raw_rmse": float(np.sqrt(np.mean(raw_resid**2))),
        "raw_mae": float(np.mean(np.abs(raw_resid))),
        "nfev": float(result.nfev),
        "success": float(bool(result.success)),
        "cost": float(result.cost),
    }
    return theta, metrics


def _pair_colors(order: list[str]) -> dict[str, tuple[float, float, float, float]]:
    cmap = plt.get_cmap("tab20")
    return {pair: cmap(idx % cmap.N) for idx, pair in enumerate(order)}


def _plot_all_pairs_scatter(
    pair_data: dict[str, dict[str, np.ndarray]],
    order: list[str],
    output_path: Path,
) -> None:
    colors = _pair_colors(order)
    fig, ax = plt.subplots(1, 1, figsize=(13, 9), constrained_layout=True)
    for pair in tqdm(order, desc="Plotting baseline scatter"):
        x = pair_data[pair]["distance"]
        y = pair_data[pair]["magnitude"]
        if x.size == 0:
            continue
        ax.scatter(x, y, s=18, alpha=0.8, color=colors[pair], label=pair)
    ax.set_title("scale_1_010 Hamiltonian block sum of squares vs edge distance")
    ax.set_xlabel("Edge distance")
    ax.set_ylabel("Hamiltonian block sum of squares")
    ax.set_yscale("log")
    ax.grid(alpha=0.2)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _plot_family_all_pairs(
    family: CurveFamily,
    pair_data: dict[str, dict[str, np.ndarray]],
    fits: dict[str, dict[str, object]],
    order: list[str],
    output_path: Path,
) -> None:
    colors = _pair_colors(order)
    fig, ax = plt.subplots(1, 1, figsize=(13, 9), constrained_layout=True)
    for pair in tqdm(order, desc=f"Plotting {family.name} all-pairs overlay"):
        x = pair_data[pair]["distance"]
        y = pair_data[pair]["magnitude"]
        if x.size == 0:
            continue
        color = colors[pair]
        ax.scatter(x, y, s=18, alpha=0.55, color=color, label=pair)
        fit = fits.get(pair)
        if fit is None:
            continue
        x_grid = np.linspace(float(np.min(x)), float(np.max(x)), 400)
        y_grid = family.predict_fn(np.asarray(fit["theta"], dtype=np.float64), x_grid)
        ax.plot(x_grid, np.maximum(y_grid, EPS), color=color, linewidth=2.2)
    ax.set_title(
        "scale_1_010 Hamiltonian block sum of squares vs edge distance\n"
        f"fit family: {family.name}"
    )
    ax.set_xlabel("Edge distance")
    ax.set_ylabel("Hamiltonian block sum of squares")
    ax.set_yscale("log")
    ax.grid(alpha=0.2)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _plot_family_per_pair(
    family: CurveFamily,
    pair_data: dict[str, dict[str, np.ndarray]],
    fits: dict[str, dict[str, object]],
    order: list[str],
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(3, 5, figsize=(19.2, 9.6), constrained_layout=True)
    colors = _pair_colors(order)
    for ax, pair in zip(
        tqdm(
            axes.flat, total=len(order), desc=f"Plotting {family.name} per-pair panels"
        ),
        order,
        strict=True,
    ):
        x = pair_data[pair]["distance"]
        y = pair_data[pair]["magnitude"]
        color = colors[pair]
        if x.size == 0:
            ax.set_title(f"{pair} (no data)")
            ax.set_yscale("log")
            ax.grid(alpha=0.2)
            continue
        ax.scatter(x, y, s=18, alpha=0.7, color=color)
        fit = fits.get(pair)
        if fit is not None:
            x_grid = np.linspace(float(np.min(x)), float(np.max(x)), 300)
            y_grid = family.predict_fn(
                np.asarray(fit["theta"], dtype=np.float64), x_grid
            )
            ax.plot(x_grid, np.maximum(y_grid, EPS), color="black", linewidth=1.8)
            ax.set_title(f"{pair}  logRMSE={fit['metrics']['log_rmse']:.3f}")
        else:
            ax.set_title(f"{pair} (fit skipped)")
        ax.set_xlabel("Distance")
        ax.set_ylabel("Sum of squares")
        ax.set_yscale("log")
        ax.grid(alpha=0.2)
    fig.suptitle(f"Per-pair radial fits: {family.name}", fontsize=16)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def _plot_family_log_residuals(
    family: CurveFamily,
    pair_data: dict[str, dict[str, np.ndarray]],
    fits: dict[str, dict[str, object]],
    order: list[str],
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(3, 5, figsize=(19.2, 9.6), constrained_layout=True)
    colors = _pair_colors(order)
    for ax, pair in zip(
        tqdm(
            axes.flat,
            total=len(order),
            desc=f"Plotting {family.name} residual panels",
        ),
        order,
        strict=True,
    ):
        x = pair_data[pair]["distance"]
        y = np.maximum(pair_data[pair]["magnitude"], EPS)
        fit = fits.get(pair)
        if x.size == 0 or fit is None:
            ax.set_title(f"{pair} (fit skipped)")
            ax.grid(alpha=0.2)
            continue
        pred = np.maximum(
            family.predict_fn(np.asarray(fit["theta"], dtype=np.float64), x), EPS
        )
        resid = np.log10(pred) - np.log10(y)
        ax.scatter(x, resid, s=18, alpha=0.7, color=colors[pair])
        ax.axhline(0.0, color="black", linewidth=1.0, linestyle="--")
        ax.set_title(pair)
        ax.set_xlabel("Distance")
        ax.set_ylabel("log10(pred) - log10(gt)")
        ax.grid(alpha=0.2)
    fig.suptitle(f"Per-pair log residuals: {family.name}", fontsize=16)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def _write_fit_summary(
    families: tuple[CurveFamily, ...],
    fits_by_family: dict[str, dict[str, dict[str, object]]],
    order: list[str],
    output_dir: Path,
) -> None:
    summary_payload: dict[str, object] = {}
    csv_rows: list[dict[str, object]] = []
    for family in families:
        fam_payload: dict[str, object] = {}
        fits = fits_by_family[family.name]
        for pair in order:
            fit = fits.get(pair)
            if fit is None:
                fam_payload[pair] = None
                continue
            fam_payload[pair] = {
                "theta": [float(v) for v in fit["theta"]],
                "metrics": fit["metrics"],
            }
            row = {"family": family.name, "pair": pair}
            row.update({k: v for k, v in fit["metrics"].items()})
            for idx, value in enumerate(fit["theta"]):
                row[f"param_{idx}"] = float(value)
            csv_rows.append(row)
        summary_payload[family.name] = fam_payload

    (output_dir / "fit_summary.json").write_text(
        json.dumps(summary_payload, indent=2, sort_keys=True)
    )
    fieldnames = sorted({key for row in csv_rows for key in row.keys()})
    with (output_dir / "fit_summary.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)


def _write_markdown_summary(
    families: tuple[CurveFamily, ...],
    fits_by_family: dict[str, dict[str, dict[str, object]]],
    order: list[str],
    output_dir: Path,
) -> None:
    lines = [
        "# ZnCuSnSeS radial fit study",
        "",
        "Primary metric used for ranking fit agreement here is `log_rmse`.",
        "",
    ]
    for family in families:
        lines.append(f"## {family.name}")
        lines.append("")
        lines.append("| Pair | log RMSE | log MAE | raw MAE |")
        lines.append("| --- | ---: | ---: | ---: |")
        fits = fits_by_family[family.name]
        for pair in order:
            fit = fits.get(pair)
            if fit is None:
                lines.append(f"| {pair} | skipped | skipped | skipped |")
            else:
                m = fit["metrics"]
                lines.append(
                    f"| {pair} | {m['log_rmse']:.4f} | {m['log_mae']:.4f} | {m['raw_mae']:.4e} |"
                )
        lines.append("")
    (output_dir / "fit_summary.md").write_text("\n".join(lines))


def _write_selected_envelope_artifact(
    family_name: str,
    fits_by_family: dict[str, dict[str, dict[str, object]]],
    output_dir: Path,
) -> None:
    fits = fits_by_family.get(family_name)
    if fits is None:
        raise ValueError(f"Requested family {family_name!r} is not available.")
    payload = {
        "family": family_name,
        "pairs": {},
    }
    for pair, fit in fits.items():
        payload["pairs"][pair] = {
            "theta": [float(v) for v in fit["theta"]],
            "metrics": fit["metrics"],
        }
    (output_dir / f"{family_name}_envelope.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True)
    )


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = args.cache_dir or (output_dir / "cache")
    cache_dir.mkdir(parents=True, exist_ok=True)

    matrix_path, info_path = _discover_snapshot_paths(
        args.snapshot_path,
        args.matrix_path,
        args.info_path,
    )
    cache_sig = _cache_signature(
        matrix_path,
        info_path,
        convention=args.convention,
        apply_cutoff=bool(args.apply_cutoff),
        cutoff_radius=float(args.cutoff_radius),
        symmetrize_targets=bool(args.symmetrize_targets),
    )
    snapshot_cache_file = cache_dir / f"processed_snapshot_{cache_sig}.pt"
    pair_cache_file = cache_dir / f"pair_data_{cache_sig}.pt"

    print(f"Loading snapshot from matrix={matrix_path} info={info_path}", flush=True)
    t0 = time.perf_counter()
    snap = _load_cached_snapshot(snapshot_cache_file)
    if snap is None:
        print(f"[CACHE] Snapshot miss -> {snapshot_cache_file}", flush=True)
        snap = _load_processed_snapshot(
            matrix_path,
            info_path,
            convention=args.convention,
            apply_cutoff=bool(args.apply_cutoff),
            cutoff_radius=float(args.cutoff_radius),
            symmetrize_targets=bool(args.symmetrize_targets),
        )
        _save_cached_snapshot(snap, snapshot_cache_file)
    else:
        print(f"[CACHE] Snapshot hit -> {snapshot_cache_file}", flush=True)
    print(
        f"Snapshot loaded in {time.perf_counter() - t0:.2f}s; collecting pair data...",
        flush=True,
    )
    cached_pair_payload = _load_cached_pair_data(pair_cache_file)
    if cached_pair_payload is None:
        print(f"[CACHE] Pair payload miss -> {pair_cache_file}", flush=True)
        pair_data, order = _collect_pair_data(snap)
        _save_cached_pair_data(pair_data, order, pair_cache_file)
    else:
        print(f"[CACHE] Pair payload hit -> {pair_cache_file}", flush=True)
        pair_data, order = cached_pair_payload
    print(
        f"Collected pair data for {len(order)} unordered pairs; writing baseline scatter...",
        flush=True,
    )

    _plot_all_pairs_scatter(
        pair_data,
        order,
        output_dir / "hamiltonian_block_magnitude_vs_distance_all_pairs_data.png",
    )
    print("Baseline scatter written.", flush=True)

    fits_by_family: dict[str, dict[str, dict[str, object]]] = {}
    for family in tqdm(CURVE_FAMILIES, desc="Curve families"):
        print(f"Fitting family: {family.name}", flush=True)
        family_start = time.perf_counter()
        family_fits: dict[str, dict[str, object]] = {}
        for pair in tqdm(order, desc=f"Fitting {family.name}", leave=False):
            x = pair_data[pair]["distance"]
            y = pair_data[pair]["magnitude"]
            if x.size < int(args.min_points_per_pair):
                continue
            try:
                theta, metrics = _fit_family(family, x, y)
            except Exception as exc:
                print(f"[WARN] fit failed for family={family.name} pair={pair}: {exc}")
                continue
            family_fits[pair] = {"theta": theta.tolist(), "metrics": metrics}
        fits_by_family[family.name] = family_fits
        print(
            f"Finished fitting family {family.name} in {time.perf_counter() - family_start:.2f}s; writing plots...",
            flush=True,
        )
        _plot_family_all_pairs(
            family,
            pair_data,
            family_fits,
            order,
            output_dir / f"{family.name}_all_pairs_overlay.png",
        )
        _plot_family_per_pair(
            family,
            pair_data,
            family_fits,
            order,
            output_dir / f"{family.name}_per_pair_panels.png",
        )
        _plot_family_log_residuals(
            family,
            pair_data,
            family_fits,
            order,
            output_dir / f"{family.name}_per_pair_log_residuals.png",
        )
        print(f"Plots written for family: {family.name}", flush=True)

    _write_fit_summary(CURVE_FAMILIES, fits_by_family, order, output_dir)
    _write_markdown_summary(CURVE_FAMILIES, fits_by_family, order, output_dir)
    _write_selected_envelope_artifact(
        "slater_soft_cutoff",
        fits_by_family,
        output_dir,
    )
    print(f"Wrote fit study artifacts to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
