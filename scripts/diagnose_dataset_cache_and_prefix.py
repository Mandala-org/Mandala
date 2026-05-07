#!/usr/bin/env python

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

import torch
from e3nn.o3 import Irreps

from core.block_irrep_mapper import BlockIrrepMapper  # noqa: E402
from data.gnn_dataset import E3GNNDataset  # noqa: E402
from data.graph_features import compute_graph_features  # noqa: E402
from data.snapshot import Snapshot  # noqa: E402
from net.common import Config  # noqa: E402


def setup_argparse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare fresh snapshot loading, cached snapshot loading, and full "
            "E3GNNDataset construction for one OpenMX pair."
        )
    )
    parser.add_argument(
        "--snapshot-path",
        type=Path,
        default=Path("data/small/ZnCu2Sn_SeS_2_scale_1_010"),
        help="Snapshot directory, or a direct matrix file path.",
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
        "--snapshot-cache-dir",
        type=Path,
        default=None,
        help=(
            "Optional existing snapshot cache dir to inspect. If omitted, a temporary "
            "cache dir is created and populated."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("eval_outputs/dataset_cache_prefix_diag"),
        help="Directory for JSON reports.",
    )
    parser.add_argument(
        "--convention",
        type=str,
        default="e3nn",
        choices=["openmx", "e3nn"],
        help="Matrix convention used by the dataset loader.",
    )
    parser.add_argument(
        "--cutoff-radius",
        type=float,
        default=11.0,
        help="Cutoff used for graph construction.",
    )
    parser.add_argument(
        "--l-max",
        type=int,
        default=5,
        help="l_max used by the dataset graph build.",
    )
    parser.add_argument(
        "--n-radial",
        type=int,
        default=8,
        help="n_radial used by the dataset graph build.",
    )
    parser.add_argument(
        "--require-exact-edge-match",
        action="store_true",
        help="Enable the exact prefix check during dataset build.",
    )
    parser.add_argument(
        "--key",
        type=str,
        default="Zn-Se",
        help="Focused key for concise prefix summaries.",
    )
    parser.add_argument(
        "--preserve-temp-cache",
        action="store_true",
        help="Keep the temporary cache dir instead of deleting it.",
    )
    return parser.parse_args()


def _discover_snapshot_paths(
    snapshot_path: Path, matrix_path: Path | None, info_path: Path | None
) -> tuple[Path, Path]:
    if matrix_path is not None and info_path is not None:
        return matrix_path.resolve(), info_path.resolve()
    if matrix_path is not None or info_path is not None:
        raise ValueError("Provide both --matrix-path and --info-path together.")

    if snapshot_path.is_file():
        matrix_candidate = snapshot_path.resolve()
        info_candidate = _discover_info_file(snapshot_path.parent)
        if info_candidate is None:
            raise FileNotFoundError(
                f"Could not infer info file next to matrix file: {snapshot_path}"
            )
        return matrix_candidate, info_candidate.resolve()

    if not snapshot_path.is_dir():
        raise FileNotFoundError(f"Snapshot path does not exist: {snapshot_path}")

    matrix_candidate = (snapshot_path / "HS.out").resolve()
    if not matrix_candidate.exists():
        raise FileNotFoundError(f"Matrix file not found: {matrix_candidate}")

    info_candidate = _discover_info_file(snapshot_path)
    if info_candidate is None:
        raise FileNotFoundError(
            f"Could not infer info file under snapshot directory: {snapshot_path}"
        )
    return matrix_candidate, info_candidate.resolve()


def _discover_info_file(snapshot_dir: Path) -> Path | None:
    preferred = ["ZnCuSeS.out", "SiO2.out", "info.dat", "info.txt"]
    for name in preferred:
        candidate = snapshot_dir / name
        if candidate.exists():
            return candidate
    for candidate in sorted(snapshot_dir.glob("*.out")):
        if candidate.name not in {"HS.out", "log.out"}:
            return candidate
    fallback = snapshot_dir / "log.out"
    if fallback.exists():
        return fallback
    return None


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str))


def _key_prefix_summary(
    snapshot: Snapshot,
    mapper: BlockIrrepMapper,
    cfg: Config,
    *,
    key: str,
) -> dict[str, Any]:
    edge_index, edge_shift, _edge_type_idx, *_ = compute_graph_features(
        positions=snapshot.positions,
        box=snapshot.box,
        atoms=snapshot.density.atoms,
        cfg=cfg,
        sh_irreps=Irreps.spherical_harmonics(cfg.l_max),
        edge_type2idx=mapper.edge_type2idx,
    )
    graph_edges = []
    for e in range(edge_index.shape[1]):
        src = int(edge_index[0, e].item())
        dst = int(edge_index[1, e].item())
        edge_key = f"{snapshot.density.atoms[src]}-{snapshot.density.atoms[dst]}"
        if edge_key != key:
            continue
        graph_edges.append(
            (
                int(edge_shift[0, e].item()),
                int(edge_shift[1, e].item()),
                int(edge_shift[2, e].item()),
                src,
                dst,
            )
        )
    target_edges = [
        tuple(map(int, row))
        for row in snapshot.hamiltonian.pair_edges[key].t().tolist()
    ]
    prefix_len = min(len(target_edges), len(graph_edges))
    first_mismatch_idx = None
    for idx in range(prefix_len):
        if target_edges[idx] != graph_edges[idx]:
            first_mismatch_idx = idx
            break
    return {
        "key": key,
        "target_len": len(target_edges),
        "graph_len": len(graph_edges),
        "prefix_exact": first_mismatch_idx is None
        and len(graph_edges) >= len(target_edges),
        "first_mismatch_idx": first_mismatch_idx,
        "target_head": [list(edge) for edge in target_edges[:12]],
        "graph_head": [list(edge) for edge in graph_edges[:12]],
    }


def _pair_edges_equal(a: Snapshot, b: Snapshot) -> dict[str, Any]:
    keys_a = set(a.hamiltonian.pair_edges)
    keys_b = set(b.hamiltonian.pair_edges)
    if keys_a != keys_b:
        return {
            "same_key_set": False,
            "keys_only_in_a": sorted(keys_a - keys_b),
            "keys_only_in_b": sorted(keys_b - keys_a),
        }
    per_key = {}
    same_all = True
    for key in sorted(keys_a):
        eq = torch.equal(a.hamiltonian.pair_edges[key], b.hamiltonian.pair_edges[key])
        per_key[key] = eq
        same_all = same_all and eq
    return {"same_key_set": True, "all_equal": same_all, "per_key": per_key}


def _build_cfg(
    cache_dir: str | None, require_exact_edge_match: bool, args: argparse.Namespace
) -> Config:
    return Config(
        cutoff_radius=args.cutoff_radius,
        l_max=args.l_max,
        n_radial=args.n_radial,
        safety_checks=True,
        radial_embedding_scale="none",
        snapshot_cache_dir=cache_dir,
        require_exact_edge_match=require_exact_edge_match,
        precompute_edge_features=True,
        separate_shifted_self=True,
        apply_cutoff_to_targets=False,
        shuffle_snapshot_load_order=False,
        matrix_targets=["hamiltonian", "overlap", "density"],
        train_target="matrix",
        verbosity=0,
    )


def main() -> None:
    args = setup_argparse()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    matrix_path, info_path = _discover_snapshot_paths(
        args.snapshot_path, args.matrix_path, args.info_path
    )
    print(f"[INFO] matrix={matrix_path}", flush=True)
    print(f"[INFO] info={info_path}", flush=True)

    temp_cache_root: tempfile.TemporaryDirectory[str] | None = None
    if args.snapshot_cache_dir is None:
        temp_cache_root = tempfile.TemporaryDirectory(prefix="mandala-snap-cache-")
        cache_dir = Path(temp_cache_root.name)
    else:
        cache_dir = args.snapshot_cache_dir.resolve()
        cache_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] cache_dir={cache_dir}", flush=True)

    cfg_fresh = _build_cfg(None, args.require_exact_edge_match, args)
    fresh_snapshot = Snapshot.from_openmx(
        matrix_path,
        info_path,
        convention=args.convention,
        cutoff_radius=None,
        cfg=cfg_fresh,
    )
    mapper = BlockIrrepMapper(
        fresh_snapshot.hamiltonian.orbital_cfg,
        diagonal=False,
        device="cpu",
        dtype=cfg_fresh.dtype,
    )

    cfg_cache = _build_cfg(str(cache_dir), args.require_exact_edge_match, args)
    ds_probe = object.__new__(E3GNNDataset)
    ds_probe.cfg = cfg_cache
    ds_probe.convention = args.convention
    ds_probe.mapper = mapper
    ds_probe.dtype = cfg_cache.dtype
    snapshot_cache_file = E3GNNDataset._snapshot_cache_file(
        ds_probe, matrix_path, info_path
    )
    preprocessed_cache_file = E3GNNDataset._preprocessed_sample_cache_file(
        ds_probe, matrix_path, info_path
    )

    if snapshot_cache_file is not None:
        print(f"[INFO] snapshot_cache_file={snapshot_cache_file}", flush=True)
    if preprocessed_cache_file is not None:
        print(f"[INFO] preprocessed_cache_file={preprocessed_cache_file}", flush=True)

    # First cached dataset build: populate cache or use existing.
    ds_first_status = {"ok": True, "error": None}
    try:
        E3GNNDataset(
            [(matrix_path, info_path)],
            mapper,
            cfg_cache,
            args.convention,
        )
    except Exception as exc:
        ds_first_status = {"ok": False, "error": repr(exc)}

    cache_file_exists = snapshot_cache_file is not None and snapshot_cache_file.exists()
    cached_snapshot = (
        Snapshot.load(snapshot_cache_file, device="cpu")
        if cache_file_exists and snapshot_cache_file is not None
        else None
    )

    ds_second_status = {"ok": True, "error": None}
    try:
        ds_second = E3GNNDataset(
            [(matrix_path, info_path)],
            mapper,
            cfg_cache,
            args.convention,
        )
        dataset_sample = ds_second[0]
        dataset_summary = {
            "x_edge_count": int(dataset_sample[0]["edge_index"].shape[1]),
            "target_key_count": len(dataset_sample[1]["hamiltonian"].pair_edges),
        }
    except Exception as exc:
        ds_second_status = {"ok": False, "error": repr(exc)}
        dataset_summary = None

    payload: dict[str, Any] = {
        "matrix_path": matrix_path,
        "info_path": info_path,
        "cache_dir": cache_dir,
        "snapshot_cache_file": snapshot_cache_file,
        "preprocessed_cache_file": preprocessed_cache_file,
        "snapshot_cache_file_exists_after_first_build": cache_file_exists,
        "fresh_prefix_summary": _key_prefix_summary(
            fresh_snapshot, mapper, cfg_fresh, key=args.key
        ),
        "cached_prefix_summary": (
            _key_prefix_summary(cached_snapshot, mapper, cfg_fresh, key=args.key)
            if cached_snapshot is not None
            else None
        ),
        "fresh_vs_cached_pair_edges": (
            _pair_edges_equal(fresh_snapshot, cached_snapshot)
            if cached_snapshot is not None
            else None
        ),
        "dataset_first_build": ds_first_status,
        "dataset_second_build": ds_second_status,
        "dataset_sample_summary": dataset_summary,
    }
    _write_json(output_dir / "dataset_cache_prefix_report.json", payload)

    print("", flush=True)
    print("[SUMMARY] fresh snapshot", flush=True)
    fresh_summary = payload["fresh_prefix_summary"]
    print(
        f"  key={fresh_summary['key']} prefix_exact={fresh_summary['prefix_exact']} "
        f"first_mismatch_idx={fresh_summary['first_mismatch_idx']} "
        f"target_len={fresh_summary['target_len']} graph_len={fresh_summary['graph_len']}",
        flush=True,
    )
    if payload["cached_prefix_summary"] is not None:
        cached_summary = payload["cached_prefix_summary"]
        print("[SUMMARY] cached snapshot", flush=True)
        print(
            f"  key={cached_summary['key']} prefix_exact={cached_summary['prefix_exact']} "
            f"first_mismatch_idx={cached_summary['first_mismatch_idx']} "
            f"target_len={cached_summary['target_len']} graph_len={cached_summary['graph_len']}",
            flush=True,
        )
    print("[SUMMARY] dataset first build", payload["dataset_first_build"], flush=True)
    print("[SUMMARY] dataset second build", payload["dataset_second_build"], flush=True)
    print(f"[DONE] wrote report to {output_dir}", flush=True)

    if temp_cache_root is not None and args.preserve_temp_cache:
        print(f"[INFO] preserved temp cache at {cache_dir}", flush=True)
    if temp_cache_root is not None and not args.preserve_temp_cache:
        temp_cache_root.cleanup()


if __name__ == "__main__":
    main()
