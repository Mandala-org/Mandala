from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import torch
from e3nn.o3 import Irreps

from net.common import Config
from data.graph_features import compute_graph_features

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from _common import (  # noqa: E402
    build_minimal_graph_inputs,
    edge_tuples,
    load_water_snapshot,
    metric_stats,
    set_seed,
)


def _to_builtin(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _to_builtin(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_builtin(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, torch.Tensor):
        if obj.numel() == 1:
            return obj.item()
        return obj.detach().cpu().tolist()
    return obj


def _dump_report(report: Dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import yaml  # type: ignore

        with output_path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(_to_builtin(report), f, sort_keys=False)
    except Exception:
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(_to_builtin(report), f, indent=2)


def _mismatch_examples(
    left: List[tuple[int, int, int, int, int]],
    right: List[tuple[int, int, int, int, int]],
    limit: int = 16,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    max_n = max(len(left), len(right))
    for i in range(max_n):
        l = left[i] if i < len(left) else None
        r = right[i] if i < len(right) else None
        if l != r:
            out.append({"idx": i, "minimal": l, "src": r})
        if len(out) >= limit:
            break
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Priority check #1: SH/radial/edge parity between minimal and src graph feature paths."
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path("data/small/H2O/original/H2O.matrix"),
    )
    parser.add_argument(
        "--info-path",
        type=Path,
        default=Path("data/small/H2O/original/H2O.info.out"),
    )
    parser.add_argument("--cutoff-radius", type=float, default=7.0)
    parser.add_argument("--n-radial", type=int, default=32)
    parser.add_argument("--l-max", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tol", type=float, default=1e-6)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("studies/deeph-e3-comparison-study/reports/check1_sh_parity.yaml"),
    )
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cpu")

    snapshot, atoms_list, positions, box, _h, _s, _d, mapper = load_water_snapshot(
        data_path=args.data_path,
        info_path=args.info_path,
        device=device,
        convention="e3nn",
        cutoff_radius=args.cutoff_radius,
    )

    minimal_graph = build_minimal_graph_inputs(
        atoms_list=atoms_list,
        positions=positions,
        box=box,
        mapper=mapper,
        cutoff_radius=args.cutoff_radius,
        n_radial=args.n_radial,
        l_max=args.l_max,
        device=device,
    )

    src_cfg = Config(
        cutoff_radius=args.cutoff_radius,
        n_radial=args.n_radial,
        safety_checks=False,
    )
    sh_irreps = Irreps.spherical_harmonics(args.l_max)
    (
        src_edge_index,
        src_edge_shift,
        src_edge_type_idx,
        src_edge_length_emb,
        src_edge_sh,
        src_num_self,
    ) = compute_graph_features(
        positions=positions,
        box=box,
        atoms=tuple(atoms_list),
        cfg=src_cfg,
        sh_irreps=sh_irreps,
        edge_type2idx=mapper.edge_type2idx,
    )

    minimal_edge_tuples = edge_tuples(
        minimal_graph.edge_index, minimal_graph.edge_shift
    )
    src_edge_tuples = edge_tuples(src_edge_index, src_edge_shift)
    edges_exact_match = minimal_edge_tuples == src_edge_tuples

    if not edges_exact_match:
        first_mismatch = _mismatch_examples(
            minimal_edge_tuples, src_edge_tuples, limit=20
        )
    else:
        first_mismatch = []

    # Minimal path scales radial embedding by sqrt(n_radial); src path does not.
    radial_minimal_scaled = minimal_graph.edge_length_emb
    radial_minimal_unscaled = minimal_graph.edge_length_emb / (args.n_radial**0.5)

    if edges_exact_match:
        sh_current_vs_src = metric_stats(minimal_graph.edge_sh_current, src_edge_sh)
        sh_aligned_vs_src = metric_stats(minimal_graph.edge_sh_aligned, src_edge_sh)
        radial_scaled_vs_src = metric_stats(radial_minimal_scaled, src_edge_length_emb)
        radial_unscaled_vs_src = metric_stats(
            radial_minimal_unscaled, src_edge_length_emb
        )
        edge_type_idx_match = bool(
            torch.equal(minimal_graph.edge_type_idx, src_edge_type_idx)
        )
        edge_dist_vs_src = metric_stats(
            minimal_graph.edge_dist,
            torch.linalg.norm(
                positions[src_edge_index[1]]
                - positions[src_edge_index[0]]
                + (
                    src_edge_shift.T.to(positions.dtype) @ box
                    if box is not None
                    else 0.0
                ),
                dim=1,
            ),
        )
    else:
        sh_current_vs_src = None
        sh_aligned_vs_src = None
        radial_scaled_vs_src = None
        radial_unscaled_vs_src = None
        edge_type_idx_match = False
        edge_dist_vs_src = None

    pass_aligned_sh = bool(
        sh_aligned_vs_src is not None and sh_aligned_vs_src["max_abs"] <= args.tol
    )
    pass_unscaled_radial = bool(
        radial_unscaled_vs_src is not None
        and radial_unscaled_vs_src["max_abs"] <= args.tol
    )
    current_sh_large_delta = bool(
        sh_current_vs_src is not None and sh_current_vs_src["mean_abs"] > 1e-3
    )

    report: Dict[str, Any] = {
        "check": "priority_1_sh_parity",
        "inputs": {
            "data_path": args.data_path,
            "info_path": args.info_path,
            "cutoff_radius": args.cutoff_radius,
            "n_radial": args.n_radial,
            "l_max": args.l_max,
            "seed": args.seed,
            "tolerance": args.tol,
        },
        "shape_summary": {
            "num_atoms": len(atoms_list),
            "num_edges_minimal": len(minimal_edge_tuples),
            "num_edges_src": len(src_edge_tuples),
            "num_self_edges_src": int(src_num_self),
            "edge_sh_shape_minimal": list(minimal_graph.edge_sh_current.shape),
            "edge_sh_shape_src": list(src_edge_sh.shape),
            "edge_length_emb_shape_minimal": list(minimal_graph.edge_length_emb.shape),
            "edge_length_emb_shape_src": list(src_edge_length_emb.shape),
        },
        "edge_set_parity": {
            "exact_match": bool(edges_exact_match),
            "edge_type_idx_match": bool(edge_type_idx_match),
            "first_mismatches": first_mismatch,
        },
        "feature_parity": {
            "sh_current_vs_src": sh_current_vs_src,
            "sh_aligned_vs_src": sh_aligned_vs_src,
            "radial_scaled_vs_src": radial_scaled_vs_src,
            "radial_unscaled_vs_src": radial_unscaled_vs_src,
            "edge_distance_stats": edge_dist_vs_src,
        },
        "conclusion": {
            "pass_aligned_sh_parity": pass_aligned_sh,
            "pass_unscaled_radial_parity": pass_unscaled_radial,
            "current_sh_is_different_from_src": current_sh_large_delta,
            "overall_pass": bool(
                edges_exact_match and pass_aligned_sh and pass_unscaled_radial
            ),
            "summary": (
                "Minimal aligned SH path and unscaled radial embedding match src path."
                if (edges_exact_match and pass_aligned_sh and pass_unscaled_radial)
                else "Detected a mismatch in edge ordering and/or feature conventions."
            ),
        },
    }

    _dump_report(report, args.output)
    print(f"[check1] report saved to: {args.output}")
    print(f"[check1] edge exact match: {edges_exact_match}")
    if sh_current_vs_src is not None and sh_aligned_vs_src is not None:
        print(
            "[check1] mean_abs SH: "
            f"current={sh_current_vs_src['mean_abs']:.6e}, "
            f"aligned={sh_aligned_vs_src['mean_abs']:.6e}"
        )
        print(
            "[check1] max_abs radial: "
            f"scaled={radial_scaled_vs_src['max_abs']:.6e}, "
            f"unscaled={radial_unscaled_vs_src['max_abs']:.6e}"
        )


if __name__ == "__main__":
    main()
