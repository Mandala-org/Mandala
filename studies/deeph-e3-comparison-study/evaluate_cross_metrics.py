#!/usr/bin/env python3
"""
Strict cross-evaluation of DeepH-E3 vs external "my model" predictions.

The script intentionally has no fallback behavior:
- missing files cause hard errors
- incompatible model/data atomic species cause hard errors
- key-set/shape mismatches cause hard errors
- DeepH eval runtime errors cause hard errors
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np


MEV_PER_EV = 1000.0
EV_PER_HARTREE = 27.2113845


@dataclass
class ScalarMetrics:
    mae_ev: float
    mse_ev2: float
    mae_mev: float
    mse_mev2: float
    n_elements: int


def unit_to_ev_scale(unit: str) -> float:
    unit_norm = unit.strip().lower()
    if unit_norm == "ev":
        return 1.0
    if unit_norm == "hartree":
        return EV_PER_HARTREE
    raise ValueError(f"Unsupported unit '{unit}'. Expected one of: ev, hartree")


def _to_builtin(x: Any) -> Any:
    if isinstance(x, np.floating):
        return float(x)
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, dict):
        return {str(k): _to_builtin(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_to_builtin(v) for v in x]
    return x


def require_exists(path: Path, kind: str = "path") -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing required {kind}: {path}")


def parse_element_dat(path: Path) -> list[int]:
    require_exists(path, "element.dat")
    vals: list[int] = []
    with path.open("r") as f:
        for line in f:
            line = line.strip()
            if line:
                vals.append(int(line))
    if not vals:
        raise ValueError(f"element.dat has no values: {path}")
    return sorted(set(vals))


def load_deeph_dataset_info(result_dir: Path) -> dict[str, Any]:
    p = result_dir / "src" / "dataset_info.json"
    require_exists(p, "DeepH dataset_info.json")
    with p.open("r") as f:
        return json.load(f)


def build_eval_ini(
    result_dir: Path,
    processed_data_root: Path,
    output_dir: Path,
    structure_dataset_name: str,
) -> str:
    return (
        textwrap.dedent(
            f"""
        [basic]
        device = cpu
        dtype = float
        trained_model_dir = {result_dir.resolve()}
        output_dir = {output_dir.resolve()}
        target = hamiltonian
        inference = False
        test_only = False

        [data]
        graph_dir =
        DFT_data_dir =
        processed_data_dir = {processed_data_root.resolve()}
        save_graph_dir = {(output_dir / "graph").resolve()}
        target_data = hamiltonian
        dataset_name = {structure_dataset_name}
        get_overlap = True
        """
        ).strip()
        + "\n"
    )


def run_deeph_eval_strict(
    deeph_repo_dir: Path,
    result_dir: Path,
    processed_data_root: Path,
    output_dir: Path,
    structure_id: str,
) -> tuple[Path, Path]:
    require_exists(deeph_repo_dir, "DeepH repo directory")
    require_exists(result_dir, "DeepH result directory")
    require_exists(processed_data_root, "processed_data_dir root")

    output_dir.mkdir(parents=True, exist_ok=True)
    ini_path = output_dir / "eval_generated.ini"
    ini_path.write_text(
        build_eval_ini(
            result_dir=result_dir,
            processed_data_root=processed_data_root,
            output_dir=output_dir,
            structure_dataset_name=f"cross_eval_{structure_id}",
        )
    )

    cmd = [sys.executable, "deephe3-eval.py", str(ini_path.resolve())]
    proc = subprocess.run(
        cmd,
        cwd=str(deeph_repo_dir.resolve()),
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            "DeepH eval failed.\n"
            f"Command: {' '.join(cmd)}\n"
            f"Return code: {proc.returncode}\n"
            f"Stdout tail:\n{proc.stdout[-4000:]}\n"
            f"Stderr tail:\n{proc.stderr[-4000:]}\n"
        )

    pred_h5 = output_dir / structure_id / "hamiltonians_pred.h5"
    test_result_h5 = output_dir / "test_result.h5"
    require_exists(pred_h5, "DeepH-generated hamiltonians_pred.h5")
    require_exists(test_result_h5, "DeepH-generated test_result.h5")
    return pred_h5, test_result_h5


def scalar_metrics_from_diff_ev(diff_ev: np.ndarray) -> ScalarMetrics:
    n = int(diff_ev.size)
    if n <= 0:
        raise ValueError("No elements available for metrics.")
    mae_ev = float(np.mean(np.abs(diff_ev)))
    mse_ev2 = float(np.mean(diff_ev * diff_ev))
    return ScalarMetrics(
        mae_ev=mae_ev,
        mse_ev2=mse_ev2,
        mae_mev=mae_ev * MEV_PER_EV,
        mse_mev2=mse_ev2 * (MEV_PER_EV**2),
        n_elements=n,
    )


def metrics_from_test_result_h5_strict(
    test_result_h5: Path, data_unit: str
) -> dict[str, Any]:
    require_exists(test_result_h5, "test_result.h5")
    scale_ev = unit_to_ev_scale(data_unit)
    with h5py.File(test_result_h5, "r") as h5:
        structure_ids = sorted(h5.keys())
        if not structure_ids:
            raise ValueError(f"test_result.h5 has no structures: {test_result_h5}")

        abs_sum_global = 0.0
        sq_sum_global = 0.0
        n_global = 0
        mae_edge_weighted_sum = 0.0
        mse_edge_weighted_sum = 0.0
        edge_weight_total = 0.0
        mae_struct_sum = 0.0
        mse_struct_sum = 0.0
        per_structure: list[dict[str, Any]] = []

        for sid in structure_ids:
            g = h5[sid]
            for name in ("H_pred", "label", "mask"):
                if name not in g:
                    raise KeyError(f"{test_result_h5}:{sid} missing dataset '{name}'")
            pred = np.array(g["H_pred"])
            label = np.array(g["label"])
            mask = np.array(g["mask"]).astype(bool)
            if pred.shape != label.shape or pred.shape != mask.shape:
                raise ValueError(
                    f"Shape mismatch in {sid}: "
                    f"H_pred={pred.shape}, label={label.shape}, mask={mask.shape}"
                )
            if not (np.isfinite(pred).all() and np.isfinite(label).all()):
                raise ValueError(f"Non-finite values in {sid} test_result tensors.")

            diff_sel = (pred - label)[mask] * scale_ev
            n_sel = int(diff_sel.size)
            if n_sel <= 0:
                raise ValueError(f"Mask selected zero elements for structure {sid}")
            mae_s = float(np.mean(np.abs(diff_sel)))
            mse_s = float(np.mean(diff_sel * diff_sel))
            edges = int(pred.shape[0])

            per_structure.append(
                {
                    "structure_id": sid,
                    "n_selected": n_sel,
                    "n_edges": edges,
                    "mask_true_ratio": float(mask.mean()),
                    "mae_ev": mae_s,
                    "mse_ev2": mse_s,
                    "mae_mev": mae_s * MEV_PER_EV,
                    "mse_mev2": mse_s * (MEV_PER_EV**2),
                }
            )

            abs_sum_global += float(np.sum(np.abs(diff_sel)))
            sq_sum_global += float(np.sum(diff_sel * diff_sel))
            n_global += n_sel
            mae_edge_weighted_sum += mae_s * edges
            mse_edge_weighted_sum += mse_s * edges
            edge_weight_total += edges
            mae_struct_sum += mae_s
            mse_struct_sum += mse_s

    if n_global <= 0:
        raise ValueError("No selected elements in test_result global masked metric.")

    mae_global = abs_sum_global / n_global
    mse_global = sq_sum_global / n_global
    mae_edge_w = mae_edge_weighted_sum / edge_weight_total
    mse_edge_w = mse_edge_weighted_sum / edge_weight_total
    n_struct = len(per_structure)
    mae_struct = mae_struct_sum / n_struct
    mse_struct = mse_struct_sum / n_struct

    return {
        "n_structures": n_struct,
        "global_masked": {
            "mae_ev": mae_global,
            "mse_ev2": mse_global,
            "mae_mev": mae_global * MEV_PER_EV,
            "mse_mev2": mse_global * (MEV_PER_EV**2),
            "n_elements": n_global,
        },
        "edge_weighted_structure_mean": {
            "mae_ev": mae_edge_w,
            "mse_ev2": mse_edge_w,
            "mae_mev": mae_edge_w * MEV_PER_EV,
            "mse_mev2": mse_edge_w * (MEV_PER_EV**2),
        },
        "equal_structure_mean": {
            "mae_ev": mae_struct,
            "mse_ev2": mse_struct,
            "mae_mev": mae_struct * MEV_PER_EV,
            "mse_mev2": mse_struct * (MEV_PER_EV**2),
        },
        "per_structure": per_structure,
    }


def load_block_h5(path: Path) -> dict[str, np.ndarray]:
    require_exists(path, "block h5")
    out: dict[str, np.ndarray] = {}
    with h5py.File(path, "r") as h5:
        for k in h5.keys():
            out[k] = np.array(h5[k])
    if not out:
        raise ValueError(f"Empty block h5: {path}")
    return out


def metrics_from_block_h5_strict(
    pred_h5: Path,
    gt_h5: Path,
    pred_unit: str,
    gt_unit: str,
) -> dict[str, Any]:
    pred = load_block_h5(pred_h5)
    gt = load_block_h5(gt_h5)
    pred_scale_ev = unit_to_ev_scale(pred_unit)
    gt_scale_ev = unit_to_ev_scale(gt_unit)

    pred_keys = set(pred.keys())
    gt_keys = set(gt.keys())
    if pred_keys != gt_keys:
        missing_in_pred = sorted(gt_keys - pred_keys)
        extra_in_pred = sorted(pred_keys - gt_keys)
        raise ValueError(
            "Block key set mismatch.\n"
            f"  pred={pred_h5}\n"
            f"  gt={gt_h5}\n"
            f"  missing_in_pred(first20)={missing_in_pred[:20]}\n"
            f"  extra_in_pred(first20)={extra_in_pred[:20]}"
        )

    diffs: list[np.ndarray] = []
    for k in sorted(gt_keys):
        p = pred[k]
        g = gt[k]
        if p.shape != g.shape:
            raise ValueError(
                f"Block shape mismatch for key {k}: pred={p.shape}, gt={g.shape}"
            )
        if not (np.isfinite(p).all() and np.isfinite(g).all()):
            raise ValueError(f"Non-finite values found in block key {k}")
        p_ev = p * pred_scale_ev
        g_ev = g * gt_scale_ev
        diffs.append((p_ev - g_ev).reshape(-1))

    diff = np.concatenate(diffs, axis=0)
    scalar = scalar_metrics_from_diff_ev(diff)
    return {
        "scalar": {
            "mae_ev": scalar.mae_ev,
            "mse_ev2": scalar.mse_ev2,
            "mae_mev": scalar.mae_mev,
            "mse_mev2": scalar.mse_mev2,
            "n_elements": scalar.n_elements,
        }
    }


def write_report_md(path: Path, report: dict[str, Any]) -> None:
    compatibility = report["compatibility_check"]
    units = report["inputs"]["units"]
    metrics = report["metrics"]
    lines = [
        "# Cross Evaluation Report (Strict)",
        "",
        "## Compatibility",
        f"- model_index_to_Z: `{compatibility['model_index_to_Z']}`",
        f"- data_unique_element_Z: `{compatibility['data_unique_element_Z']}`",
        f"- compatible_exact: `{compatibility['compatible_exact']}`",
        "",
        "## Units",
        f"- gt_unit: `{units['gt_unit']}`",
        f"- deeph_pred_unit: `{units['deeph_pred_unit']}`",
        f"- deeph_test_result_unit: `{units['deeph_test_result_unit']}`",
        f"- my_pred_unit: `{units['my_pred_unit']}`",
        "",
        "## Metrics (meV / meV^2)",
        "",
    ]

    order = [
        "their_model_in_their_way",
        "their_model_in_my_way",
        "my_model_in_their_way",
        "my_model_in_my_way",
    ]
    for name in order:
        entry = metrics[name]
        scalar = entry["scalar"] if "scalar" in entry else entry["global_masked"]
        lines.append(f"### {name}")
        lines.append(f"- MAE: `{scalar['mae_mev']}` meV")
        lines.append(f"- MSE: `{scalar['mse_mev2']}` meV^2")
        lines.append(f"- N elements: `{scalar['n_elements']}`")
        lines.append("")

    path.write_text("\n".join(lines))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Strict DeepH-vs-mine cross evaluation"
    )
    parser.add_argument(
        "--deeph-result-dir",
        type=Path,
        default=Path("external/DeepH-E3/results/2025-12-08_17-18-39_silicon"),
    )
    parser.add_argument(
        "--deeph-repo-dir",
        type=Path,
        default=Path("external/DeepH-E3"),
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("external/DeepH-E3/data"),
    )
    parser.add_argument(
        "--structure-id",
        type=str,
        default="00",
    )
    parser.add_argument(
        "--my-pred-h5",
        type=Path,
        required=True,
        help="Path to your model predictions in full block-h5 format for the same structure.",
    )
    parser.add_argument(
        "--gt-unit",
        type=str,
        choices=("ev", "hartree"),
        default="ev",
        help="Unit of values in ground-truth hamiltonians.h5.",
    )
    parser.add_argument(
        "--deeph-pred-unit",
        type=str,
        choices=("ev", "hartree"),
        default="ev",
        help="Unit of values in DeepH hamiltonians_pred.h5.",
    )
    parser.add_argument(
        "--deeph-test-result-unit",
        type=str,
        choices=("ev", "hartree"),
        default="ev",
        help="Unit of values in DeepH test_result.h5 tensors.",
    )
    parser.add_argument(
        "--my-pred-unit",
        type=str,
        choices=("ev", "hartree"),
        default="hartree",
        help="Unit of values in --my-pred-h5.",
    )
    parser.add_argument(
        "--skip-deeph-eval",
        action="store_true",
        default=False,
        help="Do not run DeepH eval; require explicit --deeph-pred-h5 and --deeph-test-result-h5.",
    )
    parser.add_argument(
        "--deeph-pred-h5",
        type=Path,
        default=None,
        help="Required when --skip-deeph-eval is set.",
    )
    parser.add_argument(
        "--deeph-test-result-h5",
        type=Path,
        default=None,
        help="Required when --skip-deeph-eval is set.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "studies/deeph-e3-comparison-study/reports/cross_eval_2025-12-08_17-18-39_silicon_00"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    structure_dir = args.data_root / args.structure_id
    gt_h5 = structure_dir / "hamiltonians.h5"
    element_dat = structure_dir / "element.dat"
    require_exists(structure_dir, "structure directory")
    require_exists(gt_h5, "ground-truth hamiltonians.h5")
    require_exists(element_dat, "element.dat")
    require_exists(args.my_pred_h5, "my model prediction h5")

    model_info = load_deeph_dataset_info(args.deeph_result_dir)
    data_elements = parse_element_dat(element_dat)
    model_elements = sorted(int(z) for z in model_info.get("index_to_Z", []))
    compatibility = {
        "model_index_to_Z": model_elements,
        "data_unique_element_Z": data_elements,
        "compatible_exact": model_elements == data_elements,
    }
    if not compatibility["compatible_exact"]:
        raise ValueError(
            "DeepH model atomic species do not match provided data.\n"
            f"  model index_to_Z={model_elements}\n"
            f"  data unique Z={data_elements}"
        )

    if args.skip_deeph_eval:
        if args.deeph_pred_h5 is None or args.deeph_test_result_h5 is None:
            raise ValueError(
                "--skip-deeph-eval requires both --deeph-pred-h5 and --deeph-test-result-h5"
            )
        deeph_pred_h5 = args.deeph_pred_h5
        deeph_test_result_h5 = args.deeph_test_result_h5
        require_exists(deeph_pred_h5, "DeepH prediction h5")
        require_exists(deeph_test_result_h5, "DeepH test_result.h5")
        deeph_eval_run = {
            "attempted": False,
            "pred_h5": str(deeph_pred_h5),
            "test_result_h5": str(deeph_test_result_h5),
        }
    else:
        deeph_eval_output_dir = args.output_dir / "deeph_eval_run"
        deeph_pred_h5, deeph_test_result_h5 = run_deeph_eval_strict(
            deeph_repo_dir=args.deeph_repo_dir,
            result_dir=args.deeph_result_dir,
            processed_data_root=args.data_root,
            output_dir=deeph_eval_output_dir,
            structure_id=args.structure_id,
        )
        deeph_eval_run = {
            "attempted": True,
            "pred_h5": str(deeph_pred_h5),
            "test_result_h5": str(deeph_test_result_h5),
            "eval_ini": str(deeph_eval_output_dir / "eval_generated.ini"),
        }

    their_in_their = metrics_from_test_result_h5_strict(
        deeph_test_result_h5,
        data_unit=args.deeph_test_result_unit,
    )
    their_in_my = metrics_from_block_h5_strict(
        deeph_pred_h5,
        gt_h5,
        pred_unit=args.deeph_pred_unit,
        gt_unit=args.gt_unit,
    )
    my_in_their = metrics_from_block_h5_strict(
        args.my_pred_h5,
        gt_h5,
        pred_unit=args.my_pred_unit,
        gt_unit=args.gt_unit,
    )
    my_in_my = metrics_from_block_h5_strict(
        args.my_pred_h5,
        gt_h5,
        pred_unit=args.my_pred_unit,
        gt_unit=args.gt_unit,
    )

    report = {
        "inputs": {
            "deeph_result_dir": str(args.deeph_result_dir),
            "deeph_repo_dir": str(args.deeph_repo_dir),
            "data_root": str(args.data_root),
            "structure_id": args.structure_id,
            "gt_h5": str(gt_h5),
            "my_pred_h5": str(args.my_pred_h5),
            "units": {
                "gt_unit": args.gt_unit,
                "deeph_pred_unit": args.deeph_pred_unit,
                "deeph_test_result_unit": args.deeph_test_result_unit,
                "my_pred_unit": args.my_pred_unit,
                "output_metrics": "meV / meV^2 (internally computed in eV)",
            },
        },
        "compatibility_check": compatibility,
        "deeph_eval_run": deeph_eval_run,
        "metrics": {
            "their_model_in_their_way": their_in_their,
            "their_model_in_my_way": their_in_my,
            "my_model_in_their_way": {
                **my_in_their,
                "note": "On block-h5 data this is computed as strict global element-wise MAE/MSE.",
            },
            "my_model_in_my_way": my_in_my,
        },
        "notes": {
            "unit_assumption": (
                "All inputs are converted to eV using explicit per-input unit flags "
                "before computing errors."
            ),
            "hartree_to_ev": EV_PER_HARTREE,
            "strict_mode": "No fallbacks; all missing/mismatched inputs raise errors.",
        },
    }

    json_path = args.output_dir / "cross_eval_report.json"
    md_path = args.output_dir / "cross_eval_report.md"
    json_path.write_text(json.dumps(_to_builtin(report), indent=2))
    write_report_md(md_path, report)

    print(f"Wrote: {json_path}")
    print(f"Wrote: {md_path}")
    print(f"DeepH pred source: {deeph_eval_run['pred_h5']}")
    print(f"DeepH test_result source: {deeph_eval_run['test_result_h5']}")


if __name__ == "__main__":
    main()
