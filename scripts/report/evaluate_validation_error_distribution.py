#!/usr/bin/env python
"""Error distributions on the exact validation populations of the paper MAEs."""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
from contextlib import ExitStack
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO_ROOT), str(REPO_ROOT / "src")]

import torch
from tqdm.auto import tqdm

from net.checkpoint_compat import set_checkpoint_film_semantics
from net.operator_metrics import OperatorErrorDistribution
from scripts.report.evaluate_validation_relative_metrics import (
    E3GNN,
    HARTREE_TO_EV,
    _build_dataset_bundle,
    _evaluate_shared_step,
    _move_to_device,
    _patch_config_unpickling,
    _restore_config,
    build_parser,
    load_checkpoint_payload,
    strip_mapper_keys,
)

PAPER_MAES = {
    "ZnCuSnSeS": {"hamiltonian": 0.0030121},
    "siox": {"hamiltonian": 0.0019959},
    "silicon_scales": {
        "hamiltonian": 0.00043516,
        "density": 5.7577e-5,
        "overlap": 1.3898e-5,
    },
}
MAE_RELATIVE_TOLERANCE = 0.01


class DistributionModel(E3GNN):
    """Observe the physical matrices of the unchanged shared validation step."""

    def _apply_matrix_envelope_mode(self, *, name, pred_matrix, target_matrix, x):
        result = super()._apply_matrix_envelope_mode(
            name=name, pred_matrix=pred_matrix, target_matrix=target_matrix, x=x
        )
        # result[2] is the native-unit physical prediction BEFORE projection.
        # Keep the original reference, including its observed-support masks.
        self.validation_matrices[name] = result[2], target_matrix
        return result


def project_prediction(matrix, x):
    alignment = x.get("pred_reverse_alignment", x.get("pred_trace_alignment"))
    if alignment is None:
        raise ValueError("Transpose symmetrization requires reverse alignment")
    return matrix.add_aligned(matrix.transpose_aligned(alignment).conjugate()) * 0.5


def paper_metadata_path(directory):
    """The canonical-name-only discovery used for the original paper splits.

    Source: scripts/dataset.py at 79fb028. Incomplete calculation directories
    have log/slurm .out files but no canonical metadata; they must be skipped,
    not promoted into the paper population by the newer fallback discovery.
    """
    for name in ("Si.out", "SiO2.out", "ZnCuSeS.out", "info.dat", "info.txt"):
        candidate = Path(directory) / name
        if candidate.is_file():
            return candidate
    return None


def reference_targets(targets, density_convention):
    """Restore the documented paper density convention, without fitting a scale.

    e78a7c0's parser used D + D.transpose(), whereas the current native-spin0
    parser uses 0.5*(D + D.transpose()). The committed historical silicon
    physical-constraints contract explicitly declares the factor of two.
    """
    if density_convention == "paper_restricted_total" and "density" in targets:
        return dict(targets, density=targets["density"] * 2.0)
    return targets


def check_paper_maes(metrics, dataset_kind):
    expected = PAPER_MAES[dataset_kind]
    if set(metrics) != set(expected):
        raise RuntimeError(
            f"Operator set {sorted(metrics)} differs from paper {sorted(expected)}"
        )
    checks = {}
    for name, reference in expected.items():
        measured = metrics[name]["mae"]
        relative_difference = abs(measured - reference) / reference
        checks[name] = {
            "paper_mae": reference,
            "measured_mae": measured,
            "relative_difference": relative_difference,
            "passed": math.isfinite(measured)
            and relative_difference <= MAE_RELATIVE_TOLERANCE,
        }
    return {
        "relative_tolerance": MAE_RELATIVE_TOLERANCE,
        "passed": all(item["passed"] for item in checks.values()),
        "operators": checks,
    }


def parse_args():
    parser = build_parser()
    parser.description = __doc__
    parser.add_argument(
        "--film-semantics", choices=["o3", "legacy_2026_07"], required=True
    )
    parser.add_argument("--scratch-dir", type=Path)
    parser.add_argument(
        "--metadata-policy", choices=["current", "paper_2026_07"], default="current"
    )
    parser.add_argument("--envelope-path", type=Path)
    parser.add_argument(
        "--density-convention",
        choices=["native_spin0", "paper_restricted_total"],
        required=True,
    )
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="First validation graph only; no paper result or MAE gate",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", "4")))
    checkpoint_path = args.checkpoint.expanduser().resolve()
    _patch_config_unpickling()
    checkpoint = load_checkpoint_payload(checkpoint_path)
    cfg = deepcopy(_restore_config(checkpoint))
    cfg.snapshot_cache_dir = (
        None if args.snapshot_cache_dir is None
        else str(args.snapshot_cache_dir.expanduser().resolve())
    )
    cfg.dataset_device = args.dataset_device
    # Old checkpoints pickle torch.device; the current configuration resolver
    # requires its equivalent primitive spelling.
    cfg.device = str(args.device)
    cfg.data_split_seed = int(args.data_split_seed)
    cfg.verbosity = 0
    # Replay the entry-weighted H diagnostic used to generate the committed
    # paper JSONs (also explicit in publication_retrain_eval_79fb028).
    cfg.log_hamiltonian_pair_contrib_metrics = True
    if args.envelope_path is not None:
        cfg.hamiltonian_envelope_path = str(args.envelope_path.expanduser().resolve())
    if not (cfg.symmetrize_output or cfg.symmetrize_validation_output):
        raise RuntimeError("Paper validation must enable transpose symmetrization")

    with ExitStack() as loading:
        if args.metadata_policy == "paper_2026_07":
            loading.enter_context(patch(
                "scripts.dataset._resolve_single_snapshot_info_path", paper_metadata_path
            ))
        train_ds, val_ds, mapper = _build_dataset_bundle(args, cfg)
    if not len(val_ds):
        raise RuntimeError("The reconstructed validation split is empty")
    device = torch.device(args.device)
    model = DistributionModel(mapper=mapper, cfg=deepcopy(cfg)).to(device).eval()
    model.load_state_dict(strip_mapper_keys(checkpoint.get("state_dict", checkpoint)), strict=True)
    film_report = set_checkpoint_film_semantics(model, args.film_semantics)
    del checkpoint
    if args.preflight_only:
        # Exercise the actual first validation structure without publishing metrics.
        # This is a smoke check, never a substitute for the full paper-MAE gate.
        indices = range(1)
    else:
        indices = range(len(val_ds))
    if args.scratch_dir:
        args.scratch_dir.mkdir(parents=True, exist_ok=True)
    maes = {name: [] for name in cfg.matrix_targets}
    with ExitStack() as stack:
        distributions = {
            name: stack.enter_context(OperatorErrorDistribution(scratch_dir=args.scratch_dir))
            for name in cfg.matrix_targets
        }
        for index in tqdm(indices, desc="Validation structures"):
            x, y = val_ds[index]
            y = reference_targets(y, args.density_convention)
            x, y = _move_to_device(x, device), _move_to_device(y, device)
            model.validation_matrices = {}
            _, metrics = _evaluate_shared_step(model, (x, y), batch_idx=index)
            if set(model.validation_matrices) != set(distributions):
                raise RuntimeError("Validation step did not expose every paper operator")
            for name, (prediction, target) in model.validation_matrices.items():
                prediction = project_prediction(prediction, x)
                scale = HARTREE_TO_EV if name == "hamiltonian" else 1.0
                element_mae = distributions[name].update(
                    prediction, target, scale=scale, edge_selector=model._partial_train_mask
                )
                # Prove that the pooled errors use the trainer's projected population.
                logged = float(metrics[f"val/{name}_mae_elementwise_projected"])
                if not math.isclose(element_mae, logged, rel_tol=1e-6, abs_tol=1e-12):
                    raise RuntimeError(f"{name}: projected population differs from trainer")
                maes[name].append(float(metrics[f"val/{name}_mae"]))
            model.validation_matrices.clear()
        operator_metrics = {
            name: {
                "mae": sum(maes[name]) / len(maes[name]),
                **distribution.compute(),
                "unit": "eV" if name == "hamiltonian" else "dimensionless",
            }
            for name, distribution in distributions.items()
        }
    payload = {
        "schema_version": 1,
        "checkpoint": str(checkpoint_path),
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True, timeout=10
        ).strip(),
        "dataset_kind": args.dataset_kind,
        "data_path": str(args.data_path.expanduser().resolve()),
        "data_split_seed": args.data_split_seed,
        "metadata_policy": args.metadata_policy,
        "density_convention": args.density_convention,
        "density_reference_factor": (
            2.0 if args.density_convention == "paper_restricted_total" else 1.0
        ),
        "density_convention_provenance": (
            "e78a7c0:src/data/openmx_parser.py; "
            "docs/data/physical_constraints_20261104/silicon/historical_contract.json"
        ),
        "training_structures": len(train_ds),
        "validation_structures": len(val_ds),
        "evaluated_structures": len(indices),
        "film_semantics": args.film_semantics,
        "film_semantics_report": film_report,
        "definition": (
            "MAE is the unchanged val/<operator>_mae from the shared validation "
            "step, first per structure then averaged equally across structures; "
            "H uses the paper entry-weighted contribution diagnostic; historical "
            "within-structure S/D diagnostic reductions are preserved. "
            "RMSE, maximum, p95, p99 and R2 pool every observed shift-resolved "
            "sparse reference element across all validation structures, including "
            "both directed reverse entries and excluding padding/unobserved entries "
            "and prediction-only support. Predictions use physical envelope units "
            "and 0.5*(M + M.transpose().conjugate()) with reverse-edge alignment. "
            "H is converted from Hartree to eV; S and D retain native dimensionless "
            "units. For paper_restricted_total, the native spin=0 reference D is "
            "multiplied by two to restore the historical parser's documented "
            "spin-summed convention; this is a fixed provenance choice, not fitted "
            "rescaling. R2 = 1 - pooled SSE / sum(abs(reference - pooled_reference_mean)^2); "
            "R2 is null for a constant reference. mae_elementwise_pooled is an "
            "additional diagnostic, not the paper MAE. Percentiles are exact NumPy "
            "linear quantiles via in-place selection on a float64 temporary memory "
            "map; no histogram approximation or full validation tensor is used."
        ),
        "split_arguments": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "metrics": operator_metrics,
        "preflight_only": args.preflight_only,
    }
    if not args.preflight_only:
        payload["paper_mae_check"] = check_paper_maes(operator_metrics, args.dataset_kind)
        if not payload["paper_mae_check"]["passed"]:
            # Keep diagnostic evidence separate from accepted paper results.
            output = args.output_json.with_suffix(".failed-mae-check.json")
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
            raise RuntimeError(
                f"Paper MAE differs by more than 1%; stopped without tuning. See {output}"
            )
        output = args.output_json.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
        print(f"Wrote {output}", flush=True)
    print(json.dumps(payload, indent=2, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
