from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import torch
import torch.nn.functional as F

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from _common import (  # noqa: E402
    build_minimal_graph_inputs,
    flatten_gradients,
    forward_to_blocks,
    load_water_snapshot,
    loss_global_element_mse,
    loss_per_key_sum_mse,
    make_network,
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


def _per_key_stats(pred_matrix, target_matrix) -> Dict[str, Any]:
    rows: Dict[str, Any] = {}
    total_numel = 0
    total_sum_sq = 0.0

    for key in target_matrix.pair_blocks.keys():
        if key not in pred_matrix.pair_blocks:
            continue
        pred = pred_matrix.pair_blocks[key]
        targ = target_matrix.pair_blocks[key]
        min_n = min(pred.shape[0], targ.shape[0])
        if min_n == 0:
            continue
        diff = pred[:min_n] - targ[:min_n]
        numel = int(diff.numel())
        sum_sq = float((diff**2).sum().item())
        mse = float(F.mse_loss(pred[:min_n], targ[:min_n]).item())
        rows[key] = {
            "num_edges": int(min_n),
            "numel": numel,
            "mse": mse,
            "sum_sq": sum_sq,
        }
        total_numel += numel
        total_sum_sq += sum_sq

    for key, row in rows.items():
        row["global_weight"] = (
            float(row["numel"] / total_numel) if total_numel > 0 else 0.0
        )
        row["per_key_weight"] = 1.0
        row["global_contribution"] = float(row["global_weight"] * row["mse"])
        row["per_key_contribution"] = float(row["mse"])

    return {
        "keys": rows,
        "total_numel": total_numel,
        "total_sum_sq": total_sum_sq,
    }


def _build_network_from_state(
    state_dict: Dict[str, torch.Tensor],
    *,
    mapper,
    n_radial: int,
    l_max: int,
    hidden_dim: int,
    num_layers: int,
    device: torch.device,
):
    net = make_network(
        mapper=mapper,
        n_radial=n_radial,
        l_max=l_max,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        device=device,
    )
    net.load_state_dict(state_dict, strict=True)
    return net


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Priority check #2: loss-definition parity and gradient comparison on fixed predictions."
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
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--use-aligned-sh", action="store_true", default=False)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "studies/deeph-e3-comparison-study/reports/check2_loss_parity.yaml"
        ),
    )
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cpu")

    (
        snapshot,
        atoms_list,
        positions,
        box,
        target_h,
        _s,
        _d,
        mapper,
    ) = load_water_snapshot(
        data_path=args.data_path,
        info_path=args.info_path,
        device=device,
        convention="e3nn",
        cutoff_radius=args.cutoff_radius,
    )

    graph = build_minimal_graph_inputs(
        atoms_list=atoms_list,
        positions=positions,
        box=box,
        mapper=mapper,
        cutoff_radius=args.cutoff_radius,
        n_radial=args.n_radial,
        l_max=args.l_max,
        device=device,
    )

    base_net = make_network(
        mapper=mapper,
        n_radial=args.n_radial,
        l_max=args.l_max,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        device=device,
    )
    base_state = copy.deepcopy(base_net.state_dict())

    # Fixed predictions for scalar objective comparison.
    with torch.no_grad():
        _, pred_blocks_fixed = forward_to_blocks(
            network=base_net,
            graph=graph,
            atoms_list=atoms_list,
            mapper=mapper,
            basis=snapshot.hamiltonian.basis,
            orbital_cfg=snapshot.hamiltonian.orbital_cfg,
            use_aligned_sh=args.use_aligned_sh,
        )
    loss_per_key = float(loss_per_key_sum_mse(pred_blocks_fixed, target_h).item())
    loss_global = float(loss_global_element_mse(pred_blocks_fixed, target_h).item())

    key_stats = _per_key_stats(pred_blocks_fixed, target_h)

    # Gradient comparison on same initialization.
    net_pk = _build_network_from_state(
        base_state,
        mapper=mapper,
        n_radial=args.n_radial,
        l_max=args.l_max,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        device=device,
    )
    net_gl = _build_network_from_state(
        base_state,
        mapper=mapper,
        n_radial=args.n_radial,
        l_max=args.l_max,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        device=device,
    )

    net_pk.zero_grad(set_to_none=True)
    _, pred_pk = forward_to_blocks(
        network=net_pk,
        graph=graph,
        atoms_list=atoms_list,
        mapper=mapper,
        basis=snapshot.hamiltonian.basis,
        orbital_cfg=snapshot.hamiltonian.orbital_cfg,
        use_aligned_sh=args.use_aligned_sh,
    )
    loss_pk_tensor = loss_per_key_sum_mse(pred_pk, target_h)
    loss_pk_tensor.backward()
    grad_pk = flatten_gradients(net_pk)

    net_gl.zero_grad(set_to_none=True)
    _, pred_gl = forward_to_blocks(
        network=net_gl,
        graph=graph,
        atoms_list=atoms_list,
        mapper=mapper,
        basis=snapshot.hamiltonian.basis,
        orbital_cfg=snapshot.hamiltonian.orbital_cfg,
        use_aligned_sh=args.use_aligned_sh,
    )
    loss_gl_tensor = loss_global_element_mse(pred_gl, target_h)
    loss_gl_tensor.backward()
    grad_gl = flatten_gradients(net_gl)

    grad_pk_norm = (
        float(torch.linalg.norm(grad_pk).item()) if grad_pk.numel() > 0 else 0.0
    )
    grad_gl_norm = (
        float(torch.linalg.norm(grad_gl).item()) if grad_gl.numel() > 0 else 0.0
    )
    if grad_pk.numel() == grad_gl.numel() and grad_pk.numel() > 0:
        cosine = float(
            F.cosine_similarity(
                grad_pk.unsqueeze(0), grad_gl.unsqueeze(0), dim=1
            ).item()
        )
    else:
        cosine = None

    per_param: List[Dict[str, Any]] = []
    for (name_pk, p_pk), (name_gl, p_gl) in zip(
        net_pk.named_parameters(), net_gl.named_parameters()
    ):
        if name_pk != name_gl:
            continue
        g_pk = p_pk.grad
        g_gl = p_gl.grad
        if g_pk is None or g_gl is None:
            continue
        n_pk = float(torch.linalg.norm(g_pk).item())
        n_gl = float(torch.linalg.norm(g_gl).item())
        ratio = n_gl / (n_pk + 1e-20)
        per_param.append(
            {
                "name": name_pk,
                "grad_norm_per_key": n_pk,
                "grad_norm_global": n_gl,
                "global_over_per_key_ratio": float(ratio),
            }
        )
    per_param.sort(
        key=lambda r: abs(
            torch.log10(torch.tensor(r["global_over_per_key_ratio"] + 1e-20)).item()
        ),
        reverse=True,
    )

    report: Dict[str, Any] = {
        "check": "priority_2_loss_parity",
        "inputs": {
            "data_path": args.data_path,
            "info_path": args.info_path,
            "cutoff_radius": args.cutoff_radius,
            "n_radial": args.n_radial,
            "l_max": args.l_max,
            "hidden_dim": args.hidden_dim,
            "num_layers": args.num_layers,
            "seed": args.seed,
            "use_aligned_sh": bool(args.use_aligned_sh),
        },
        "objective_values_on_fixed_prediction": {
            "loss_per_key_sum_mse": loss_per_key,
            "loss_global_element_mse": loss_global,
            "global_over_per_key_ratio": float(loss_global / (loss_per_key + 1e-20)),
        },
        "gradient_comparison": {
            "grad_norm_per_key_sum_mse": grad_pk_norm,
            "grad_norm_global_element_mse": grad_gl_norm,
            "global_over_per_key_grad_norm_ratio": float(
                grad_gl_norm / (grad_pk_norm + 1e-20)
            ),
            "gradient_cosine_similarity": cosine,
            "num_grad_elements": int(grad_pk.numel()),
            "largest_param_norm_ratio_diffs": per_param[:15],
        },
        "per_key_weighting_breakdown": key_stats,
        "conclusion": {
            "summary": (
                "Objectives are not equivalent: per-key sum gives each key equal weight, "
                "while global element MSE weights by number of elements."
            ),
            "same_scalar_loss": bool(abs(loss_per_key - loss_global) <= 1e-12),
            "same_gradient_direction": bool(cosine is not None and cosine > 0.9999),
        },
    }

    _dump_report(report, args.output)
    print(f"[check2] report saved to: {args.output}")
    print(
        "[check2] loss(per_key, global)=("
        f"{loss_per_key:.6e}, {loss_global:.6e}), ratio={loss_global/(loss_per_key+1e-20):.6e}"
    )
    print(
        "[check2] grad cosine="
        f"{'None' if cosine is None else f'{cosine:.6f}'}"
        f", norm ratio={grad_gl_norm/(grad_pk_norm+1e-20):.6e}"
    )


if __name__ == "__main__":
    main()
