from __future__ import annotations

import argparse
import copy
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from _common import (  # noqa: E402
    build_minimal_graph_inputs,
    forward_to_blocks,
    load_water_snapshot,
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


def _summarize(history: List[Dict[str, Any]]) -> Dict[str, Any]:
    losses = [r["loss"] for r in history]
    lrs = [r["lr"] for r in history]
    best_loss = min(losses)
    best_epoch = losses.index(best_loss)
    final_loss = losses[-1]
    first_lr_drop_epoch = None
    lr_drop_count = 0
    for i in range(1, len(lrs)):
        if lrs[i] < lrs[i - 1] - 1e-15:
            lr_drop_count += 1
            if first_lr_drop_epoch is None:
                first_lr_drop_epoch = i
    return {
        "best_loss": float(best_loss),
        "best_epoch": int(best_epoch),
        "final_loss": float(final_loss),
        "improvement_from_start": float(losses[0] - final_loss),
        "relative_improvement_from_start": float(
            (losses[0] - final_loss) / (abs(losses[0]) + 1e-20)
        ),
        "first_lr_drop_epoch": first_lr_drop_epoch,
        "lr_drop_count": int(lr_drop_count),
        "final_lr": float(lrs[-1]),
        "avg_epoch_ms": float(
            sum(r["epoch_ms"] for r in history) / max(len(history), 1)
        ),
    }


def _train_variant(
    *,
    network,
    target_h,
    graph,
    atoms_list,
    mapper,
    basis,
    orbital_cfg,
    epochs: int,
    lr: float,
    lr_factor: float,
    lr_patience: int,
    lr_cooldown: int,
    min_lr: float,
    scheduler_before_optimizer: bool,
    use_aligned_sh: bool,
) -> List[Dict[str, Any]]:
    optimizer = Adam(network.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=lr_factor,
        patience=lr_patience,
        threshold=1e-4,
        threshold_mode="rel",
        cooldown=lr_cooldown,
        min_lr=min_lr,
        verbose=False,
    )
    history: List[Dict[str, Any]] = []

    for epoch in range(epochs):
        t0 = time.perf_counter()
        network.train()
        optimizer.zero_grad(set_to_none=True)

        _, pred_blocks = forward_to_blocks(
            network=network,
            graph=graph,
            atoms_list=atoms_list,
            mapper=mapper,
            basis=basis,
            orbital_cfg=orbital_cfg,
            use_aligned_sh=use_aligned_sh,
        )
        loss = loss_per_key_sum_mse(pred_blocks, target_h)

        if scheduler_before_optimizer:
            scheduler.step(loss.detach())

        loss.backward()
        optimizer.step()

        if not scheduler_before_optimizer:
            scheduler.step(loss.detach())

        epoch_ms = (time.perf_counter() - t0) * 1000.0
        history.append(
            {
                "epoch": int(epoch),
                "loss": float(loss.item()),
                "lr": float(optimizer.param_groups[0]["lr"]),
                "epoch_ms": float(epoch_ms),
            }
        )
    return history


def _save_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _save_plot(
    path: Path, before: List[Dict[str, Any]], after: List[Dict[str, Any]]
) -> None:
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    epochs = [r["epoch"] for r in before]
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].plot(epochs, [r["loss"] for r in before], label="scheduler_before_optimizer")
    ax[0].plot(epochs, [r["loss"] for r in after], label="scheduler_after_optimizer")
    ax[0].set_title("Loss")
    ax[0].set_xlabel("Epoch")
    ax[0].set_ylabel("Loss")
    ax[0].set_yscale("log")
    ax[0].legend()

    ax[1].plot(epochs, [r["lr"] for r in before], label="scheduler_before_optimizer")
    ax[1].plot(epochs, [r["lr"] for r in after], label="scheduler_after_optimizer")
    ax[1].set_title("Learning Rate")
    ax[1].set_xlabel("Epoch")
    ax[1].set_ylabel("LR")
    ax[1].set_yscale("log")
    ax[1].legend()

    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Priority check #3: A/B test for scheduler.step() ordering."
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
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lr-factor", type=float, default=0.5)
    parser.add_argument("--lr-patience", type=int, default=20)
    parser.add_argument("--lr-cooldown", type=int, default=10)
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--use-aligned-sh", action="store_true", default=False)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "studies/deeph-e3-comparison-study/reports/check3_scheduler_order_ab.yaml"
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

    base_network = make_network(
        mapper=mapper,
        n_radial=args.n_radial,
        l_max=args.l_max,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        device=device,
    )
    base_state = copy.deepcopy(base_network.state_dict())

    net_before = make_network(
        mapper=mapper,
        n_radial=args.n_radial,
        l_max=args.l_max,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        device=device,
    )
    net_before.load_state_dict(base_state, strict=True)
    hist_before = _train_variant(
        network=net_before,
        target_h=target_h,
        graph=graph,
        atoms_list=atoms_list,
        mapper=mapper,
        basis=snapshot.hamiltonian.basis,
        orbital_cfg=snapshot.hamiltonian.orbital_cfg,
        epochs=args.epochs,
        lr=args.lr,
        lr_factor=args.lr_factor,
        lr_patience=args.lr_patience,
        lr_cooldown=args.lr_cooldown,
        min_lr=args.min_lr,
        scheduler_before_optimizer=True,
        use_aligned_sh=args.use_aligned_sh,
    )

    net_after = make_network(
        mapper=mapper,
        n_radial=args.n_radial,
        l_max=args.l_max,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        device=device,
    )
    net_after.load_state_dict(base_state, strict=True)
    hist_after = _train_variant(
        network=net_after,
        target_h=target_h,
        graph=graph,
        atoms_list=atoms_list,
        mapper=mapper,
        basis=snapshot.hamiltonian.basis,
        orbital_cfg=snapshot.hamiltonian.orbital_cfg,
        epochs=args.epochs,
        lr=args.lr,
        lr_factor=args.lr_factor,
        lr_patience=args.lr_patience,
        lr_cooldown=args.lr_cooldown,
        min_lr=args.min_lr,
        scheduler_before_optimizer=False,
        use_aligned_sh=args.use_aligned_sh,
    )

    summary_before = _summarize(hist_before)
    summary_after = _summarize(hist_after)

    deltas = {
        "delta_final_loss_after_minus_before": float(
            summary_after["final_loss"] - summary_before["final_loss"]
        ),
        "delta_best_loss_after_minus_before": float(
            summary_after["best_loss"] - summary_before["best_loss"]
        ),
        "delta_final_lr_after_minus_before": float(
            summary_after["final_lr"] - summary_before["final_lr"]
        ),
        "delta_avg_epoch_ms_after_minus_before": float(
            summary_after["avg_epoch_ms"] - summary_before["avg_epoch_ms"]
        ),
    }

    report: Dict[str, Any] = {
        "check": "priority_3_scheduler_order_ab",
        "inputs": {
            "data_path": args.data_path,
            "info_path": args.info_path,
            "cutoff_radius": args.cutoff_radius,
            "n_radial": args.n_radial,
            "l_max": args.l_max,
            "hidden_dim": args.hidden_dim,
            "num_layers": args.num_layers,
            "seed": args.seed,
            "epochs": args.epochs,
            "lr": args.lr,
            "lr_factor": args.lr_factor,
            "lr_patience": args.lr_patience,
            "lr_cooldown": args.lr_cooldown,
            "min_lr": args.min_lr,
            "use_aligned_sh": bool(args.use_aligned_sh),
        },
        "variant_before_optimizer": summary_before,
        "variant_after_optimizer": summary_after,
        "deltas": deltas,
        "conclusion": {
            "summary": (
                "Compare final/best loss and LR traces to quantify impact of "
                "calling scheduler.step() before vs after optimizer.step()."
            ),
            "recommended_order": "after_optimizer",
        },
    }

    _dump_report(report, args.output)

    out_dir = args.output.parent
    _save_csv(out_dir / "check3_scheduler_before_history.csv", hist_before)
    _save_csv(out_dir / "check3_scheduler_after_history.csv", hist_after)
    _save_plot(out_dir / "check3_scheduler_order_ab.png", hist_before, hist_after)

    print(f"[check3] report saved to: {args.output}")
    print(
        "[check3] final_loss(before, after)=("
        f"{summary_before['final_loss']:.6e}, {summary_after['final_loss']:.6e})"
    )
    print(
        "[check3] best_loss(before, after)=("
        f"{summary_before['best_loss']:.6e}, {summary_after['best_loss']:.6e})"
    )


if __name__ == "__main__":
    main()
