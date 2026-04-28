from __future__ import annotations

from _bootstrap import add_repo_root_to_path

add_repo_root_to_path()

import torch

from demos.common import demo_config, load_h2o_dataset
from net.e3gnn import E3GNN
from net.observable_metrics import build_observable_predictions, observable_loss


def main() -> None:
    cfg = demo_config(
        matrix_targets=["hamiltonian", "overlap", "density"],
        enable_energy=True,
        enable_num_electrons=True,
        train_on_energy=True,
        train_on_num_electrons=True,
        loss_coef_observables=1e-5,
    )
    train_ds, _, mapper = load_h2o_dataset(cfg=cfg, convention="e3nn")
    x, y = train_ds[0]
    model = E3GNN(mapper=mapper, cfg=cfg)
    preds = model(x)
    preds_matrix = {name: pred.to_blocks(mapper) for name, pred in preds.items()}
    observable_values = build_observable_predictions(
        preds_matrix,
        pred_trace_alignment=x["pred_trace_alignment"],
        H_true=y["hamiltonian"],
        D_true=y["density"],
        S_true=y["overlap"],
    )

    loss_E, loss_N = observable_loss(
        cfg=cfg,
        observable_values=observable_values,
        energy_target=y["energy"],
        num_electrons_target=y["num_electrons"],
        mse_fn=torch.nn.functional.mse_loss,
        device=torch.device("cpu"),
    )

    print("Observable predictions")
    print("----------------------")
    for key, value in sorted(observable_values.items()):
        print(f"{key}: {float(value.item()):.6f}")
    print()
    print("Observable losses")
    print("-----------------")
    print(f"energy loss: {float(loss_E.item()):.6e}")
    print(f"num_electrons loss: {float(loss_N.item()):.6e}")
    print()
    print("Ground truth")
    print("------------")
    print(f"energy target: {float(y['energy'].item()):.6f}")
    print(f"num_electrons target: {float(y['num_electrons'].item()):.6f}")


if __name__ == "__main__":
    main()
