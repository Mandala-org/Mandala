from __future__ import annotations

from _bootstrap import add_repo_root_to_path

add_repo_root_to_path()

from collections import defaultdict

import torch

from demos.common import demo_config, load_h2o_dataset
from net.e3gnn import E3GNN


def main() -> None:
    cfg = demo_config(
        matrix_targets=["hamiltonian"],
        train_target="matrix",
        enable_energy=False,
        enable_num_electrons=False,
        train_on_energy=False,
        train_on_num_electrons=False,
        max_epochs=5,
        lr=1e-3,
    )
    train_ds, _, mapper = load_h2o_dataset(cfg=cfg, convention="e3nn")
    x, y = train_ds[0]
    model = E3GNN(mapper=mapper, cfg=cfg)
    model.log = lambda *args, **kwargs: None  # type: ignore[assignment]
    model.log_dict = lambda *args, **kwargs: None  # type: ignore[assignment]
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    history: dict[str, list[float]] = defaultdict(list)
    for step in range(cfg.max_epochs):
        optimizer.zero_grad(set_to_none=True)
        loss = model.training_step((x, y), batch_idx=step)
        loss.backward()
        optimizer.step()
        history["loss"].append(float(loss.item()))
        print(f"step={step:02d} loss={history['loss'][-1]:.6f}")

    print()
    print("Training history")
    print("----------------")
    print(history["loss"])


if __name__ == "__main__":
    main()
