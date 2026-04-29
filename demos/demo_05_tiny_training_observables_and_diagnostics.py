from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from data.factory import DatasetFactory
from net.artifacts import (
    compute_distance_error_curve,
    save_dos_comparison_plot,
    save_distance_error_curve_plot,
    save_matrix_comparison_plot,
)
from net.common import Config
from net.e3gnn import E3GNN
from net.observable_metrics import build_observable_predictions, observable_loss

# %%
matrix_path = REPO_ROOT / "data" / "small" / "H2O" / "original" / "H2O.matrix"
info_path = REPO_ROOT / "data" / "small" / "H2O" / "original" / "H2O.info.out"
output_dir = REPO_ROOT / "demos" / "_outputs" / "tiny_training"
output_dir.mkdir(parents=True, exist_ok=True)

cfg = Config(
    cutoff_radius=7.0,
    matrix_targets=["hamiltonian", "overlap", "density"],
    train_target="matrix",
    train_on_energy=True,
    train_on_num_electrons=True,
    train_observables_on_gt=False,
    enable_energy=True,
    enable_num_electrons=True,
    loss_coef_observables=1e-5,
    max_epochs=3,
    lr=1e-3,
    num_layers_gnn=1,
    hidden_base_dim=16,
    l_max=2,
    n_radial=16,
    radial_layers=(32,),
    verbosity=0,
)

factory = DatasetFactory(cfg, convention="e3nn")
factory.add_snapshot(matrix_path, info_path, purpose="train")
train_ds, _, mapper = factory.create()
x, y = train_ds[0]

model = E3GNN(mapper=mapper, cfg=cfg)
model.log = lambda *args, **kwargs: None
model.log_dict = lambda *args, **kwargs: None
optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

# %%
loss_history = []
for step in range(cfg.max_epochs):
    optimizer.zero_grad(set_to_none=True)
    loss = model.training_step((x, y), batch_idx=step)
    loss.backward()
    optimizer.step()
    loss_history.append(float(loss.item()))

predictions = model(x)
predictions_matrix = {
    name: pred.to_blocks(mapper) for name, pred in predictions.items()
}
observable_values = build_observable_predictions(
    predictions_matrix,
    pred_trace_alignment=x["pred_trace_alignment"],
    H_true=y["hamiltonian"],
    D_true=y["density"],
    S_true=y["overlap"],
)
energy_loss, num_electron_loss = observable_loss(
    cfg=cfg,
    observable_values=observable_values,
    energy_target=y["energy"],
    num_electrons_target=y["num_electrons"],
    mse_fn=torch.nn.functional.mse_loss,
    device=torch.device("cpu"),
)

# %%
matrix_plot_path = output_dir / "hamiltonian_comparison.png"
dos_plot_path = output_dir / "dos_comparison.png"
distance_plot_path = output_dir / "distance_error_curve.png"

save_matrix_comparison_plot(
    predictions_matrix["hamiltonian"],
    y["hamiltonian"],
    matrix_plot_path,
    title="Hamiltonian comparison",
    max_atoms=6,
)
save_dos_comparison_plot(
    predictions_matrix["hamiltonian"],
    y["hamiltonian"],
    y["overlap"],
    dos_plot_path,
    title="DOS comparison",
)
distance_curve = compute_distance_error_curve(
    predictions_matrix["hamiltonian"],
    y["hamiltonian"],
    positions=x["positions"],
    box=x["box"],
    n_bins=12,
)
if distance_curve is not None:
    save_distance_error_curve_plot(
        distance_curve, distance_plot_path, title="Distance error curve"
    )

# %%
print("loss_history:", [round(v, 6) for v in loss_history])
print("energy_loss:", float(energy_loss.item()))
print("num_electron_loss:", float(num_electron_loss.item()))
print("observable keys:", sorted(observable_values.keys()))
print("saved_matrix_plot:", matrix_plot_path)
print("saved_dos_plot:", dos_plot_path)
print(
    "saved_distance_plot:", distance_plot_path if distance_curve is not None else None
)
