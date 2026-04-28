from __future__ import annotations

from _bootstrap import add_repo_root_to_path

add_repo_root_to_path()


import matplotlib

matplotlib.use("Agg")

from demos.common import demo_config, ensure_output_dir, load_h2o_dataset
from net.artifacts import (
    compute_distance_error_curve,
    save_dos_comparison_plot,
    save_distance_error_curve_plot,
    save_matrix_comparison_plot,
)
from net.e3gnn import E3GNN


def main() -> None:
    cfg = demo_config(matrix_targets=["hamiltonian", "overlap", "density"])
    train_ds, _, mapper = load_h2o_dataset(cfg=cfg, convention="e3nn")
    x, y = train_ds[0]
    model = E3GNN(mapper=mapper, cfg=cfg)
    preds = model(x)
    pred_h = preds["hamiltonian"].to_blocks(mapper)

    out_dir = ensure_output_dir("artifacts_and_diagnostics")
    matrix_path = out_dir / "hamiltonian_comparison.png"
    dos_path = out_dir / "dos_comparison.png"
    dist_path = out_dir / "distance_error_curve.png"

    save_matrix_comparison_plot(
        pred_h,
        y["hamiltonian"],
        matrix_path,
        title="Hamiltonian comparison",
        max_atoms=6,
    )
    dos_stats = save_dos_comparison_plot(
        pred_h,
        y["hamiltonian"],
        y["overlap"],
        dos_path,
        title="H DOS comparison",
    )
    curve = compute_distance_error_curve(
        pred_h,
        y["hamiltonian"],
        positions=x["positions"],
        box=x["box"],
        n_bins=12,
    )
    if curve is not None:
        save_distance_error_curve_plot(curve, dist_path, title="Distance error curve")

    print("Artifacts")
    print("---------")
    print(f"matrix plot: {matrix_path}")
    print(f"dos plot: {dos_path}")
    if curve is not None:
        print(f"distance curve: {dist_path}")
    print()
    print("DOS stats")
    print("---------")
    print(dos_stats)
    print()
    print("What this demo shows")
    print("--------------------")
    print("The artifact helpers turn predictions into readable diagnostics and plots.")


if __name__ == "__main__":
    main()
