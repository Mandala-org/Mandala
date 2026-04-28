from __future__ import annotations

from _bootstrap import add_repo_root_to_path

add_repo_root_to_path()

from demos.common import demo_config, load_h2o_dataset
from net.e3gnn import E3GNN


def main() -> None:
    cfg = demo_config(matrix_targets=["hamiltonian", "overlap", "density"])
    train_ds, _, mapper = load_h2o_dataset(cfg=cfg, convention="e3nn")
    x, y = train_ds[0]
    model = E3GNN(mapper=mapper, cfg=cfg)
    preds = model(x)

    print("Model summary")
    print("-------------")
    print(f"heads: {sorted(model.heads.keys())}")
    print(f"num_message_blocks: {len(model.mp_blocks)}")
    print(f"final_node_irreps: {model.final_node_irreps}")
    print(f"final_edge_irreps: {model.final_edge_irreps}")
    print()
    print("Forward output")
    print("--------------")
    for name, pred in preds.items():
        print(f"{name}:")
        print(f"  pair keys: {sorted(pred.pair_vectors.keys())}")
        print(f"  H-H shape: {tuple(pred['H-H'].shape)}")
        print(f"  H-O shape: {tuple(pred['H-O'].shape)}")
    print()
    print("Targets")
    print("-------")
    for name in ("hamiltonian", "overlap", "density"):
        print(f"{name}: H-H shape {tuple(y[name]['H-H'].shape)}")


if __name__ == "__main__":
    main()
