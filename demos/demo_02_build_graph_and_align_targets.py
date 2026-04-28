from __future__ import annotations

from _bootstrap import add_repo_root_to_path

add_repo_root_to_path()

from demos.common import demo_config, load_h2o_dataset, summarize_dataset_sample


def main() -> None:
    cfg = demo_config()
    train_ds, _, mapper = load_h2o_dataset(cfg=cfg, convention="e3nn")
    x, y = train_ds[0]

    print("Dataset sample")
    print("--------------")
    print(summarize_dataset_sample(x, y))
    print()
    print("Graph tensors")
    print("-------------")
    print(f"edge_index: {tuple(x['edge_index'].shape)}")
    print(f"edge_shift: {tuple(x['edge_shift'].shape)}")
    print(f"edge_type_idx: {tuple(x['edge_type_idx'].shape)}")
    print(f"edge_length_emb: {tuple(x['edge_length_emb'].shape)}")
    print(f"edge_sh: {tuple(x['edge_sh'].shape)}")
    print()
    print("Target alignment")
    print("----------------")
    for name in ("hamiltonian", "overlap", "density"):
        target = y[name]
        print(f"{name}:")
        for key in sorted(target.pair_edges.keys())[:4]:
            edge_count = target.pair_edges[key].shape[1]
            pred_count = x["pred_pair_edges_static"][key].shape[1]
            print(f"  {key}: target_edges={edge_count}, pred_edges={pred_count}")
    print()
    print("Trace alignment keys")
    print("--------------------")
    print(sorted(x["pred_trace_alignment"].keys()))
    print()
    print("Pair partitions example")
    print("-----------------------")
    first_key = sorted(x["edge_partitions"].keys())[0]
    print(first_key)
    print(
        {
            name: tuple(value.shape)
            for name, value in x["edge_partitions"][first_key].items()
        }
    )


if __name__ == "__main__":
    main()
