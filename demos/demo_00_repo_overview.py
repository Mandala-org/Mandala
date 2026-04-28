from __future__ import annotations

from _bootstrap import add_repo_root_to_path

add_repo_root_to_path()

from demos.common import REPO_ROOT, demo_config


def main() -> None:
    cfg = demo_config()
    print("Mandala demo overview")
    print("=====================")
    print(f"repo_root: {REPO_ROOT}")
    print(f"main_config_cutoff: {cfg.cutoff_radius}")
    print(f"main_config_hidden_base_dim: {cfg.hidden_base_dim}")
    print(f"main_config_l_max: {cfg.l_max}")
    print()
    print("Pipeline")
    print("--------")
    print("1. Parse backend files into Snapshot objects")
    print("2. Canonicalize basis, edges, and sparse blocks")
    print("3. Build graph features and align targets")
    print("4. Map sparse blocks to irreps when needed")
    print("5. Run the E(3)-equivariant model")
    print("6. Compute matrix losses and observable losses")
    print("7. Save diagnostic plots and training artifacts")
    print()
    print("Where to look")
    print("-------------")
    print("- src/data: snapshot parsing, basis conversion, graph alignment")
    print("- src/core: block/irrep mappings and sparse physics helpers")
    print("- src/net: model, heads, variants, observables, artifacts")
    print("- scripts: runnable training and sweep entry points")
    print("- studies: research notebooks, sweeps, and ablations")


if __name__ == "__main__":
    main()
