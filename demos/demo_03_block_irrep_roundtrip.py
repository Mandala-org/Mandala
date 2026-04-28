from __future__ import annotations

from _bootstrap import add_repo_root_to_path

add_repo_root_to_path()

from demos.common import demo_config, load_h2o_snapshot
from core.block_irrep_mapper import BlockIrrepMapper


def main() -> None:
    cfg = demo_config()
    snap = load_h2o_snapshot(cfg=cfg, convention="e3nn")
    mapper = BlockIrrepMapper(snap.density.orbital_cfg, dtype=cfg.dtype)

    print("Irrep dimensions by pair")
    print("------------------------")
    for key in sorted(snap.hamiltonian.keys()):
        print(
            f"{key}: n_vec={mapper.vector_dim(key)}, block_dims={mapper.block_dims(key)}"
        )

    print()
    print("Roundtrip errors")
    print("----------------")
    for key in ("H-H", "H-O", "O-H", "O-O"):
        blocks = snap.hamiltonian[key]
        vecs = mapper.blocks_to_vectors(key, blocks)
        recon = mapper.vectors_to_blocks(key, vecs)
        err = (recon - blocks).abs().max().item()
        print(f"{key}: {err:.3e}")

    print()
    print("Block-vs-irrep view")
    print("-------------------")
    ham_vec = snap.hamiltonian.to_vectors(mapper)
    print(f"H-H vectors: {tuple(ham_vec['H-H'].shape)}")
    print(f"H-H blocks:  {tuple(snap.hamiltonian['H-H'].shape)}")


if __name__ == "__main__":
    main()
