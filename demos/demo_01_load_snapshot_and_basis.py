from __future__ import annotations

from _bootstrap import add_repo_root_to_path

add_repo_root_to_path()

from core.block_irrep_mapper import BlockIrrepMapper
from demos.common import (
    demo_config,
    load_h2o_snapshot,
    summarize_block_matrix,
    summarize_snapshot,
)


def _roundtrip_error(snapshot, mapper, key: str) -> float:
    blocks = snapshot[key]
    vecs = blocks.to_vectors(mapper)
    recon = vecs.to_blocks(mapper)
    diff = recon.pair_blocks[key] - blocks.pair_blocks[key]
    return float(diff.abs().max().item())


def main() -> None:
    cfg = demo_config()
    snap_openmx = load_h2o_snapshot(cfg=cfg, convention="openmx")
    mapper = BlockIrrepMapper(snap_openmx.density.orbital_cfg, dtype=cfg.dtype)
    snap_e3nn = snap_openmx.to_e3nn()

    print("Snapshot summary")
    print("----------------")
    print(summarize_snapshot(snap_openmx))
    print()
    print("Hamiltonian blocks")
    print("------------------")
    print(summarize_block_matrix(snap_openmx.hamiltonian))
    print()
    print("Basis conversion")
    print("----------------")
    print(f"openmx basis -> {snap_openmx.density.basis}")
    print(f"e3nn basis    -> {snap_e3nn.density.basis}")
    print(
        f"H-O roundtrip max error: {_roundtrip_error(snap_e3nn, mapper, 'hamiltonian'):.3e}"
    )
    print()
    print("Example block shapes")
    print("--------------------")
    print(f"H[H-O] shape: {tuple(snap_e3nn.hamiltonian['H-O'].shape)}")
    print(f"D[O-H] shape: {tuple(snap_e3nn.density['O-H'].shape)}")


if __name__ == "__main__":
    main()
