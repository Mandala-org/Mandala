from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from core.block_irrep_mapper import BlockIrrepMapper
from data.snapshot import Snapshot
from net.common import Config

# %%
cfg = Config(
    cutoff_radius=7.0,
    allow_openmx_positions_box_from_out=True,
    verbosity=0,
)

snapshot = Snapshot.from_openmx(
    matrix_path=REPO_ROOT / "data" / "small" / "H2O" / "original" / "H2O.matrix",
    info_path=REPO_ROOT / "data" / "small" / "H2O" / "original" / "H2O.info.out",
    convention="e3nn",
    cutoff_radius=cfg.cutoff_radius,
    cfg=cfg,
)

mapper = BlockIrrepMapper(snapshot.density.orbital_cfg, dtype=cfg.dtype)
hamiltonian_vectors = snapshot.hamiltonian.to_vectors(mapper)
hamiltonian_blocks = hamiltonian_vectors.to_blocks(mapper)

# %%
print("Pair dimensions")
for key in sorted(snapshot.hamiltonian.keys()):
    print(
        key,
        "-> vector_dim =",
        mapper.vector_dim(key),
        "block_dims =",
        mapper.block_dims(key),
    )

print(
    "Roundtrip max error for H-H:",
    float((hamiltonian_blocks["H-H"] - snapshot.hamiltonian["H-H"]).abs().max().item()),
)
print(
    "Roundtrip max error for H-O:",
    float((hamiltonian_blocks["H-O"] - snapshot.hamiltonian["H-O"]).abs().max().item()),
)
print("H-H vectors shape:", tuple(hamiltonian_vectors["H-H"].shape))
print("H-H blocks shape:", tuple(snapshot.hamiltonian["H-H"].shape))
