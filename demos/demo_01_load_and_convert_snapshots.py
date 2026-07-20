from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from net.common import Config
from data.snapshot import Snapshot

# %%
cfg = Config(
    cutoff_radius=7.0,
    allow_openmx_positions_box_from_out=True,
    verbosity=0,
    safety_checks=True,
)

openmx_snapshot = Snapshot.from_openmx(
    matrix_path=REPO_ROOT / "data" / "small" / "H2O" / "original" / "H2O.matrix",
    info_path=REPO_ROOT / "data" / "small" / "H2O" / "original" / "H2O.info.out",
    convention="e3nn",
    cutoff_radius=cfg.cutoff_radius,
    cfg=cfg,
)

fhiaims_snapshot = Snapshot.from_fhiaims(
    geometry_path=REPO_ROOT
    / "data"
    / "fhi-aims"
    / "original"
    / "basis_small"
    / "geometry.in",
    basis_path=REPO_ROOT
    / "data"
    / "fhi-aims"
    / "original"
    / "basis_small"
    / "basis-indices.out",
    hamiltonian_path=REPO_ROOT
    / "data"
    / "fhi-aims"
    / "original"
    / "basis_small"
    / "H_spin_01_kpt_000001.csc",
    overlap_path=REPO_ROOT
    / "data"
    / "fhi-aims"
    / "original"
    / "basis_small"
    / "S_spin_01_kpt_000001.csc",
    density_path=REPO_ROOT
    / "data"
    / "fhi-aims"
    / "original"
    / "basis_small"
    / "D_spin_01_kpt_000001.csc",
    convention="e3nn",
    cfg=cfg,
)

openmx_back_to_native = openmx_snapshot.to_openmx()
openmx_e3nn = openmx_snapshot.to_e3nn()

# %%
print("OpenMX snapshot")
print(openmx_snapshot)
print("H2O energy:", float(openmx_snapshot.get_energy().item()))
print("H2O electrons:", float(openmx_snapshot.get_number_of_electrons().item()))
print("OpenMX basis:", openmx_snapshot.density.basis)
print("OpenMX -> e3nn basis:", openmx_e3nn.density.basis)
print("OpenMX roundtrip basis:", openmx_back_to_native.density.basis)

# %%
print("FHI-aims snapshot")
print(fhiaims_snapshot)
print("FHI-aims basis:", fhiaims_snapshot.density.basis)
print("FHI-aims H block keys:", sorted(fhiaims_snapshot.hamiltonian.keys()))
