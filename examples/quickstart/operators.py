"""Sparse electronic-structure operators: load, inspect, contract, diagonalize.

Run from the repository root::

    PYTHONPATH=src:. python examples/quickstart/operators.py
"""

from pathlib import Path

import torch

from core.block_irrep_mapper import BlockIrrepMapper
from core.sparse_math import trace_matmul_sparse_block_matrix
from data.snapshot import Snapshot
from net.common import Config
from utils.units import HARTREE_TO_EV

ROOT = Path(__file__).resolve().parents[2]
SILICON = ROOT / "data" / "small" / "Si"

# 1. Load an OpenMX calculation (HS.out + its output file) into the common
#    e3nn real-spherical-harmonic convention. All stored blocks are kept.
cfg = Config(verbosity=0)
snapshot = Snapshot.from_openmx(
    SILICON / "HS.out", SILICON / "Si.out", convention="e3nn", cfg=cfg
)
H, S, D = snapshot.hamiltonian, snapshot.overlap, snapshot.density
print("atoms:", len(H.atoms), "| species pairs:", sorted(H.keys()))
print("orbital irreps per species:", snapshot.density.orbital_cfg.to_dict())
print("Si-Si blocks (edges, rows, cols):", tuple(H["Si-Si"].shape))

# 2. Every block is labelled by its directed periodic edge (L, i, j).
edges = H.edges("Si-Si")  # rows: Lx, Ly, Lz, i, j
onsite = (edges[:3] == 0).all(0) & (edges[3] == edges[4])
print("onsite blocks:", int(onsite.sum()), "| total blocks:", edges.shape[1])
local = snapshot.filter_by_distance(8.0)
print("blocks within 8 A:", local.hamiltonian["Si-Si"].shape[0])

# 3. Observables are sparse trace contractions Tr(AB) = sum tr(A_ij^L B_ji^-L).
electrons = float(trace_matmul_sparse_block_matrix(D, S))
band_energy = float(trace_matmul_sparse_block_matrix(D, H)) * HARTREE_TO_EV
print(f"Tr(DS) = {electrons:.6f} electrons | Tr(DH) = {band_energy:.4f} eV")

# 4. Hermiticity of the stored real-space operator: H_ij^L = (H_ji^-L)^T.
asymmetry = float((H - H.adjoint()).to_dense().abs().max())
print(f"max |H - H^T| = {asymmetry:.2e} Hartree")

# 5. Change of basis into irreducible representations and back.
mapper = BlockIrrepMapper(snapshot.density.orbital_cfg, dtype=torch.float32)
vectors = H.to_vectors(mapper)
roundtrip = float((vectors.to_blocks(mapper)["Si-Si"] - H["Si-Si"]).abs().max())
print("Si-Si irreps:", mapper.get_pair_irreps("Si-Si"))
print(f"block -> irreps -> block error: {roundtrip:.1e}")

# 6. Generalized eigenvalue problem H(k) c = e S(k) c along a band path.
bands = snapshot.get_band_structure(path="GXMGR", npoints=60)
energies = bands.eigenvalues * HARTREE_TO_EV
print("band structure:", tuple(energies.shape), "(k-points, bands)")
