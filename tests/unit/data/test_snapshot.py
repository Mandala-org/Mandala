import pytest
from pathlib import Path
import torch
from collections import Counter
import math
import copy

from core.sparse_math import trace_matmul_sparse_snap_vectorized
from data.snapshot import Snapshot
from data.block_matrix import BlockMatrix
from core.orbital_irrep_config import OrbitalIrrepConfig
from net.common import Config
from data.openmx_parser import parse_openmx_scfout


def _load_snapshot():
    # cfg = OrbitalIrrepConfig.from_dict({"H": "3s2p", "O": "3s3p2d"})
    # sample = Path("./data/small/H2O/original/H2O.matrix")
    # atoms = list("HHHHOO")
    # snap = parse_openmx_scfout(sample, atoms, cfg, convention="openmx")
    cfg = Config(cutoff_radius=8.0, allow_openmx_positions_box_from_out=True)
    snap = Snapshot.from_openmx(
        Path("./data/small/H2O/original/H2O.matrix"),
        Path("./data/small/H2O/original/H2O.info.out"),
        cfg=cfg,
        convention="openmx",
    )
    return snap


@pytest.mark.unit
def test_energy_and_electron_count(small_angular_snapshot_e3nn):
    snap = small_angular_snapshot_e3nn
    E = snap.get_energy()
    Ne = snap.get_number_of_electrons()

    # reference using sparse helper directly
    E_ref = trace_matmul_sparse_snap_vectorized(snap.hamiltonian, snap.density)
    Ne_ref = trace_matmul_sparse_snap_vectorized(snap.density, snap.overlap)

    assert torch.allclose(E, E_ref, atol=1e-6)
    assert torch.allclose(Ne, Ne_ref, atol=1e-6)


@pytest.mark.unit
def test_symmetrize_matrices_keeps_energy_and_electrons(h2o_orbital_cfg):
    snap_raw = parse_openmx_scfout(
        Path("./data/small/H2O/original/H2O.matrix"),
        list("HHHHOO"),
        h2o_orbital_cfg,
        convention="e3nn",
        symmetrize_density=False,
    )
    snap_sym = snap_raw.symmetrize_matrices()

    assert torch.allclose(snap_raw.get_energy(), snap_sym.get_energy(), atol=1e-6)
    assert torch.allclose(
        snap_raw.get_number_of_electrons(),
        snap_sym.get_number_of_electrons(),
        atol=1e-6,
    )
    for name in ("overlap", "density"):
        mat = getattr(snap_sym, name)
        mat_t = mat.transpose()
        for key in mat.keys():
            assert torch.allclose(mat[key], mat_t[key], atol=1e-6)


@pytest.mark.unit
def test_save_load_roundtrip(tmp_path, small_angular_snapshot_e3nn):
    snap = small_angular_snapshot_e3nn
    file = tmp_path / "snapshot.pt"
    snap.save(file)

    snap2 = Snapshot.load(file)
    assert torch.allclose(snap2.get_energy(), snap.get_energy(), atol=1e-6)
    assert torch.allclose(
        snap2.get_number_of_electrons(), snap.get_number_of_electrons(), atol=1e-6
    )


@pytest.mark.unit
def test_deepcopy_is_independent_and_preserves_dynamic_matrix_aliases(
    small_angular_snapshot_e3nn,
):
    snap = small_angular_snapshot_e3nn
    cloned = copy.deepcopy(snap)

    assert cloned is not snap
    assert cloned.hamiltonian is cloned["hamiltonian"]
    assert cloned.hamiltonian is not snap.hamiltonian
    original = snap.hamiltonian.pair_blocks["H-H"].clone()
    cloned.hamiltonian.pair_blocks["H-H"][0, 0, 0] += 1.0
    assert torch.equal(snap.hamiltonian.pair_blocks["H-H"], original)


@pytest.mark.unit
def test_snapshot_load_uses_weights_only_false(monkeypatch, tmp_path):
    file = tmp_path / "snapshot.pt"
    file.write_bytes(b"placeholder")

    captured = {}

    def fake_torch_load(path, map_location=None, weights_only=None, **kwargs):
        captured["path"] = path
        captured["map_location"] = map_location
        captured["weights_only"] = weights_only
        return {
            "mats": {
                "hamiltonian": {
                    "atoms": ("H",),
                    "atom_counts": {"H": 1},
                    "pair_blocks": {"H-H": torch.ones(1, 1, 1)},
                    "pair_edges": {"H-H": torch.tensor([[0], [0], [0], [0], [0]])},
                    "lookup": {(0, 0, 0, 0, 0): ("H-H", 0)},
                    "orbital_cfg": {"H": "1s"},
                    "basis": "e3nn",
                },
                "overlap": {
                    "atoms": ("H",),
                    "atom_counts": {"H": 1},
                    "pair_blocks": {"H-H": torch.ones(1, 1, 1)},
                    "pair_edges": {"H-H": torch.tensor([[0], [0], [0], [0], [0]])},
                    "lookup": {(0, 0, 0, 0, 0): ("H-H", 0)},
                    "orbital_cfg": {"H": "1s"},
                    "basis": "e3nn",
                },
                "density": {
                    "atoms": ("H",),
                    "atom_counts": {"H": 1},
                    "pair_blocks": {"H-H": torch.ones(1, 1, 1)},
                    "pair_edges": {"H-H": torch.tensor([[0], [0], [0], [0], [0]])},
                    "lookup": {(0, 0, 0, 0, 0): ("H-H", 0)},
                    "orbital_cfg": {"H": "1s"},
                    "basis": "e3nn",
                },
            }
        }

    monkeypatch.setattr(torch, "load", fake_torch_load)

    snap = Snapshot.load(file)

    assert captured["path"] == file
    assert captured["map_location"] == "cpu"
    assert captured["weights_only"] is False
    assert snap.density.atoms == ("H",)


@pytest.mark.unit
def test_stress_transforms_under_basis_change_and_rotation():
    atoms = ("H",)
    orb_cfg = OrbitalIrrepConfig.from_dict({"H": "1s"})
    pair_edges = {"H-H": torch.tensor([[0], [0], [0], [0], [0]], dtype=torch.long)}
    pair_blocks = {"H-H": torch.ones(1, 1, 1)}
    lookup = {(0, 0, 0, 0, 0): ("H-H", 0)}
    bm = BlockMatrix(
        atoms=atoms,
        atom_counts=Counter(atoms),
        pair_blocks=pair_blocks,
        pair_edges=pair_edges,
        lookup=lookup,
        orbital_cfg=orb_cfg,
        basis="openmx",
    )
    positions = torch.tensor([[1.0, 2.0, 3.0]])
    forces = torch.tensor([[0.3, -0.4, 0.5]])
    box = torch.tensor([[2.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 4.0]])
    stress = torch.tensor([[1.0, 0.2, -0.1], [0.3, 2.0, 0.4], [-0.2, 0.5, 3.0]])
    snap = Snapshot(
        hamiltonian=bm,
        overlap=bm,
        density=bm,
        positions=positions,
        forces=forces,
        box=box,
        stress=stress,
    )

    cob = torch.eye(3)[[2, 0, 1]]
    snap_e3 = snap.to_e3nn()
    assert torch.allclose(snap_e3.positions, positions @ cob, atol=1e-6)
    assert torch.allclose(snap_e3.forces, forces @ cob, atol=1e-6)
    assert torch.allclose(snap_e3.box, box @ cob, atol=1e-6)
    assert torch.allclose(snap_e3.stress, cob.T @ stress @ cob, atol=1e-6)
    assert torch.allclose(snap_e3.to_openmx().stress, stress, atol=1e-6)

    theta = math.pi / 2.0
    R = torch.tensor(
        [
            [math.cos(theta), -math.sin(theta), 0.0],
            [math.sin(theta), math.cos(theta), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
    )
    snap_rot = snap.rotate(R)
    assert torch.allclose(snap_rot.positions, positions @ R.T, atol=1e-6)
    assert torch.allclose(snap_rot.forces, forces @ R.T, atol=1e-6)
    assert torch.allclose(snap_rot.box, box @ R.T, atol=1e-6)
    assert torch.allclose(snap_rot.stress, R @ stress @ R.T, atol=1e-6)


# ---------------------------------------------------------------- basis conversion


@pytest.mark.unit
def test_basis_conversion_roundtrip():
    snap_open = _load_snapshot()  # native OpenMX basis
    snap_e3 = snap_open.to_e3nn()

    # sanity checks
    assert snap_e3.density.basis == "e3nn"
    assert snap_e3.to_e3nn() is snap_e3  # idempotent

    # back-conversion
    snap_back = snap_e3.to_openmx()
    assert snap_back.density.basis == "openmx"

    # numerical invariants -------------------------------------------------
    # choose one representative block (H-O first edge)
    key = "H-O"
    assert torch.allclose(
        snap_back.density[key][0], snap_open.density[key][0], atol=1e-6
    )
    assert torch.allclose(
        snap_back.hamiltonian[key][0], snap_open.hamiltonian[key][0], atol=1e-6
    )
    assert torch.allclose(
        snap_back.overlap[key][0], snap_open.overlap[key][0], atol=1e-6
    )
    assert torch.allclose(snap_back.positions, snap_open.positions, atol=1e-6)
    assert torch.allclose(snap_back.box, snap_open.box, atol=1e-6)

    # physics helpers unchanged
    assert torch.allclose(snap_back.get_energy(), snap_open.get_energy(), atol=1e-6)
    assert torch.allclose(
        snap_back.get_number_of_electrons(),
        snap_open.get_number_of_electrons(),
        atol=1e-6,
    )
    assert torch.allclose(snap_e3.get_energy(), snap_open.get_energy(), atol=1e-6)
    assert torch.allclose(
        snap_e3.get_number_of_electrons(),
        snap_open.get_number_of_electrons(),
        atol=1e-6,
    )


@pytest.mark.unit
def test_canonical_edge_ordering_tiebreaker():
    # 1. Setup Geometry
    # Atom 0: "H" at (0, 0, 0)
    # Atom 1: "H" at (2, 0, 0) -> dist 2
    # Atom 2: "H" at (0, 2, 0) -> dist 2
    # Atom 3: "H" at (1, 0, 0) -> dist 1

    # Edges of interest: (0, 1) and (0, 2). Both have dist=2.
    # Canonical order should be determined by indices: (0, 1) < (0, 2).

    atoms = ("H", "H", "H", "H")
    positions = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [1.0, 0.0, 0.0],
        ]
    )
    # Box is needed for neighbor_list, make it large enough to avoid PBC issues for this test
    box = torch.eye(3) * 10.0

    # 2. Create Dummy BlockMatrix
    # We only need one key "H-H"
    # Edges:
    # 0: (0, 2) - dist 2
    # 1: (0, 1) - dist 2
    # 2: (0, 3) - dist 1
    # 3: (0, 0) - dist 0 (diagonal)

    # Expected order:
    # 1. Diagonal: (0, 0)
    # 2. Off-diagonal sorted by distance, then lex:
    #    - (0, 3) [dist 1]
    #    - (0, 1) [dist 2, dst=1]
    #    - (0, 2) [dist 2, dst=2]

    # Input edges (shuffled/reverse order to test sorting)
    # src, dst, sx, sy, sz
    edges_data = [
        [0, 0, 0, 0, 2],  # dist 2
        [0, 0, 0, 0, 1],  # dist 2
        [0, 0, 0, 0, 3],  # dist 1
        [0, 0, 0, 0, 0],  # dist 0
    ]
    edges_tensor = torch.tensor(edges_data).T  # (5, 4)

    # Dummy blocks (1x1)
    blocks_tensor = torch.randn(4, 1, 1)

    pair_edges = {"H-H": edges_tensor}
    pair_blocks = {"H-H": blocks_tensor}

    # Lookup not strictly needed for canonicalize_edges but good for consistency
    lookup = {}
    for idx, row in enumerate(edges_data):
        lookup[tuple(row)] = ("H-H", idx)

    orb_cfg = OrbitalIrrepConfig.from_dict({"H": "1s"})

    bm = BlockMatrix(
        atoms=atoms,
        atom_counts=Counter(atoms),
        pair_blocks=pair_blocks,
        pair_edges=pair_edges,
        lookup=lookup,
        orbital_cfg=orb_cfg,
        basis="e3nn",
    )

    # 3. Create Snapshot
    cfg = Config()
    cfg.cutoff_radius = 5.0  # Large enough

    snap = Snapshot(
        hamiltonian=bm,
        overlap=bm,
        density=bm,
        positions=positions,
        box=box,
        cfg=cfg,
    )

    # 4. Run Canonicalization
    snap_canon = snap.canonicalize_edges()

    # 5. Verify Order
    new_edges = snap_canon.hamiltonian.pair_edges["H-H"]
    # Expected:
    # Index 0: (0, 0) - Diag
    # Index 1: (0, 3) - Dist 1
    # Index 2: (0, 1) - Dist 2, dst 1
    # Index 3: (0, 2) - Dist 2, dst 2

    # Check src/dst pairs
    pairs = new_edges[3:, :].T.tolist()
    assert pairs[0] == [0, 0], f"Expected [0, 0], got {pairs[0]}"
    assert pairs[1] == [0, 3], f"Expected [0, 3], got {pairs[1]}"
    assert pairs[2] == [0, 1], f"Expected [0, 1], got {pairs[2]}"
    assert pairs[3] == [0, 2], f"Expected [0, 2], got {pairs[3]}"
