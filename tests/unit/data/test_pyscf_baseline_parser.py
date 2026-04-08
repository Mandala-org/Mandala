from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from data.pyscf_baseline_parser import load_pyscf_baseline, load_pyscf_snapshot


def _write_artifacts(
    root: Path,
    *,
    matrix_key: str = "hamiltonian_ao",
    orbital_set: dict[str, str] | None = None,
    with_shifts: bool = True,
    include_observables_in_npz: bool = True,
) -> tuple[
    Path, Path, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray
]:
    atoms = [
        {
            "index": 1,
            "element": "H",
            "x_angstrom": 0.0,
            "y_angstrom": 0.0,
            "z_angstrom": 0.0,
        },
        {
            "index": 2,
            "element": "H",
            "x_angstrom": 1.0,
            "y_angstrom": 0.0,
            "z_angstrom": 0.0,
        },
        {
            "index": 3,
            "element": "O",
            "x_angstrom": 0.0,
            "y_angstrom": 1.0,
            "z_angstrom": 0.0,
        },
    ]
    if orbital_set is None:
        orbital_set = {"H": "1s", "O": "1s"}

    ham = np.array(
        [[-1.1, 0.2, -0.3], [0.2, -0.9, 0.1], [-0.3, 0.1, -1.5]],
        dtype=np.float32,
    )
    overlap = np.eye(3, dtype=np.float32)
    dm = np.array(
        [[1.0, 0.05, 0.0], [0.05, 1.0, 0.0], [0.0, 0.0, 1.8]],
        dtype=np.float32,
    )
    positions = np.array(
        [
            [atoms[0]["x_angstrom"], atoms[0]["y_angstrom"], atoms[0]["z_angstrom"]],
            [atoms[1]["x_angstrom"], atoms[1]["y_angstrom"], atoms[1]["z_angstrom"]],
            [atoms[2]["x_angstrom"], atoms[2]["y_angstrom"], atoms[2]["z_angstrom"]],
        ],
        dtype=np.float32,
    )
    box = np.diag([4.0, 5.0, 6.0]).astype(np.float32)
    forces = np.array(
        [[0.1, -0.2, 0.3], [-0.4, 0.5, -0.6], [0.7, -0.8, 0.9]], dtype=np.float32
    )
    stress = np.array(
        [[1.0, 0.2, -0.1], [0.3, 2.0, 0.4], [-0.2, 0.5, 3.0]], dtype=np.float32
    )

    npz_path = root / "sample.npz"
    json_path = root / "sample.json"

    payload_npz = {
        matrix_key: ham,
        "overlap_ao": overlap,
        "dm_ao": dm,
        "positions_angstrom": positions,
    }
    if include_observables_in_npz:
        payload_npz["box_angstrom"] = box
        payload_npz["forces_hartree_per_bohr"] = forces
        payload_npz["stress_hartree_per_angstrom3"] = stress
    if with_shifts:
        shifts = np.array([[0, 0, 0], [1, 0, 0]], dtype=np.int64)
        payload_npz["shifts"] = shifts
        payload_npz["hamiltonian_shifted"] = np.stack([ham, ham + 10.0], axis=0)
        payload_npz["overlap_shifted"] = np.stack([overlap, overlap * 2.0], axis=0)
        payload_npz["density_shifted"] = np.stack([dm, dm + 1.0], axis=0)
    np.savez_compressed(npz_path, **payload_npz)
    payload = {
        "snapshot": {"atoms": atoms, "box_angstrom": box.tolist()},
        "settings": {"orbital_set": orbital_set},
        "pyscf": {
            "forces_hartree_per_bohr": forces.tolist(),
            "stress_hartree_per_angstrom3": stress.tolist(),
        },
    }
    json_path.write_text(json.dumps(payload))
    return npz_path, json_path, ham, overlap, dm, box, forces, stress


@pytest.mark.unit
def test_load_pyscf_baseline_and_snapshot(tmp_path: Path):
    npz_path, json_path, ham, overlap, dm, box, forces, stress = _write_artifacts(
        tmp_path, with_shifts=True
    )

    data = load_pyscf_baseline(npz_path, json_path=json_path)
    assert data.atoms == ("H", "H", "O")
    assert data.hamiltonian_ao.shape == (3, 3)
    assert data.overlap_ao.shape == (3, 3)
    assert data.density_ao.shape == (3, 3)
    assert data.positions is not None
    assert data.positions.shape == (3, 3)
    assert data.box is not None
    assert torch.allclose(data.box, torch.tensor(box))
    assert data.forces is not None
    assert torch.allclose(data.forces, torch.tensor(forces))
    assert data.stress is not None
    assert torch.allclose(data.stress, torch.tensor(stress))

    snap = data.to_snapshot()
    assert torch.isclose(
        snap.hamiltonian[(0, 0, 0, 0, 1)].squeeze(),
        torch.tensor(ham[0, 1], dtype=torch.float32),
    )
    assert torch.isclose(
        snap.overlap[(0, 0, 0, 0, 0)].squeeze(),
        torch.tensor(overlap[0, 0], dtype=torch.float32),
    )
    assert torch.isclose(
        snap.density[(0, 0, 0, 2, 2)].squeeze(),
        torch.tensor(dm[2, 2], dtype=torch.float32),
    )
    assert snap.box is not None and torch.allclose(snap.box, torch.tensor(box))
    assert snap.forces is not None and torch.allclose(snap.forces, torch.tensor(forces))
    assert snap.stress is not None and torch.allclose(snap.stress, torch.tensor(stress))
    assert Path(str(snap.matrix_path)).name == "sample.npz"
    assert Path(str(snap.info_path)).name == "sample.json"

    snap2 = load_pyscf_snapshot(npz_path, json_path=json_path)
    assert torch.isclose(
        snap2.hamiltonian[(1, 0, 0, 0, 1)].squeeze(),
        torch.tensor(ham[0, 1] + 10.0, dtype=torch.float32),
    )


@pytest.mark.unit
def test_load_uses_fock_key_when_hamiltonian_missing(tmp_path: Path):
    npz_path, json_path, ham, _, _, _, _, _ = _write_artifacts(
        tmp_path,
        matrix_key="fock_ao",
        with_shifts=True,
    )
    data = load_pyscf_baseline(npz_path, json_path=json_path)
    assert torch.allclose(data.hamiltonian_ao, torch.tensor(ham))


@pytest.mark.unit
def test_load_fails_on_ao_dimension_mismatch(tmp_path: Path):
    npz_path, json_path, *_ = _write_artifacts(
        tmp_path,
        orbital_set={"H": "1s", "O": "2s"},  # expected dim 1 + 1 + 2 = 4
        with_shifts=True,
    )
    with pytest.raises(ValueError, match="AO dimension mismatch"):
        _ = load_pyscf_baseline(npz_path, json_path=json_path)


@pytest.mark.unit
def test_shift_resolved_loading_builds_shifted_block_matrices(tmp_path: Path):
    npz_path, json_path, ham, _, _, _, _, _ = _write_artifacts(
        tmp_path, with_shifts=True
    )
    data = load_pyscf_baseline(npz_path, json_path=json_path)
    assert data.shifts is not None
    assert data.hamiltonian_shifted is not None
    assert data.hamiltonian_shifted.shape == (2, 3, 3)

    snap = data.to_snapshot()
    # The loader closes the graph under Hermitian reverse edges, so the
    # H-H key contains shift 0 plus both +/- periodic directions.
    assert snap.hamiltonian["H-H"].shape[0] == 12
    # Direct edge lookup must respect periodic shift and its synthesized reverse.
    blk_shift0 = snap.hamiltonian[(0, 0, 0, 0, 1)]
    blk_shift1 = snap.hamiltonian[(1, 0, 0, 0, 1)]
    blk_shiftm1 = snap.hamiltonian[(-1, 0, 0, 1, 0)]
    assert torch.isclose(
        blk_shift0.squeeze(), torch.tensor(ham[0, 1], dtype=torch.float32)
    )
    assert torch.isclose(
        blk_shift1.squeeze(), torch.tensor(ham[0, 1] + 10.0, dtype=torch.float32)
    )
    assert torch.isclose(blk_shiftm1.squeeze(), blk_shift1.squeeze())


@pytest.mark.unit
def test_load_fails_when_shift_payload_is_missing(tmp_path: Path):
    npz_path, json_path, *_ = _write_artifacts(tmp_path, with_shifts=False)
    with pytest.raises(ValueError, match="requires `shifts`"):
        _ = load_pyscf_baseline(npz_path, json_path=json_path)


@pytest.mark.unit
def test_load_falls_back_to_json_for_box_forces_and_stress(tmp_path: Path):
    npz_path, json_path, _, _, _, box, forces, stress = _write_artifacts(
        tmp_path,
        with_shifts=True,
        include_observables_in_npz=False,
    )
    data = load_pyscf_baseline(npz_path, json_path=json_path)
    assert data.box is not None and torch.allclose(data.box, torch.tensor(box))
    assert data.forces is not None and torch.allclose(data.forces, torch.tensor(forces))
    assert data.stress is not None and torch.allclose(data.stress, torch.tensor(stress))
