import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch
from ase import Atoms
from ase.io import write

from core.block_irrep_mapper import BlockIrrepMapper
from core.orbital_irrep_config import OrbitalIrrepConfig
from core.sparse_math import build_trace_alignment_from_pair_edges
from data.openmx_parser import parse_openmx_scfout
from data.openmx_writer import (
    build_openmx_translation_table,
    export_model_openmx,
    openmx_cutoff_layout,
    write_openmx_from_structure,
)
from data.structure_inference import build_model_input_from_structure
from net.common import Config
from net.e3gnn import E3GNN


@pytest.mark.parametrize("n", [0, 1, 2, 3])
def test_translation_tables_are_complete_inverses(n):
    table = build_openmx_translation_table(n)
    assert table.tcpy_cell == (2 * n + 1) ** 3 - 1
    assert table.ratv.shape == (2 * n + 1,) * 3
    assert table.rn((0, 0, 0)) == 0
    assert len({tuple(row[1:]) for row in table.atv_ijk}) == table.tcpy_cell + 1
    for rn, row in enumerate(table.atv_ijk):
        assert table.rn(tuple(row[1:])) == rn
    with pytest.raises(ValueError, match="within CpyCell"):
        table.rn((n + 1, 0, 0))
    with pytest.raises(ValueError):
        table.rn((0.5, 0, 0))


def test_generation_atv_known_openmx_order():
    table = build_openmx_translation_table(1)
    expected = [
        (0, 0, 0),
        (-1, -1, -1),
        (-1, -1, 0),
        (-1, -1, 1),
        (-1, 0, -1),
        (-1, 0, 0),
        (-1, 0, 1),
        (-1, 1, -1),
        (-1, 1, 0),
        (-1, 1, 1),
        (0, -1, -1),
        (0, -1, 0),
        (0, -1, 1),
        (0, 0, -1),
        (0, 0, 1),
        (0, 1, -1),
        (0, 1, 0),
        (0, 1, 1),
        (1, -1, -1),
        (1, -1, 0),
        (1, -1, 1),
        (1, 0, -1),
        (1, 0, 0),
        (1, 0, 1),
        (1, 1, -1),
        (1, 1, 0),
        (1, 1, 1),
    ]
    assert table.atv_ijk[:, 1:].tolist() == [list(s) for s in expected]
    assert table.rn((-1, 0, 0)) == 5
    table2 = build_openmx_translation_table(2)
    assert table2.rn((-1, 0, 0)) == 38
    assert table2.rn((-2, -2, -2)) == 1
    assert table2.rn((2, 2, 2)) == 124


def test_skew_lattice_and_unwrapped_atoms_match_bruteforce():
    lattice = np.array([[2.0, 0.0, 0.0], [1.9, 0.4, 0.0], [0.0, 0.0, 5.0]])
    positions = np.array([[0.0, 0.0, 0.0], [4.2, 0.2, 0.0]])
    table, edges = openmx_cutoff_layout(("H", "H"), positions, lattice, 0.9)
    expected = set()
    for i in range(2):
        for j in range(2):
            for a in range(-6, 7):
                for b in range(-6, 7):
                    for c in range(-1, 2):
                        if (
                            np.linalg.norm(
                                positions[j]
                                - positions[i]
                                + np.array([a, b, c]) @ lattice
                            )
                            < 0.9
                        ):
                            expected.add((a, b, c, i, j))
    assert set(edges) == expected
    assert table.cpy_cell == max(abs(n) for edge in expected for n in edge[:3])
    assert table.cpy_cell > 1  # cutoff/shortest lattice vector would miss these.
    for i in range(2):
        assert next(edge for edge in edges if edge[3] == i) == (0, 0, 0, i, i)


@pytest.mark.parametrize("cutoff", [0.0, -1.0, float("nan"), float("inf")])
def test_invalid_cutoff_rejected(cutoff):
    with pytest.raises(ValueError, match="cutoff"):
        openmx_cutoff_layout(("H",), [[0.0, 0.0, 0.0]], np.eye(3), cutoff)


@pytest.fixture
def model_structure():
    orbital_cfg = OrbitalIrrepConfig.from_dict({"H": "1s", "O": "1s1p"})
    cfg = Config(
        matrix_targets=["hamiltonian", "overlap", "density"],
        cutoff_radius=2.1,
        hidden_base_dim=2,
        l_max=1,
        n_radial=4,
        radial_layers=[4],
        num_layers_gnn=1,
        neck_depth=1,
        internal_e3mlp_layers=1,
        head_e3mlp_layers=1,
        verbosity=0,
    )
    torch.manual_seed(37)
    model = E3GNN(BlockIrrepMapper(orbital_cfg, dtype=cfg.dtype), cfg)
    atoms = ("H", "O")
    positions = torch.tensor([[0.0, 0.0, 0.0], [0.8, 0.2, 0.1]])
    lattice = torch.diag(torch.tensor([2.0, 4.0, 5.0]))
    return model, atoms, positions, lattice


def test_model_only_export_roundtrip_and_no_density_rescale(model_structure, tmp_path):
    model, atoms, positions, lattice = model_structure
    output = tmp_path / "standalone.matrix"
    stats = export_model_openmx(
        model, atoms=atoms, positions=positions, lattice=lattice, output_path=output
    )
    assert model.training  # original mode restored
    text = output.read_text()
    assert "CpyCell=1\nTCpyCell=26" in text
    assert "Overlap matrix with position" not in text
    assert "Overlap matrix with momentum" not in text
    assert "local index=0 (global=1, Rn=0)" in text
    loaded = parse_openmx_scfout(
        output, atoms, model.mapper.orbital_cfg, convention="e3nn"
    )
    cob = torch.eye(3)[[2, 0, 1]]
    x = build_model_input_from_structure(
        atoms=atoms,
        positions=positions @ cob,
        box=lattice @ cob,
        cfg=model.cfg,
        mapper=model.mapper,
    )
    model.eval()
    with torch.no_grad():
        expected = model.predict_matrices(x, physical=True)
    for name, matrix in expected.items():
        matrix = matrix.symmetrize_aligned(
            build_trace_alignment_from_pair_edges(matrix.pair_edges)
        )
        actual = getattr(loaded, name)
        assert set(actual.lookup) == set(matrix.lookup)
        assert stats.blocks_written[name] == len(matrix.lookup)
        for edge, (key, index) in matrix.lookup.items():
            ak, ai = actual.lookup[edge]
            torch.testing.assert_close(
                actual.pair_blocks[ak][ai],
                matrix.pair_blocks[key][index],
                atol=1e-6,
                rtol=1e-5,
            )
    # Invalid support or NaNs fail before replacing an existing output.
    before = output.read_bytes()
    matrix = expected["density"]
    key = next(iter(matrix.pair_blocks))
    matrix.pair_blocks[key][0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        write_openmx_from_structure(
            output,
            {"density": matrix},
            atoms=atoms,
            positions=positions,
            lattice=lattice,
            cutoff_radius=model.cfg.cutoff_radius,
        )
    assert output.read_bytes() == before


def test_molecular_subset_export_and_missing_support(model_structure, tmp_path):
    model, atoms, positions, _ = model_structure
    output = tmp_path / "molecule.matrix"
    export_model_openmx(
        model, atoms=atoms, positions=positions, lattice=None, output_path=output
    )
    loaded = parse_openmx_scfout(
        output, atoms, model.mapper.orbital_cfg, convention="openmx"
    )
    density = loaded.density
    write_openmx_from_structure(
        output,
        {"density": density},
        atoms=atoms,
        positions=positions,
        lattice=None,
        cutoff_radius=model.cfg.cutoff_radius,
    )
    text = output.read_text()
    assert "CpyCell=0\nTCpyCell=0" in text
    assert "Density matrix spin=0" in text
    assert "Kohn-Sham Hamiltonian" not in text
    assert "Overlap matrix" not in text
    before = output.read_bytes()
    with pytest.raises(ValueError, match="outside-cutoff"):
        write_openmx_from_structure(
            output,
            {"density": density},
            atoms=atoms,
            positions=positions,
            lattice=None,
            cutoff_radius=0.1,
        )
    assert output.read_bytes() == before


def test_checkpoint_xyz_cli_without_reference_files(model_structure, tmp_path):
    model, atoms, positions, lattice = model_structure
    assert "orbital_cfg" in model.hparams
    checkpoint = tmp_path / "model.pt"
    torch.save(
        {"hyper_parameters": dict(model.hparams), "state_dict": model.state_dict()},
        checkpoint,
    )
    structure = tmp_path / "input.xyz"
    write(
        structure,
        Atoms(
            symbols=atoms, positions=positions.numpy(), cell=lattice.numpy(), pbc=True
        ),
        format="extxyz",
    )
    script = Path(__file__).resolve().parents[3] / "scripts/export_prediction.py"
    spec = importlib.util.spec_from_file_location("standalone_export_cli", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    output = tmp_path / "output.matrix"
    module.main(
        [
            "--checkpoint",
            str(checkpoint),
            "--structure",
            str(structure),
            "--output",
            str(output),
            "--device",
            "cpu",
        ]
    )
    loaded = parse_openmx_scfout(
        output, atoms, model.mapper.orbital_cfg, convention="openmx"
    )
    assert all(
        torch.isfinite(block).all() for block in loaded.density.pair_blocks.values()
    )


@pytest.mark.parametrize(
    "obsolete_args",
    [
        ["--matrix-path", "reference.matrix"],
        ["--info-path", "reference.out"],
        ["--missing-edges", "zero"],
        ["--use-info-geometry"],
    ],
)
def test_cli_rejects_removed_template_options(obsolete_args, capsys):
    script = Path(__file__).resolve().parents[3] / "scripts/export_prediction.py"
    spec = importlib.util.spec_from_file_location("openmx_export_cli", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    with pytest.raises(SystemExit) as exc:
        module.main(
            [
                "--checkpoint",
                "model.pt",
                "--structure",
                "input.xyz",
                "--output",
                "output.matrix",
                *obsolete_args,
            ]
        )
    assert exc.value.code == 2
    assert "unrecognized arguments" in capsys.readouterr().err
