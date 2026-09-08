from collections import Counter

import pytest
import torch

from core.orbital_irrep_config import OrbitalIrrepConfig
from data.block_matrix import BlockMatrix
from data.openmx_parser import parse_openmx_scfout
from data.openmx_writer import write_openmx_hsout_prediction


def _matrix(value: float, orbital_cfg: OrbitalIrrepConfig) -> BlockMatrix:
    edges = torch.tensor([[0], [0], [0], [0], [0]], dtype=torch.long)
    return BlockMatrix(
        atoms=("Si",),
        atom_counts=Counter({"Si": 1}),
        pair_blocks={"Si-Si": torch.tensor([[[value]]])},
        pair_edges={"Si-Si": edges},
        lookup={(0, 0, 0, 0, 0): ("Si-Si", 0)},
        orbital_cfg=orbital_cfg,
        basis="openmx",
    )


@pytest.mark.unit
def test_write_openmx_hsout_prediction_preserves_layout_and_density_scale(tmp_path):
    template = tmp_path / "HS.out"
    output = tmp_path / "HS_pred.out"
    template.write_text(
        "preamble\n"
        "Kohn-Sham Hamiltonian spin=0\n"
        "global index=1  local index=0 (global=1, Rn=0)\n"
        "  1.0000000000000000 \n"
        "Overlap matrix\n"
        "global index=1  local index=0 (global=1, Rn=0)\n"
        "  1.0000000000000000 \n"
        "Overlap matrix with position operator x\n"
        "global index=1  local index=0 (global=1, Rn=0 0 0 0)\n"
        "  9.0000000000000000 \n"
        "Density matrix spin=0\n"
        "global index=1  local index=0 (global=1, Rn=0)\n"
        "  1.0000000000000000 \n"
        "Density matrix spin=0\n"
        "global index=1  local index=0 (global=1, Rn=0)\n"
        "  7.0000000000000000 \n",
        encoding="utf-8",
    )
    orbital_cfg = OrbitalIrrepConfig.from_dict({"Si": "1s"})
    predictions = {
        "hamiltonian": _matrix(2.0, orbital_cfg),
        "overlap": _matrix(3.0, orbital_cfg),
        "density": _matrix(4.0, orbital_cfg),
    }

    stats = write_openmx_hsout_prediction(template, output, predictions)

    template_headers = [
        line for line in template.read_text().splitlines() if not line.startswith(" ")
    ]
    output_headers = [
        line for line in output.read_text().splitlines() if not line.startswith(" ")
    ]
    assert output_headers == template_headers
    assert output.stat().st_mode & 0o777 == template.stat().st_mode & 0o777
    assert output.read_text().count("Density matrix spin=0") == 2
    assert stats.blocks_written == {"hamiltonian": 1, "overlap": 1, "density": 1}
    assert stats.zero_filled_blocks == {
        "hamiltonian": 0,
        "overlap": 0,
        "density": 0,
    }

    loaded = parse_openmx_scfout(
        output,
        atoms=["Si"],
        orbital_cfg=orbital_cfg,
        convention="openmx",
        symmetrize_density=True,
    )
    assert loaded.hamiltonian[0, 0].item() == pytest.approx(2.0)
    assert loaded.overlap[0, 0].item() == pytest.approx(3.0)
    assert loaded.density[0, 0].item() == pytest.approx(4.0)
    raw = parse_openmx_scfout(
        output,
        atoms=["Si"],
        orbital_cfg=orbital_cfg,
        convention="openmx",
        symmetrize_density=False,
    )
    assert raw.density[0, 0].item() == pytest.approx(4.0)
    assert output.read_text().splitlines()[-1].strip() == "0.0000000000000000"
