from __future__ import annotations

from ase import Atoms
from ase.io import write

from scripts.visualize_crystal import build_crystal_figure


def test_crystal_visualization_uses_shaded_spheres_and_cell_edges(tmp_path) -> None:
    structure = Atoms(
        "Si2",
        scaled_positions=[(0.0, 0.0, 0.0), (0.25, 0.25, 0.25)],
        cell=[5.43, 5.43, 5.43],
        pbc=True,
    )
    input_path = tmp_path / "silicon.cif"
    write(input_path, structure)

    figure = build_crystal_figure(input_path, title="Silicon")

    assert figure.data[0].type == "mesh3d"
    assert figure.data[0].name == "Si"
    assert figure.data[0].flatshading is False
    assert figure.data[0].lighting.diffuse > 0.0
    assert figure.data[-1].type == "scatter3d"
    assert figure.data[-1].name == "Simulation cell"
    assert figure.layout.scene.aspectmode == "data"
