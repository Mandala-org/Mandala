from __future__ import annotations

from ase import Atoms
from ase.data import covalent_radii
from ase.io import write
import numpy as np

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
    vertices = np.column_stack([figure.data[0][axis] for axis in ("x", "y", "z")])
    # The two same-species spheres share one mesh. Check each sphere against
    # its atom position and radius, including separation of their vertices.
    spheres = vertices.reshape(2, -1, 3)
    for sphere, position in zip(spheres, structure.positions):
        np.testing.assert_allclose(
            np.linalg.norm(sphere - position, axis=1),
            covalent_radii[14] * 0.55,
            atol=1e-7,
        )
    cell = np.column_stack([figure.data[-1][axis] for axis in ("x", "y", "z")])
    assert cell.shape == (36, 3)
    assert np.isnan(cell.reshape(12, 3, 3)[:, 2]).all()
    actual_edges = {
        frozenset((tuple(segment[0]), tuple(segment[1])))
        for segment in cell.reshape(12, 3, 3)
    }
    # Enumerate the cube edges independently by binary corner coordinates.
    expected_edges = set()
    for bits in np.ndindex(2, 2, 2):
        for axis in range(3):
            if bits[axis] == 0:
                neighbor = list(bits)
                neighbor[axis] = 1
                expected_edges.add(
                    frozenset(
                        (tuple(np.array(bits) * 5.43), tuple(np.array(neighbor) * 5.43))
                    )
                )
    assert actual_edges == expected_edges
