#!/usr/bin/env python3
"""Render an ASE-readable structure as shaded atoms inside its simulation cell."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
import shutil
import subprocess

import numpy as np
import plotly.graph_objects as go
from ase.data import covalent_radii
from ase.data.colors import jmol_colors
from ase.io import read


def _sphere_mesh(
    center: np.ndarray,
    radius: float,
    *,
    longitude_count: int,
    latitude_count: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return vertices and triangular faces for one UV sphere."""
    longitude = np.linspace(0.0, 2.0 * np.pi, longitude_count, endpoint=False)
    latitude = np.linspace(0.0, np.pi, latitude_count)
    lon_grid, lat_grid = np.meshgrid(longitude, latitude)
    vertices = np.column_stack(
        (
            np.sin(lat_grid).ravel() * np.cos(lon_grid).ravel(),
            np.sin(lat_grid).ravel() * np.sin(lon_grid).ravel(),
            np.cos(lat_grid).ravel(),
        )
    )
    normals = vertices.copy()
    vertices = center[None, :] + radius * vertices

    faces: list[tuple[int, int, int]] = []
    for lat_idx in range(latitude_count - 1):
        row = lat_idx * longitude_count
        next_row = (lat_idx + 1) * longitude_count
        for lon_idx in range(longitude_count):
            next_lon = (lon_idx + 1) % longitude_count
            faces.append((row + lon_idx, next_row + lon_idx, next_row + next_lon))
            faces.append((row + lon_idx, next_row + next_lon, row + next_lon))
    return vertices, np.asarray(faces, dtype=np.int32), normals


def _element_mesh(
    centers: list[np.ndarray],
    radii: list[float],
    *,
    longitude_count: int,
    latitude_count: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    vertices: list[np.ndarray] = []
    faces: list[np.ndarray] = []
    normals: list[np.ndarray] = []
    vertex_offset = 0
    for center, radius in zip(centers, radii, strict=True):
        sphere_vertices, sphere_faces, sphere_normals = _sphere_mesh(
            center,
            radius,
            longitude_count=longitude_count,
            latitude_count=latitude_count,
        )
        vertices.append(sphere_vertices)
        faces.append(sphere_faces + vertex_offset)
        normals.append(sphere_normals)
        vertex_offset += sphere_vertices.shape[0]
    return np.concatenate(vertices), np.concatenate(faces), np.concatenate(normals)


def _studio_vertex_colors(base_color: np.ndarray, normals: np.ndarray) -> list[str]:
    """Bake a soft three-light studio rig into RGB vertex colors."""
    light_directions = np.asarray(
        [
            [0.45, 0.70, 0.55],  # broad key light
            [-0.75, 0.30, 0.45],  # cooler fill light
            [0.10, -0.65, 0.75],  # top/back rim light
        ],
        dtype=float,
    )
    light_directions /= np.linalg.norm(light_directions, axis=1, keepdims=True)
    strengths = np.asarray([0.58, 0.25, 0.22])
    diffuse = np.maximum(normals @ light_directions.T, 0.0) @ strengths
    illumination = np.clip(0.38 + diffuse, 0.28, 1.18)

    rgb = np.clip(base_color[None, :] * illumination[:, None], 0.0, 255.0)
    # A gentle cool fill keeps shadowed sides readable without washing out colors.
    rgb += np.maximum(0.0, 0.55 - illumination[:, None]) * np.asarray([12, 16, 24])
    rgb = np.clip(rgb, 0.0, 255.0).astype(np.uint8)
    return [f"rgb({r},{g},{b})" for r, g, b in rgb]


def _cell_segments(cell: np.ndarray) -> np.ndarray:
    corners = np.asarray(
        [
            np.zeros(3),
            cell[0],
            cell[1],
            cell[2],
            cell[0] + cell[1],
            cell[0] + cell[2],
            cell[1] + cell[2],
            cell[0] + cell[1] + cell[2],
        ]
    )
    edges = (
        (0, 1),
        (0, 2),
        (0, 3),
        (1, 4),
        (1, 5),
        (2, 4),
        (2, 6),
        (3, 5),
        (3, 6),
        (4, 7),
        (5, 7),
        (6, 7),
    )
    points: list[np.ndarray] = []
    nan_point = np.full(3, np.nan)
    for start, stop in edges:
        points.extend((corners[start], corners[stop], nan_point))
    return np.asarray(points)


def build_crystal_figure(
    input_path: Path,
    *,
    title: str | None = None,
    radius_scale: float = 0.55,
    sphere_resolution: int = 18,
) -> go.Figure:
    """Build a Plotly figure for any periodic structure readable by ASE."""
    atoms = read(input_path)
    if not np.any(atoms.pbc):
        raise ValueError(f"Structure has no periodic cell: {input_path}")
    atoms.wrap()
    cell = np.asarray(atoms.cell, dtype=float)
    positions = np.asarray(atoms.positions, dtype=float)
    atomic_numbers = np.asarray(atoms.numbers, dtype=int)

    centers_by_element: dict[int, list[np.ndarray]] = defaultdict(list)
    radii_by_element: dict[int, list[float]] = defaultdict(list)
    for center, atomic_number in zip(positions, atomic_numbers, strict=True):
        radius = max(float(covalent_radii[atomic_number]) * radius_scale, 0.25)
        centers_by_element[int(atomic_number)].append(center)
        radii_by_element[int(atomic_number)].append(radius)

    figure = go.Figure()
    symbols = atoms.get_chemical_symbols()
    for atomic_number in sorted(centers_by_element):
        vertices, faces, normals = _element_mesh(
            centers_by_element[atomic_number],
            radii_by_element[atomic_number],
            longitude_count=max(8, sphere_resolution),
            latitude_count=max(6, sphere_resolution // 2 + 1),
        )
        color = np.clip(jmol_colors[atomic_number] * 255.0, 0, 255).astype(int)
        element = symbols[int(np.flatnonzero(atomic_numbers == atomic_number)[0])]
        figure.add_trace(
            go.Mesh3d(
                x=vertices[:, 0],
                y=vertices[:, 1],
                z=vertices[:, 2],
                i=faces[:, 0],
                j=faces[:, 1],
                k=faces[:, 2],
                vertexcolor=_studio_vertex_colors(color, normals),
                name=element,
                showlegend=True,
                hoverinfo="name",
                flatshading=False,
                lighting={
                    "ambient": 0.62,
                    "diffuse": 0.48,
                    "specular": 0.52,
                    "roughness": 0.24,
                    "fresnel": 0.12,
                },
                lightposition={"x": 140, "y": 220, "z": 260},
            )
        )

    segments = _cell_segments(cell)
    figure.add_trace(
        go.Scatter3d(
            x=segments[:, 0],
            y=segments[:, 1],
            z=segments[:, 2],
            mode="lines",
            line={"color": "#30343B", "width": 5},
            name="Simulation cell",
            hoverinfo="skip",
            showlegend=False,
        )
    )

    camera_distance = 1.18
    axis_style = {
        "visible": False,
        "showbackground": False,
        "showgrid": False,
        "zeroline": False,
    }
    figure.update_layout(
        title=title,
        template="plotly_white",
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        margin={"l": 0, "r": 0, "t": 50 if title else 0, "b": 0},
        legend={
            "x": 0.02,
            "y": 0.98,
            "bgcolor": "rgba(255,255,255,0.78)",
            "bordercolor": "rgba(0,0,0,0.18)",
            "borderwidth": 1,
        },
        scene={
            "xaxis": axis_style,
            "yaxis": axis_style,
            "zaxis": axis_style,
            "aspectmode": "data",
            "camera": {
                "eye": {
                    "x": camera_distance,
                    "y": camera_distance,
                    "z": 0.9 * camera_distance,
                },
                "center": {"x": 0.0, "y": 0.0, "z": 0.0},
                "projection": {"type": "perspective"},
            },
            "bgcolor": "#FFFFFF",
        },
    )
    return figure


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input", type=Path, help="Any periodic structure readable by ASE"
    )
    parser.add_argument("--output-html", type=Path, required=True)
    parser.add_argument("--output-png", type=Path)
    parser.add_argument("--title")
    parser.add_argument("--radius-scale", type=float, default=0.55)
    parser.add_argument("--sphere-resolution", type=int, default=36)
    parser.add_argument("--width", type=int, default=1200)
    parser.add_argument("--height", type=int, default=900)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    figure = build_crystal_figure(
        args.input,
        title=args.title,
        radius_scale=args.radius_scale,
        sphere_resolution=args.sphere_resolution,
    )
    args.output_html.parent.mkdir(parents=True, exist_ok=True)
    figure.write_html(
        args.output_html,
        include_plotlyjs=True,
        full_html=True,
        config={"responsive": True, "displaylogo": False, "displayModeBar": False},
    )
    if args.output_png is not None:
        args.output_png.parent.mkdir(parents=True, exist_ok=True)
        try:
            figure.write_image(
                args.output_png, width=args.width, height=args.height, scale=2
            )
        except Exception as exc:
            chrome = shutil.which("google-chrome") or shutil.which("chromium")
            if chrome is None:
                raise RuntimeError(
                    "PNG export requires Plotly's Kaleido package or a Chrome/Chromium "
                    "executable; the HTML output was written."
                ) from exc
            command = [
                chrome,
                "--headless=new",
                "--no-sandbox",
                "--disable-dev-shm-usage",
                "--hide-scrollbars",
                "--enable-webgl",
                "--use-angle=swiftshader",
                "--virtual-time-budget=5000",
                f"--window-size={args.width},{args.height}",
                f"--screenshot={args.output_png.resolve()}",
                args.output_html.resolve().as_uri(),
            ]
            completed = subprocess.run(command, capture_output=True, text=True)
            if completed.returncode != 0 or not args.output_png.is_file():
                raise RuntimeError(
                    "Both Kaleido and the headless Chrome PNG fallback failed. "
                    f"Chrome stderr: {completed.stderr.strip()}"
                ) from exc


if __name__ == "__main__":
    main()
