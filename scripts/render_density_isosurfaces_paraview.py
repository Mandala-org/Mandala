#!/usr/bin/env pvpython
"""Render comparable GT, prediction, and signed-error density isosurfaces."""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import subprocess

from paraview.simple import (  # type: ignore[import-not-found]
    ColorBy,
    Contour,
    CreateRenderView,
    GaussianCubeReader,
    Hide,
    Outline,
    OutputPort,
    Render,
    ResetSession,
    SaveScreenshot,
    Show,
    Text,
)


def _parse_levels(value: str) -> list[float]:
    levels = [float(item) for item in value.split(",") if item.strip()]
    if not levels or any(level <= 0.0 for level in levels):
        raise argparse.ArgumentTypeError(
            "Isovalues must be positive comma-separated numbers"
        )
    return sorted(levels)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ground-truth-cube", type=Path, required=True)
    parser.add_argument("--predicted-cube", type=Path, required=True)
    parser.add_argument("--error-cube", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--density-isovalues", type=_parse_levels, default=[0.02, 0.05])
    parser.add_argument("--error-isovalue", type=float, default=1.0e-4)
    parser.add_argument("--width", type=int, default=1000)
    parser.add_argument("--height", type=int, default=900)
    parser.add_argument(
        "--only",
        choices=("all", "ground_truth", "prediction", "error"),
        default="all",
        help="Render one panel; 'all' isolates all three in child pvpython processes",
    )
    return parser.parse_args()


def _new_view(width: int, height: int):
    view = CreateRenderView()
    view.ViewSize = [width, height]
    view.Background = [1.0, 1.0, 1.0]
    view.UseColorPaletteForBackground = 0
    view.OrientationAxesVisibility = 0
    view.CenterAxesVisibility = 0
    view.CameraParallelProjection = 1
    return view


def _volume_port(cube_path: Path):
    reader = GaussianCubeReader(FileName=[str(cube_path.resolve())])
    reader.UpdatePipeline()
    grid = OutputPort(reader, 1)
    grid.UpdatePipeline()
    information = grid.GetDataInformation()
    if information.GetNumberOfPoints() == 0:
        raise RuntimeError(f"ParaView read no volume points from {cube_path}")
    point_data = information.GetPointDataInformation()
    if point_data.GetNumberOfArrays() != 1:
        raise RuntimeError(
            f"Expected one scalar array in {cube_path}, found {point_data.GetNumberOfArrays()}"
        )
    array = point_data.GetArrayInformation(0)
    return (
        reader,
        grid,
        array.GetName(),
        array.GetComponentRange(0),
        information.GetBounds(),
    )


def _set_camera(view, bounds) -> None:
    center = [(bounds[2 * axis] + bounds[2 * axis + 1]) * 0.5 for axis in range(3)]
    spans = [bounds[2 * axis + 1] - bounds[2 * axis] for axis in range(3)]
    diagonal = sum(span * span for span in spans) ** 0.5
    view.CameraFocalPoint = center
    view.CameraPosition = [
        center[0] + 1.45 * diagonal,
        center[1] + 1.45 * diagonal,
        center[2] + 1.10 * diagonal,
    ]
    view.CameraViewUp = [-0.30, -0.30, 0.90]
    view.CameraParallelScale = 0.64 * diagonal


def _add_cell_outline(grid, view) -> None:
    outline = Outline(Input=grid)
    display = Show(outline, view)
    display.LineWidth = 3.0
    display.AmbientColor = [0.12, 0.14, 0.17]
    display.DiffuseColor = [0.12, 0.14, 0.17]


def _add_contour(
    grid, view, array_name: str, value: float, color, opacity: float
) -> None:
    contour = Contour(Input=grid)
    contour.ContourBy = ["POINTS", array_name]
    contour.Isosurfaces = [float(value)]
    contour.PointMergeMethod = "Uniform Binning"
    display = Show(contour, view)
    ColorBy(display, None)
    display.Representation = "Surface"
    display.DiffuseColor = color
    display.AmbientColor = color
    display.Opacity = opacity
    display.Specular = 0.28
    display.SpecularPower = 24.0


def _add_title(view, title: str, subtitle: str) -> None:
    text = Text()
    text.Text = f"{title}\n{subtitle}"
    display = Show(text, view)
    display.WindowLocation = "Upper Center"
    display.FontSize = 22
    display.Color = [0.08, 0.10, 0.13]
    display.Bold = 1


def _render_density(
    cube_path: Path,
    output_path: Path,
    *,
    title: str,
    levels: list[float],
    width: int,
    height: int,
) -> None:
    reader, grid, array_name, scalar_range, bounds = _volume_port(cube_path)
    if levels[-1] >= scalar_range[1]:
        raise ValueError(
            f"Largest density isovalue {levels[-1]:.6g} is outside {cube_path} "
            f"range [{scalar_range[0]:.6g}, {scalar_range[1]:.6g}]"
        )
    view = _new_view(width, height)
    Hide(reader, view)
    colors = ([0.39, 0.76, 0.72], [0.03, 0.32, 0.61], [0.95, 0.55, 0.18])
    opacities = (0.35, 0.88, 0.95)
    for index, level in enumerate(levels):
        _add_contour(
            grid,
            view,
            array_name,
            level,
            colors[min(index, len(colors) - 1)],
            opacities[min(index, len(opacities) - 1)],
        )
    _add_cell_outline(grid, view)
    _add_title(
        view,
        title,
        "Isovalues: " + ", ".join(f"{level:g}" for level in levels) + " e/bohr^3",
    )
    # The first render initializes ParaView's camera from visible bounds.
    # Apply our shared camera only after that initialization.
    Render(view)
    _set_camera(view, bounds)
    Render(view)
    SaveScreenshot(str(output_path.resolve()), view, ImageResolution=[width, height])
    ResetSession()


def _render_error(
    cube_path: Path,
    output_path: Path,
    *,
    isovalue: float,
    width: int,
    height: int,
) -> None:
    reader, grid, array_name, scalar_range, bounds = _volume_port(cube_path)
    if not (scalar_range[0] < -isovalue and scalar_range[1] > isovalue):
        raise ValueError(
            f"Signed error range {scalar_range} does not span +/-{isovalue:.6g}"
        )
    view = _new_view(width, height)
    Hide(reader, view)
    _add_contour(grid, view, array_name, -isovalue, [0.12, 0.35, 0.82], 0.88)
    _add_contour(grid, view, array_name, isovalue, [0.84, 0.16, 0.16], 0.88)
    _add_cell_outline(grid, view)
    _add_title(
        view,
        "Signed density error",
        f"blue = -{isovalue:g}, red = +{isovalue:g} e/bohr^3",
    )
    Render(view)
    _set_camera(view, bounds)
    Render(view)
    SaveScreenshot(str(output_path.resolve()), view, ImageResolution=[width, height])
    ResetSession()


def main() -> None:
    args = _parse_args()
    if args.error_isovalue <= 0.0:
        raise ValueError("--error-isovalue must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.only == "all":
        pvpython = shutil.which("pvpython")
        if pvpython is None:
            raise RuntimeError("Could not locate pvpython for isolated panel rendering")
        base_command = [
            pvpython,
            str(Path(__file__).resolve()),
            "--ground-truth-cube",
            str(args.ground_truth_cube),
            "--predicted-cube",
            str(args.predicted_cube),
            "--error-cube",
            str(args.error_cube),
            "--output-dir",
            str(args.output_dir),
            "--density-isovalues",
            ",".join(str(value) for value in args.density_isovalues),
            "--error-isovalue",
            str(args.error_isovalue),
            "--width",
            str(args.width),
            "--height",
            str(args.height),
        ]
        for kind in ("ground_truth", "prediction", "error"):
            subprocess.run([*base_command, "--only", kind], check=True)
    elif args.only == "ground_truth":
        _render_density(
            args.ground_truth_cube,
            args.output_dir / "ground_truth_density_isosurfaces.png",
            title="Ground-truth density",
            levels=args.density_isovalues,
            width=args.width,
            height=args.height,
        )
    elif args.only == "prediction":
        _render_density(
            args.predicted_cube,
            args.output_dir / "predicted_density_isosurfaces.png",
            title="Predicted density",
            levels=args.density_isovalues,
            width=args.width,
            height=args.height,
        )
    else:
        _render_error(
            args.error_cube,
            args.output_dir / "signed_density_error_isosurfaces.png",
            isovalue=args.error_isovalue,
            width=args.width,
            height=args.height,
        )
    print(f"Wrote ParaView renders to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
