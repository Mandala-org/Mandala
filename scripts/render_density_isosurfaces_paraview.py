#!/usr/bin/env pvpython
"""Render comparable GT, prediction, and signed-error density isosurfaces."""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import subprocess

import numpy as np
import vtk  # type: ignore[import-not-found]
from paraview.simple import (  # type: ignore[import-not-found]
    CreateRenderView,
    GaussianCubeReader,
    Hide,
    OutputPort,
    Render,
    ResetSession,
    SaveScreenshot,
    Show,
    Text,
    TrivialProducer,
)
from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy  # type: ignore[import-not-found]


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
    parser.add_argument("--density-isovalues", type=_parse_levels, default=[0.01, 0.03])
    parser.add_argument("--error-isovalue", type=float, default=5.0e-5)
    parser.add_argument("--repeat", type=int, default=2)
    parser.add_argument("--width", type=int, default=1600)
    parser.add_argument("--height", type=int, default=1200)
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
    view.MultiSamples = 8
    return view


def _volume_data(cube_path: Path):
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
    volume = reader.GetClientSideObject().GetOutputDataObject(1)
    if not volume.IsA("vtkImageData"):
        raise RuntimeError(
            f"Expected vtkImageData from {cube_path}, found {volume.GetClassName()}"
        )
    return reader, volume, array.GetName(), array.GetComponentRange(0)


def _tile_volume(volume, array_name: str, repeat: int):
    """Build a periodic 3-D image with one guard cell around the visible volume."""
    if repeat < 1:
        raise ValueError("--repeat must be at least 1")
    dimensions = volume.GetDimensions()
    spacing = volume.GetSpacing()
    origin = volume.GetOrigin()
    source_values = vtk_to_numpy(volume.GetPointData().GetArray(array_name))
    source_values = source_values.reshape(dimensions[2], dimensions[1], dimensions[0])

    # One guard cell on either side allows periodic contours to close before cropping.
    work_repeat = repeat + 2
    tiled_values = np.tile(source_values, (work_repeat, work_repeat, work_repeat))
    windows = []
    for size in dimensions[::-1]:
        indices = np.arange(size * work_repeat, dtype=float)
        visible_start = float(size)
        visible_stop = float(size * (repeat + 1))
        inside_distance = np.minimum(
            indices - visible_start,
            visible_stop - indices,
        )
        phase = np.clip(inside_distance / (0.08 * size), 0.0, 1.0)
        windows.append(0.5 * (1.0 - np.cos(np.pi * phase)))
    tiled_values *= (
        windows[0][:, None, None]
        * windows[1][None, :, None]
        * windows[2][None, None, :]
    )
    tiled = vtk.vtkImageData()
    tiled.SetDimensions(
        dimensions[0] * work_repeat,
        dimensions[1] * work_repeat,
        dimensions[2] * work_repeat,
    )
    tiled.SetSpacing(spacing)
    tiled.SetOrigin(
        origin[0] - dimensions[0] * spacing[0],
        origin[1] - dimensions[1] * spacing[1],
        origin[2] - dimensions[2] * spacing[2],
    )
    values = numpy_to_vtk(tiled_values.ravel(), deep=True)
    values.SetName(array_name)
    tiled.GetPointData().SetScalars(values)

    visible_bounds = tuple(
        coordinate
        for axis in range(3)
        for coordinate in (
            origin[axis],
            origin[axis] + dimensions[axis] * spacing[axis] * repeat,
        )
    )
    return tiled, visible_bounds, dimensions, spacing, origin


def _contour_with_boundary_taper(
    volume,
    array_name: str,
    value: float,
    _visible_bounds,
):
    contour = vtk.vtkContourFilter()
    contour.SetInputData(volume)
    contour.SetInputArrayToProcess(0, 0, 0, 0, array_name)
    contour.SetValue(0, float(value))
    contour.ComputeNormalsOn()
    contour.Update()

    output = vtk.vtkPolyData()
    output.ShallowCopy(contour.GetOutput())
    output.GetPointData().Initialize()
    output.GetCellData().Initialize()
    return output


def _producer_for(data):
    producer = TrivialProducer()
    producer.GetClientSideObject().SetOutput(data)
    producer.UpdatePipeline()
    return producer


def _set_camera(view, bounds) -> None:
    center = [(bounds[2 * axis] + bounds[2 * axis + 1]) * 0.5 for axis in range(3)]
    spans = [bounds[2 * axis + 1] - bounds[2 * axis] for axis in range(3)]
    diagonal = sum(span * span for span in spans) ** 0.5
    view.CameraFocalPoint = center
    view.CameraPosition = [
        center[0] + 1.28 * diagonal,
        center[1] + 0.82 * diagonal,
        center[2] + 0.62 * diagonal,
    ]
    view.CameraViewUp = [-0.18, -0.22, 0.96]
    view.CameraParallelScale = 0.61 * diagonal


def _cell_grid(bounds, dimensions, spacing, origin, repeat: int):
    points = vtk.vtkPoints()
    lines = vtk.vtkCellArray()
    cell_lengths = [dimensions[axis] * spacing[axis] for axis in range(3)]
    ranges = [range(repeat + 1) for _ in range(3)]

    def add_line(start, end) -> None:
        line = vtk.vtkLine()
        line.GetPointIds().SetId(0, points.InsertNextPoint(start))
        line.GetPointIds().SetId(1, points.InsertNextPoint(end))
        lines.InsertNextCell(line)

    for ix in ranges[0]:
        for iy in ranges[1]:
            x = origin[0] + ix * cell_lengths[0]
            y = origin[1] + iy * cell_lengths[1]
            add_line((x, y, bounds[4]), (x, y, bounds[5]))
    for ix in ranges[0]:
        for iz in ranges[2]:
            x = origin[0] + ix * cell_lengths[0]
            z = origin[2] + iz * cell_lengths[2]
            add_line((x, bounds[2], z), (x, bounds[3], z))
    for iy in ranges[1]:
        for iz in ranges[2]:
            y = origin[1] + iy * cell_lengths[1]
            z = origin[2] + iz * cell_lengths[2]
            add_line((bounds[0], y, z), (bounds[1], y, z))
    polydata = vtk.vtkPolyData()
    polydata.SetPoints(points)
    polydata.SetLines(lines)
    return polydata


def _add_cell_outline(cell_grid, view) -> None:
    display = Show(_producer_for(cell_grid), view)
    display.LineWidth = 3.0
    display.RenderLinesAsTubes = 1
    display.AmbientColor = [0.12, 0.14, 0.17]
    display.DiffuseColor = [0.12, 0.14, 0.17]


def _add_contour(polydata, view, color, opacity: float) -> None:
    display = Show(_producer_for(polydata), view)
    display.Representation = "Surface"
    display.DiffuseColor = color
    display.AmbientColor = color
    display.Opacity = opacity
    display.Specular = 0.28
    display.SpecularPower = 24.0
    display.Diffuse = 0.82
    display.Ambient = 0.18


def _add_title(view, title: str, subtitle: str) -> None:
    text = Text()
    text.Text = f"{title}\n{subtitle}"
    display = Show(text, view)
    display.WindowLocation = "Upper Center"
    display.FontSize = 30
    display.Color = [0.08, 0.10, 0.13]
    display.Bold = 1
    display.Shadow = 1


def _render_density(
    cube_path: Path,
    output_path: Path,
    *,
    title: str,
    levels: list[float],
    repeat: int,
    width: int,
    height: int,
) -> None:
    reader, volume, array_name, scalar_range = _volume_data(cube_path)
    if levels[-1] >= scalar_range[1]:
        raise ValueError(
            f"Largest density isovalue {levels[-1]:.6g} is outside {cube_path} "
            f"range [{scalar_range[0]:.6g}, {scalar_range[1]:.6g}]"
        )
    view = _new_view(width, height)
    Hide(reader, view)
    tiled, bounds, dimensions, spacing, origin = _tile_volume(
        volume, array_name, repeat
    )
    cell_grid = _cell_grid(bounds, dimensions, spacing, origin, repeat)
    colors = ([0.39, 0.76, 0.72], [0.18, 0.50, 0.76], [0.95, 0.55, 0.18])
    opacities = (0.20, 0.72, 0.88)
    for index, level in enumerate(levels):
        polydata = _contour_with_boundary_taper(
            tiled,
            array_name,
            level,
            bounds,
        )
        _add_contour(
            polydata,
            view,
            colors[min(index, len(colors) - 1)],
            opacities[min(index, len(opacities) - 1)],
        )
    _add_cell_outline(cell_grid, view)
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
    repeat: int,
    width: int,
    height: int,
) -> None:
    reader, volume, array_name, scalar_range = _volume_data(cube_path)
    if not (scalar_range[0] < -isovalue and scalar_range[1] > isovalue):
        raise ValueError(
            f"Signed error range {scalar_range} does not span +/-{isovalue:.6g}"
        )
    view = _new_view(width, height)
    Hide(reader, view)
    tiled, bounds, dimensions, spacing, origin = _tile_volume(
        volume, array_name, repeat
    )
    negative = _contour_with_boundary_taper(tiled, array_name, -isovalue, bounds)
    positive = _contour_with_boundary_taper(tiled, array_name, isovalue, bounds)
    _add_contour(negative, view, [0.12, 0.35, 0.82], 0.88)
    _add_contour(positive, view, [0.84, 0.16, 0.16], 0.88)
    _add_cell_outline(_cell_grid(bounds, dimensions, spacing, origin, repeat), view)
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
            "--repeat",
            str(args.repeat),
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
            repeat=args.repeat,
            width=args.width,
            height=args.height,
        )
    elif args.only == "prediction":
        _render_density(
            args.predicted_cube,
            args.output_dir / "predicted_density_isosurfaces.png",
            title="Predicted density",
            levels=args.density_isovalues,
            repeat=args.repeat,
            width=args.width,
            height=args.height,
        )
    else:
        _render_error(
            args.error_cube,
            args.output_dir / "signed_density_error_isosurfaces.png",
            isovalue=args.error_isovalue,
            repeat=args.repeat,
            width=args.width,
            height=args.height,
        )
    print(f"Wrote ParaView renders to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
