#!/usr/bin/env python
"""Reload ``HS_pred.out`` and validate its layout and predicted observables."""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import sys

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from data.snapshot import Snapshot  # noqa: E402
from predict_openmx_hsout import (  # noqa: E402
    DEFAULT_CHECKPOINT,
    DEFAULT_SNAPSHOT_DIR,
    _device,
    load_hsd_prediction,
)

_SECTION_RE = re.compile(
    r"^(Kohn-Sham Hamiltonian spin=0|Overlap matrix|Density matrix spin=0|"
    r"Overlap matrix with .+)$"
)
_BLOCK_HEADER_RE = re.compile(
    r"^global index=\d+\s+local index=\d+\s+"
    r"\(global=\d+,\s*Rn=-?\d+(?:\s+-?\d+\s+-?\d+\s+-?\d+)?\)$"
)


def _layout(path: Path) -> tuple[int, list[str], list[str]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    sections = [line for line in lines if _SECTION_RE.fullmatch(line)]
    blocks = [line for line in lines if _BLOCK_HEADER_RE.fullmatch(line)]
    return len(lines), sections, blocks


def _assert_only_hsd_values_changed(template: Path, prediction: Path) -> None:
    template_lines = template.read_text(encoding="utf-8").splitlines()
    prediction_lines = prediction.read_text(encoding="utf-8").splitlines()
    if len(template_lines) != len(prediction_lines):
        raise AssertionError("HS_pred.out line count differs from HS.out")

    replaceable_section = False
    for line_number, (original, predicted) in enumerate(
        zip(template_lines, prediction_lines), start=1
    ):
        if original in {
            "Kohn-Sham Hamiltonian spin=0",
            "Overlap matrix",
            "Density matrix spin=0",
        }:
            replaceable_section = True
        elif original.startswith("Overlap matrix with "):
            replaceable_section = False
        if original == predicted:
            continue
        if not replaceable_section:
            raise AssertionError(
                f"Non-H/S/D content changed at line {line_number}: {original!r}"
            )
        try:
            original_values = [float(value) for value in original.split()]
            predicted_values = [float(value) for value in predicted.split()]
        except ValueError as exc:
            raise AssertionError(
                f"H/S/D header or non-numeric content changed at line {line_number}"
            ) from exc
        if not original_values or len(original_values) != len(predicted_values):
            raise AssertionError(
                f"H/S/D row shape changed at line {line_number}: "
                f"{len(original_values)} != {len(predicted_values)}"
            )


def _max_block_error(actual, expected) -> tuple[float, int, int]:
    maximum = 0.0
    compared = 0
    zero_checked = 0
    for key, blocks in actual.pair_blocks.items():
        for index, edge_values in enumerate(actual.pair_edges[key].t().tolist()):
            edge = tuple(int(value) for value in edge_values)
            actual_block = blocks[index]
            location = expected.lookup.get(edge)
            if location is None:
                maximum = max(maximum, float(actual_block.abs().max()))
                zero_checked += 1
                continue
            expected_key, expected_index = location
            difference = (
                actual_block - expected.pair_blocks[expected_key][expected_index]
            )
            maximum = max(maximum, float(difference.abs().max()))
            compared += 1
    return maximum, compared, zero_checked


def _project_to_layout(layout, prediction):
    projected_blocks = {}
    for key, layout_blocks in layout.pair_blocks.items():
        blocks = []
        for edge_values in layout.pair_edges[key].t().tolist():
            edge = tuple(int(value) for value in edge_values)
            location = prediction.lookup.get(edge)
            if location is None:
                blocks.append(torch.zeros_like(layout_blocks[0]))
            else:
                prediction_key, prediction_index = location
                blocks.append(prediction.pair_blocks[prediction_key][prediction_index])
        projected_blocks[key] = torch.stack(blocks)
    return layout._replace_pair_blocks(projected_blocks, basis=layout.basis)


def _second_density_max_abs(path: Path) -> float:
    density_section = 0
    maximum = 0.0
    in_second_density = False
    for line in path.read_text(encoding="utf-8").splitlines():
        if line == "Density matrix spin=0":
            density_section += 1
            in_second_density = density_section == 2
            continue
        if in_second_density and _SECTION_RE.fullmatch(line):
            in_second_density = False
        if not in_second_density or _BLOCK_HEADER_RE.fullmatch(line):
            continue
        values = line.split()
        if values:
            try:
                maximum = max(maximum, max(abs(float(value)) for value in values))
            except ValueError:
                pass
    return maximum


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--snapshot-dir", type=Path, default=DEFAULT_SNAPSHOT_DIR)
    parser.add_argument("--template", type=Path, default=None)
    parser.add_argument("--prediction", type=Path, default=None)
    parser.add_argument("--info-path", type=Path, default=None)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--atol", type=float, default=2.0e-6)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    template_path = args.template or args.snapshot_dir / "HS.out"
    prediction_path = args.prediction or args.snapshot_dir / "HS_pred.out"
    info_path = args.info_path or args.snapshot_dir / "Si.out"

    template_layout = _layout(template_path)
    prediction_layout = _layout(prediction_path)
    _assert_only_hsd_values_changed(template_path, prediction_path)
    if prediction_layout != template_layout:
        raise AssertionError("HS_pred.out section/block layout differs from HS.out")

    expected, _, cfg = load_hsd_prediction(
        checkpoint_path=args.checkpoint,
        matrix_path=template_path,
        info_path=info_path,
        device=_device(args.device),
    )
    loaded = Snapshot.from_openmx(
        matrix_path=prediction_path,
        info_path=info_path,
        convention="e3nn",
        symmetrize_density=True,
        cfg=cfg,
    )

    second_density_max = _second_density_max_abs(prediction_path)
    if second_density_max != 0.0:
        raise AssertionError(
            "The second 'Density matrix spin=0' section is not exactly zero: "
            f"max={second_density_max:.6e}"
        )

    errors: dict[str, float] = {}
    coverage: dict[str, tuple[int, int]] = {}
    for name in ("hamiltonian", "overlap", "density"):
        error, compared, zero_checked = _max_block_error(loaded[name], expected[name])
        errors[name] = error
        coverage[name] = (compared, zero_checked)
        if error > args.atol:
            raise AssertionError(
                f"{name} reload error {error:.6e} exceeds tolerance {args.atol:.6e}"
            )

    projected = {
        name: _project_to_layout(loaded[name], expected[name])
        for name in ("hamiltonian", "overlap", "density")
    }
    expected_snapshot = Snapshot(
        hamiltonian=projected["hamiltonian"],
        overlap=projected["overlap"],
        density=projected["density"],
    )
    expected_energy = expected_snapshot.get_energy()
    loaded_energy = loaded.get_energy()
    expected_electrons = expected_snapshot.get_number_of_electrons()
    loaded_electrons = loaded.get_number_of_electrons()
    energy_error = float(torch.abs(loaded_energy - expected_energy))
    electron_error = float(torch.abs(loaded_electrons - expected_electrons))
    if energy_error > args.atol:
        raise AssertionError(f"Band-energy reload error is {energy_error:.6e}")
    if electron_error > args.atol:
        raise AssertionError(f"Electron-count reload error is {electron_error:.6e}")

    print(f"layout: {prediction_layout[0]} lines, {len(prediction_layout[1])} sections")
    print(f"block headers: {len(prediction_layout[2])}")
    print(f"second density section max abs: {second_density_max:.1f}")
    for name in ("hamiltonian", "overlap", "density"):
        compared, zero_checked = coverage[name]
        print(
            f"{name}: max reload error={errors[name]:.6e}, "
            f"predicted blocks={compared}, zero-filled blocks={zero_checked}"
        )
    print(
        f"band energy: expected={float(expected_energy):.12f} Ha, "
        f"loaded={float(loaded_energy):.12f} Ha, error={energy_error:.6e}"
    )
    print(
        f"electron count: expected={float(expected_electrons):.12f}, "
        f"loaded={float(loaded_electrons):.12f}, error={electron_error:.6e}"
    )
    print("validation: PASS")


if __name__ == "__main__":
    main()
