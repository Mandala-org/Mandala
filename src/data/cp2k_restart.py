"""CP2K 2023.1 version-1 k-point AO-density restart records.

This is the real-space image format implemented in CP2K ``kpoint_io.F`` and
``cp_fm_types.F``, not the Gamma MO ``.wfn`` format. Matrices supplied here
must already use CP2K's basis, spin sum, image and AO conventions. Reading a
native template and replacing only its density payload is recommended.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.io import FortranEOFError, FortranFile


@dataclass
class CP2KImageDensity:
    cell: tuple[int, int, int]
    matrix: np.ndarray


@dataclass
class CP2KDensityRestart:
    natom: int
    nao: int
    nset_max: int
    nshell_max: int
    nset_info: np.ndarray
    nshell_info: np.ndarray
    nso_info: np.ndarray
    spins: list[list[CP2KImageDensity]]


def _integers(file: FortranFile, size: int) -> np.ndarray:
    values = file.read_ints(np.dtype("<i4"))
    if values.size != size:
        raise ValueError(f"Expected {size} integers, found {values.size}")
    return values


def read_cp2k_density_restart(path: str | Path) -> CP2KDensityRestart:
    """Read the GCC/x86 Fortran sequential format used by the ROSI build."""
    with FortranFile(path, "r", header_dtype=np.dtype("<u4")) as file:
        version = int(_integers(file, 1)[0])
        if version != 1:
            raise ValueError(f"Unsupported CP2K density restart version {version}")
        natom, nao, nset_max, nshell_max = map(int, _integers(file, 4))
        if min(natom, nao, nset_max, nshell_max) < 1:
            raise ValueError("Invalid CP2K basis dimensions")
        nset = _integers(file, natom)
        nshell = _integers(file, nset_max * natom).reshape((nset_max, natom), order="F")
        nso = _integers(file, nshell_max * nset_max * natom).reshape(
            (nshell_max, nset_max, natom), order="F"
        )
        spins: list[list[CP2KImageDensity]] = []
        nspin = None
        while nspin is None or len(spins) < nspin:
            ispin, current_nspin, nimages = map(int, _integers(file, 3))
            if nspin is None:
                nspin = current_nspin
            if (
                nspin not in (1, 2)
                or current_nspin != nspin
                or ispin != len(spins) + 1
                or nimages < 1
            ):
                raise ValueError("Invalid CP2K spin/image header")
            images = []
            cells = set()
            for index in range(nimages):
                image, *cell = map(int, _integers(file, 4))
                if image != index + 1 or tuple(cell) in cells:
                    raise ValueError("Invalid or duplicate CP2K image index")
                cells.add(tuple(cell))
                columns = [file.read_reals(np.dtype("<f8")) for _ in range(nao)]
                if any(column.size != nao for column in columns):
                    raise ValueError("Invalid CP2K density column length")
                matrix = np.column_stack(columns)
                if not np.isfinite(matrix).all():
                    raise ValueError("Nonfinite CP2K density")
                images.append(CP2KImageDensity(tuple(cell), matrix))
            spins.append(images)
        try:
            file.read_record(np.uint8)
        except FortranEOFError:
            pass
        else:
            raise ValueError("Unexpected records after CP2K density")
    return CP2KDensityRestart(
        natom, nao, nset_max, nshell_max, nset, nshell, nso, spins
    )


def write_cp2k_density_restart(path: str | Path, restart: CP2KDensityRestart) -> None:
    """Write a new file, refusing overwrite or complex/nonfinite AO payloads."""
    if min(restart.natom, restart.nao, restart.nset_max, restart.nshell_max) < 1:
        raise ValueError("Invalid CP2K basis dimensions")
    expected = (
        (restart.natom,),
        (restart.nset_max, restart.natom),
        (restart.nshell_max, restart.nset_max, restart.natom),
    )
    metadata = (restart.nset_info, restart.nshell_info, restart.nso_info)
    for array, shape in zip(metadata, expected):
        if np.asarray(array).shape != shape or np.asarray(array).dtype.kind not in "iu":
            raise ValueError("Invalid CP2K basis metadata")
    if len(restart.spins) not in (1, 2):
        raise ValueError("Only restricted or collinear spin densities are supported")
    for images in restart.spins:
        if not images or len({x.cell for x in images}) != len(images):
            raise ValueError("Missing or duplicate CP2K images")
        for image in images:
            if len(image.cell) != 3 or any(
                not isinstance(n, (int, np.integer)) for n in image.cell
            ):
                raise ValueError("Cells must contain three integers")
            matrix = np.asarray(image.matrix)
            if (
                matrix.shape != (restart.nao, restart.nao)
                or not np.isfinite(matrix).all()
            ):
                raise ValueError("Invalid CP2K density matrix")
            if np.iscomplexobj(matrix):
                raise ValueError(
                    "Version-1 image records require real AO density matrices"
                )
    # Opening in exclusive mode protects a converged native template from loss.
    with Path(path).open("xb") as stream:
        with FortranFile(stream, "w", header_dtype=np.dtype("<u4")) as file:

            def ints(value):
                return np.asarray(value, dtype="<i4").ravel(order="F")

            file.write_record(ints([1]))
            file.write_record(
                ints([restart.natom, restart.nao, restart.nset_max, restart.nshell_max])
            )
            for array in metadata:
                file.write_record(ints(array))
            for ispin, images in enumerate(restart.spins, 1):
                file.write_record(ints([ispin, len(restart.spins), len(images)]))
                for index, image in enumerate(images, 1):
                    file.write_record(ints([index, *image.cell]))
                    for column in np.asarray(image.matrix).T:
                        file.write_record(np.asarray(column, dtype="<f8"))
