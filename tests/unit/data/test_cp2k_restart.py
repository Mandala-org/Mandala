"""Independent record-layout checks for the installed CP2K density format."""

import struct

import numpy as np
import pytest

from data.cp2k_restart import read_cp2k_density_restart, write_cp2k_density_restart


def record(data):
    return struct.pack("<I", len(data)) + data + struct.pack("<I", len(data))


def native_records():
    # Layout follows the CP2K 2023.1 Fortran WRITE statements. A deliberately
    # nonsymmetric image block detects an accidental transpose on serialization.
    records = [
        struct.pack("<i", 1),
        struct.pack("<4i", 1, 2, 1, 2),
        struct.pack("<i", 1),
        struct.pack("<i", 2),
        struct.pack("<2i", 1, 1),
        struct.pack("<3i", 1, 1, 2),
    ]
    for index, cell, values in [
        (1, (0, 0, 0), ((1.0, 2.0), (3.0, 4.0))),
        (2, (-1, 0, 0), ((5.0, 6.0), (7.0, 8.0))),
    ]:
        records += [struct.pack("<4i", index, *cell)]
        records += [struct.pack("<2d", *column) for column in np.asarray(values).T]
    return b"".join(record(value) for value in records)


def test_independent_native_layout_and_byte_exact_roundtrip(tmp_path):
    source = tmp_path / "native.kp"
    source.write_bytes(native_records())
    density = read_cp2k_density_restart(source)
    assert density.spins[0][1].cell == (-1, 0, 0)
    np.testing.assert_array_equal(density.spins[0][0].matrix, [[1, 2], [3, 4]])
    output = tmp_path / "rewritten.kp"
    write_cp2k_density_restart(output, density)
    assert output.read_bytes() == source.read_bytes()
    with pytest.raises(FileExistsError):
        write_cp2k_density_restart(output, density)


def test_rejects_unsupported_version_truncation_and_trailing_records(tmp_path):
    source = tmp_path / "bad.kp"
    data = native_records()
    source.write_bytes(record(struct.pack("<i", 2)) + data[12:])
    with pytest.raises(ValueError, match="version"):
        read_cp2k_density_restart(source)
    source.write_bytes(data[:-5])
    with pytest.raises(Exception):
        read_cp2k_density_restart(source)
    source.write_bytes(data + record(b"ignored"))
    with pytest.raises(ValueError, match="Unexpected"):
        read_cp2k_density_restart(source)


def test_refuses_complex_payload_in_image_records(tmp_path):
    source = tmp_path / "native.kp"
    source.write_bytes(native_records())
    density = read_cp2k_density_restart(source)
    density.spins[0][0].matrix = density.spins[0][0].matrix.astype(complex)
    with pytest.raises(ValueError, match="real AO"):
        write_cp2k_density_restart(tmp_path / "complex.kp", density)
