"""End-to-end tests for ``src.data.openmx_info_parser``.
"""

from __future__ import annotations
import pytest

import re
from pathlib import Path


from data.openmx_info_parser import parse_info_out

# --------------------------------------------------------------------------- test data path
TEST_FILE = Path("data/small/H2O/original/H2O.info.out")


# --------------------------------------------------------------------------- fixtures
@pytest.fixture(scope="module")
def parsed_data():
    """Parse the reference file once per test module."""
    return parse_info_out(TEST_FILE)


# --------------------------------------------------------------------------- basic content checks
@pytest.mark.parametrize("key", ["Ukin", "UH0", "UH1", "Una"])
@pytest.mark.unit
def test_energy_keys_present(parsed_data, key):
    """Selected energy labels must be present in the header dictionary."""
    assert key in parsed_data.energies


@pytest.mark.unit
def test_energy_values_are_scalars(parsed_data):
    """Energy tensors should be 0-dimensional scalars."""
    for val in parsed_data.energies.values():
        # PyTorch tensors expose ``ndim`` while the NumPy fallback yields ``ndim`` as well.
        assert getattr(val, "shape", ()) == () or getattr(val, "ndim", 0) == 0


@pytest.mark.unit
def test_orbital_set(parsed_data):
    """The compact orbital specification must exactly match the expected reference."""
    expected = {"H": "3s2p", "O": "3s3p2d"}
    assert parsed_data.orbital_set == expected


@pytest.mark.unit
def test_occupancies_shape(parsed_data):
    """Two spin channels per raw row ⇒ second dimension must be 2."""
    assert parsed_data.occupancies.ndim == 2
    assert parsed_data.occupancies.shape[1] == 2
    # There should be *some* records - a file without occupancies would be fishy.
    assert parsed_data.occupancies.shape[0] > 0


@pytest.mark.unit
def test_occupancy_by_element_helper(parsed_data):
    """Check that the convenience mask function returns only the requested element."""
    h_occ = parsed_data.occupancy_by_element("H")
    o_occ = parsed_data.occupancy_by_element("O")
    # The shapes should add up to the master tensor's length
    assert h_occ.shape[0] + o_occ.shape[0] == parsed_data.occupancies.shape[0]


# --------------------------------------------------------------------------- raw-row accounting (no aggregates)
RAW_ROW_RE = re.compile(r"^\s*[A-Za-z][^\s]*\s+\d+\s+[-+0-9Ee\.]+\s+[-+0-9Ee\.]+")


def count_raw_rows(path: Path) -> int:
    """Return the number of *raw* occupancy lines in *path* (skipping aggregates)."""
    n = 0
    with path.open(errors="ignore") as fh:
        for line in fh:
            if RAW_ROW_RE.match(line) and not line.lstrip().startswith(
                ("sum", "multiple")
            ):
                n += 1
    return n


@pytest.mark.unit
def test_raw_row_count_matches_tensor(parsed_data):
    """Every raw occupancy line in the file must appear once in the tensor."""
    expected_rows = count_raw_rows(TEST_FILE)
    assert parsed_data.occupancies.shape[0] == expected_rows


@pytest.mark.unit
def test_chemical_potential_is_parsed_as_fermi_level(tmp_path):
    info_path = tmp_path / "SiO2.out"
    info_path.write_text("Chemical potential (Hartree)      -0.184928095654\n")

    parsed = parse_info_out(info_path)

    assert parsed.fermi_level.item() == pytest.approx(-0.184928095654)


# --------------------------------------------------------------------------- edge-case sanity
@pytest.mark.parametrize(
    "element, part",
    [
        ("H", "s"),
        ("H", "p"),
        ("O", "d"),
    ],
)
@pytest.mark.unit
def test_orbital_parts_present(parsed_data, element: str, part: str):
    """Ensure every expected orbital letter appears in the element's compact spec."""
    assert part in parsed_data.orbital_set[element]
