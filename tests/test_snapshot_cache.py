import os
import time

import torch
import pytest

import sys

# Ensure src/ directory is in PYTHONPATH
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
)
import data.snapshot as snapshotmod
from data.snapshot import Snapshot
from core.orbital_irrep_config import OrbitalIrrepConfig
from data.block_matrix import BlockMatrix


class DummyInfo:
    def __init__(self):
        # minimal fields
        self.elements = ["H"]
        self.xyz = torch.tensor([])
        self.box = torch.tensor([])
        self.orbital_set = {"H": ["1x0e"]}


@pytest.fixture(autouse=True)
def patch_cache_dir(tmp_path, monkeypatch):
    # Redirect cache to tmp
    monkeypatch.setattr(snapshotmod, "CACHE_ROOT", tmp_path / "cache")
    yield


def make_dummy_snapshot(matrix_path, info_path):
    # Create a minimal Snapshot for testing caching logic
    orb_cfg = OrbitalIrrepConfig.from_dict({"H": ["1x0e"]})
    bm = BlockMatrix.empty(("H",), orb_cfg, basis="openmx")
    snap = Snapshot(bm, bm, bm)
    snap.positions = None
    snap.box = None
    return snap


def test_cache_hit_and_miss(tmp_path, monkeypatch):
    # Prepare dummy files
    mat = tmp_path / "m.out"
    info = tmp_path / "i.out"
    mat.write_text("data")
    info.write_text("data")

    calls = []
    # Stub parse_info_out and parse_openmx_scfout
    import data.openmx_info_parser as info_parser
    import data.openmx_parser as openmx_parser

    def stub_parse_info(path):
        calls.append("info")
        return DummyInfo()

    def stub_parse_scf(matrix_path, atoms, orb_cfg, convention, symmetrize_density):
        calls.append("scf")
        return make_dummy_snapshot(matrix_path, atoms)

    monkeypatch.setattr(info_parser, "parse_info_out", stub_parse_info)
    monkeypatch.setattr(openmx_parser, "parse_openmx_scfout", stub_parse_scf)

    # First load: should call parsers and cache
    # First load: should call parsers and cache
    _ = Snapshot.from_openmx(mat, info)
    assert calls == ["info", "scf"]
    cache_files = list((snapshotmod.CACHE_ROOT).glob("*.pkl"))
    assert len(cache_files) == 1

    # Second load: should hit cache and return Snapshot without error
    calls.clear()
    s2 = Snapshot.from_openmx(mat, info)
    # only one cache file should exist
    cache_files2 = list((snapshotmod.CACHE_ROOT).glob("*.pkl"))
    assert len(cache_files2) == 1
    assert isinstance(s2, Snapshot)

    # Modify matrix file to invalidate cache
    time.sleep(0.01)
    mat.write_text("new")
    calls.clear()
    # Third load: after file change, should re-parse
    _ = Snapshot.from_openmx(mat, info)
    assert calls == ["info", "scf"]


def test_disable_cache(tmp_path, monkeypatch):
    mat = tmp_path / "m2.out"
    info = tmp_path / "i2.out"
    mat.write_text("x")
    info.write_text("y")

    calls = []
    import data.openmx_info_parser as info_parser
    import data.openmx_parser as openmx_parser

    # Stub parse_info_out and parse_openmx_scfout to count calls
    def stub_parse_info(path):
        calls.append("info")
        return DummyInfo()

    def stub_parse_scfout(matrix_path, atoms, orb_cfg, convention, symmetrize_density):
        calls.append("scf")
        return make_dummy_snapshot(matrix_path, atoms)

    monkeypatch.setattr(info_parser, "parse_info_out", stub_parse_info)
    monkeypatch.setattr(openmx_parser, "parse_openmx_scfout", stub_parse_scfout)

    # Call with use_cache=False twice: always parse
    # Call with use_cache=False twice: always parse both times
    calls.clear()
    # Two loads with use_cache=False: always parse
    _ = Snapshot.from_openmx(mat, info, use_cache=False)
    _ = Snapshot.from_openmx(mat, info, use_cache=False)
    # expected two info and two scf calls
    assert calls.count("info") == 2
    assert calls.count("scf") == 2
