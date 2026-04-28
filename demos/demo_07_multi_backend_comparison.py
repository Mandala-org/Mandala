from __future__ import annotations

from _bootstrap import add_repo_root_to_path

add_repo_root_to_path()

from pathlib import Path

from demos.common import (
    FHIAIMS_ROOT,
    PYSCF_RESULTS_ROOT,
    demo_config,
    load_h2o_snapshot,
    summarize_block_matrix,
    summarize_snapshot,
)
from data.snapshot import Snapshot


def _first_existing(*paths: Path) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


def main() -> None:
    cfg = demo_config()
    openmx = load_h2o_snapshot(cfg=cfg, convention="e3nn")
    fhiaims = Snapshot.from_fhiaims(
        FHIAIMS_ROOT / "geometry.in",
        FHIAIMS_ROOT / "basis-indices.out",
        FHIAIMS_ROOT / "H_spin_01_kpt_000001.csc",
        FHIAIMS_ROOT / "S_spin_01_kpt_000001.csc",
        FHIAIMS_ROOT / "D_spin_01_kpt_000001.csc",
        convention="e3nn",
        cfg=cfg,
    )

    pyscf_npz = _first_existing(
        PYSCF_RESULTS_ROOT / "sample.npz",
        PYSCF_RESULTS_ROOT / "h2o.npz",
        PYSCF_RESULTS_ROOT / "baseline.npz",
    )
    pyscf_json = None if pyscf_npz is None else pyscf_npz.with_suffix(".json")
    pyscf = None
    if pyscf_npz is not None and pyscf_json is not None and pyscf_json.exists():
        pyscf = Snapshot.from_pyscf(pyscf_npz, pyscf_json, convention="e3nn", cfg=cfg)

    print("OpenMX")
    print("------")
    print(summarize_snapshot(openmx))
    print(summarize_block_matrix(openmx.hamiltonian))
    print()
    print("FHI-aims")
    print("--------")
    print(summarize_snapshot(fhiaims))
    print(summarize_block_matrix(fhiaims.hamiltonian))
    print()
    print("PySCF")
    print("-----")
    if pyscf is None:
        print(
            "No local PySCF baseline artifact found under data/pyscf_baseline/results."
        )
    else:
        print(summarize_snapshot(pyscf))
        print(summarize_block_matrix(pyscf.hamiltonian))
    print()
    print("Common theme")
    print("------------")
    print("The loaders normalize each backend into the same Snapshot interface.")


if __name__ == "__main__":
    main()
