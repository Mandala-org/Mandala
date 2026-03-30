"""
gnn_dataset.py
==============

In-memory dataset that converts :class:`Snapshot` objects into graph dict:

* **x**    -> Full graph. Sub-graph corresponding to smaller, message-passing edge index available through index_gnn_cutoff

Targets *y* (Hamiltonian / Overlap / Density + energy, electrons)
are provided as snapshot-level information; training code can access
diagonal / off-diagonal parts through the ``.diag()`` and ``.offdiag()``
helpers.
"""

from __future__ import annotations
import hashlib
import os
from pathlib import Path
import tempfile

from typing import Dict, List, Sequence, Tuple

import torch
import time  # needed for __getitem__ timing
from torch.utils.data import Dataset
from e3nn.o3 import Irreps

from core.block_irrep_mapper import BlockIrrepMapper
from net.common import Config

from data.snapshot import Snapshot
from data.graph_features import compute_graph_features
from tqdm.auto import tqdm


# ═══════════════════════════════════════════════════════════════════════════════
class E3GNNDataset(Dataset):
    """
    Fully in-memory dataset that yields **(x, y)** tuples.

    *x*    - no self-edges
    *y*    - targets
    """

    # --------------------------------------------------------------------- init
    def __init__(
        self,
        snapshot_paths: Sequence[Tuple[Path, Path]],
        mapper: BlockIrrepMapper,
        cfg: Config,
        convention: str = "e3nn",
    ):
        self.cfg = cfg
        self.convention = convention
        self.snapshot_paths = list(snapshot_paths)

        if not self.snapshot_paths:
            raise ValueError("At least one snapshot path must be provided")

        self.dtype = self.cfg.dtype

        if cfg.train_on_forces and not self.cfg.enable_forces:
            raise Exception("Forces must be enabled to train on them")
        if cfg.train_on_stress and not self.cfg.enable_stress:
            raise Exception("Stress must be enabled to train on it")

        # shared, **externally-provided** mapper ------------------------------
        self.mapper: BlockIrrepMapper = mapper
        self.orbital_cfg = mapper.orbital_cfg
        self.sh_irreps: Irreps = Irreps.spherical_harmonics(self.cfg.l_max)

        # preprocess all snapshots
        self.snapshots: List[Tuple[Dict, Dict, Dict]] = []
        self.snapshot_cache_hits = 0
        self.snapshot_cache_misses = 0
        for matrix_path, info_path in tqdm(
            self.snapshot_paths, desc="Loading snapshots"
        ):
            snapshot = self._load_snapshot(matrix_path, info_path)
            sample = self._process_snapshot_to_sample(snapshot)
            self.snapshots.append(sample)
        if getattr(self.cfg, "snapshot_cache_dir", None) is None:
            print(
                f"[CACHE] Snapshot loading: cache disabled, loaded {len(self.snapshot_paths)} snapshot(s)"
            )
        else:
            total = self.snapshot_cache_hits + self.snapshot_cache_misses
            print(
                "[CACHE] Snapshot loading: "
                f"{self.snapshot_cache_hits} hit(s), {self.snapshot_cache_misses} miss(es), "
                f"{total} total"
            )

    # ---------------------------------------------------------------- snapshot caching & helpers
    def _snapshot_cache_file(self, matrix_path: Path, info_path: Path) -> Path | None:
        snapshot_cache_dir = getattr(self.cfg, "snapshot_cache_dir", None)
        if snapshot_cache_dir is None:
            return None
        cache_dir = Path(snapshot_cache_dir).expanduser()
        mat_stat = matrix_path.stat()
        info_stat = info_path.stat()
        key = "|".join(
            [
                str(matrix_path.resolve()),
                str(info_path.resolve()),
                self.convention,
                str(self.cfg.cutoff_radius),
                str(self.dtype),
                str(mat_stat.st_mtime_ns),
                str(mat_stat.st_size),
                str(info_stat.st_mtime_ns),
                str(info_stat.st_size),
            ]
        )
        key_hash = hashlib.md5(key.encode("utf-8")).hexdigest()
        return cache_dir / f"{matrix_path.stem}_{key_hash}.pt"

    def _load_snapshot(
        self,
        matrix_path: Path,
        info_path: Path,
    ) -> Snapshot:
        cache_file = self._snapshot_cache_file(matrix_path, info_path)
        if cache_file is not None and cache_file.exists():
            try:
                self.snapshot_cache_hits += 1
                return Snapshot.load(cache_file, device="cpu")
            except Exception:
                self.snapshot_cache_hits -= 1
                pass

        self.snapshot_cache_misses += 1
        snapshot = Snapshot.from_openmx(
            matrix_path=matrix_path,
            info_path=info_path,
            convention=self.convention,
            symmetrize_density=True,
            cutoff_radius=self.cfg.cutoff_radius,
            cfg=self.cfg,
        )
        if cache_file is not None:
            try:
                cache_file.parent.mkdir(parents=True, exist_ok=True)
                with tempfile.NamedTemporaryFile(
                    dir=cache_file.parent,
                    prefix=f"{cache_file.stem}.",
                    suffix=".tmp",
                    delete=False,
                ) as tmp:
                    tmp_path = Path(tmp.name)
                try:
                    snapshot.save(tmp_path)
                    os.replace(tmp_path, cache_file)
                finally:
                    if tmp_path.exists():
                        tmp_path.unlink()
            except Exception:
                pass
        return snapshot

    # ---------- main per-snapshot routine -----------------------------------
    def _process_snapshot_to_sample(
        self, snap: Snapshot
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Build graph inputs and targets from one Snapshot.
        """
        # snap.positions = snap.positions.to(self.dtype)
        # snap.box = snap.box.to(self.dtype)
        # snap.forces = snap.forces.to(self.dtype)
        # snap.stress = snap.stress.to(self.dtype)

        if self.cfg.enable_forces:
            snap.positions.requires_grad_()
        if self.cfg.enable_stress:
            snap.box.requires_grad_()
        if self.cfg.train_on_forces:
            snap.forces.requires_grad_()
        if self.cfg.train_on_stress:
            snap.stress.requires_grad_()

        atoms = snap.density.atoms
        elem2idx = {el: i for i, el in enumerate(self.orbital_cfg.elements())}
        node_type_idx = torch.tensor([elem2idx[el] for el in atoms], dtype=torch.long)

        x = {
            "node_type_idx": node_type_idx,
            "positions": snap.positions,
            "box": snap.box,
            "atoms": atoms,
        }

        if self.cfg.precompute_edge_features:
            (
                edge_index,
                edge_shift,
                edge_type_idx,
                edge_length_emb,
                edge_sh,
                num_self_edges,
            ) = compute_graph_features(
                positions=snap.positions,
                box=snap.box,
                atoms=snap.density.atoms,
                cfg=self.cfg,
                sh_irreps=self.sh_irreps,
                edge_type2idx=self.mapper.edge_type2idx,
            )
            x["edge_index"] = edge_index
            x["edge_shift"] = edge_shift
            x["edge_type_idx"] = edge_type_idx
            x["edge_length_emb"] = edge_length_emb
            x["edge_sh"] = edge_sh
            x["num_self_edges"] = num_self_edges

        with torch.no_grad():
            if self.cfg.train_target == "matrix":
                hamiltonian_target = snap.hamiltonian
                overlap_target = snap.overlap
                density_target = snap.density
            elif self.cfg.train_target == "irreps":
                hamiltonian_target = snap.hamiltonian.to_vectors(self.mapper)
                overlap_target = snap.overlap.to_vectors(self.mapper)
                density_target = snap.density.to_vectors(self.mapper)
            else:
                raise ValueError(
                    f"Unknown train_target {self.cfg.train_target}, must be 'irreps' or 'matrix'"
                )

            y = {
                "hamiltonian": hamiltonian_target,
                "overlap": overlap_target,
                "density": density_target,
                "energy": snap.get_energy(),
                "num_electrons": snap.get_number_of_electrons(),
                "forces": snap.forces,
                "stress": snap.stress,
            }
        return x, y

    # ------------------- torch Dataset interface ---------------------------
    def __len__(self) -> int:
        return len(self.snapshots)

    def __getitem__(
        self, idx: int
    ) -> Tuple[
        Dict[str, torch.Tensor], Dict[str, torch.Tensor], Dict[str, torch.Tensor]
    ]:
        t0 = time.perf_counter()
        snapshot = self.snapshots[idx]
        t1 = time.perf_counter()
        try:
            self.loader_times.append(t1 - t0)
        except Exception:
            pass
        return snapshot

    def to(self, device: torch.device | str) -> E3GNNDataset:
        """
        Move all dataset inputs and targets to the specified device.
        """
        device = torch.device(device)
        if device == self.device:
            return self
        self.device = device
        # Explicitly move known fields
        for idx, (x, y) in enumerate(self.snapshots):
            # x
            if "node_type_idx" in x:
                x["node_type_idx"] = x["node_type_idx"].to(device)
            if "edge_index" in x:
                x["edge_index"] = x["edge_index"].to(device)
            if "edge_shift" in x:
                x["edge_shift"] = x["edge_shift"].to(device)
            if "edge_type_idx" in x:
                x["edge_type_idx"] = x["edge_type_idx"].to(device)
            if "edge_length_emb" in x:
                x["edge_length_emb"] = x["edge_length_emb"].to(device)
            if "edge_sh" in x:
                x["edge_sh"] = x["edge_sh"].to(device)

            # y targets
            y["hamiltonian"] = y["hamiltonian"].to(device)
            y["overlap"] = y["overlap"].to(device)
            y["density"] = y["density"].to(device)
            y["energy"] = y["energy"].to(device)
            y["num_electrons"] = y["num_electrons"].to(device)

            self.snapshots[idx] = (x, y)
        return self
