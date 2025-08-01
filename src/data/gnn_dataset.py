"""
gnn_dataset.py
==============

In-memory dataset that converts :class:`Snapshot` objects into graph dict:

* **x**    → Full graph. Sub-graph corresponding to smaller, message-passing edge index available through index_gnn_cutoff

Targets *y* (Hamiltonian / Overlap / Density + energy, electrons)
are provided as snapshot-level information; training code can access
diagonal / off-diagonal parts through the ``.diag()`` and ``.offdiag()``
helpers.
"""

from __future__ import annotations
import hashlib
import pickle
from pathlib import Path

from typing import Dict, List, Sequence, Tuple

import torch
import time  # needed for __getitem__ timing
from torch.utils.data import Dataset
from e3nn.o3 import Irreps, spherical_harmonics
from e3nn.math import soft_one_hot_linspace

from core.block_irrep_mapper import BlockIrrepMapper
from net.common import Config

from data.snapshot import Snapshot
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
    ):
        self.cfg = cfg
        if cfg.cutoff_gnn >= cfg.cutoff_matrix:
            raise ValueError("cutoff_gnn must be < cutoff_matrix")

        if not snapshot_paths:
            raise ValueError("At least one snapshot path must be provided")

        self.device = torch.device(self.cfg.device)
        self.dtype = self.cfg.dtype

        if cfg.train_on_forces and not self.cfg.enable_forces:
            raise Exception("Forces must be enabled to train on them")
        if cfg.train_on_stress and not self.cfg.enable_stress:
            raise Exception("Stress must be enabled to train on it")

        if cfg.cache_root and (self.cfg.enable_forces or self.cfg.enable_stress):
            raise Exception("Caching must be disabled for forces and stress to work")

        # shared, **externally-provided** mapper ------------------------------
        self.mapper: BlockIrrepMapper = mapper
        self.orbital_cfg = mapper.orbital_cfg
        self.sh_irreps: Irreps = Irreps.spherical_harmonics(self.cfg.l_max_gnn)

        # edge-type encoding (ordered pairs)
        elems = self.orbital_cfg.elements()
        self.edge_types: List[str] = [f"{a}-{b}" for a in elems for b in elems]
        self.edge_type2idx: Dict[str, int] = {
            k: i for i, k in enumerate(self.edge_types)
        }
        self.n_edge_types = len(self.edge_types)

        # configure cache root (if None, caching is disabled)
        if cfg.cache_root is not None:
            self.cfg.cache_root = Path(self.cfg.cache_root).expanduser()

        # preprocess all snapshots
        self.samples: List[Tuple[Dict, Dict, Dict]] = []
        for matrix_path, info_path in tqdm(snapshot_paths, desc="Loading snapshots"):
            sample = self._load_or_process_snapshot(matrix_path, info_path)
            self.samples.append(sample)

    # ---------------------------------------------------------------- snapshot caching & helpers
    def _load_or_process_snapshot(
        self,
        matrix_path: Path,
        info_path: Path,
    ) -> Tuple[Dict, Dict, Dict]:
        """
        Load processed snapshot from cache if available, otherwise process and cache it.
        """
        # build cache key from snapshot payload and model settings
        if self.cfg.cache_root is not None:
            key_obj = (
                matrix_path.resolve(),
                info_path.resolve(),
                self.cfg.n_radial,
                self.cfg.cutoff_gnn,
                self.cfg.cutoff_matrix,
                self.cfg.l_max_gnn,
            )
            key_hash = hashlib.md5(pickle.dumps(key_obj)).hexdigest()
            cache_file = self.cfg.cache_root / f"{key_hash}.pt"
            # attempt load from cache
            if cache_file.exists():
                try:
                    return torch.load(cache_file)
                except Exception:
                    pass

        # process snapshot
        snapshot = Snapshot.from_openmx(
            matrix_path=matrix_path,
            info_path=info_path,
            convention="e3nn",
            symmetrize_density=True,
            cutoff_radius=self.cfg.cutoff_matrix,
        )
        sample = self._process_snapshot(snapshot)
        # save to cache if enabled
        if self.cfg.cache_root is not None:
            try:
                self.cfg.cache_root.mkdir(parents=True, exist_ok=True)
                torch.save(sample, cache_file)
            except Exception:
                pass
        return sample

    # ---------------------------------------------------------------- helpers
    # ---------- minimal-image displacements ----------------------------------
    @staticmethod
    def _minimal_disp(
        pos: torch.Tensor,
        edges: torch.Tensor,
        box: torch.Tensor | None,
        inv_box: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if box is None:
            return pos[edges[1]] - pos[edges[0]]
        if inv_box is None:
            inv_box = torch.inverse(box)
        delta = pos[edges[1]] - pos[edges[0]]  # cart
        frac = delta @ inv_box
        frac = frac - torch.round(frac)
        return frac @ box

    # ---------- build edge tensors for a given cfg.cutoff_gnn ------------------------
    def _edge_tensors(
        self,
        snap: Snapshot,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
        """
        Returns
        -------
        edge_index : (2, E_total) long
        edge_type_idx : (E_total,) long
        edge_length_emb : (E_total, cfg.n_radial) float
        edge_sh : (E_total, sh_dim) float
        index_gnn_cutoff : int
        """
        # 1. Collect all edges and their properties
        edges = []
        for key, pair_edges in snap.density.pair_edges.items():
            for i in range(pair_edges.shape[1]):
                src, dst = pair_edges[:, i]
                edges.append(
                    {
                        "src": src.item(),
                        "dst": dst.item(),
                        "key": key,
                    }
                )

        # 2. Separate self-edges and off-diagonal edges
        self_edges = sorted(
            [e for e in edges if e["src"] == e["dst"]], key=lambda e: e["src"]
        )
        offdiag_edges = [e for e in edges if e["src"] != e["dst"]]

        # 3. Calculate lengths for off-diagonal edges and sort them
        offdiag_edge_index = torch.tensor(
            [[e["src"] for e in offdiag_edges], [e["dst"] for e in offdiag_edges]],
            dtype=torch.long,
            device=self.device,
        )
        disp = self._minimal_disp(
            snap.positions,
            offdiag_edge_index,
            snap.box,
        )
        lengths = torch.linalg.norm(disp, dim=-1)
        sorted_indices = torch.argsort(lengths)
        offdiag_edges = [offdiag_edges[i] for i in sorted_indices]

        # 4. Combine edges in the specified order
        all_edges = self_edges + offdiag_edges
        edge_index = torch.tensor(
            [[e["src"] for e in all_edges], [e["dst"] for e in all_edges]],
            dtype=torch.long,
            device=self.device,
        )
        edge_type_idx = torch.tensor(
            [self.edge_type2idx[e["key"]] for e in all_edges],
            dtype=torch.long,
            device=self.device,
        )

        # 5. Calculate geometric features for the final edge order
        disp = self._minimal_disp(snap.positions, edge_index, snap.box)
        lengths = torch.linalg.norm(disp, dim=-1)
        edge_sh = spherical_harmonics(
            self.sh_irreps, disp, normalize=True, normalization="component"
        )
        edge_length_emb = soft_one_hot_linspace(
            lengths,
            start=0.0,
            end=self.cfg.cutoff_matrix,
            number=self.cfg.n_radial,
            basis="gaussian",
            cutoff=False,
        )

        # 6. Determine the GNN cfg.cutoff_gnn index
        index_gnn_cutoff = torch.sum(lengths <= self.cfg.cutoff_gnn).item()

        return (
            edge_index,
            edge_type_idx,
            edge_length_emb,
            edge_sh,
            index_gnn_cutoff,
            len(self_edges),
        )

    # ---------- main per-snapshot routine -----------------------------------
    def _process_snapshot(
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

        snap.positions = snap.positions.to(self.device)
        snap.box = snap.box.to(self.device)
        snap.hamiltonian = snap.hamiltonian.to(self.device)
        snap.overlap = snap.overlap.to(self.device)
        snap.density = snap.density.to(self.device)

        (
            edge_index,
            edge_type_idx,
            edge_length_emb,
            edge_sh,
            index_gnn_cutoff,
            num_self_edges,
        ) = self._edge_tensors(snap)

        edge_index = edge_index.to(self.device)
        edge_type_idx = edge_type_idx.to(self.device)
        edge_length_emb = edge_length_emb.to(self.device)
        edge_sh = edge_sh.to(self.device)

        atoms = snap.density.atoms
        elem2idx = {el: i for i, el in enumerate(self.orbital_cfg.elements())}
        node_type_idx = torch.tensor(
            [elem2idx[el] for el in atoms], dtype=torch.long, device=self.device
        )

        x = {
            "node_type_idx": node_type_idx,
            "edge_index": edge_index,
            "edge_type_idx": edge_type_idx,
            "index_gnn_cutoff": index_gnn_cutoff,
            "num_self_edges": num_self_edges,
            "edge_length_emb": edge_length_emb,
            "edge_sh": edge_sh,
            "positions": snap.positions,
            "box": snap.box,
            "atoms": atoms,
        }
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
                "hamiltonian": hamiltonian_target.to(self.device),
                "overlap": overlap_target.to(self.device),
                "density": density_target.to(self.device),
                "energy": snap.get_energy().to(self.device),
                "num_electrons": snap.get_number_of_electrons().to(self.device),
                "forces": snap.forces.to(self.device),
                "stress": snap.stress.to(self.device),
            }

        return x, y

    # ------------------- torch Dataset interface ---------------------------
    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(
        self, idx: int
    ) -> Tuple[
        Dict[str, torch.Tensor], Dict[str, torch.Tensor], Dict[str, torch.Tensor]
    ]:
        t0 = time.perf_counter()
        sample = self.samples[idx]
        t1 = time.perf_counter()
        try:
            self.loader_times.append(t1 - t0)
        except Exception:
            pass
        return sample

    def to(self, device: torch.device | str) -> E3GNNDataset:
        """
        Move all dataset inputs and targets to the specified device.
        """
        device = torch.device(device)
        if device == self.device:
            return self
        self.device = device
        # Explicitly move known fields
        for idx, (x, y) in enumerate(self.samples):
            # x
            x["node_type_idx"] = x["node_type_idx"].to(device)
            x["edge_index"] = x["edge_index"].to(device)
            x["edge_type_idx"] = x["edge_type_idx"].to(device)
            x["edge_length_emb"] = x["edge_length_emb"].to(device)
            x["edge_sh"] = x["edge_sh"].to(device)
            x["index_gnn_cutoff"] = x["index_gnn_cutoff"].to(device)

            # y targets
            y["hamiltonian"] = y["hamiltonian"].to(device)
            y["overlap"] = y["overlap"].to(device)
            y["density"] = y["density"].to(device)
            y["energy"] = y["energy"].to(device)
            y["num_electrons"] = y["num_electrons"].to(device)

            self.samples[idx] = (x, y)
        return self
