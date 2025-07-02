"""
gnn_dataset.py
==============

In-memory dataset that converts :class:`Snapshot` objects into two graph inputs:

* **x_gnn**    → small-cutoff graph used by message-passing layers
* **x_matrix** → larger graph (no self-edges) used only by the readout head

Diagonal / off-diagonal overlap features are kept **as IrrepsBlockData**
instead of being concatenated, because different pair-keys carry different
irreps.

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
from data.snapshot import Snapshot
from tqdm.auto import tqdm


# ═══════════════════════════════════════════════════════════════════════════════
class E3GNNDataset(Dataset):
    """
    Fully in-memory dataset that yields **(x_gnn, x_matrix, y)** tuples.

    *x_gnn*    - tensors built with the *small* cutoff (no self-edges)
    *x_matrix* - tensors built with the *large* cutoff (superset of edges)
    *y*        - targets (unchanged Snap-level information)
    """

    # --------------------------------------------------------------------- init
    def __init__(
        self,
        snapshot_paths: Sequence[Tuple[Path, Path]],
        mapper: BlockIrrepMapper,
        *,
        cutoff_gnn: float = 5.0,
        cutoff_matrix: float = 7.5,
        l_max_sh: int = 3,
        n_radial: int = 64,
        device: torch.device | str = "cpu",
        cache_root: str | Path | None = None,
        enable_positions_grad: bool = False,
    ):
        if cutoff_gnn >= cutoff_matrix:
            raise ValueError("cutoff_gnn must be < cutoff_matrix")

        if not snapshot_paths:
            raise ValueError("At least one snapshot path must be provided")

        self.device = torch.device(device)
        self.enable_positions_grad = enable_positions_grad

        # shared, **externally-provided** mapper ------------------------------
        self.mapper: BlockIrrepMapper = mapper
        self.global_cfg = mapper.orbital_cfg
        self.l_max_sh = int(l_max_sh)
        self.sh_irreps: Irreps = Irreps.spherical_harmonics(self.l_max_sh)
        self.n_radial = int(n_radial)
        self.cut_gnn = float(cutoff_gnn)
        self.cut_mat = float(cutoff_matrix)

        # edge-type encoding (ordered pairs)
        elems = self.global_cfg.elements()
        self.edge_types: List[str] = [f"{a}-{b}" for a in elems for b in elems]
        self.edge_type2idx: Dict[str, int] = {
            k: i for i, k in enumerate(self.edge_types)
        }
        self.n_edge_types = len(self.edge_types)

        # configure cache root (if None, caching is disabled)
        if cache_root is None:
            self.cache_root = None
        else:
            self.cache_root = Path(cache_root).expanduser()
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
        if self.cache_root is not None:
            key_obj = (
                matrix_path.resolve(),
                info_path.resolve(),
                self.n_radial,
                self.cut_gnn,
                self.cut_mat,
                self.l_max_sh,
            )
            key_hash = hashlib.md5(pickle.dumps(key_obj)).hexdigest()
            cache_file = self.cache_root / f"{key_hash}.pt"
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
            cutoff_radius=self.cut_mat,
        )
        sample = self._process_snapshot(snapshot)
        # save to cache if enabled
        if self.cache_root is not None:
            try:
                self.cache_root.mkdir(parents=True, exist_ok=True)
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

    # ---------- build edge tensors for a given cutoff ------------------------
    def _edge_tensors(
        self,
        snap: Snapshot,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
        """
        Returns
        -------
        edge_index : (2, E_total) long
        edge_type_idx : (E_total,) long
        edge_length_emb : (E_total, n_radial) float
        edge_sh : (E_total, sh_dim) float
        gnn_edge_cutoff_idx : int
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
        if offdiag_edges:
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
            end=self.cut_mat,
            number=self.n_radial,
            basis="gaussian",
            cutoff=False,
        )

        # 6. Determine the GNN cutoff index
        gnn_edge_cutoff_idx = (
            len(self_edges) + torch.sum(lengths <= self.cut_gnn).item()
        )

        return edge_index, edge_type_idx, edge_length_emb, edge_sh, gnn_edge_cutoff_idx

    # ---------- main per-snapshot routine -----------------------------------
    def _process_snapshot(
        self, snap: Snapshot
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Build graph inputs and targets from one Snapshot.
        """
        if self.enable_positions_grad:
            snap.positions.requires_grad_(True)

        (
            edge_index,
            edge_type_idx,
            edge_length_emb,
            edge_sh,
            gnn_edge_cutoff_idx,
        ) = self._edge_tensors(snap)

        atoms = snap.density.atoms
        elem2idx = {el: i for i, el in enumerate(self.global_cfg.elements())}
        node_type_idx = torch.tensor(
            [elem2idx[el] for el in atoms], dtype=torch.long, device=self.device
        )

        x = {
            "node_type_idx": node_type_idx,
            "edge_index": edge_index,
            "edge_type_idx": edge_type_idx,
            "edge_length_emb": edge_length_emb,
            "edge_sh": edge_sh,
            "gnn_edge_cutoff_idx": gnn_edge_cutoff_idx,
            "positions": snap.positions,
            "box": snap.box,
            "atoms": atoms,
        }

        y = {
            "hamiltonian": snap.hamiltonian.to_vectors(self.mapper).to(self.device),
            "overlap": snap.overlap.to_vectors(self.mapper).to(self.device),
            "density": snap.density.to_vectors(self.mapper).to(self.device),
            "energy": snap.get_energy().to(self.device),
            "num_electrons": snap.get_number_of_electrons().to(self.device),
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
        for idx, (x_gnn, x_matrix, y) in enumerate(self.samples):
            # x_gnn
            x_gnn["node_type_idx"] = x_gnn["node_type_idx"].to(device)
            x_gnn["edge_index"] = x_gnn["edge_index"].to(device)
            x_gnn["edge_type_idx"] = x_gnn["edge_type_idx"].to(device)
            x_gnn["edge_length_emb"] = x_gnn["edge_length_emb"].to(device)
            x_gnn["edge_sh"] = x_gnn["edge_sh"].to(device)
            for k, v in x_gnn.get("overlap_vectors_diag", {}).items():
                x_gnn["overlap_vectors_diag"][k] = v.to(device)
            for k, v in x_gnn.get("overlap_vectors_offdiag", {}).items():
                x_gnn["overlap_vectors_offdiag"][k] = v.to(device)
            # atoms list remains unchanged

            # x_matrix
            x_matrix["edge_index"] = x_matrix["edge_index"].to(device)
            x_matrix["edge_type_idx"] = x_matrix["edge_type_idx"].to(device)
            x_matrix["edge_length_emb"] = x_matrix["edge_length_emb"].to(device)
            x_matrix["edge_sh"] = x_matrix["edge_sh"].to(device)
            for k, v in x_matrix.get("overlap_vectors_diag", {}).items():
                x_matrix["overlap_vectors_diag"][k] = v.to(device)
            for k, v in x_matrix.get("overlap_vectors_offdiag", {}).items():
                x_matrix["overlap_vectors_offdiag"][k] = v.to(device)

            # y targets
            y["hamiltonian"] = y["hamiltonian"].to(device)
            y["overlap"] = y["overlap"].to(device)
            y["density"] = y["density"].to(device)
            y["energy"] = y["energy"].to(device)
            y["num_electrons"] = y["num_electrons"].to(device)

            self.samples[idx] = (x_gnn, x_matrix, y)
        return self
