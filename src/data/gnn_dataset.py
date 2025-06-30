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
    ):
        if cutoff_gnn >= cutoff_matrix:
            raise ValueError("cutoff_gnn must be < cutoff_matrix")

        if not snapshot_paths:
            raise ValueError("At least one snapshot path must be provided")

        self.device = torch.device(device)

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
        dist_dict: Dict[str, torch.Tensor],
        cutoff: float,
    ) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]
    ]:
        """
        Returns
        -------
        edge_index : (2,E)  long
        edge_one_hot : (E,n_types)  float
        edge_length_emb : (E,n_radial)  float
        edge_sh : (E, sh_dim)  float
        keep_mask_dict : dict[key] → bool mask (E_key,)  (needed to subset vectors)
        """
        edge_src, edge_dst, etype_idx = [], [], []
        keep_mask_dict: Dict[str, torch.Tensor] = {}

        for key in sorted(snap.density.keys()):
            edges = snap.density.pair_edges[key]  # (2,E_key)
            dists = dist_dict[key]  # (E_key,)
            diag_mask = edges[0] == edges[1]
            keep = (~diag_mask) & (dists <= cutoff)  # off-diag & within cutoff
            if torch.any(keep):
                keep_mask_dict[key] = keep
                edge_src.extend(edges[0][keep].tolist())
                edge_dst.extend(edges[1][keep].tolist())
                etype_idx.extend([self.edge_type2idx[key]] * int(keep.sum()))

        edge_index = torch.tensor(
            [edge_src, edge_dst], dtype=torch.long, device=self.device
        )
        etype_idx = torch.tensor(etype_idx, dtype=torch.long, device=self.device)
        edge_one_hot = torch.nn.functional.one_hot(
            etype_idx, num_classes=self.n_edge_types
        ).to(torch.float32)

        # geometric encodings -------------------------------------------------
        # minimal-image displacement: compute box and its inverse once
        if snap.box is not None:
            box = snap.box.to(self.device)
            inv_box = torch.inverse(box)
        else:
            box = None
            inv_box = None
        disp = self._minimal_disp(
            snap.positions.to(self.device),
            edge_index,
            box,
            inv_box,
        )
        dists_kept = torch.linalg.norm(disp, dim=-1)

        edge_length_emb = soft_one_hot_linspace(
            dists_kept,
            start=0.0,
            end=self.cut_mat,  # upper bound irrelevant due to Gaussian tail
            number=self.n_radial,
            basis="gaussian",
            cutoff=False,
        ).to(torch.float32)

        edge_sh = spherical_harmonics(
            self.sh_irreps, disp, normalize=True, normalization="component"
        ).to(torch.float32)

        return edge_index, edge_one_hot, edge_length_emb, edge_sh, keep_mask_dict

    # ---------- main per-snapshot routine -----------------------------------
    def _process_snapshot(
        self, snap: Snapshot
    ) -> Tuple[
        Dict[str, torch.Tensor], Dict[str, torch.Tensor], Dict[str, torch.Tensor]
    ]:
        """
        Build graph inputs (small and large cutoff) and targets from one Snapshot.

        Returns:
          Tuple of (x_gnn, x_matrix, y), where x_gnn and x_matrix are input dicts
          for the GNN and readout, and y contains the target IrrepsBlockData.
        """
        # pre-compute distances once for the largest graph
        dist_dict = snap._edge_distances(snap.density)

        # ---------- build matrices for *both* cutoffs ------------------------
        (ei_gnn, eo_gnn, el_gnn, es_gnn, keep_gnn) = self._edge_tensors(
            snap, dist_dict, self.cut_gnn
        )

        (ei_mat, eo_mat, el_mat, es_mat, keep_mat) = self._edge_tensors(
            snap, dist_dict, self.cut_mat
        )

        # ---------- node one-hot --------------------------------------------
        atoms = snap.density.atoms
        elem2idx = {el: i for i, el in enumerate(self.global_cfg.elements())}
        N = len(atoms)
        node_oh = torch.zeros(N, len(elem2idx), device=self.device)
        for i, el in enumerate(atoms):
            node_oh[i, elem2idx[el]] = 1.0

        # ---------- overlap vectors split diag / offdiag --------------------
        overlap_ir = snap.overlap.to_vectors(self.mapper)  # IrrepsBlockData
        overlap_diag = overlap_ir.diag()  # dict

        # helper to subset off-diag vectors to the kept edges ----------------
        def _subset_offdiag(
            mask_dict: Dict[str, torch.Tensor]
        ) -> Dict[str, torch.Tensor]:
            out: Dict[str, torch.Tensor] = {}
            for key, mask in mask_dict.items():
                out[key] = overlap_ir.pair_vectors[key][mask]
            return out

        overlap_off_gnn = _subset_offdiag(keep_gnn)
        overlap_off_mat = _subset_offdiag(keep_mat)

        # ========== assemble x_gnn / x_matrix ===============================
        x_gnn = {
            "node_one_hot": node_oh,
            "edge_index": ei_gnn,
            "edge_one_hot": eo_gnn,
            "edge_length_emb": el_gnn,
            "edge_sh": es_gnn,
            "overlap_vectors_diag": overlap_diag,
            "overlap_vectors_offdiag": overlap_off_gnn,
            # per-node element symbols
            "atoms": atoms,
        }
        x_matrix = {
            "edge_index": ei_mat,
            "edge_one_hot": eo_mat,
            "edge_length_emb": el_mat,
            "edge_sh": es_mat,
            "overlap_vectors_diag": overlap_diag,  # same dict
            "overlap_vectors_offdiag": overlap_off_mat,  # superset of gnn
        }

        # ========== targets y ===============================================
        y = {
            "hamiltonian": snap.hamiltonian.to_vectors(self.mapper).to(self.device),
            "overlap": overlap_ir.to(self.device),
            "density": snap.density.to_vectors(self.mapper).to(self.device),
            "energy": snap.get_energy().to(self.device),
            "num_electrons": snap.get_number_of_electrons().to(self.device),
        }

        return x_gnn, x_matrix, y

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
            x_gnn["node_one_hot"] = x_gnn["node_one_hot"].to(device)
            x_gnn["edge_index"] = x_gnn["edge_index"].to(device)
            x_gnn["edge_one_hot"] = x_gnn["edge_one_hot"].to(device)
            x_gnn["edge_length_emb"] = x_gnn["edge_length_emb"].to(device)
            x_gnn["edge_sh"] = x_gnn["edge_sh"].to(device)
            for k, v in x_gnn.get("overlap_vectors_diag", {}).items():
                x_gnn["overlap_vectors_diag"][k] = v.to(device)
            for k, v in x_gnn.get("overlap_vectors_offdiag", {}).items():
                x_gnn["overlap_vectors_offdiag"][k] = v.to(device)
            # atoms list remains unchanged

            # x_matrix
            x_matrix["edge_index"] = x_matrix["edge_index"].to(device)
            x_matrix["edge_one_hot"] = x_matrix["edge_one_hot"].to(device)
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
