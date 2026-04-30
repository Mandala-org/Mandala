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
import json
import os
import secrets
import random
from pathlib import Path
import tempfile

from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn.functional as F
import time  # needed for __getitem__ timing
from torch.utils.data import Dataset
from e3nn.o3 import Irreps

from core.block_irrep_mapper import BlockIrrepMapper
from net.common import Config

from data.edge_alignment import (
    build_prediction_edge_metadata,
    reconcile_graph_edges_to_target,
    strict_edge_alignment_check,
    strict_reverse_edge_check,
)
from data.graph_features import (
    compute_edge_geometry_from_static_edges,
    compute_graph_features,
)
from data.snapshot import Snapshot
from tqdm.auto import tqdm


PREPROCESSED_SAMPLE_CACHE_VERSION = "v1"


def _serialize_orbital_cfg_key(mapper: BlockIrrepMapper) -> str:
    return json.dumps(mapper.orbital_cfg.to_dict(), sort_keys=True)


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
        self.device = torch.device("cpu")

        # preprocess all snapshots
        self.snapshots: List[Tuple[Dict, Dict, Dict]] = []
        self.snapshot_cache_hits = 0
        self.snapshot_cache_misses = 0
        self.preprocessed_cache_hits = 0
        self.preprocessed_cache_misses = 0
        load_indices = list(range(len(self.snapshot_paths)))
        if getattr(self.cfg, "shuffle_snapshot_load_order", True):
            warmup_seed = secrets.randbits(64)
            random.Random(warmup_seed).shuffle(load_indices)
            print(
                f"--- Snapshot warmup order shuffled with ephemeral seed={warmup_seed} ---"
            )
        else:
            print("--- Snapshot warmup order preserved ---")
        loaded_samples: list[tuple[dict, dict] | None] = [None] * len(
            self.snapshot_paths
        )
        for idx in tqdm(load_indices, desc="Loading snapshots"):
            matrix_path, info_path = self.snapshot_paths[idx]
            snapshot = self._load_snapshot(matrix_path, info_path)
            try:
                sample = self._load_or_build_preprocessed_sample(
                    matrix_path, info_path, snapshot
                )
            except Exception as exc:
                raise RuntimeError(
                    f"Failed to preprocess snapshot matrix={matrix_path} info={info_path}: {exc}"
                ) from exc
            loaded_samples[idx] = sample
        if any(sample is None for sample in loaded_samples):
            raise RuntimeError("Internal error: some snapshots failed to load.")
        self.snapshots = [sample for sample in loaded_samples if sample is not None]
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
        if getattr(self.cfg, "snapshot_cache_dir", None) is None:
            print(
                f"[CACHE] Preprocessed samples: cache disabled, built {len(self.snapshot_paths)} sample(s)"
            )
        else:
            total = self.preprocessed_cache_hits + self.preprocessed_cache_misses
            print(
                "[CACHE] Preprocessed samples: "
                f"{self.preprocessed_cache_hits} hit(s), {self.preprocessed_cache_misses} miss(es), "
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
                str(self.cfg.apply_cutoff_to_targets),
                str(self.dtype),
                str(mat_stat.st_mtime_ns),
                str(mat_stat.st_size),
                str(info_stat.st_mtime_ns),
                str(info_stat.st_size),
            ]
        )
        key_hash = hashlib.md5(key.encode("utf-8")).hexdigest()
        return cache_dir / f"{matrix_path.stem}_{key_hash}.pt"

    def _preprocessed_sample_cache_file(
        self,
        matrix_path: Path,
        info_path: Path,
    ) -> Path | None:
        snapshot_cache_dir = getattr(self.cfg, "snapshot_cache_dir", None)
        if snapshot_cache_dir is None:
            return None
        cache_root = Path(snapshot_cache_dir).expanduser() / "preprocessed_samples"
        mat_stat = matrix_path.stat()
        info_stat = info_path.stat()
        key_payload = {
            "version": PREPROCESSED_SAMPLE_CACHE_VERSION,
            "matrix_path": str(matrix_path.resolve()),
            "info_path": str(info_path.resolve()),
            "matrix_mtime_ns": mat_stat.st_mtime_ns,
            "matrix_size": mat_stat.st_size,
            "info_mtime_ns": info_stat.st_mtime_ns,
            "info_size": info_stat.st_size,
            "orbital_cfg": _serialize_orbital_cfg_key(self.mapper),
            "convention": self.convention,
            "l_max": int(self.cfg.l_max),
            "n_radial": int(self.cfg.n_radial),
            "radial_embedding_scale": self.cfg.radial_embedding_scale,
            "cutoff_radius": float(self.cfg.cutoff_radius),
            "apply_cutoff_to_targets": bool(self.cfg.apply_cutoff_to_targets),
            "matrix_targets": list(self.cfg.matrix_targets),
            "train_target": self.cfg.train_target,
            "dtype": str(self.dtype),
            "symmetrize_hamiltonian_targets": bool(
                self.cfg.symmetrize_hamiltonian_targets
            ),
            "require_exact_edge_match": bool(self.cfg.require_exact_edge_match),
            "precompute_edge_features": bool(self.cfg.precompute_edge_features),
            "separate_shifted_self": bool(self.cfg.separate_shifted_self),
        }
        key_hash = hashlib.md5(
            json.dumps(key_payload, sort_keys=True).encode("utf-8")
        ).hexdigest()
        return cache_root / f"{matrix_path.stem}_{key_hash}.pt"

    def _load_preprocessed_sample(self, cache_file: Path) -> tuple[dict, dict] | None:
        try:
            return torch.load(cache_file, map_location="cpu", weights_only=False)
        except Exception as exc:
            print(
                f"[CACHE] Failed to load preprocessed sample cache {cache_file}: {exc!r}. "
                "Deleting cache entry and rebuilding."
            )
            try:
                cache_file.unlink()
            except Exception:
                pass
            return None

    def _save_preprocessed_sample(
        self, cache_file: Path, sample: tuple[dict, dict]
    ) -> None:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            dir=cache_file.parent,
            prefix=f"{cache_file.stem}.",
            suffix=".tmp",
            delete=False,
        ) as tmp:
            tmp_path = Path(tmp.name)
        try:
            torch.save(sample, tmp_path)
            os.replace(tmp_path, cache_file)
        finally:
            if tmp_path.exists():
                tmp_path.unlink()

    def _load_or_build_preprocessed_sample(
        self,
        matrix_path: Path,
        info_path: Path,
        snapshot: Snapshot,
    ) -> tuple[dict, dict]:
        cache_file = self._preprocessed_sample_cache_file(matrix_path, info_path)
        if cache_file is not None and cache_file.exists():
            cached = self._load_preprocessed_sample(cache_file)
            if cached is not None:
                self.preprocessed_cache_hits += 1
                return cached

        self.preprocessed_cache_misses += 1
        sample = self._process_snapshot_to_sample(
            snapshot,
            snapshot_label=matrix_path.name,
        )
        if cache_file is not None and not cache_file.exists():
            self._save_preprocessed_sample(cache_file, sample)
        return sample

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
        snapshot_cutoff = None
        if not self.cfg.apply_cutoff_to_targets:
            snapshot_cutoff = self.cfg.cutoff_radius
        if matrix_path.suffix == ".npz" or info_path.suffix == ".json":
            snapshot = Snapshot.from_pyscf(
                npz_path=matrix_path,
                json_path=info_path,
                convention=self.convention,
                cutoff_radius=snapshot_cutoff,
                dtype=self.dtype,
                cfg=self.cfg,
            )
        else:
            snapshot = Snapshot.from_openmx(
                matrix_path=matrix_path,
                info_path=info_path,
                convention=self.convention,
                symmetrize_density=True,
                cutoff_radius=snapshot_cutoff,
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
        self,
        snap: Snapshot,
        *,
        snapshot_label: str = "snapshot",
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Build graph inputs and targets from one Snapshot.
        """
        with torch.no_grad():
            target_edges_before_cutoff = sum(
                edges.shape[1] for edges in snap.hamiltonian.pair_edges.values()
            )
            target_edges_after_cutoff = target_edges_before_cutoff
            if self.cfg.apply_cutoff_to_targets and self.cfg.cutoff_radius is not None:
                snap = snap.filter_by_distance(self.cfg.cutoff_radius)
                target_edges_after_cutoff = sum(
                    edges.shape[1] for edges in snap.hamiltonian.pair_edges.values()
                )

            snap = snap.symmetrize_matrices(
                hamiltonian=self.cfg.symmetrize_hamiltonian_targets,
                overlap=True,
                density=True,
            )

            hamiltonian_target_matrix = snap.hamiltonian

            if self.cfg.train_target == "matrix":
                hamiltonian_target = hamiltonian_target_matrix
                overlap_target = snap.overlap
                density_target = snap.density
            elif self.cfg.train_target == "irreps":
                hamiltonian_target = hamiltonian_target_matrix.to_vectors(self.mapper)
                overlap_target = snap.overlap.to_vectors(self.mapper)
                density_target = snap.density.to_vectors(self.mapper)
            else:
                raise ValueError(
                    f"Unknown train_target {self.cfg.train_target}, must be 'irreps' or 'matrix'"
                )

            if self.cfg.enable_forces:
                snap.positions.requires_grad_()
            if self.cfg.enable_stress and snap.box is not None:
                snap.box.requires_grad_()
            if self.cfg.train_on_forces and snap.forces is not None:
                snap.forces.requires_grad_()
            if self.cfg.train_on_stress and snap.stress is not None:
                snap.stress.requires_grad_()

            atoms = snap.density.atoms
            atoms_tuple = tuple(atoms)
            atom_counts = dict(snap.density.atom_counts)
            elem2idx = {el: i for i, el in enumerate(self.orbital_cfg.elements())}
            node_type_idx = torch.tensor(
                [elem2idx[el] for el in atoms], dtype=torch.long
            )

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

            if self.cfg.require_exact_edge_match:
                edge_index, edge_shift, _ = reconcile_graph_edges_to_target(
                    edge_index=edge_index,
                    edge_shift=edge_shift,
                    reference_matrix=snap.hamiltonian,
                    atoms_list=list(atoms),
                    mapper=self.mapper,
                    positions=snap.positions,
                    box=snap.box,
                    snapshot_label=snapshot_label,
                )
                edge_type_idx = torch.tensor(
                    [
                        self.mapper.edge_type2idx[f"{atoms[i]}-{atoms[j]}"]
                        for i, j in zip(edge_index[0].tolist(), edge_index[1].tolist())
                    ],
                    dtype=torch.long,
                )
                edge_length_emb, edge_sh, _ = compute_edge_geometry_from_static_edges(
                    positions=snap.positions,
                    box=snap.box,
                    edge_index=edge_index,
                    edge_shift=edge_shift,
                    sh_irreps=self.sh_irreps,
                    cutoff_radius=self.cfg.cutoff_radius,
                    n_radial=self.cfg.n_radial,
                    radial_embedding_scale=self.cfg.radial_embedding_scale,
                )
                num_self_edges = int(
                    ((edge_index[0] == edge_index[1]) & (edge_shift == 0).all(dim=0))
                    .sum()
                    .item()
                )

            strict_reverse_edge_check(edge_index, edge_shift, edge_set_name="graph")
            for matrix_name, matrix_target in (
                ("hamiltonian", snap.hamiltonian),
                ("overlap", snap.overlap),
                ("density", snap.density),
            ):
                strict_edge_alignment_check(
                    matrix_target,
                    edge_index=edge_index,
                    edge_shift=edge_shift,
                    edge_type_idx=edge_type_idx,
                    atoms_list=list(atoms),
                    edge_types=self.mapper.edge_types,
                    matrix_name=matrix_name,
                    require_exact=self.cfg.require_exact_edge_match,
                )

            num_species = len(self.orbital_cfg.elements())
            node_one_hot = F.one_hot(node_type_idx, num_classes=num_species).to(
                dtype=self.dtype
            )
            src_type = node_type_idx[edge_index[0]]
            dst_type = node_type_idx[edge_index[1]]
            edge_one_hot = F.one_hot(
                src_type * num_species + dst_type, num_classes=num_species * num_species
            ).to(dtype=self.dtype)
            pred_metadata = build_prediction_edge_metadata(
                edge_index=edge_index,
                edge_shift=edge_shift,
                edge_type_idx=edge_type_idx,
                atoms=atoms_tuple,
                edge_types=self.mapper.edge_types,
                edge_type2idx=self.mapper.edge_type2idx,
                separate_shifted_self=bool(self.cfg.separate_shifted_self),
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

            x = {
                "node_type_idx": node_type_idx,
                "node_one_hot": node_one_hot,
                "positions": snap.positions,
                "box": snap.box,
                "atoms": atoms,
                "atoms_tuple": atoms_tuple,
                "atom_counts": atom_counts,
                "edge_index": edge_index,
                "edge_shift": edge_shift,
                "edge_type_idx": edge_type_idx,
                "edge_one_hot": edge_one_hot,
                "num_self_edges": num_self_edges,
                "target_edges_before_cutoff": target_edges_before_cutoff,
                "target_edges_after_cutoff": target_edges_after_cutoff,
                **pred_metadata,
            }
            if self.cfg.precompute_edge_features:
                x["edge_length_emb"] = edge_length_emb
                x["edge_sh"] = edge_sh
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
            if "node_one_hot" in x:
                x["node_one_hot"] = x["node_one_hot"].to(device)
            if "edge_index" in x:
                x["edge_index"] = x["edge_index"].to(device)
            if "edge_shift" in x:
                x["edge_shift"] = x["edge_shift"].to(device)
            if "edge_type_idx" in x:
                x["edge_type_idx"] = x["edge_type_idx"].to(device)
            if "edge_one_hot" in x:
                x["edge_one_hot"] = x["edge_one_hot"].to(device)
            if "edge_length_emb" in x:
                x["edge_length_emb"] = x["edge_length_emb"].to(device)
            if "edge_sh" in x:
                x["edge_sh"] = x["edge_sh"].to(device)
            if "pred_pair_edges_static" in x:
                x["pred_pair_edges_static"] = {
                    k: v.to(device) for k, v in x["pred_pair_edges_static"].items()
                }
            if "pred_trace_alignment" in x:
                x["pred_trace_alignment"] = {
                    k: (rev_key, idx.to(device))
                    for k, (rev_key, idx) in x["pred_trace_alignment"].items()
                }
            if "edge_partitions" in x:
                x["edge_partitions"] = {
                    key: {name: value.to(device) for name, value in parts.items()}
                    for key, parts in x["edge_partitions"].items()
                }

            # y targets
            y["hamiltonian"] = y["hamiltonian"].to(device)
            y["overlap"] = y["overlap"].to(device)
            y["density"] = y["density"].to(device)
            y["energy"] = y["energy"].to(device)
            y["num_electrons"] = y["num_electrons"].to(device)
            if y["forces"] is not None:
                y["forces"] = y["forces"].to(device)
            if y["stress"] is not None:
                y["stress"] = y["stress"].to(device)

            self.snapshots[idx] = (x, y)
        return self
