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
import time
from torch.utils.data import Dataset
from e3nn.o3 import Irreps

from core.block_irrep_mapper import BlockIrrepMapper
from net.common import Config

from data.edge_alignment import (
    build_prediction_edge_metadata,
    strict_edge_alignment_check,
    strict_reverse_edge_check,
)
from data.envelope import (
    build_edge_envelope,
    load_pair_envelope_table,
)
from data.graph_features import (
    compute_graph_features,
    compute_graph_features_from_target_edges,
)
from data.snapshot import Snapshot
from data.deeph_e3_parser import is_deeph_e3_snapshot
from data.spectral_fermi_cache import (
    load_spectral_fermi_cache,
    lookup_spectral_fermi_record,
)
from net.spectral_loss import build_spectral_reference
from tqdm.auto import tqdm

# Native spin=0 density normalization; invalidate old doubled-density targets.
SNAPSHOT_CACHE_VERSION = "v5"
PREPROCESSED_SAMPLE_CACHE_VERSION = "v8"


def _serialize_orbital_cfg_key(mapper: BlockIrrepMapper) -> str:
    return json.dumps(mapper.orbital_cfg.to_dict(), sort_keys=True)


def _stat_payload(path: Path | None) -> dict[str, str | int | None]:
    if path is None or not path.exists():
        return {
            "path": None,
            "mtime_ns": None,
            "size": None,
        }
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "mtime_ns": int(stat.st_mtime_ns),
        "size": int(stat.st_size),
    }


def _geometry_source_payload(info_path: Path, cfg: Config) -> dict[str, object]:
    if (
        info_path.name == "info.json"
        and (info_path.parent / "site_positions.dat").is_file()
    ):
        return {
            "source_format": "deeph_e3",
            "elements": _stat_payload(info_path.parent / "element.dat"),
            "orbitals": _stat_payload(info_path.parent / "orbital_types.dat"),
            "positions": _stat_payload(info_path.parent / "site_positions.dat"),
            "lattice": _stat_payload(info_path.parent / "lat.dat"),
            "operator_files": {
                operator: _stat_payload(info_path.parent / filename)
                for operator, filename in cfg.data.get("matrix_files", {}).items()
            },
            "frames": _stat_payload(info_path.parent / "rc.h5"),
            "support": _stat_payload(
                info_path.parent / cfg.data.get("support_file", "hamiltonians.h5")
            ),
            "overlap": _stat_payload(info_path.parent / "overlaps.h5"),
            "density": _stat_payload(info_path.parent / "density_matrixs.h5"),
        }
    cif_path = info_path.with_suffix(".cif")
    return {
        "allow_openmx_positions_box_from_out": bool(
            getattr(cfg, "allow_openmx_positions_box_from_out", False)
        ),
        "cif": _stat_payload(cif_path),
    }


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
        *,
        preprocessing=None,
        split="train",
    ):
        self.cfg = cfg
        self.device = torch.device("cpu")
        self.convention = convention
        self.preprocessing = preprocessing
        from data.sample import collate_graph_samples

        self.collate_fn = collate_graph_samples
        self.split = split
        self.snapshot_paths = list(snapshot_paths)

        if not self.snapshot_paths:
            raise ValueError("At least one snapshot path must be provided")

        # shared, **externally-provided** mapper ------------------------------
        self.mapper: BlockIrrepMapper = mapper
        self.orbital_cfg = mapper.orbital_cfg

        self.species_order = tuple(
            self.cfg.species.get("order", mapper.orbital_cfg.elements())
        )
        self.edge_types = tuple(
            f"{a}-{b}" for a in self.species_order for b in self.species_order
        )
        self.edge_type2idx = {pair: index for index, pair in enumerate(self.edge_types)}
        self.dtype = self.cfg.dtype

        if self.cfg.train_target not in {"matrix", "irreps"}:
            raise ValueError("train_target must be one of 'matrix' or 'irreps'.")

        envelope_mode = str(
            getattr(self.cfg, "hamiltonian_envelope_mode", "off")
        ).lower()
        pair_distance_normalization = str(
            getattr(self.cfg, "pair_distance_normalization", "off")
        ).lower()
        if envelope_mode not in {"off", "normalize_target", "multiply_prediction"}:
            raise ValueError(
                "hamiltonian_envelope_mode must be one of 'off', "
                "'normalize_target', or 'multiply_prediction'."
            )
        if pair_distance_normalization not in {"off", "pair_r0"}:
            raise ValueError(
                "pair_distance_normalization must be one of 'off' or 'pair_r0'."
            )
        self.hamiltonian_envelope_mode = envelope_mode
        self.pair_distance_normalization = pair_distance_normalization
        envelope_path = getattr(self.cfg, "hamiltonian_envelope_path", None)
        loss_weighting_mode = str(
            getattr(self.cfg, "loss_weighting_mode", "off")
        ).lower()
        self.loss_weighting_mode = loss_weighting_mode
        envelope_weight_modes = {
            "envelope_inverse_sqrt_clipped",
            "envelope_inverse_clipped",
        }
        need_envelope_table = (
            envelope_path is not None
            or envelope_mode != "off"
            or pair_distance_normalization != "off"
            or loss_weighting_mode in envelope_weight_modes
        )
        self.envelope_table = None
        self.edge_type_r0 = None
        if need_envelope_table:
            if envelope_path is None:
                raise ValueError(
                    "hamiltonian_envelope_path is required when envelope-based "
                    "modes are enabled."
                )
            self.envelope_table = load_pair_envelope_table(
                envelope_path,
                pair_order=self.edge_types,
                dtype=self.dtype,
                device="cpu",
            )
            if self.pair_distance_normalization == "pair_r0":
                self.edge_type_r0 = self.envelope_table.r0.clone().detach()

        if cfg.train_on_forces and not self.cfg.enable_forces:
            raise Exception("Forces must be enabled to train on them")
        if cfg.train_on_stress and not self.cfg.enable_stress:
            raise Exception("Stress must be enabled to train on it")

        self.sh_irreps: Irreps = Irreps.spherical_harmonics(self.cfg.l_max)
        self.device = torch.device("cpu")
        self.spectral_loss_enabled = bool(
            getattr(self.cfg, "spectral_loss_enabled", False)
        )
        self.spectral_fermi_cache_payload: dict[str, object] | None = None
        self.spectral_fermi_cache_hits = 0
        self.spectral_fermi_cache_misses = 0
        if self.spectral_loss_enabled:
            print(
                "--- Spectral loss dataset precompute enabled: "
                f"kmesh={self.cfg.spectral_loss_kmesh}, "
                f"window=±{float(self.cfg.spectral_loss_window_ev):.2f} eV, "
                f"taper={float(self.cfg.spectral_loss_taper_ev):.2f} eV, "
                f"overlap_psd_cleanup={bool(getattr(self.cfg, 'spectral_loss_overlap_psd_cleanup', True))}, "
                f"overlap_jitter={bool(getattr(self.cfg, 'spectral_loss_overlap_jitter', False))} ---"
            )
            spectral_fermi_cache_path = getattr(
                self.cfg, "spectral_fermi_cache_path", None
            )
            if spectral_fermi_cache_path not in (None, ""):
                self.spectral_fermi_cache_payload = load_spectral_fermi_cache(
                    spectral_fermi_cache_path
                )
                print(
                    "--- Loaded spectral Fermi cache: "
                    f"path={spectral_fermi_cache_path}, "
                    f"records={len(self.spectral_fermi_cache_payload['records'])}, "
                    f"failures={len(self.spectral_fermi_cache_payload['failures'])} ---"
                )

        # preprocess all snapshots
        self.snapshots: List[Tuple[Dict, Dict, Dict]] = []
        self.snapshot_cache_hits = 0
        self.snapshot_cache_misses = 0
        self.preprocessed_cache_hits = 0
        self.preprocessed_cache_misses = 0
        self.loader_times: list[float] | None = None
        self.skipped_snapshot_paths: list[tuple[Path, Path, str]] = []
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
            try:
                sample = self._load_preprocessed_sample_if_available(
                    matrix_path,
                    info_path,
                )
                if sample is None:
                    snapshot = self._load_snapshot(matrix_path, info_path)
                    sample = self._build_and_cache_preprocessed_sample(
                        matrix_path,
                        info_path,
                        snapshot,
                    )
            except Exception as exc:
                if not bool(getattr(self.cfg, "allow_incomplete_dataset", False)):
                    raise RuntimeError(
                        "Snapshot load/preprocess failure encountered while "
                        "allow_incomplete_dataset=False. "
                        f"matrix={matrix_path} info={info_path}"
                    ) from exc
                message = (
                    "!!! WARNING: skipping snapshot due to load/preprocess failure !!! "
                    f"matrix={matrix_path} info={info_path} error={exc}"
                )
                print(message)
                self.skipped_snapshot_paths.append((matrix_path, info_path, str(exc)))
                loaded_samples[idx] = None
                continue
            loaded_samples[idx] = sample
        self.snapshots = [sample for sample in loaded_samples if sample is not None]
        if not self.snapshots:
            raise RuntimeError(
                "All snapshots failed to load or preprocess. "
                f"Skipped={len(self.skipped_snapshot_paths)}."
            )
        if self.skipped_snapshot_paths:
            print(
                "!!! WARNING: skipped "
                f"{len(self.skipped_snapshot_paths)} / {len(self.snapshot_paths)} snapshot(s) "
                "during dataset construction !!!"
            )
            for matrix_path, info_path, error in self.skipped_snapshot_paths[:10]:
                print(
                    "!!! SKIPPED SNAPSHOT !!! "
                    f"matrix={matrix_path} info={info_path} error={error}"
                )
            if len(self.skipped_snapshot_paths) > 10:
                print(
                    "!!! WARNING: additional skipped snapshots not shown: "
                    f"{len(self.skipped_snapshot_paths) - 10}"
                )
        if getattr(self.cfg, "snapshot_cache_dir", None) is None:
            print(
                f"[CACHE] Snapshot loading: cache disabled, loaded {len(self.snapshots)} snapshot(s)"
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
                f"[CACHE] Preprocessed samples: cache disabled, built {len(self.snapshots)} sample(s)"
            )
        else:
            total = self.preprocessed_cache_hits + self.preprocessed_cache_misses
            print(
                "[CACHE] Preprocessed samples: "
                f"{self.preprocessed_cache_hits} hit(s), {self.preprocessed_cache_misses} miss(es), "
                f"{total} total"
            )
        if self.spectral_loss_enabled:
            print(
                "[CACHE] Spectral Fermi lookup: "
                f"{self.spectral_fermi_cache_hits} hit(s), "
                f"{self.spectral_fermi_cache_misses} miss(es)"
            )

    def _resolve_spectral_fermi_level(
        self,
        *,
        snap: Snapshot,
        matrix_path: Path,
        info_path: Path,
    ) -> tuple[torch.Tensor, str]:
        direct = getattr(getattr(snap, "info", None), "fermi_level", None)
        if direct is not None:
            return direct, "snapshot.info"
        if self.spectral_fermi_cache_payload is not None:
            record = lookup_spectral_fermi_record(
                self.spectral_fermi_cache_payload,
                matrix_path,
                info_path,
            )
            if record is not None:
                self.spectral_fermi_cache_hits += 1
                return (
                    torch.tensor(
                        float(record["fermi_level_hartree"]),
                        dtype=snap.box.dtype,
                        device=snap.box.device,
                    ),
                    "spectral_fermi_cache",
                )
            self.spectral_fermi_cache_misses += 1
        raise ValueError(
            "spectral_loss_enabled requires a Fermi level, but neither "
            "snapshot.info.fermi_level nor spectral_fermi_cache_path provided one "
            f"for matrix={matrix_path} info={info_path}."
        )

    # ---------------------------------------------------------------- snapshot caching & helpers
    def _snapshot_cache_file(self, matrix_path: Path, info_path: Path) -> Path | None:
        snapshot_cache_dir = getattr(self.cfg, "snapshot_cache_dir", None)
        if snapshot_cache_dir is None:
            return None
        cache_dir = Path(snapshot_cache_dir).expanduser()
        key = json.dumps(
            {
                "version": SNAPSHOT_CACHE_VERSION,
                "loader_policy": self.cfg.data,
                "symmetrize_density": self.cfg.symmetrize_density_targets,
                "preprocessing": (
                    self.preprocessing.signature()
                    if self.preprocessing is not None
                    else None
                ),
                "matrix": _stat_payload(matrix_path),
                "info": _stat_payload(info_path),
                "geometry": _geometry_source_payload(info_path, self.cfg),
                "convention": self.convention,
                "dtype": str(self.dtype),
            },
            sort_keys=True,
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
        key_payload = {
            "version": PREPROCESSED_SAMPLE_CACHE_VERSION,
            "loader_policy": self.cfg.data,
            "symmetrize_overlap": self.cfg.symmetrize_overlap_targets,
            "symmetrize_density": self.cfg.symmetrize_density_targets,
            "enable_forces": self.cfg.enable_forces,
            "enable_stress": self.cfg.enable_stress,
            "matrix": _stat_payload(matrix_path),
            "info": _stat_payload(info_path),
            "geometry": _geometry_source_payload(info_path, self.cfg),
            "orbital_cfg": _serialize_orbital_cfg_key(self.mapper),
            "preprocessing": (
                self.preprocessing.signature()
                if self.preprocessing is not None
                else None
            ),
            "split": self.split,
            "convention": self.convention,
            "l_max": int(self.cfg.l_max),
            "n_radial": int(self.cfg.n_radial),
            "radial_embedding_scale": self.cfg.radial_embedding_scale,
            "cutoff_radius": float(self.cfg.cutoff_radius),
            "apply_cutoff_to_targets": bool(self.cfg.apply_cutoff_to_targets),
            "graph_source": str(getattr(self.cfg, "graph_source", "geometry_cutoff")),
            "matrix_targets": list(self.cfg.matrix_targets),
            "train_target": self.cfg.train_target,
            "dtype": str(self.dtype),
            "symmetrize_hamiltonian_targets": bool(
                self.cfg.symmetrize_hamiltonian_targets
            ),
            "require_exact_edge_match": bool(self.cfg.require_exact_edge_match),
            "precompute_edge_features": bool(self.cfg.precompute_edge_features),
            "separate_shifted_self": bool(self.cfg.separate_shifted_self),
            "hamiltonian_envelope_mode": self.hamiltonian_envelope_mode,
            "hamiltonian_envelope_path": _stat_payload(
                Path(self.cfg.hamiltonian_envelope_path)
                if getattr(self.cfg, "hamiltonian_envelope_path", None)
                else None
            ),
            "pair_distance_normalization": self.pair_distance_normalization,
            "loss_weighting_mode": self.loss_weighting_mode,
            "spectral_loss_enabled": bool(
                getattr(self.cfg, "spectral_loss_enabled", False)
            ),
            "spectral_loss_kmesh": str(getattr(self.cfg, "spectral_loss_kmesh", "")),
            "spectral_loss_window_ev": float(
                getattr(self.cfg, "spectral_loss_window_ev", 0.0)
            ),
            "spectral_loss_taper_ev": float(
                getattr(self.cfg, "spectral_loss_taper_ev", 0.0)
            ),
            "spectral_loss_huber_delta_ev": float(
                getattr(self.cfg, "spectral_loss_huber_delta_ev", 0.0)
            ),
            "spectral_loss_overlap_psd_cleanup": bool(
                getattr(self.cfg, "spectral_loss_overlap_psd_cleanup", True)
            ),
            "spectral_loss_overlap_jitter": bool(
                getattr(self.cfg, "spectral_loss_overlap_jitter", False)
            ),
            "spectral_fermi_cache_path": _stat_payload(
                Path(self.cfg.spectral_fermi_cache_path)
                if getattr(self.cfg, "spectral_fermi_cache_path", None)
                else None
            ),
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

    def _load_preprocessed_sample_if_available(
        self,
        matrix_path: Path,
        info_path: Path,
    ) -> tuple[dict, dict] | None:
        cache_file = self._preprocessed_sample_cache_file(matrix_path, info_path)
        if cache_file is None or not cache_file.exists():
            return None
        cached = self._load_preprocessed_sample(cache_file)
        if cached is None:
            return None
        self.preprocessed_cache_hits += 1
        return cached

    def _build_and_cache_preprocessed_sample(
        self,
        matrix_path: Path,
        info_path: Path,
        snapshot: Snapshot,
    ) -> tuple[dict, dict]:
        cache_file = self._preprocessed_sample_cache_file(matrix_path, info_path)
        self.preprocessed_cache_misses += 1
        process = (
            self._process_snapshot_to_sample
            if self.preprocessing is None
            else self.preprocessing
        )
        if self.cfg.apply_cutoff_to_targets and self.cfg.cutoff_radius is not None:
            snapshot = snapshot.filter_by_distance(self.cfg.cutoff_radius)
        snapshot = snapshot.symmetrize_matrices(
            hamiltonian=self.cfg.symmetrize_hamiltonian_targets,
            overlap=self.cfg.symmetrize_overlap_targets,
            density=self.cfg.symmetrize_density_targets,
        )
        sample = process(
            snapshot,
            matrix_path=matrix_path,
            info_path=info_path,
            snapshot_label=matrix_path.name,
            **({"split": self.split} if self.preprocessing is not None else {}),
        )
        sample = self._add_scientific_features(
            sample, snapshot, matrix_path, info_path, matrix_path.name
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
        from net.common import get_torch_dtype

        geometry_dtype = get_torch_dtype(
            self.cfg.data.get("geometry_dtype", self.dtype)
        )
        if is_deeph_e3_snapshot(matrix_path, info_path):
            snapshot = Snapshot.from_deeph_e3(
                info_path.parent,
                convention=self.convention,
                cutoff_radius=snapshot_cutoff,
                dtype=geometry_dtype,
                cfg=self.cfg,
                matrix_files=self.cfg.data.get("matrix_files"),
                validate_hermitian=self.cfg.data.get("validate_hermitian", True),
                validate_support=self.cfg.data.get("validate_support", True),
                canonicalize=self.cfg.data.get("canonicalize_edges", True),
                symmetrize_density=self.cfg.symmetrize_density_targets,
            )
        elif matrix_path.suffix == ".npz" or info_path.suffix == ".json":
            snapshot = Snapshot.from_pyscf(
                npz_path=matrix_path,
                json_path=info_path,
                convention=self.convention,
                cutoff_radius=snapshot_cutoff,
                dtype=self.dtype,
                cfg=self.cfg,
            )
        else:
            try:
                snapshot = Snapshot.from_openmx(
                    matrix_path=matrix_path,
                    info_path=info_path,
                    convention=self.convention,
                    symmetrize_density=self.cfg.symmetrize_density_targets,
                    dtype=geometry_dtype,
                    cutoff_radius=snapshot_cutoff,
                    cfg=self.cfg,
                )
            except Exception as exc:
                raise RuntimeError(
                    "Failed to load OpenMX snapshot "
                    f"matrix_path={matrix_path} info_path={info_path}: {exc}"
                ) from exc
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
        matrix_path: Path,
        info_path: Path,
        snapshot_label: str = "snapshot",
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Build graph inputs and targets from one Snapshot.
        """
        with torch.no_grad():
            available_matrices = set(
                getattr(
                    getattr(snap, "info", None),
                    "available_matrices",
                    {"hamiltonian", "overlap", "density"},
                )
            )
            unavailable_targets = set(self.cfg.matrix_targets) - available_matrices
            if unavailable_targets:
                raise ValueError(
                    "Requested matrix targets are absent from this snapshot: "
                    f"{sorted(unavailable_targets)}. Available: "
                    f"{sorted(available_matrices)}"
                )
            if self.spectral_loss_enabled and "overlap" not in available_matrices:
                raise ValueError(
                    "spectral_loss_enabled requires overlaps.h5 for DeepH-E3 data."
                )
            if (self.cfg.train_on_energy or self.cfg.train_on_num_electrons) and not {
                "hamiltonian",
                "overlap",
                "density",
            }.issubset(available_matrices):
                raise ValueError(
                    "Observable training requires Hamiltonian, overlap, and density "
                    "matrices, but this DeepH-E3 snapshot provides only "
                    f"{sorted(available_matrices)}."
                )
            target_edges_before_cutoff = sum(
                edges.shape[1] for edges in snap.hamiltonian.pair_edges.values()
            )
            target_edges_after_cutoff = target_edges_before_cutoff
            hamiltonian_target_matrix = snap.hamiltonian
            hamiltonian_target = hamiltonian_target_matrix
            overlap_target = snap.overlap
            density_target = snap.density

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
            elem2idx = {el: i for i, el in enumerate(self.species_order)}
            node_type_idx = torch.tensor(
                [elem2idx[el] for el in atoms], dtype=torch.long
            )

            graph_source = str(
                getattr(self.cfg, "graph_source", "geometry_cutoff")
            ).lower()
            declared_graph = self.cfg.data.get("graph_source")
            if declared_graph is not None:
                graph_source = {
                    "stored": "target_edges",
                    "radius": "geometry_cutoff",
                }.get(declared_graph, declared_graph)
            graph_kwargs = {
                "positions": snap.positions,
                "box": snap.box,
                "atoms": snap.density.atoms,
                "cfg": self.cfg,
                "sh_irreps": self.sh_irreps,
                "edge_type2idx": self.edge_type2idx,
                "edge_type_r0": (
                    self.edge_type_r0
                    if self.pair_distance_normalization == "pair_r0"
                    else None
                ),
            }
            if (
                declared_graph == "radius"
                and self.cfg.data.get("graph_radius") is not None
            ):
                from dataclasses import replace
                import math

                radius = float(self.cfg.data["graph_radius"])
                if not math.isfinite(radius) or radius <= 0:
                    raise ValueError(
                        "Radius graph requires a finite positive graph_radius"
                    )
                graph_kwargs["cfg"] = replace(self.cfg, cutoff_radius=radius)
            if graph_source == "target_edges":
                support_operator = self.cfg.data.get("support_operator") or {
                    "hamiltonians.h5": "hamiltonian",
                    "overlaps.h5": "overlap",
                    "density_matrixs.h5": "density",
                    **{
                        filename: name
                        for name, filename in self.cfg.data.get(
                            "matrix_files", {}
                        ).items()
                    },
                }.get(self.cfg.data.get("support_file", "hamiltonians.h5"))
                if support_operator not in available_matrices:
                    raise ValueError(
                        "Stored graph requires its declared support operator"
                    )
                graph_features = compute_graph_features_from_target_edges(
                    pair_edges=snap[support_operator].pair_edges,
                    **graph_kwargs,
                )
            elif graph_source == "geometry_cutoff":
                graph_features = compute_graph_features(**graph_kwargs)
            else:
                raise ValueError(
                    "graph_source must be 'geometry_cutoff' or 'target_edges', "
                    f"got {graph_source!r}."
                )
            (
                edge_index,
                edge_shift,
                edge_type_idx,
                edge_length_emb,
                edge_sh,
                num_self_edges,
                edge_lengths,
            ) = graph_features

            # Graph builders may partition stored edges differently from the file.
            # Align targets explicitly; storage order is not a scientific policy.
            graph_keys = torch.cat((edge_shift, edge_index)).T.tolist()
            graph_order = {tuple(edge): index for index, edge in enumerate(graph_keys)}

            def align_target(matrix):
                orders = {
                    pair: torch.tensor(
                        sorted(
                            range(edges.shape[1]),
                            key=lambda row: graph_order.get(
                                tuple(edges[:, row].tolist()), len(graph_order)
                            ),
                        ),
                        device=edges.device,
                    )
                    for pair, edges in matrix.pair_edges.items()
                }
                return matrix.reorder_edges(orders)

            hamiltonian_target_matrix = align_target(hamiltonian_target_matrix)
            hamiltonian_target = hamiltonian_target_matrix
            overlap_target = align_target(overlap_target)
            density_target = align_target(density_target)
            strict_reverse_edge_check(edge_index, edge_shift, edge_set_name="graph")
            for matrix_name, matrix_target in (
                ("hamiltonian", hamiltonian_target),
                ("overlap", overlap_target),
                ("density", density_target),
            ):
                strict_edge_alignment_check(
                    matrix_target,
                    edge_index=edge_index,
                    edge_shift=edge_shift,
                    edge_type_idx=edge_type_idx,
                    atoms_list=list(atoms),
                    edge_types=self.edge_types,
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
            edge_envelope = None
            edge_r0 = None
            if self.envelope_table is not None:
                edge_envelope = build_edge_envelope(
                    edge_lengths=edge_lengths,
                    edge_type_idx=edge_type_idx,
                    envelope_table=self.envelope_table,
                ).to(dtype=self.dtype)
                if self.pair_distance_normalization == "pair_r0":
                    edge_r0 = self.envelope_table.r0.index_select(0, edge_type_idx).to(
                        dtype=self.dtype
                    )
            pred_metadata = build_prediction_edge_metadata(
                edge_index=edge_index,
                edge_shift=edge_shift,
                edge_type_idx=edge_type_idx,
                atoms=atoms_tuple,
                edge_types=self.edge_types,
                edge_type2idx=self.edge_type2idx,
                separate_shifted_self=bool(self.cfg.separate_shifted_self),
                target_pair_edges=hamiltonian_target_matrix.pair_edges,
            )

            has_observable_inputs = {
                "hamiltonian",
                "overlap",
                "density",
            }.issubset(available_matrices)
            y = {
                "hamiltonian": (
                    hamiltonian_target
                    if self.cfg.train_target == "matrix"
                    else hamiltonian_target.to_vectors(self.mapper)
                ),
                "overlap": (
                    overlap_target
                    if self.cfg.train_target == "matrix"
                    else overlap_target.to_vectors(self.mapper)
                ),
                "density": (
                    density_target
                    if self.cfg.train_target == "matrix"
                    else density_target.to_vectors(self.mapper)
                ),
                "energy": snap.get_energy() if has_observable_inputs else None,
                "num_electrons": (
                    snap.get_number_of_electrons() if has_observable_inputs else None
                ),
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
                "edge_length": edge_lengths.to(dtype=self.dtype),
                "num_self_edges": num_self_edges,
                "target_edges_before_cutoff": target_edges_before_cutoff,
                "target_edges_after_cutoff": target_edges_after_cutoff,
                **pred_metadata,
            }
            if edge_envelope is not None:
                x["edge_envelope"] = edge_envelope
            if edge_r0 is not None:
                x["edge_r0"] = edge_r0
            if self.cfg.precompute_edge_features:
                x["edge_length_emb"] = edge_length_emb
                x["edge_sh"] = edge_sh
        return x, y

    def _add_scientific_features(
        self, sample, snap, matrix_path, info_path, snapshot_label
    ):
        """Optional physical targets and geometry apply to every representation."""
        from data.sample import GraphSample
        from data.graph_features import complete_graph_metadata

        x, y = sample
        available = set(
            getattr(
                snap.info, "available_matrices", {"hamiltonian", "overlap", "density"}
            )
        )
        for name in available:
            y.setdefault(name, snap[name])
        physical = {"hamiltonian", "overlap", "density"}.issubset(available)
        if (
            self.cfg.train_on_energy or self.cfg.train_on_num_electrons
        ) and not physical:
            raise ValueError(
                "Observable guidance requires Hamiltonian, overlap and density references"
            )
        for name, value in {
            "energy": snap.get_energy() if physical else None,
            "num_electrons": snap.get_number_of_electrons() if physical else None,
            "forces": snap.forces,
            "stress": snap.stress,
        }.items():
            y[name] = value
        if self.cfg.train_on_forces and snap.forces is None:
            raise ValueError(
                "Force supervision requested but snapshot has no force labels"
            )
        if self.cfg.train_on_stress and snap.stress is None:
            raise ValueError(
                "Stress supervision requested but snapshot has no stress labels"
            )
        if self.cfg.enable_forces:
            x["positions"].requires_grad_()
        if self.cfg.enable_stress and x.get("box") is not None:
            x["box"].requires_grad_()
        if self.cfg.spectral_loss_enabled and "overlap" not in available:
            raise ValueError("Spectral loss requires an overlap reference")
        complete_graph_metadata(
            x,
            self.mapper,
            species_order=self.species_order,
            separate_shifted_self=self.cfg.separate_shifted_self,
        )
        from data.graph_topology import prepare_graph_topology

        prepare_graph_topology(x, self.cfg.data.get("supports"))
        x["_geometry_reference"] = {
            key: x[key].detach().clone()
            for key in ("positions", "box")
            if x.get(key) is not None
        }
        lengths = x.get("edge_length", x.get("edge_lengths"))
        x["edge_length"] = lengths
        if self.envelope_table is not None:
            x["edge_envelope"] = build_edge_envelope(
                edge_lengths=lengths,
                edge_type_idx=x["edge_type_idx"],
                envelope_table=self.envelope_table,
            ).to(self.dtype)
            x["edge_r0"] = self.envelope_table.r0.index_select(
                0, x["edge_type_idx"]
            ).to(self.dtype)
        hamiltonian_target, overlap_target = snap.hamiltonian, snap.overlap
        if self.spectral_loss_enabled:
            fermi_level, fermi_source = self._resolve_spectral_fermi_level(
                snap=snap,
                matrix_path=matrix_path,
                info_path=info_path,
            )
            print(
                f"--- [{snapshot_label}] Starting spectral reference precompute "
                f"(fermi source: {fermi_source}) ---",
                flush=True,
            )
            spectral_t0 = time.perf_counter()
            spectral_payload = build_spectral_reference(
                hamiltonian=hamiltonian_target,
                overlap=overlap_target,
                box=snap.box,
                fermi_level_hartree=fermi_level,
                kmesh_spec=str(self.cfg.spectral_loss_kmesh),
                window_ev=float(self.cfg.spectral_loss_window_ev),
                taper_ev=float(self.cfg.spectral_loss_taper_ev),
                overlap_psd_cleanup=bool(
                    getattr(self.cfg, "spectral_loss_overlap_psd_cleanup", True)
                ),
                overlap_jitter=bool(
                    getattr(self.cfg, "spectral_loss_overlap_jitter", False)
                ),
                progress_label=snapshot_label,
                verbose=True,
            )
            print(
                f"--- [{snapshot_label}] Finished spectral reference precompute "
                f"in {time.perf_counter() - spectral_t0:.2f}s ---",
                flush=True,
            )
            x.update(spectral_payload)
        return (
            sample
            if isinstance(sample, GraphSample)
            else GraphSample(
                x, y, {"id": str(matrix_path.parent.resolve()), "split": self.split}
            )
        )

    # ------------------- torch Dataset interface ---------------------------
    def __len__(self) -> int:
        return len(self.snapshots)

    def __getitem__(
        self, idx: int
    ) -> Tuple[
        Dict[str, torch.Tensor], Dict[str, torch.Tensor], Dict[str, torch.Tensor]
    ]:
        if self.loader_times is None:
            return self.snapshots[idx]
        t0 = time.perf_counter()
        snapshot = self.snapshots[idx]
        self.loader_times.append(time.perf_counter() - t0)
        return snapshot

    def to(self, device: torch.device | str) -> E3GNNDataset:
        """
        Move all dataset inputs and targets to the specified device.
        """
        from data.sample import GraphSample, move_tree

        device = torch.device(device)
        if device != self.device:
            self.snapshots = [
                (
                    sample.to(device)
                    if isinstance(sample, GraphSample)
                    else move_tree(sample, device)
                )
                for sample in self.snapshots
            ]
            self.device = device
        return self

    @property
    def samples(self):
        return self.snapshots
