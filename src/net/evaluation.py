"""Small, stateless helpers for checkpoint and snapshot evaluation.

These functions deliberately compose the existing :class:`Config`,
:class:`Snapshot`, :class:`BlockIrrepMapper`, and :class:`E3GNN` classes.  They
do not introduce a new model wrapper or change the Lightning checkpoint
format.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import torch

from core.block_irrep_mapper import BlockIrrepMapper
from data.snapshot import Snapshot
from data.structure_inference import build_model_input_from_structure
from net.common import Config, get_torch_dtype
from net.e3gnn import E3GNN
from net.observable_metrics import build_observable_predictions


def _patch_config_unpickling() -> None:
    """Allow checkpoints written with older slotted ``Config`` objects to load."""
    if getattr(Config, "_mandala_legacy_unpickle_patch", False):
        return

    def __setstate__(self: Config, state: Any) -> None:
        state_map: dict[str, Any] = {}
        if isinstance(state, tuple) and len(state) == 2:
            dict_state, slot_state = state
            if isinstance(dict_state, dict):
                state_map.update(dict_state)
            if isinstance(slot_state, dict):
                state_map.update(slot_state)
        elif isinstance(state, dict):
            state_map.update(state)

        defaults = Config()
        for name in Config.__dataclass_fields__:
            object.__setattr__(self, name, state_map.get(name, getattr(defaults, name)))

    Config.__setstate__ = __setstate__  # type: ignore[attr-defined]
    Config._mandala_legacy_unpickle_patch = True  # type: ignore[attr-defined]


def load_checkpoint(path: str | Path) -> dict[str, Any]:
    """Load a Lightning checkpoint on CPU without changing its payload."""
    _patch_config_unpickling()
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Unexpected checkpoint payload type: {type(checkpoint)!r}")
    return checkpoint


def restore_config(checkpoint: dict[str, Any]) -> Config:
    """Restore the saved ``Config`` object from a Mandala checkpoint."""
    cfg = checkpoint.get("hyper_parameters", {}).get("cfg")
    if not isinstance(cfg, Config):
        raise ValueError(
            "Checkpoint does not contain a Config instance in "
            "hyper_parameters['cfg']."
        )
    cfg = copy.deepcopy(cfg)
    if isinstance(cfg.dtype, str):
        dtype_name = cfg.dtype.removeprefix("torch.")
        cfg.dtype = get_torch_dtype(dtype_name)
    if isinstance(cfg.matrix_targets, str):
        cfg.matrix_targets = [cfg.matrix_targets]
    return cfg


def build_mapper(snapshot: Snapshot, cfg: Config) -> BlockIrrepMapper:
    """Build the mapper required by a model from one reference snapshot."""
    return BlockIrrepMapper(
        snapshot.hamiltonian.orbital_cfg,
        dtype=get_torch_dtype(cfg.dtype),
    )


def _move_to_device(obj: Any, device: torch.device) -> Any:
    if isinstance(obj, dict):
        return {key: _move_to_device(value, device) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        values = [_move_to_device(value, device) for value in obj]
        return type(obj)(values)
    if hasattr(obj, "to"):
        try:
            return obj.to(device)
        except TypeError:
            return obj.to(device=device)
    return obj


def snapshot_model_input(
    snapshot: Snapshot,
    cfg: Config,
    *,
    mapper: BlockIrrepMapper,
) -> dict[str, Any]:
    """Construct graph/model inputs from a structure-only ``Snapshot``."""
    if snapshot.positions is None:
        raise ValueError("Snapshot requires positions for model inference.")
    return build_model_input_from_structure(
        atoms=tuple(snapshot.hamiltonian.atoms),
        positions=snapshot.positions,
        box=snapshot.box,
        cfg=cfg,
        mapper=mapper,
    )


def predict_snapshot(
    model: E3GNN,
    snapshot: Snapshot,
) -> dict[str, Any]:
    """Run structure-only inference and return predicted block matrices."""
    device = next(model.parameters()).device
    x = snapshot_model_input(snapshot, model.cfg, mapper=model.mapper)
    x = _move_to_device(x, device)
    return model.predict_matrices(x)


def predictions_to_snapshot(
    predictions: dict[str, Any],
    reference: Snapshot,
    *,
    reference_matrices: tuple[str, ...] = ("overlap", "density"),
) -> Snapshot:
    """Combine predicted matrices with explicitly selected reference matrices."""
    matrices = {
        name: (
            predictions.get(name, getattr(reference, name))
            if name in reference_matrices
            else predictions[name]
        )
        for name in ("hamiltonian", "overlap", "density")
    }
    return Snapshot(
        **matrices,
        positions=reference.positions,
        box=reference.box,
        cfg=reference.cfg,
    )


def restore_model_for_snapshot(
    checkpoint_path: str | Path,
    snapshot: Snapshot,
    *,
    device: str | torch.device = "cpu",
) -> E3GNN:
    """Restore a model using the orbital mapper inferred from ``snapshot``."""
    checkpoint = load_checkpoint(checkpoint_path)
    cfg = restore_config(checkpoint)
    mapper = build_mapper(snapshot, cfg)
    model = E3GNN(mapper=mapper, cfg=cfg)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    return model.to(device).eval()


def evaluate_sample(
    model: E3GNN,
    x: dict[str, Any],
    y: dict[str, Any],
) -> dict[str, Any]:
    """Evaluate one already-aligned dataset sample and its observables."""
    predictions = model.predict_matrices(x)
    observables = build_observable_predictions(
        predictions,
        pred_trace_alignment=x["pred_trace_alignment"],
        H_true=y.get("hamiltonian"),
        D_true=y.get("density"),
        S_true=y.get("overlap"),
    )
    return {"predictions": predictions, "targets": y, "observables": observables}


__all__ = [
    "build_mapper",
    "evaluate_sample",
    "load_checkpoint",
    "predict_snapshot",
    "predictions_to_snapshot",
    "restore_config",
    "restore_model_for_snapshot",
    "snapshot_model_input",
]
