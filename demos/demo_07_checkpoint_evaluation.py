"""Minimal checkpoint evaluation workflow on the bundled H2O snapshot."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from data.snapshot import Snapshot
from net.common import Config
from net.evaluation import (
    predict_snapshot,
    predictions_to_snapshot,
    restore_model_for_snapshot,
)

snapshot = Snapshot.from_openmx(
    matrix_path=REPO_ROOT / "data" / "small" / "H2O" / "original" / "H2O.matrix",
    info_path=REPO_ROOT / "data" / "small" / "H2O" / "original" / "H2O.info.out",
    convention="e3nn",
    cfg=Config(allow_openmx_positions_box_from_out=True, verbosity=0),
)

checkpoint = REPO_ROOT / "best_model.pt"
if not checkpoint.exists():
    print("Set checkpoint to a trained .pt file to run this demo:", checkpoint)
else:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = restore_model_for_snapshot(checkpoint, snapshot, device=device)
    predictions = predict_snapshot(model, snapshot)
    predicted = predictions_to_snapshot(
        predictions, snapshot, reference_matrices=("overlap", "density")
    )
    print("predicted matrices:", sorted(predictions))
    if predicted.box is not None:
        bands = predicted.get_band_structure(path="GXWKG", npoints=200)
        print("band eigenvalues shape:", tuple(bands.eigenvalues.shape))
