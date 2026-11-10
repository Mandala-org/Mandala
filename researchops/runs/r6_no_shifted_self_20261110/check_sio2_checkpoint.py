"""Strict CPU load of the live SiO2 checkpoint, without writing to the run."""

import json
from pathlib import Path

import torch

from core.block_irrep_mapper import BlockIrrepMapper
from core.orbital_irrep_config import OrbitalIrrepConfig
from net.common import get_torch_dtype
from net.e3gnn import E3GNN
from net.evaluation import load_checkpoint, restore_config

path = Path(
    "/bigdata/casus/wdm/hamiltonian_learning/models/checkpoints/publication_retrain_v2/sio2/latest_checkpoint.pt"
)
payload = load_checkpoint(path)
cfg = restore_config(payload)
torch.set_default_dtype(get_torch_dtype(cfg.dtype))
mapper = BlockIrrepMapper(
    OrbitalIrrepConfig.from_dict(payload["hyper_parameters"]["orbital_cfg"]),
    dtype=get_torch_dtype(cfg.dtype),
)
model = E3GNN.from_checkpoint_config(mapper, cfg).eval()
model.load_state_dict(payload["state_dict"], strict=True)
print(
    json.dumps(
        {
            "checkpoint": str(path),
            "epoch": payload["epoch"],
            "strict_load": "passed",
            "device": str(model.device),
            "legacy_separate_shifted_self": model.legacy_separate_shifted_self,
            "state_dict_keys": len(model.state_dict()),
        }
    ),
    flush=True,
)
