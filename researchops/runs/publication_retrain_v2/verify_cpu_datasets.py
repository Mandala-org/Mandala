"""Exercise frozen recipe dataset construction without training or a CUDA device.

Run from the repository root with PYTHONPATH=src:. and the MANDALA environment.
Only device selection is overridden; split manifests and numerical settings stay
as declared by each recipe. This does not create training runs or contracts.
"""

from pathlib import Path
import gc
import json

import torch

from scripts.reproduce_examples.train import load_recipe
from scripts.train import (
    _as_namespace,
    _build_dataset_bundle,
    _normalize_dataset_bundle,
    _populate_config_from_args,
    _resolve_accelerator_and_device,
)


def main():
    results = []
    for material in ("silicon", "sio2", "zncusnses"):
        path = Path(f"scripts/reproduce_examples/{material}_v2.json").resolve()
        recipe = load_recipe(path)
        values = recipe["config"] | recipe["training"] | {"device": "cpu", "gpus": 0}
        args = _as_namespace(values)
        cfg = _populate_config_from_args(args)
        _resolve_accelerator_and_device(cfg)
        assert isinstance(cfg.device, torch.device)
        bundle = _build_dataset_bundle(args, cfg, None)
        train, val, test, mapper = _normalize_dataset_bundle(bundle)
        result = {
            "material": material,
            "device": str(cfg.device),
            "train": len(train),
            "val": len(val),
            "test": len(test),
        }
        results.append(result)
        print("CPU_DATASET_PASS " + json.dumps(result), flush=True)
        del bundle, train, val, test, mapper, cfg, args
        gc.collect()
    print("CPU_DATASET_RESULTS " + json.dumps(results), flush=True)


if __name__ == "__main__":
    main()
