"""Exercise frozen recipe dataset construction without training or a CUDA device.

Run from the repository root with PYTHONPATH=src:. and the MANDALA environment.
Only device selection is overridden; split manifests and numerical settings stay
as declared by each recipe. This does not create training runs or contracts.
"""

from pathlib import Path
import argparse
import gc
import json
from unittest.mock import patch

import torch

from net.common import Config
from scripts.reproduce_examples.train import load_recipe
from scripts.train import (
    _as_namespace,
    _build_dataset_bundle,
    _normalize_dataset_bundle,
    _populate_config_from_args,
    _resolve_accelerator_and_device,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--resolve-only", action="store_true",
        help="Stop immediately after the actual factory configuration resolution.",
    )
    options = parser.parse_args()
    results = []
    for material in ("silicon", "sio2", "zncusnses"):
        path = Path(f"scripts/reproduce_examples/{material}_v2.json").resolve()
        recipe = load_recipe(path)
        values = recipe["config"] | recipe["training"] | {"device": "cpu", "gpus": 0}
        args = _as_namespace(values)
        cfg = _populate_config_from_args(args)
        _resolve_accelerator_and_device(cfg)
        assert isinstance(cfg.device, torch.device)
        if options.resolve_only:
            # Run the real resolver with the real manifest's orbital metadata.
            # Stop before costly cold snapshot parsing, not before resolution.
            class ConfigurationResolved(Exception):
                pass

            resolve = Config.resolve
            result = {"material": material, "device": str(cfg.device)}

            def resolve_and_stop(*args, **kwargs):
                resolved, _ = resolve(*args, **kwargs)
                assert resolved["device"] == "cpu"
                result["species"] = list(resolved["species"]["shells"])
                result["boundary"] = "factory_configuration_resolved"
                raise ConfigurationResolved

            with patch.object(Config, "resolve", side_effect=resolve_and_stop):
                try:
                    _build_dataset_bundle(args, cfg, None)
                except ConfigurationResolved:
                    pass
                else:
                    raise AssertionError("Dataset factory did not resolve configuration")
            results.append(result)
            print("CPU_CONFIGURATION_PASS " + json.dumps(result), flush=True)
            continue
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
