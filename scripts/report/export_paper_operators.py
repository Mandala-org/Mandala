#!/usr/bin/env python
"""Strict, explicitly pinned historical inference control for paper checkpoints.

This executable is launched in a separate interpreter. It deliberately imports
the requested frozen source, not the develop implementation. Physical tests are
performed later by develop. Never use it as an implicit production fallback.
"""

import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--matrix-path", type=Path, required=True)
    parser.add_argument("--info-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--envelope-path", type=Path)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    sys.path[:0] = [str(args.source_root / "src"), str(args.source_root)]
    import torch
    from net.common import Config
    from net.e3gnn import E3GNN
    from data.factory import DatasetFactory
    import net.e3gnn as implementation

    if (
        not Path(implementation.__file__)
        .resolve()
        .is_relative_to(args.source_root.resolve())
    ):
        raise RuntimeError("Historical source import escaped the frozen source root")

    def setstate(self, state):
        values = {}
        for part in state if isinstance(state, tuple) else [state]:
            if isinstance(part, dict):
                values.update(part)
        defaults = Config()
        for name in Config.__dataclass_fields__:
            object.__setattr__(self, name, values.get(name, getattr(defaults, name)))

    Config.__setstate__ = setstate
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = copy.deepcopy(payload["hyper_parameters"]["cfg"])
    cfg.device, cfg.dataset_device, cfg.gpus = (
        args.device,
        None,
        int(args.device.startswith("cuda")),
    )
    cfg.snapshot_cache_dir = None
    cfg.spectral_loss_enabled, cfg.spectral_fermi_cache_path = False, None
    cfg.verbosity = 0
    if args.envelope_path:
        cfg.hamiltonian_envelope_path = str(args.envelope_path)
    factory = DatasetFactory(cfg, convention="e3nn")
    # Historical factory needs training metadata. This performs no fit, loss,
    # gradient update, model selection, or training-statistics estimation.
    factory.add_snapshot(args.matrix_path, args.info_path, purpose="train")
    dataset, _, mapper = factory.create()
    x, _ = dataset[0]
    from scripts.check_compat_reload_equivalence import _move_to_device

    model = E3GNN.from_checkpoint_config(mapper=mapper, cfg=cfg).to(args.device).eval()
    model.load_state_dict(payload["state_dict"], strict=True)
    with torch.no_grad():
        x = _move_to_device(x, torch.device(args.device))
        matrices = model.predicted_irreps_to_block_matrices(model(x), x, physical=True)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    hashes = {}
    for name, matrix in matrices.items():
        path = args.output_dir / f"{name}.pt"
        matrix.to("cpu").save(path)
        hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    (args.output_dir / "export.json").write_text(
        json.dumps(
            {
                "strict_load": True,
                "implementation": str(implementation.__file__),
                "operator_sha256": hashes,
                "postprocessing": "radial envelope only; raw directed predictions",
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
