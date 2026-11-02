#!/usr/bin/env python3
"""Freeze pinned upstream raw-file preprocessing and three fresh Adam updates.

Unlike generate_reference_fixtures.py, every model input here is produced by
its upstream graph builder from native files. MANDALA forwards, geometry,
losses, and matrix encoders are never used to generate numerical expectations.
MACE-H output scaling is deliberately identity: leaked full-dataset statistics
are outside the accepted reproduction policy and separately tested train-only.
"""

from __future__ import annotations

import argparse
import importlib
from importlib.metadata import version
import json
from pathlib import Path
import tempfile
import sys

import gzip
import h5py
import numpy as np
import torch

from net.configuration import resolve_configuration
from scripts.generate_reference_fixtures import (
    configuration,
    load_source,
    source_model,
    pack,
    PINS,
)


def generate(kind, root):
    torch.set_num_threads(1)
    # Preprocess and evaluate in the declared double precision. DeepH's seeded
    # model initialization below separately follows its float32 source default.
    torch.set_default_dtype(torch.float64)
    config = configuration(kind)
    if kind == "deeph":
        config = resolve_configuration(
            Path(__file__).resolve().parents[1] / "configs/reference/deeph.yaml",
            species_shells={"H": [0, 1]},
            overrides={"parameters": config["parameters"]},
        )[0]
    source, hashes = load_source(kind, root)
    graphmod = importlib.import_module(f"{source.__package__}.graph")
    if kind == "deeph":
        sys.modules["deeph"] = sys.modules[source.__package__]
        sys.modules["deeph"].load_orbital_types = graphmod.load_orbital_types
    positions = np.array([[0.0, 0.0, 0.0], [1.2, 0.3, 0.2], [-0.4, 1.3, 0.7]])
    box = np.array([[5.1, 0.2, 0.0], [0.1, 6.3, 0.3], [0.2, 0.0, 7.2]])
    shells = [0, 1]
    dim = sum(2 * l + 1 for l in shells)
    atomic_number = 1 if kind == "deeph" else 14
    entries = {}
    for i in range(3):
        for j in range(3):
            block = (
                0.01 * np.arange(dim * dim).reshape(dim, dim)
                + 0.07 * (i + 1)
                - 0.02 * j
            )
            entries[str([0, 0, 0, i + 1, j + 1])] = block.tolist()
    # Nonzero image bonds expose lattice/coordinate preprocessing mismatches.
    for sign in [-1, 1]:
        entries[str([sign, 0, 0, 1 if sign > 0 else 2, 2 if sign > 0 else 1])] = (
            0.03 * np.eye(dim) + 0.02 * sign
        ).tolist()
    raw = {
        "positions": positions.tolist(),
        "box": box.tolist(),
        "atomic_numbers": [atomic_number] * 3,
        "shells": shells,
        "hamiltonians": entries,
    }
    with tempfile.TemporaryDirectory() as folder:
        directory = Path(folder)
        np.savetxt(directory / "site_positions.dat", positions.T)
        np.savetxt(directory / "lat.dat", box.T)
        np.savetxt(directory / "element.dat", [atomic_number] * 3, fmt="%d")
        (directory / "orbital_types.dat").write_text(
            (" ".join(map(str, shells)) + "\n") * 3
        )
        (directory / "info.json").write_text('{"isspinful":false}')
        with h5py.File(directory / "hamiltonians.h5", "w") as handle:
            for key, block in entries.items():
                handle[key] = block
        if kind == "deeph":
            import shutil

            shutil.copyfile(directory / "hamiltonians.h5", directory / "overlaps.h5")
            geom = importlib.import_module(f"{source.__package__}.preprocess.get_rc")
            rotate = importlib.import_module(f"{source.__package__}.rotate")
            geom.get_rc(folder, folder, radius=-1)
            rotate.get_rh(folder, folder)
            with h5py.File(directory / "rc.h5") as h:
                raw["frames"] = {
                    key: np.asarray(value).tolist() for key, value in h.items()
                }
            graph = graphmod.get_graph(
                torch.tensor(positions),
                torch.tensor(positions @ np.linalg.inv(box)),
                torch.tensor([atomic_number] * 3),
                "raw_oracle",
                -1,
                0,
                1e-8,
                torch.tensor(box),
                torch.float64,
                folder,
                "h5",
                2,
                True,
                True,
                False,
            )
            slices = {0: slice(0, 1), 1: slice(1, 4)}
            target = torch.cat(
                [
                    graph.term_real[:, slices[l], slices[r]].flatten(1)
                    for l, r in config["layout"]["shell_pairs"]
                ],
                dim=1,
            )
        else:
            graph = graphmod.get_graph(
                torch.tensor(positions),
                torch.tensor(positions @ np.linalg.inv(box)),
                torch.tensor([atomic_number] * 3),
                "raw_oracle",
                -1,
                0,
                True,
                torch.tensor(box),
                torch.float64,
                folder,
                target_file_name="hamiltonians.h5",
            )
            slices = {0: slice(0, 1), 1: slice(1, 4)}
            target = torch.cat(
                [
                    graph.Aij[:, slices[l], slices[r]].flatten(1)
                    for l, r in config["layout"]["shell_pairs"]
                ],
                dim=1,
            )
    seed = 42 if kind == "deeph" else 337
    torch.set_default_dtype(torch.float32 if kind == "deeph" else torch.float64)
    torch.manual_seed(seed)
    if kind == "deeph":
        model = source.HGNN(
            1,
            4,
            8,
            config["layout"]["packed_dim"],
            "GaussianBasis",
            6.0,
            True,
            False,
            True,
            True,
            "LayerNorm",
            "CGConv",
            False,
            False,
            False,
            num_l=2,
        ).double()
        for layer in model.modules():
            if isinstance(layer, source.CGConv):
                layer.propagate = layer._orig_propagate
    else:
        model = source_model(kind, config, source).double()
    torch.set_default_dtype(torch.float64)
    graph.x = torch.zeros(3, dtype=torch.long)
    graph.batch = torch.zeros(3, dtype=torch.long)
    if kind == "deeph":
        graph.edge_attr = graph.edge_attr.double()
        sub = graph.subgraph_dict

        def forward():
            return model(
                graph.x,
                graph.edge_index,
                graph.edge_attr,
                graph.batch,
                sub["subgraph_atom_idx"],
                sub["subgraph_edge_idx"],
                sub["subgraph_edge_ang"].double(),
                sub["subgraph_index"],
            )

    else:
        module = importlib.import_module(f"{source.__package__}.e3modules")
        codec = module.e3TensorDecomp(
            config["layout"]["coupled_irreps"],
            [tuple(p) for p in config["layout"]["shell_pairs"]],
            torch.float64,
        )
        if kind == "mace_h":
            model.mean_tensor.zero_()
            model.std_tensor.fill_(1)

        def forward():
            return codec.get_H(model(graph)[1])

    optimizer_options = dict(
        lr=0.001 if kind == "deeph" else 0.002, betas=[0.9, 0.999], eps=1e-8
    )
    optimizer = torch.optim.Adam(model.parameters(), **optimizer_options)
    initial = {name: pack(p) for name, p in model.named_parameters()}
    steps = []
    for _ in range(3):
        optimizer.zero_grad()
        output = forward()
        loss = (output - target).abs().square().mean()
        loss.backward()
        grads = {
            name: None if p.grad is None else pack(p.grad)
            for name, p in model.named_parameters()
        }
        if kind == "deeph":
            torch.nn.utils.clip_grad_norm_(model.parameters(), 4.2)
        optimizer.step()
        steps.append(
            dict(
                output=pack(output),
                loss=loss.item(),
                gradients=grads,
                updated_parameters={
                    name: pack(p) for name, p in model.named_parameters()
                },
            )
        )
    return dict(
        schema_version=1,
        environment={
            name: version(name)
            for name in ("torch", "e3nn", "torch-geometric", "numpy", "scipy")
        },
        preprocessing_dtype="float64",
        evaluation_dtype="float64",
        kind=kind,
        source=dict(commit=PINS[kind], sha256=hashes),
        configuration=config,
        seed=seed,
        raw=raw,
        edge_index=pack(graph.edge_index),
        edge_attr=pack(graph.edge_attr),
        target=pack(target),
        optimizer=optimizer_options,
        initial_parameters=initial,
        steps=steps,
        initialization_dtype="float32" if kind == "deeph" else "float64",
        source_subgraph=(
            {k: pack(v) for k, v in graph.subgraph_dict.items()}
            if kind == "deeph"
            else None
        ),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deeph-root", type=Path, required=True)
    parser.add_argument("--deeph-e3-root", type=Path, required=True)
    parser.add_argument("--mace-h-root", type=Path, required=True)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("tests/fixtures/raw_training")
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for kind, key in [
        ("deeph", "deeph_root"),
        ("deeph_e3", "deeph_e3_root"),
        ("mace_h", "mace_h_root"),
    ]:
        result = generate(kind, getattr(args, key))
        path = args.output_dir / f"{kind}.json.gz"
        path.write_bytes(
            gzip.compress(
                json.dumps(result, separators=(",", ":"), allow_nan=False).encode(),
                mtime=0,
            )
        )
        print(path, path.stat().st_size)


if __name__ == "__main__":
    main()
