"""Export native directed operator blocks from a modular E3GNN checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil

import h5py
import torch

from core.block_irrep_mapper import BlockIrrepMapper
from core.orbital_irrep_config import OrbitalIrrepConfig
from data.operator_dataset import load_operator_sample
from net.configuration import resolve_configuration
from net.e3gnn import E3GNN


def predict(checkpoint, snapshot, output, *, graph_radius=None, device="cpu"):
    checkpoint, snapshot, output = map(Path, (checkpoint, snapshot, output))
    specification = json.loads((checkpoint.parent / "specification.json").read_text())
    frozen = specification["configuration"]
    dtype = getattr(torch, frozen["training"].get("dtype", "float32"))
    previous_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(dtype)
        document, codec = resolve_configuration(
            frozen,
            species_shells=frozen["species"]["shells"],
            spinful=frozen["codec"]["spinful"],
        )
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        orbital_spec = payload.get("hyper_parameters", {}).get("orbital_cfg")
        if orbital_spec is None:
            raise ValueError("Modular inference requires checkpointed orbital metadata")
        mapper = BlockIrrepMapper(
            OrbitalIrrepConfig.from_dict(orbital_spec), dtype=dtype
        )
        model = E3GNN.load_from_checkpoint_with_mapper(
            str(checkpoint), mapper, map_location="cpu"
        )
        model.to(device=device, dtype=dtype)
        data = document["data"]
        sample = load_operator_sample(
            snapshot,
            codec,
            species=document["species"]["order"],
            split="inference",
            matrix_files={},
            support_file=data.get("support_file", "hamiltonians.h5"),
            local_coordinates=data.get("local_coordinates", False),
            num_l=data.get("num_l", 5),
            frame_source=data.get("frame_source", "derive"),
            target_frame=data.get("target_frame", "global"),
            graph_source=(
                "radius"
                if graph_radius is not None
                else data.get("graph_source", "stored")
            ),
            graph_radius=(
                graph_radius if graph_radius is not None else data.get("graph_radius")
            ),
            dtype=dtype,
        )
        predictions = model.predict_operator_blocks(sample, codec, return_masks=True)
        if output.exists() and any(output.iterdir()):
            raise FileExistsError(
                f"Prediction output must be a new or empty directory: {output}"
            )
        output.mkdir(parents=True, exist_ok=True)
        for filename in (
            "element.dat",
            "orbital_types.dat",
            "site_positions.dat",
            "lat.dat",
        ):
            shutil.copyfile(snapshot / filename, output / filename)
        names = {
            "hamiltonian": "hamiltonians.h5",
            "overlap": "overlaps.h5",
            "density": "density_matrixs.h5",
        }
        for name, (blocks, masks) in predictions.items():
            filename = names.get(name, f"{name}.h5")
            with (
                h5py.File(output / filename, "w") as values,
                h5py.File(output / f"{name}_coverage.h5", "w") as coverage,
            ):
                for key, block in blocks.items():
                    native_key = str([*key[:3], key[3] + 1, key[4] + 1])
                    values[native_key] = block.detach().cpu().numpy()
                    coverage[native_key] = masks[key].detach().cpu().numpy()
        digest = hashlib.sha256()
        with checkpoint.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024**2), b""):
                digest.update(chunk)
        info = json.loads((snapshot / "info.json").read_text())
        sources = {
            name: hashlib.sha256((snapshot / name).read_bytes()).hexdigest()
            for name in (
                "element.dat",
                "orbital_types.dat",
                "site_positions.dat",
                "lat.dat",
                "info.json",
            )
        }
        info["prediction"] = {
            "checkpoint": str(checkpoint.resolve()),
            "sha256": digest.hexdigest(),
            "units": {"energy": "eV", "length": "Angstrom"},
            "orbital_basis": "openmx",
            "frame": "global",
            "hermitian_projection": False,
            "source_snapshot": str(snapshot.resolve()),
            "source_sha256": sources,
            "graph": {
                "source": (
                    "radius"
                    if graph_radius is not None
                    else data.get("graph_source", "stored")
                ),
                "radius": (
                    graph_radius
                    if graph_radius is not None
                    else data.get("graph_radius")
                ),
                "support_file": (
                    None
                    if graph_radius is not None
                    else data.get("support_file", "hamiltonians.h5")
                ),
            },
            "coverage_files": {name: f"{name}_coverage.h5" for name in predictions},
        }
        (output / "info.json").write_text(json.dumps(info, indent=2) + "\n")
        return predictions
    finally:
        torch.set_default_dtype(previous_dtype)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--graph-radius", type=float)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)
    predict(
        args.checkpoint,
        args.snapshot,
        args.output,
        graph_radius=args.graph_radius,
        device=args.device,
    )


if __name__ == "__main__":
    main()
