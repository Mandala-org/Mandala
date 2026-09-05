#!/usr/bin/env python3
"""Plot validation six-atom Hamiltonian comparisons for the three Stage-5 leaders."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.torch_version import TorchVersion

from core.orbital_irrep_config import OrbitalIrrepConfig
from data.envelope import build_edge_envelope, load_pair_envelope_table
from pair_descriptors import AtomicNeighborDensity
from pair_hamiltonian.hamgnn_sio2 import HARTREE_TO_MEV
from pair_hamiltonian.output_schema import FullBlockIrrepTransform
from pair_hamiltonian.sio2_cache import CANONICAL_PAIR_NAMES
from pair_mappers import ClosedFormM0PairMapper, NativeACEPairMapper

SPECIES_BY_Z = {8: "O", 14: "Si"}
CANONICAL_PAIRS = tuple(tuple(value.split("-")) for value in CANONICAL_PAIR_NAMES)
IRREP_LABELS = ("0e", "1o", "1e", "2o", "2e", "3o", "3e", "4e")
AO_PER_ATOM = 13


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-cache-dir", type=Path, required=True)
    parser.add_argument("--baseline-summary", type=Path, required=True)
    parser.add_argument("--shard-registry", type=Path, required=True)
    parser.add_argument("--range-envelope", type=Path, required=True)
    parser.add_argument("--native-config", type=Path, required=True)
    parser.add_argument("--native-checkpoint", type=Path, required=True)
    parser.add_argument("--d4-cache-dir", type=Path, required=True)
    parser.add_argument("--d4-schemas", type=Path, required=True)
    parser.add_argument("--m0-high-checkpoint", type=Path, required=True)
    parser.add_argument("--m0-compact-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--atom-count", type=int, default=6)
    parser.add_argument("--color-limit-hartree", type=float, default=0.05)
    parser.add_argument("--envelope-floor-hartree", type=float, default=1.0e-8)
    return parser.parse_args()


def atomic_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def first_validation_row(path: Path) -> dict[str, str]:
    with path.open(newline="") as stream:
        for row in csv.DictReader(stream):
            if row["split"] == "validation":
                return row
    raise ValueError("registry contains no validation structures")


def load_baseline(path: Path) -> dict[str, np.ndarray]:
    names = (
        "atomic_numbers",
        "descriptor",
        "onsite_target_irreps_hartree",
        "offsite_source",
        "offsite_target",
        "offsite_image",
        "offsite_displacement_angstrom",
        "offsite_pair_type",
        "offsite_target_irreps_hartree",
    )
    with h5py.File(path, "r") as handle:
        if not handle.attrs.get("complete", False):
            raise ValueError(f"incomplete shard: {path}")
        return {name: handle[name][:] for name in names}


def load_d4(path: Path, keys: tuple[str, ...]) -> dict[str, np.ndarray]:
    with h5py.File(path, "r") as handle:
        if not handle.attrs.get("complete", False):
            raise ValueError(f"incomplete descriptor shard: {path}")
        return {key: handle["descriptors"][key][:] for key in keys}


def make_native_model(
    cache_metadata: dict[str, object], config: dict[str, object]
) -> NativeACEPairMapper:
    orbital_config = OrbitalIrrepConfig.from_dict({"O": "2s2p1d", "Si": "2s2p1d"})
    transform = FullBlockIrrepTransform(orbital_config, dtype=torch.float32)
    density = AtomicNeighborDensity(
        (8, 14),
        n_radial=int(cache_metadata["density_radial_count"]),
        l_max=int(cache_metadata["density_l_max"]),
        cutoff=float(cache_metadata["descriptor_cutoff_angstrom"]),
    )
    return NativeACEPairMapper(
        transform,
        density.layout,
        onsite_correlation_order=int(config["onsite_correlation_order"]),
        onsite_max_degree=int(config["onsite_max_degree"]),
        bond_n_radial=int(config["bond_radial_count"]),
        bond_l_max=int(config["bond_l_max"]),
        bond_cutoff=float(config["bond_cutoff_angstrom"]),
        offsite_max_degree=int(config["offsite_max_degree"]),
        ridge=float(config["ridge"]),
        dtype=torch.float32,
    )


def make_m0_model(schema: dict[str, object]) -> ClosedFormM0PairMapper:
    transform = FullBlockIrrepTransform(
        OrbitalIrrepConfig.from_dict({"O": "2s2p1d", "Si": "2s2p1d"}),
        dtype=torch.float64,
    )
    return ClosedFormM0PairMapper(
        transform,
        str(schema["irreps_out"]),
        bond_n_radial=2,
        bond_l_max=4,
        bond_cutoff=6.5,
        ridge=1.0e-8,
        dtype=torch.float64,
    )


def selected_edges(data: dict[str, np.ndarray], atom_count: int) -> np.ndarray:
    image_zero = np.all(data["offsite_image"] == 0, axis=1)
    return np.flatnonzero(
        image_zero
        & (data["offsite_source"] < atom_count)
        & (data["offsite_target"] < atom_count)
    )


@torch.no_grad()
def predict(
    model: NativeACEPairMapper | ClosedFormM0PairMapper,
    descriptor: np.ndarray,
    data: dict[str, np.ndarray],
    envelope_table,
    envelope_floor: float,
) -> tuple[np.ndarray, np.ndarray]:
    features = torch.from_numpy(descriptor)
    atomic_numbers = torch.from_numpy(data["atomic_numbers"].astype(np.int64))
    source = torch.from_numpy(data["offsite_source"].astype(np.int64))
    target = torch.from_numpy(data["offsite_target"].astype(np.int64))
    displacement = torch.from_numpy(data["offsite_displacement_angstrom"])
    pair_types = torch.from_numpy(data["offsite_pair_type"].astype(np.int64))
    onsite = torch.empty_like(torch.from_numpy(data["onsite_target_irreps_hartree"]))
    offsite = torch.empty_like(torch.from_numpy(data["offsite_target_irreps_hartree"]))
    for atomic_number, species in SPECIES_BY_Z.items():
        index = torch.nonzero(atomic_numbers == atomic_number).flatten()
        onsite[index] = model.predict_onsite(species, features[index]).to(onsite.dtype)
    distance = torch.linalg.vector_norm(displacement, dim=-1)
    envelope = build_edge_envelope(distance, pair_types, envelope_table).clamp_min(
        envelope_floor
    )
    for pair_index, pair in enumerate(CANONICAL_PAIRS):
        index = torch.nonzero(pair_types == pair_index).flatten()
        offsite[index] = (
            model.predict_offsite(
                pair,
                features[source[index]],
                displacement[index],
                features[target[index]],
            ).to(offsite.dtype)
            * envelope[index, None]
        )
    return onsite.numpy(), offsite.numpy()


def label_mask(
    transform: FullBlockIrrepTransform, pair: tuple[str, str], label: str
) -> np.ndarray:
    mask = np.zeros(transform.schema(pair).vector_dimension, dtype=bool)
    for copy in transform.schema(pair).copies:
        if copy.irrep_label == label:
            mask[copy.vector_start : copy.vector_stop] = True
    return mask


def assemble(
    transform: FullBlockIrrepTransform,
    data: dict[str, np.ndarray],
    onsite: np.ndarray,
    offsite: np.ndarray,
    atom_count: int,
    edge_index: np.ndarray,
    label: str | None = None,
) -> np.ndarray:
    dimension = atom_count * AO_PER_ATOM
    matrix = np.zeros((dimension, dimension), dtype=np.float64)
    numbers = data["atomic_numbers"]
    for atom in range(atom_count):
        species = SPECIES_BY_Z[int(numbers[atom])]
        pair = (species, species)
        vector = onsite[atom].copy()
        if label is not None:
            vector[~label_mask(transform, pair, label)] = 0.0
        block = transform.irreps_to_blocks(pair, torch.from_numpy(vector)).numpy()
        start = atom * AO_PER_ATOM
        matrix[start : start + AO_PER_ATOM, start : start + AO_PER_ATOM] = block
    occupied: set[tuple[int, int]] = set()
    for edge in edge_index:
        source = int(data["offsite_source"][edge])
        target = int(data["offsite_target"][edge])
        if (source, target) in occupied or (target, source) in occupied:
            raise ValueError(
                f"duplicate L=0 atom-pair block for atoms {source}, {target}"
            )
        occupied.add((source, target))
        pair = CANONICAL_PAIRS[int(data["offsite_pair_type"][edge])]
        vector = offsite[edge].copy()
        if label is not None:
            vector[~label_mask(transform, pair, label)] = 0.0
        block = transform.irreps_to_blocks(pair, torch.from_numpy(vector)).numpy()
        row = slice(source * AO_PER_ATOM, (source + 1) * AO_PER_ATOM)
        column = slice(target * AO_PER_ATOM, (target + 1) * AO_PER_ATOM)
        matrix[row, column] = block
        matrix[column, row] = block.T
    return matrix


def coefficient_mae(
    transform: FullBlockIrrepTransform,
    data: dict[str, np.ndarray],
    predicted_onsite: np.ndarray,
    predicted_offsite: np.ndarray,
    atom_count: int,
    edge_index: np.ndarray,
    label: str,
) -> tuple[float, int]:
    errors: list[np.ndarray] = []
    numbers = data["atomic_numbers"]
    for atom in range(atom_count):
        species = SPECIES_BY_Z[int(numbers[atom])]
        mask = label_mask(transform, (species, species), label)
        errors.append(
            np.abs(
                predicted_onsite[atom, mask]
                - data["onsite_target_irreps_hartree"][atom, mask]
            )
        )
    for edge in edge_index:
        pair = CANONICAL_PAIRS[int(data["offsite_pair_type"][edge])]
        mask = label_mask(transform, pair, label)
        errors.append(
            np.abs(
                predicted_offsite[edge, mask]
                - data["offsite_target_irreps_hartree"][edge, mask]
            )
        )
    joined = np.concatenate(errors)
    return float(joined.mean() * HARTREE_TO_MEV), int(joined.size)


def plot_triptych(
    truth: np.ndarray,
    prediction: np.ndarray,
    path: Path,
    title: str,
    atom_labels: list[str],
    limit: float,
) -> None:
    residual = prediction - truth
    figure, axes = plt.subplots(1, 3, figsize=(16.2, 5.2), constrained_layout=True)
    shown = None
    for axis, matrix, panel_title in zip(
        axes,
        (truth, prediction, residual),
        ("Ground truth", "Prediction", "Residual (prediction − truth)"),
    ):
        shown = axis.imshow(
            matrix,
            cmap="bwr",
            vmin=-limit,
            vmax=limit,
            origin="upper",
            interpolation="nearest",
        )
        axis.set_title(panel_title)
        centers = np.arange(len(atom_labels)) * AO_PER_ATOM + (AO_PER_ATOM - 1) / 2
        axis.set_xticks(centers, atom_labels, rotation=45, ha="right", fontsize=8)
        axis.set_yticks(centers, atom_labels, fontsize=8)
        axis.set_xlabel("AO columns grouped by atom")
        axis.set_ylabel("AO rows grouped by atom")
        for boundary in np.arange(1, len(atom_labels)) * AO_PER_ATOM - 0.5:
            axis.axhline(boundary, color="black", linewidth=0.35, alpha=0.65)
            axis.axvline(boundary, color="black", linewidth=0.35, alpha=0.65)
    assert shown is not None
    figure.colorbar(shown, ax=axes, label="Hamiltonian element (Hartree)", shrink=0.88)
    figure.suptitle(title, fontsize=11)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path.with_suffix(".png"), dpi=220)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    if (
        args.atom_count <= 0
        or args.color_limit_hartree <= 0
        or args.envelope_floor_hartree <= 0
    ):
        raise ValueError("atom count, color limit, and envelope floor must be positive")
    output = args.output_dir.resolve()
    existing = (
        []
        if not output.exists()
        else [p for p in output.iterdir() if p.name != "launcher.log"]
    )
    if existing:
        raise FileExistsError(
            f"refusing to overwrite nonempty output directory {output}"
        )
    output.mkdir(parents=True, exist_ok=True)

    baseline_summary = json.loads(args.baseline_summary.read_text())
    native_config = json.loads(args.native_config.read_text())
    schemas = {item["key"]: item for item in json.loads(args.d4_schemas.read_text())}
    high_key = "r6p5_d4_spherical_bessel_high_n8_l6_b3"
    compact_key = "r6p5_d4_spherical_bessel_compact_n4_l3_b2"
    row = first_validation_row(args.shard_registry)
    structure_index = int(row["structure_index"])
    baseline_path = (
        args.baseline_cache_dir / "shards" / f"structure_{structure_index:04d}.h5"
    )
    d4_path = args.d4_cache_dir / "shards" / f"structure_{structure_index:04d}.h5"
    data = load_baseline(baseline_path)
    descriptors = load_d4(d4_path, (high_key, compact_key))
    if args.atom_count > len(data["atomic_numbers"]):
        raise ValueError("requested atom count exceeds structure size")
    edge_index = selected_edges(data, args.atom_count)
    if not edge_index.size:
        raise ValueError("selected six-atom L=0 submatrix contains no offsite blocks")

    envelope_table = load_pair_envelope_table(
        args.range_envelope,
        pair_order=CANONICAL_PAIR_NAMES,
        dtype=torch.float32,
        device="cpu",
    )
    native = make_native_model(baseline_summary["cache_metadata"], native_config)
    # The checkpoint is produced by this repository and contains TorchVersion in
    # its provenance dictionary.  Explicitly allow that inert metadata type while
    # retaining the restricted weights-only unpickler.
    with torch.serialization.safe_globals([TorchVersion]):
        native_payload = torch.load(
            args.native_checkpoint, map_location="cpu", weights_only=True
        )
    native.load_state_dict(native_payload["model_state_dict"])
    high = make_m0_model(schemas[high_key])
    high.load_state_dict(
        torch.load(args.m0_high_checkpoint, map_location="cpu", weights_only=True)
    )
    compact = make_m0_model(schemas[compact_key])
    compact.load_state_dict(
        torch.load(args.m0_compact_checkpoint, map_location="cpu", weights_only=True)
    )
    for model in (native, high, compact):
        model.eval()

    methods = (
        ("native_ace", "Native ACE", 157.62090261675303, native, data["descriptor"]),
        ("m0_d4_high", "M0 · D4 high", 157.8083701491333, high, descriptors[high_key]),
        (
            "m0_d4_compact",
            "M0 · D4 compact",
            157.8215031600897,
            compact,
            descriptors[compact_key],
        ),
    )
    transform = FullBlockIrrepTransform(
        OrbitalIrrepConfig.from_dict({"O": "2s2p1d", "Si": "2s2p1d"}),
        dtype=torch.float64,
    )
    truth_onsite = data["onsite_target_irreps_hartree"]
    truth_offsite = data["offsite_target_irreps_hartree"]
    full_truth = assemble(
        transform, data, truth_onsite, truth_offsite, args.atom_count, edge_index
    )
    channel_truth = {
        label: assemble(
            transform,
            data,
            truth_onsite,
            truth_offsite,
            args.atom_count,
            edge_index,
            label,
        )
        for label in IRREP_LABELS
    }
    if not np.allclose(
        sum(channel_truth.values()), full_truth, atol=2.0e-7, rtol=2.0e-6
    ):
        raise RuntimeError(
            "sum of irrep-channel truth matrices does not reproduce full matrix"
        )
    atom_labels = [
        f"{atom} {SPECIES_BY_Z[int(data['atomic_numbers'][atom])]}"
        for atom in range(args.atom_count)
    ]
    summary: dict[str, object] = {
        "passed": True,
        "partition": "validation",
        "structure_index": structure_index,
        "source_key": int(row["source_key"]),
        "atom_indices": list(range(args.atom_count)),
        "atom_labels": atom_labels,
        "periodic_block": "L=(0,0,0)",
        "selected_canonical_offsite_blocks": int(edge_index.size),
        "color_limit_hartree": args.color_limit_hartree,
        "residual_convention": "prediction_minus_ground_truth",
        "methods": {},
    }
    report_rows: list[tuple[str, str, float, float]] = []
    for slug, display, selection_mae, model, descriptor in methods:
        print(f"Predicting and plotting {display}", flush=True)
        predicted_onsite, predicted_offsite = predict(
            model, descriptor, data, envelope_table, args.envelope_floor_hartree
        )
        full_prediction = assemble(
            transform,
            data,
            predicted_onsite,
            predicted_offsite,
            args.atom_count,
            edge_index,
        )
        full_mae = float(np.mean(np.abs(full_prediction - full_truth)) * HARTREE_TO_MEV)
        method_dir = output / slug
        plot_triptych(
            full_truth,
            full_prediction,
            method_dir / "full",
            f"{display} — first validation structure, atoms 0–{args.atom_count - 1}, L=0\n"
            f"submatrix MAE = {full_mae:.3f} meV; fixed scale ±{args.color_limit_hartree:.2f} Hartree",
            atom_labels,
            args.color_limit_hartree,
        )
        arrays: dict[str, np.ndarray] = {
            "full_ground_truth": full_truth,
            "full_prediction": full_prediction,
            "full_residual": full_prediction - full_truth,
        }
        method_metrics: dict[str, object] = {
            "selection_validation_matrix_mae_mev": selection_mae,
            "six_atom_submatrix_mae_mev": full_mae,
            "per_irrep": {},
        }
        predicted_channels: dict[str, np.ndarray] = {}
        for label in IRREP_LABELS:
            channel_prediction = assemble(
                transform,
                data,
                predicted_onsite,
                predicted_offsite,
                args.atom_count,
                edge_index,
                label,
            )
            predicted_channels[label] = channel_prediction
            coeff_mae, count = coefficient_mae(
                transform,
                data,
                predicted_onsite,
                predicted_offsite,
                args.atom_count,
                edge_index,
                label,
            )
            matrix_mae = float(
                np.mean(np.abs(channel_prediction - channel_truth[label]))
                * HARTREE_TO_MEV
            )
            method_metrics["per_irrep"][label] = {
                "canonical_block_coefficient_mae_mev": coeff_mae,
                "coefficient_count": count,
                "six_atom_channel_submatrix_mae_mev": matrix_mae,
            }
            report_rows.append((display, label, coeff_mae, matrix_mae))
            plot_triptych(
                channel_truth[label],
                channel_prediction,
                method_dir / "irreps" / label,
                f"{display} — {label} channel, first validation structure, atoms 0–{args.atom_count - 1}, L=0\n"
                f"channel coefficient MAE = {coeff_mae:.3f} meV; submatrix MAE = {matrix_mae:.3f} meV",
                atom_labels,
                args.color_limit_hartree,
            )
            arrays[f"{label}_ground_truth"] = channel_truth[label]
            arrays[f"{label}_prediction"] = channel_prediction
            arrays[f"{label}_residual"] = channel_prediction - channel_truth[label]
        if not np.allclose(
            sum(predicted_channels.values()), full_prediction, atol=2.0e-7, rtol=2.0e-6
        ):
            raise RuntimeError(
                f"sum of {display} irrep predictions does not reproduce full matrix"
            )
        np.savez_compressed(method_dir / "matrices.npz", **arrays)
        atomic_json(method_dir / "metrics.json", method_metrics)
        summary["methods"][slug] = method_metrics

    summary["ground_truth_hermiticity_max_abs_hartree"] = float(
        np.max(np.abs(full_truth - full_truth.T))
    )
    atomic_json(output / "summary.json", summary)
    config = {
        "convention": "mandala-stage5-six-atom-validation-comparison-v1",
        "test_shards_read": False,
        "method_selection_basis": "lowest validation matrix MAE among completed methods",
        "missing_or_out_of_cutoff_blocks": "represented by exact zeros",
        "irrep_mae_definition": "MAE of selected canonical L=0 block coefficients, including onsite blocks",
        "inputs": {
            key: str(value.resolve())
            for key, value in vars(args).items()
            if isinstance(value, Path)
        },
        "input_hashes": {
            "baseline_summary": sha256(args.baseline_summary),
            "shard_registry": sha256(args.shard_registry),
            "range_envelope": sha256(args.range_envelope),
            "native_config": sha256(args.native_config),
            "native_checkpoint": sha256(args.native_checkpoint),
            "d4_schemas": sha256(args.d4_schemas),
            "m0_high_checkpoint": sha256(args.m0_high_checkpoint),
            "m0_compact_checkpoint": sha256(args.m0_compact_checkpoint),
            "baseline_shard": sha256(baseline_path),
            "d4_shard": sha256(d4_path),
        },
    }
    atomic_json(output / "config.json", config)
    lines = [
        "# Six-atom validation Hamiltonian comparison",
        "",
        f"Structure `{structure_index}` (source key `{row['source_key']}`), validation partition; atoms 0–{args.atom_count - 1}; periodic block L=(0,0,0).",
        "No test shard was read. The three methods were selected by their already-recorded validation matrix MAE.",
        "All panels use `bwr` with a shared fixed range of [-0.05, 0.05] Hartree. Residual means prediction minus ground truth.",
        "",
        "| Method | Irrep | coefficient MAE (meV) | six-atom channel matrix MAE (meV) |",
        "|---|---:|---:|---:|",
    ]
    lines.extend(
        f"| {method} | {label} | {coeff:.6f} | {matrix:.6f} |"
        for method, label, coeff, matrix in report_rows
    )
    (output / "report.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
