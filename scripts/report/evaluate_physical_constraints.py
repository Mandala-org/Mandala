#!/usr/bin/env python
"""Evaluate the original paper checkpoints without repairing their predictions.

Run on allocated compute nodes. Output is bounded JSON, never dense matrices.
Historical paper density heads predict restricted total D; current OpenMX
reference parsing returns one spin channel. The factor two is explicit below.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import re
import sys
import subprocess
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

import torch

from analysis.physical_constraints import (
    BlockFourier,
    diagnose_kpoint,
    fermi_consistency,
    public_record,
    summarize_records,
    uniform_kpoints,
)
from core.block_irrep_mapper import BlockIrrepMapper
from data.gnn_dataset import E3GNNDataset
from data.block_matrix import BlockMatrix
from data.snapshot import Snapshot
from net.checkpoint_compat import strip_mapper_keys, set_checkpoint_film_semantics
from net.e3gnn import E3GNN
from net.evaluation import load_checkpoint, restore_config
from utils.units import HARTREE_TO_EV


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, document):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(document, indent=2, default=str, allow_nan=False) + "\n"
    )
    temporary.replace(path)


def select_pairs(material, data_path):
    """Same sorted discovery and Random(seed) shuffle as scripts.dataset.

    Record the entire discovery population: missing historical files can change
    membership, even with the correct seed. Never assert identity from a seed.
    """
    seed = {"zncusnses": 44, "sio2": 42, "silicon": 43}[material]
    # Preserve the publication loader's metadata whitelist. Failed/empty runs
    # were skipped rather than interpreted as ambiguous generic *.out files.
    data_path = data_path if material == "sio2" else data_path / "scale_1"
    directories = ([data_path] if (data_path / "HS.out").is_file() else []) + sorted(
        p for p in data_path.iterdir() if p.is_dir()
    )
    pairs, excluded = [], []
    for directory in directories:
        matrix = directory / "HS.out"
        if not matrix.is_file():
            continue
        info = next(
            (
                directory / name
                for name in (
                    "Si.out",
                    "SiO2.out",
                    "ZnCuSeS.out",
                    "info.dat",
                    "info.txt",
                )
                if (directory / name).is_file()
            ),
            None,
        )
        if info is None:
            excluded.append(str(directory))
        else:
            pairs.append((matrix.resolve(), info.resolve()))
    discovered = list(pairs)
    random.Random(seed).shuffle(pairs)
    train_count, requested_val = (90, 10) if material == "sio2" else (80, 20)
    if len(pairs) <= train_count:
        raise ValueError("No validation structures after the registered training slice")
    train, validation = (
        pairs[:train_count],
        pairs[train_count : train_count + requested_val],
    )
    return validation, {
        "seed": seed,
        "requested_validation_count": requested_val,
        "actual_validation_count": len(validation),
        "historical_membership_certified": False,
        "excluded_without_publication_metadata": excluded,
        "membership_caveat": "Reconstructed from current discovery; compare with an original saved manifest before asserting historical identity.",
        "discovered": [[str(a), str(b)] for a, b in discovered],
        "train": [[str(a), str(b)] for a, b in train],
        "validation": [[str(a), str(b)] for a, b in validation],
    }


def input_metadata(info_path):
    """Read the nonmagnetic OpenMX input, not a charge inferred from predicted D."""
    candidates = [
        info_path.parent / "conf.dat",
        info_path.with_suffix(".dat"),
        info_path,
    ]
    for path in candidates:
        if not path.is_file():
            continue
        text = path.read_text(errors="replace")
        spin = re.search(r"^\s*scf\.SpinPolarization\s+(\S+)", text, re.M | re.I)
        coords = re.search(
            r"<Atoms\.SpeciesAndCoordinates[^\n]*\n(.*?)Atoms\.SpeciesAndCoordinates>",
            text,
            re.S | re.I,
        )
        temp = re.search(
            r"^\s*scf\.ElectronicTemperature\s+([\d.eE+-]+)", text, re.M | re.I
        )
        mesh = re.search(r"^\s*scf\.Kgrid\s+(\d+)\s+(\d+)\s+(\d+)", text, re.M | re.I)
        charge = re.search(r"^\s*scf\.system\.charge\s+([\d.eE+-]+)", text, re.M | re.I)
        if not all((spin, coords, temp, mesh)):
            continue
        if spin[1].lower() != "off":
            raise ValueError(
                "This restricted-total diagnostic requires SpinPolarization off"
            )
        rows = [
            line.split("#")[0].split()
            for line in coords[1].splitlines()
            if line.split("#")[0].strip()
        ]
        electrons = sum(float(row[5]) + float(row[6]) for row in rows) - (
            float(charge[1]) if charge else 0.0
        )
        return {
            "path": str(path),
            "sha256": sha256(path),
            "spin": "restricted_total",
            "num_electrons": electrons,
            "temperature_kelvin": float(temp[1]),
            "native_kmesh": [int(mesh[j]) for j in range(1, 4)],
            "charge": float(charge[1]) if charge else 0.0,
            "charge_source": "explicit" if charge else "OpenMX default zero",
        }
    raise ValueError(
        f"Cannot establish spin, electron count, temperature and native kmesh for {info_path}"
    )


def sparse_trace(density, operator):
    total = 0j
    for (x, y, z, i, j), (pair, index) in density.lookup.items():
        other = operator.lookup.get((-x, -y, -z, j, i))
        if other is not None:
            key, n = other
            a = density.pair_blocks[pair][index].detach().cpu().to(torch.complex128)
            b = operator.pair_blocks[key][n].detach().cpu().to(torch.complex128)
            total += complex((a * b.T).sum())
    return {"real": total.real, "imaginary": total.imag}


def matrix_error(prediction, reference):
    error = magnitude = count = 0
    for edge, (pair, index) in prediction.lookup.items():
        target = reference.lookup.get(edge)
        if target is None:
            continue
        key, n = target
        p = prediction.pair_blocks[pair][index].detach().cpu().double()
        r = reference.pair_blocks[key][n].detach().cpu().double()
        error += float((p - r).abs().sum())
        magnitude += float(r.abs().sum())
        count += p.numel()
    if count == 0:
        raise ValueError("No common operator support")
    extra_sum = extra_count = extra_max = 0
    for edge, (pair, index) in prediction.lookup.items():
        if edge not in reference.lookup:
            block = prediction.pair_blocks[pair][index].detach().abs()
            extra_sum += float(block.sum())
            extra_count += block.numel()
            extra_max = max(extra_max, float(block.max()))
    return {
        "mae": error / count,
        "relative_mae": error / max(magnitude, 1e-30),
        "common_scalar_entries": count,
        "prediction_only_scalar_entries": extra_count,
        "prediction_only_absolute_sum": extra_sum,
        "prediction_only_max_absolute": extra_max,
        "common_plus_prediction_only_mae": (error + extra_sum) / (count + extra_count),
        "prediction_only_blocks": len(set(prediction.lookup) - set(reference.lookup)),
        "reference_only_blocks": len(set(reference.lookup) - set(prediction.lookup)),
    }


def restrict_support(prediction, reference):
    """Explicit sparsity control using reference keys only, never its values."""
    edges, blocks, lookup = {}, {}, {}
    for pair, edge in prediction.pair_edges.items():
        keep = [
            i
            for i, key in enumerate(edge.detach().cpu().T.tolist())
            if tuple(key) in reference.lookup
        ]
        if not keep:
            continue
        edges[pair] = edge[:, keep]
        blocks[pair] = prediction.pair_blocks[pair][keep]
        for i, key in enumerate(edges[pair].detach().cpu().T.tolist()):
            lookup[tuple(key)] = pair, i
    return BlockMatrix(
        prediction.atoms,
        prediction.atom_counts,
        blocks,
        edges,
        lookup,
        prediction.orbital_cfg,
        prediction.basis,
    )


def evaluate_structure(
    model, mapper, cfg, pair, *, mesh, device, historical=None, diagnostic_device=None
):
    matrix_path, info_path = pair
    metadata = input_metadata(info_path)
    full = Snapshot.from_openmx(matrix_path, info_path, cfg=cfg)
    reference = {
        key: getattr(full, key) for key in ("hamiltonian", "overlap", "density")
    }
    reference["density"] = reference["density"] * 2.0
    cutoff = full.filter_by_distance(float(cfg.cutoff_radius))
    truncated = {key: getattr(cutoff, key) for key in reference}
    truncated["density"] = truncated["density"] * 2.0
    if historical is None:
        dataset = E3GNNDataset([pair], mapper, cfg, convention="e3nn", split="val")
        x, _ = dataset[0]
        predictions = model.predict_matrices(x, physical=True)
    else:
        source_root, output_dir, historical_checkpoint = historical
        command = [
            sys.executable,
            str(ROOT / "scripts/report/export_paper_operators.py"),
            "--source-root",
            str(source_root),
            "--checkpoint",
            str(historical_checkpoint),
            "--matrix-path",
            str(matrix_path),
            "--info-path",
            str(info_path),
            "--output-dir",
            str(output_dir),
            "--device",
            str(device),
        ]
        if cfg.hamiltonian_envelope_path:
            command += ["--envelope-path", str(cfg.hamiltonian_envelope_path)]
        subprocess.run(command, check=True)
        exported = json.loads((output_dir / "export.json").read_text())
        predictions = {
            key: BlockMatrix.load(output_dir / f"{key}.pt")
            for key in exported["operator_sha256"]
        }
    # physical=True only restores radial-envelope units. It does NOT project,
    # clip, jitter, rescale density, or enforce any operator constraint.
    groups = {
        "reference_full": reference,
        "reference_cutoff": truncated,
        "predicted_H_reference_SD": dict(
            reference, hamiltonian=predictions["hamiltonian"]
        ),
    }
    restricted = {
        key: restrict_support(value, reference[key])
        for key, value in predictions.items()
    }
    groups["predicted_H_reference_SD_reference_support"] = dict(
        reference, hamiltonian=restricted["hamiltonian"]
    )
    if {"overlap", "density"}.issubset(predictions):
        groups.update(
            {
                "joint_predicted": predictions,
                "joint_predicted_reference_support": restricted,
                "predicted_HS_reference_D": dict(
                    predictions, density=reference["density"]
                ),
                "reference_HS_predicted_D": dict(
                    reference, density=predictions["density"]
                ),
            }
        )
    actual_mesh = (
        tuple(metadata["native_kmesh"])
        if mesh == "native"
        else tuple(map(int, mesh.split("x")))
    )
    kpoints = uniform_kpoints(actual_mesh)
    output = {
        "matrix_path": str(matrix_path),
        "info_path": str(info_path),
        "input_metadata": metadata,
        "mesh": actual_mesh,
        "kpoint_weight": 1 / len(kpoints),
        "matrix_errors_raw_common_support": {},
        "matrix_errors_hermitian_common_support": {},
        "scenarios": {},
    }
    for key, pred in predictions.items():
        raw_error = matrix_error(pred, reference[key])
        projected_error = matrix_error((pred + pred.adjoint()) * 0.5, reference[key])
        if key == "hamiltonian":
            raw_error["mae_ev"] = raw_error["mae"] * HARTREE_TO_EV
            projected_error["mae_ev"] = projected_error["mae"] * HARTREE_TO_EV
        output["matrix_errors_raw_common_support"][key] = raw_error
        output["matrix_errors_hermitian_common_support"][key] = projected_error
    # Reuse Fourier assembly across scenarios, streaming dense matrices one k at a time.
    assemblers = {}
    for group in groups.values():
        for a in group.values():
            if id(a) not in assemblers:
                assemblers[id(a)] = BlockFourier(a, device=diagnostic_device or device)
    records = {key: [] for key in groups}
    for k in kpoints:
        dense = {identity: asm.at(k) for identity, asm in assemblers.items()}
        for label, group in groups.items():
            record = diagnose_kpoint(
                dense[id(group["hamiltonian"])],
                dense[id(group["overlap"])],
                dense[id(group["density"])],
            )
            record["fractional_k"] = k
            records[label].append(record)
    for label, group in groups.items():
        values = records[label]
        output["scenarios"][label] = {
            "operators": {
                key: (
                    "prediction"
                    if group[key] is predictions.get(key)
                    else (
                        "prediction_on_reference_support"
                        if group[key] is restricted.get(key)
                        else "reference"
                    )
                )
                for key in group
            },
            "missing_reverse_blocks": {
                key: assemblers[id(a)].reverse_missing for key, a in group.items()
            },
            "exact_realspace_electron_trace": sparse_trace(
                group["density"], group["overlap"]
            ),
            "hermitian_realspace_electron_trace": sparse_trace(
                (group["density"] + group["density"].adjoint()) * 0.5,
                (group["overlap"] + group["overlap"].adjoint()) * 0.5,
            ),
            "summary": summarize_records(values),
            "fermi_consistency": fermi_consistency(
                values,
                num_electrons=metadata["num_electrons"],
                temperature_kelvin=metadata["temperature_kelvin"],
            ),
            "kpoints": [public_record(r) for r in values],
        }
        traces = [r["density"]["electron_trace"] for r in values if "density" in r]
        if len(traces) == len(kpoints):
            mean_trace = sum(traces) / len(traces)
            output["scenarios"][label]["mesh_electron_count"] = mean_trace
            output["scenarios"][label]["mesh_electron_count_error"] = (
                mean_trace - metadata["num_electrons"]
            )
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--material", choices=["zncusnses", "sio2", "silicon"], required=True
    )
    parser.add_argument("--data-path", type=Path)
    parser.add_argument(
        "--sample-dir",
        type=Path,
        help="One representative structure, not a population claim",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--mesh",
        default="3x3x3",
        help="Frozen Gamma-centered mesh, or native from input",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--film-semantics",
        choices=["o3", "legacy_2026_07"],
        default="o3",
        help="Explicit develop historical-FiLM replay; corrected O(3) behavior remains the default",
    )
    parser.add_argument(
        "--diagnostic-device",
        help="Optional separate device for dense complex128 algebra",
    )
    parser.add_argument("--envelope-path", type=Path)
    parser.add_argument(
        "--prediction-source-root",
        type=Path,
        help="Explicit historical inference control; diagnostics still use develop",
    )
    parser.add_argument(
        "--prediction-cache",
        type=Path,
        help="Large-data path for historical raw operators",
    )
    args = parser.parse_args()
    if bool(args.data_path) == bool(args.sample_dir):
        parser.error("Provide exactly one of --data-path and --sample-dir")
    if args.prediction_source_root and not args.prediction_cache:
        parser.error("Historical control requires a large-data --prediction-cache")
    if args.prediction_source_root and args.film_semantics != "o3":
        parser.error("Select either frozen historical source or develop FiLM replay")
    torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", "4")))
    if args.output_dir.exists():
        raise FileExistsError(
            "Use a fresh output directory; existing evidence is immutable"
        )
    args.output_dir.mkdir(parents=True)
    start = time.monotonic()
    if args.sample_dir:
        from data.sources import openmx_info_path

        pairs = [(args.sample_dir / "HS.out", openmx_info_path(args.sample_dir))]
        split = {
            "type": "representative_only",
            "historical_membership_certified": False,
        }
    else:
        pairs, split = select_pairs(args.material, args.data_path)
    identities = []
    for matrix, info in pairs:
        paths = [matrix, info, info.with_suffix(".cif")]
        identities.append({str(p): sha256(p) for p in paths if p.is_file()})
    contract = {
        "schema_version": 1,
        "arguments": vars(args),
        "split": split,
        "checkpoint_sha256": sha256(args.checkpoint),
        "input_sha256": identities,
        "source_manifest_sha256": (
            sha256(ROOT / "source_manifest.json")
            if (ROOT / "source_manifest.json").is_file()
            else None
        ),
        "density_convention": "original paper prediction is restricted-total; native reference spin=0 multiplied by two",
        "postprocessing": "raw Hermiticity, then evaluation-only Hermitian projection; no clipping, jitter or rescaling",
        "tolerances": {"occupation": 1e-5, "overlap_atol": 1e-8, "overlap_rtol": 1e-8},
        "checkpoint_selection": "fixed original paper model; no selection or training",
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "threads": torch.get_num_threads(),
        },
        "publication_faithfulness": "not certified by strict weight loading; compare predictive errors with archived paper metrics",
    }
    if args.prediction_source_root:
        contract["historical_source_manifest_sha256"] = sha256(
            args.prediction_source_root / "source_manifest.json"
        )
    write_json(args.output_dir / "contract.json", contract)
    payload = load_checkpoint(args.checkpoint)
    cfg = restore_config(payload)
    cfg.dataset_device = "cpu"
    cfg.snapshot_cache_dir = None
    cfg.spectral_loss_enabled = False
    cfg.spectral_fermi_cache_path = None
    cfg.verbosity = 0
    if args.envelope_path:
        cfg.hamiltonian_envelope_path = str(args.envelope_path)
    if cfg.hamiltonian_envelope_path:
        contract["envelope_sha256"] = sha256(cfg.hamiltonian_envelope_path)
    first = Snapshot.from_openmx(*pairs[0], cfg=cfg)
    mapper = BlockIrrepMapper(first.hamiltonian.orbital_cfg, dtype=cfg.dtype)
    del first
    model = None
    if args.prediction_source_root is None:
        model = E3GNN(mapper=deepcopy(mapper), cfg=cfg).to(args.device).eval()
        model.load_state_dict(strip_mapper_keys(payload["state_dict"]), strict=True)
        contract["film_semantics"] = set_checkpoint_film_semantics(
            model, args.film_semantics
        )
    del payload
    contract["resolved_config"] = asdict(cfg)
    write_json(args.output_dir / "contract.json", contract)
    rows = []
    for index, pair in enumerate(pairs):
        print(f"Structure {index + 1}/{len(pairs)}: {pair[0]}", flush=True)
        historical = (
            None
            if args.prediction_source_root is None
            else (
                args.prediction_source_root,
                args.prediction_cache / f"structure_{index:04d}",
                args.checkpoint,
            )
        )
        result = evaluate_structure(
            model,
            mapper,
            cfg,
            pair,
            mesh=args.mesh,
            device=args.device,
            historical=historical,
            diagnostic_device=args.diagnostic_device,
        )
        write_json(args.output_dir / f"structure_{index:04d}.json", result)
        rows.append({key: value for key, value in result.items() if key != "scenarios"})
        rows[-1]["scenarios"] = {
            key: {k: v for k, v in scenario.items() if k != "kpoints"}
            for key, scenario in result["scenarios"].items()
        }
    write_json(
        args.output_dir / "summary.json",
        {
            "status": "complete",
            "material": args.material,
            "structures": len(rows),
            "elapsed_seconds": time.monotonic() - start,
            "results": rows,
        },
    )
    print(
        f"Completed {len(rows)} structures in {time.monotonic() - start:.1f} s",
        flush=True,
    )


if __name__ == "__main__":
    main()
