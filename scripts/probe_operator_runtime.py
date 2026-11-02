"""GPU allocation compatibility/memory probe, never a benchmark accuracy run.

Run through Slurm with the chosen runtime profile. An optional registered config
is reduced to its first training and validation records; no checkpoint, optimizer
step, test input, or accuracy selection is used.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import importlib
import json
import os
from pathlib import Path
import platform
import tempfile

from omegaconf import OmegaConf
import torch

SCOPE = "Technical GPU compatibility and peak-memory check only; no benchmark accuracy conclusion"


def module_origins():
    source = Path(__file__).resolve().parents[1] / "src"
    result = {}
    for name in ("data.sample", "data.factory", "data.snapshot", "net.e3gnn"):
        module = importlib.import_module(name)
        actual = getattr(module, "__file__", None)
        expected = source.joinpath(*name.split(".")).with_suffix(".py").resolve()
        if actual is None or Path(actual).resolve() != expected:
            raise ValueError(f"{name} import is outside frozen source: {actual}")
        result[name] = str(expected)
    return result


def validate_versions(
    metadata, *, expected_torch_prefix=None, expected_cuda_prefix=None
):
    for key, expected in (
        ("torch", expected_torch_prefix),
        ("cuda", expected_cuda_prefix),
    ):
        actual = metadata[key]
        if expected is not None and (
            actual is None or not str(actual).startswith(expected)
        ):
            raise ValueError(f"Expected {key} prefix {expected!r}, observed {actual!r}")


def runtime_metadata(
    *, expected_torch_prefix=None, expected_cuda_prefix=None, torch_module=None
):
    runtime = torch if torch_module is None else torch_module
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("This GPU probe requires an allocated Slurm job")
    origins = module_origins()
    versions = {"torch": str(runtime.__version__), "cuda": runtime.version.cuda}
    validate_versions(
        versions,
        expected_torch_prefix=expected_torch_prefix,
        expected_cuda_prefix=expected_cuda_prefix,
    )
    if not runtime.cuda.is_available():
        raise RuntimeError("Allocated runtime cannot access a CUDA GPU")
    properties = runtime.cuda.get_device_properties(0)
    e3nn, scatter = importlib.import_module("e3nn"), importlib.import_module(
        "torch_scatter"
    )
    return {
        **versions,
        "python": platform.python_version(),
        "python_executable": os.sys.executable,
        "e3nn": str(e3nn.__version__),
        "torch_scatter": str(scatter.__version__),
        "dependency_origins": {
            "torch": runtime.__file__,
            "e3nn": e3nn.__file__,
            "torch_scatter": scatter.__file__,
        },
        "module_origins": origins,
        "gpu": {
            "device": 0,
            "name": properties.name,
            "capability": [properties.major, properties.minor],
            "total_memory_bytes": properties.total_memory,
            "torch_arch_list": runtime.cuda.get_arch_list(),
        },
        "allocation": {
            key: os.environ.get(key)
            for key in (
                "SLURM_JOB_ID",
                "SLURM_JOB_NODELIST",
                "SLURM_JOB_PARTITION",
                "CUDA_VISIBLE_DEVICES",
            )
        },
    }


def miniature_document(document):
    """Select technical records from explicit registered train/validation lists."""
    result = deepcopy(
        OmegaConf.to_container(document, resolve=True)
        if OmegaConf.is_config(document)
        else document
    )
    splits = result.get("data", {}).get("splits")
    if splits is None:
        raise ValueError("Probe requires explicit registered data.splits")
    if splits.get("test"):
        raise ValueError("Runtime probe refuses test inputs")
    if not splits.get("train") or not splits.get("val"):
        raise ValueError("Runtime probe requires nonempty train and validation lists")
    if result.get("symmetrize_output", False):
        raise ValueError("Runtime probe requires raw training outputs")
    result["data"]["splits"] = {
        "train": [splits["train"][0]],
        "val": [splits["val"][0]],
        "test": [],
    }
    result.setdefault("training", {}).pop("split", None)
    return result


def exercise_model(document, *, device):
    """One shared training forward/backward and one raw validation forward."""
    from data.sample import move_tree
    from scripts.train import build_components

    dtype_name = document.get("training", {}).get(
        "dtype", document.get("dtype", "float32")
    )
    if dtype_name not in {"float32", "float64"}:
        raise ValueError("Unsupported probe dtype")
    torch.set_default_dtype(getattr(torch, dtype_name))
    _, _, model, datasets, inputs = build_components(document, resume=False)
    model.to(device).train()
    sample = move_tree(datasets["train"][0], device)
    model.zero_grad(set_to_none=True)
    loss, _, _ = model._graph_step(sample, 0, "train")
    if not bool(torch.isfinite(loss)):
        raise RuntimeError("Nonfinite training loss in technical probe")
    loss.backward()
    gradients = [
        parameter.grad for parameter in model.parameters() if parameter.grad is not None
    ]
    if not gradients or not all(
        bool(torch.isfinite(gradient).all()) for gradient in gradients
    ):
        raise RuntimeError("Missing or nonfinite training gradients in technical probe")
    model.zero_grad(set_to_none=True)
    model.eval()
    validation, _ = move_tree(datasets["val"][0], device)
    raw = model.predict_matrices(validation)
    matrices = {}
    for name, matrix in raw.items():
        if not all(
            bool(torch.isfinite(block).all()) for block in matrix.pair_blocks.values()
        ):
            raise RuntimeError(f"Nonfinite raw validation {name} output")
        matrices[name] = {
            "directed_blocks": len(matrix.lookup),
            "entries": sum(block.numel() for block in matrix.pair_blocks.values()),
        }
    return {
        "structures": {name: len(datasets[name]) for name in ("train", "val", "test")},
        "selected_inputs": inputs,
        "statistics_policy": "Fitted only on the single selected training record; checkpoint state is unused",
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "parameter_tensors_with_finite_gradients": len(gradients),
        "training_loss_finite": True,
        "training_backward_finite": True,
        "raw_validation_outputs_finite": True,
        "raw_validation_matrices": matrices,
        "optimizer_steps": 0,
    }


def probe(config=None, *, expected_torch_prefix=None, expected_cuda_prefix=None):
    # Refuse test membership before initializing a dataset or a GPU context.
    document = None if config is None else miniature_document(OmegaConf.load(config))
    metadata = runtime_metadata(
        expected_torch_prefix=expected_torch_prefix,
        expected_cuda_prefix=expected_cuda_prefix,
    )
    result = {"scope": SCOPE, "runtime": metadata, "test_inputs": 0, "passed": True}
    torch.cuda.reset_peak_memory_stats(0)
    if document is not None:
        result["config_sha256"] = hashlib.sha256(Path(config).read_bytes()).hexdigest()
        result["model_probe"] = exercise_model(document, device=torch.device("cuda:0"))
    torch.cuda.synchronize(0)
    result["cuda_peak_memory"] = {
        "allocated_bytes": torch.cuda.max_memory_allocated(0),
        "reserved_bytes": torch.cuda.max_memory_reserved(0),
    }
    return result


def write_exclusive_json(path, payload):
    """Publish complete JSON atomically without replacing any existing result."""
    path = Path(path)
    encoded = json.dumps(payload, indent=2, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", dir=path.parent, prefix=".runtime_probe_", delete=False
        ) as stream:
            temporary = Path(stream.name)
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--expected-torch-prefix")
    parser.add_argument("--expected-cuda-prefix")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError(args.output)
    try:
        result = probe(
            args.config,
            expected_torch_prefix=args.expected_torch_prefix,
            expected_cuda_prefix=args.expected_cuda_prefix,
        )
    except Exception as error:
        result = {
            "scope": SCOPE,
            "passed": False,
            "error_type": type(error).__name__,
            "error": str(error),
        }
    write_exclusive_json(args.output, result)
    print(
        json.dumps(
            {"output": str(args.output), "passed": result["passed"], "scope": SCOPE}
        )
    )
    if not result["passed"]:
        raise SystemExit(1)
    return result


if __name__ == "__main__":
    main()
