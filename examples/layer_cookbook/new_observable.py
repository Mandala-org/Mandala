"""Add an onsite Hamiltonian-trace objective to the cookbook model.

Run from the repository root (the Si files must be hydrated from Git LFS)::

    PYTHONPATH=src:. python examples/layer_cookbook/new_observable.py \
        --output runs/cookbook_trace --accelerator cpu

The adjacent YAML is an overlay on cookbook.yaml. The ordinary trainer writes
checkpoints and manifests; this wrapper only registers the plugin, merges the
overlay, and prints the new loss on the training structure before/after fit.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import tempfile

import torch
from omegaconf import OmegaConf

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import layers  # noqa: E402,F401 (registers the cookbook transformations)

from net.objectives import register_objective_term  # noqa: E402


def onsite_trace_per_atom(matrix):
    """Mean onsite trace in Hartree, using only (R=0, i=j) blocks.

    Requiring complete onsite reference blocks avoids silently training on a
    zero-filled missing value. Explicit edge keys exclude shifted self-images
    and make the result independent of block order.
    """
    if not matrix.atoms:
        raise ValueError("The onsite trace requires at least one atom")
    traces = []
    masks = matrix.coverage_masks
    for atom in range(len(matrix.atoms)):
        edge = (0, 0, 0, atom, atom)
        if edge not in matrix.lookup:
            raise ValueError(f"Missing onsite Hamiltonian block for atom {atom}")
        pair, row = matrix.lookup[edge]
        block = matrix.pair_blocks[pair][row]
        if not masks[pair][row].all() or not torch.isfinite(block).all():
            raise ValueError("Hamiltonian trace requires complete finite onsite blocks")
        traces.append(block.diagonal().sum().real)
    return torch.stack(traces).mean()


@register_objective_term("hamiltonian_trace")
def hamiltonian_trace_loss(*, predictions, targets, inputs):
    """Squared error of the mean onsite trace, in Hartree squared per atom squared."""
    predicted = onsite_trace_per_atom(predictions["hamiltonian"])
    reference = onsite_trace_per_atom(targets["hamiltonian"])
    return (predicted - reference).square()


def main(argv=None):
    from net.evaluation import load_checkpoint
    from scripts.train import build_components, main as train

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=HERE / "new_observable.yaml")
    parser.add_argument("--output", type=Path, default=Path("runs/cookbook_trace"))
    parser.add_argument("--accelerator", default="cpu", choices=("cpu",))
    args = parser.parse_args(argv)
    document = OmegaConf.merge(
        OmegaConf.load(HERE / "cookbook.yaml"), OmegaConf.load(args.config)
    )
    torch.set_default_dtype(getattr(torch, document.training.dtype))
    _, _, model, datasets, _ = build_components(
        OmegaConf.to_container(document, resolve=False)
    )
    graph = datasets["train"][0]

    def measure(candidate):
        candidate.eval()
        with torch.no_grad():
            _, metrics, _ = candidate._graph_step(graph, 0, "val")
        return float(metrics["val/loss_term_hamiltonian_trace"])

    before = measure(model)
    # The standard CLI accepts a single complete recipe. Keep the merged copy
    # temporary: the persistent run manifest already stores the resolved one.
    with tempfile.TemporaryDirectory(prefix="mandala-trace-") as directory:
        config = Path(directory) / "trace.yaml"
        OmegaConf.save(document, config)
        train(
            [
                "--config",
                str(config),
                "--output",
                str(args.output),
                "--accelerator",
                args.accelerator,
            ]
        )
    checkpoint = load_checkpoint(args.output / "latest_checkpoint.pt")
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    after = measure(model)
    print(f"Hamiltonian-trace loss before training: {before:.10e} Hartree^2/atom^2")
    print(
        f"Hamiltonian-trace loss after {document.training.max_epochs} epochs: "
        f"{after:.10e} Hartree^2/atom^2"
    )
    if not after < before:
        raise AssertionError("The two-epoch example did not reduce its observable loss")


if __name__ == "__main__":
    main()
