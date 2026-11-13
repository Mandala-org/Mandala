"""Locate which compiled module changes the network output.

Builds one variant, records baseline decoded outputs, then compiles one
network module at a time (each pipeline transform for MANDALA; each direct
child for native) and reports the output difference for each. Finally
compiles the whole network as a single function.
"""

from __future__ import annotations

import argparse
import contextlib
import json
from pathlib import Path
import sys

import e3nn
from omegaconf import OmegaConf
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import optimize_reference_inference as O  # noqa: E402
import scripts.profile_reference_inference as P  # noqa: E402


def compile_one(module, mode):
    module.forward = torch.compile(module.forward, mode=mode, dynamic=False)


def restore(module):
    module.__dict__.pop("forward", None)
    torch._dynamo.reset()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--material", default="graphene")
    p.add_argument("--variant", required=True)
    p.add_argument("--mode", default="default")
    p.add_argument(
        "--inside",
        help="pipeline transform whose direct children are compiled one at a time",
    )
    p.add_argument("--safe-scatter", action="store_true")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    e3nn.set_optimization_defaults(jit_mode="eager")
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    paths = list(
        OmegaConf.load(
            args.root / "configs" / f"mandala_{args.material}.yaml"
        ).data.splits.val
    )[:3]
    variant = P.build_variant(args.root, args.material, args.variant, paths)
    n = len(paths)
    reference = O.decoded_outputs(variant, n)
    result = {
        "noise": O.max_difference(reference, O.decoded_outputs(variant, n)),
        "modules": {},
    }
    model = variant["model"]
    if args.inside:
        named = list(model.pipeline.transforms[args.inside].named_children())
    else:
        named = (
            list(model.named_children())
            if variant["native"]
            else list(model.pipeline.transforms.items())
        )
    if args.safe_scatter:
        stack = contextlib.ExitStack()
        stack.enter_context(O.safe_scatter())
    for name, module in named:
        if not any(True for _ in module.parameters()) and variant["native"]:
            continue
        try:
            compile_one(module, args.mode)
            result["modules"][name] = O.max_difference(
                reference, O.decoded_outputs(variant, n)
            )
        except Exception as error:
            result["modules"][name] = {"error": repr(error)[:500]}
        finally:
            restore(module)
        print(name, result["modules"][name], flush=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    whole = (
        model.pipeline.transforms[args.inside]
        if args.inside
        else model if variant["native"] else model.pipeline
    )
    try:
        compile_one(whole, args.mode)
        result["whole"] = O.max_difference(reference, O.decoded_outputs(variant, n))
    except Exception as error:
        result["whole"] = {"error": repr(error)[:500]}
    finally:
        restore(whole)
    print("whole", result["whole"], flush=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
