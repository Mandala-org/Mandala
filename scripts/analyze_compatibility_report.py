#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path


TRANSFER_RULES = {
    "node_enc.elem_emb.weight": "embedding row-prefix copy; extra destination rows filled with source-row mean",
    "edge_enc.edge_emb.weight": "embedding row-prefix copy; extra destination rows filled with source-row mean",
    "edge_enc.tp.weight": "instruction-matched FullyConnectedTensorProduct copy by (in1 irrep, in2 irrep, out irrep, connection_mode)",
}


@dataclass
class KeySummary:
    module_family: str
    param_family: str
    transfer_rule: str


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _load_log_lines(path: Path | None) -> list[str]:
    if path is None or not path.exists():
        return []
    return path.read_text().splitlines()


def _classify_key(key: str) -> KeySummary:
    if key in TRANSFER_RULES:
        return KeySummary(key, "special", TRANSFER_RULES[key])

    if re.search(r"\.weights[12]\.\d+$", key):
        module = re.sub(r"\.weights[12]\.\d+$", "", key)
        return KeySummary(
            module_family=module,
            param_family="separate_weight_tp",
            transfer_rule="instruction-matched SeparateWeightTensorProduct copy by (in1 irrep, in2 irrep, out irrep, connection_mode); overlapping tensor prefix copied within the matched instruction",
        )

    if key.endswith(".linear.weight"):
        module = key[: -len(".weight")]
        return KeySummary(
            module_family=module,
            param_family="e3_linear_weight",
            transfer_rule="instruction-matched e3nn Linear copy by (input irrep, output irrep); overlapping multiplicity prefix copied within the matched instruction",
        )

    if key.endswith(".linear.bias"):
        module = key[: -len(".bias")]
        return KeySummary(
            module_family=module,
            param_family="linear_bias",
            transfer_rule="bias prefix copy for overlapping output channels",
        )

    if key.endswith(".weight"):
        module = key[: -len(".weight")]
        return KeySummary(
            module_family=module,
            param_family="plain_weight",
            transfer_rule="plain linear/tensor parameter overlap copy by raw prefix rectangle",
        )

    if key.endswith(".bias"):
        module = key[: -len(".bias")]
        return KeySummary(
            module_family=module,
            param_family="plain_bias",
            transfer_rule="bias prefix copy for overlapping channels",
        )

    return KeySummary(
        module_family=key,
        param_family="other",
        transfer_rule="exact copy or unmatched auxiliary parameter",
    )


def _top_module_prefixes(
    keys: list[str], depth: int = 3, limit: int = 15
) -> list[tuple[str, int]]:
    counter: Counter[str] = Counter()
    for key in keys:
        parts = key.split(".")
        prefix = ".".join(parts[:depth]) if len(parts) >= depth else key
        counter[prefix] += 1
    return counter.most_common(limit)


def _summarize_keys(
    keys: list[str],
) -> tuple[Counter[str], Counter[str], dict[str, list[str]]]:
    by_param_family: Counter[str] = Counter()
    by_module_family: Counter[str] = Counter()
    examples: dict[str, list[str]] = defaultdict(list)

    for key in keys:
        summary = _classify_key(key)
        by_param_family[summary.param_family] += 1
        by_module_family[summary.module_family] += 1
        if len(examples[summary.param_family]) < 6:
            examples[summary.param_family].append(key)

    return by_param_family, by_module_family, examples


def _extract_log_events(lines: list[str]) -> dict[str, list[str]]:
    events: dict[str, list[str]] = defaultdict(list)
    patterns = {
        "embedding": r"^\[compat\] remapped embedding ",
        "e3_linear": r"^\[compat\] remapped e3 linear ",
        "tensor_product": r"^\[compat\] remapped tensor product ",
        "separate_weight_tp": r"^\[compat\] remapped separate weight tp ",
        "shared_projector_init": r"^\[compat\] initializing shared projector ",
        "shared_projector_agg": r"^\[compat\] shared projector aggregation ",
        "summary": r"^\[compat\] load summary ",
    }
    for line in lines:
        for name, pattern in patterns.items():
            if re.search(pattern, line):
                events[name].append(line)
    return events


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize compatibility initialization behavior."
    )
    parser.add_argument(
        "--report",
        default="compatibility_report.json",
        help="Path to compatibility report JSON",
    )
    parser.add_argument(
        "--log", default="slurm_compat.out", help="Optional path to a training log"
    )
    args = parser.parse_args()

    report_path = Path(args.report)
    log_path = Path(args.log) if args.log else None

    report = _load_json(report_path)
    log_lines = _load_log_lines(log_path)
    events = _extract_log_events(log_lines)

    print("Compatibility analysis")
    print(f"report={report_path}")
    if log_path is not None and log_path.exists():
        print(f"log={log_path}")
    print()

    print("Checkpoint/source metadata")
    print(f"  checkpoint_path: {report.get('checkpoint_path')}")
    print(f"  source_targets: {report.get('source_targets')}")
    print(f"  target_clone_sources: {report.get('target_clone_sources')}")
    print(f"  source_num_species: {report.get('source_num_species')}")
    print(f"  source_head_pair_mode: {report.get('source_head_pair_mode')}")
    print()

    for field in ["exact_copied", "cloned_keys", "remapped_keys", "skipped_keys"]:
        keys = report.get(field, [])
        by_param_family, _, examples = _summarize_keys(keys)
        print(f"{field}: {len(keys)}")
        for family, count in by_param_family.most_common():
            print(f"  {family}: {count}")
            for example in examples[family][:3]:
                print(f"    example: {example}")
        print("  top prefixes:")
        for prefix, count in _top_module_prefixes(keys):
            print(f"    {count:4d} {prefix}")
        print()

    if events:
        print("Compatibility log events")
        for name in [
            "embedding",
            "tensor_product",
            "e3_linear",
            "separate_weight_tp",
            "shared_projector_init",
            "shared_projector_agg",
            "summary",
        ]:
            values = events.get(name, [])
            if not values:
                continue
            print(f"  {name}: {len(values)}")
            for line in values[:5]:
                print(f"    {line}")
        print()

    print("Interpretation guide")
    print(
        "  exact_copied: source and destination state-dict key matched exactly and shape matched exactly"
    )
    print(
        "  remapped e3_linear_weight: copied only instruction pairs with the same input and output irrep labels"
    )
    print(
        "  remapped separate_weight_tp: copied only TP instructions with the same (in1 irrep, in2 irrep, out irrep, connection mode)"
    )
    print(
        "  skipped_keys: source parameters that were not directly staged under their original key; some of them may still have contributed indirectly through remap or shared-projector aggregation"
    )


if __name__ == "__main__":
    main()
