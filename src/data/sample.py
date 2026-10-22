"""One graph sample contract for training, preprocessing and prediction."""

import torch


class GraphSample:
    def __init__(self, inputs, targets, metadata=None):
        self.inputs, self.targets = inputs, targets
        self.metadata = {} if metadata is None else metadata

    def __iter__(self):
        yield self.inputs
        yield self.targets

    def __len__(self):
        return 2

    def __contains__(self, key):
        return key in {"inputs", "targets"} or key in self.metadata

    def __getitem__(self, key):
        if key in (0, "inputs"):
            return self.inputs
        if key in (1, "targets"):
            return self.targets
        return self.metadata[key]

    def to(self, device):
        return GraphSample(
            move_tree(self.inputs, device),
            move_tree(self.targets, device),
            move_tree(self.metadata, device),
        )


def move_tree(value, device, dtype=None):
    if isinstance(value, dict):
        return {key: move_tree(item, device, dtype) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(move_tree(item, device, dtype) for item in value)
    if isinstance(value, list):
        return [move_tree(item, device, dtype) for item in value]
    if dtype is not None:
        import torch

        if isinstance(value, torch.Tensor):
            selected = (
                (torch.complex128 if dtype == torch.float64 else torch.complex64)
                if value.is_complex()
                else dtype if value.is_floating_point() else value.dtype
            )
            return value.to(device=device, dtype=selected)
    return value.to(device) if hasattr(value, "to") else value


class GraphBatch:
    """Independent periodic cells; preserves geometry, frames and physical targets."""

    def __init__(self, samples):
        self.samples = samples

    def to(self, device):
        return GraphBatch(
            [
                (
                    sample.to(device)
                    if isinstance(sample, GraphSample)
                    else move_tree(sample, device)
                )
                for sample in self.samples
            ]
        )


def collate_graph_samples(samples):
    if not samples:
        raise ValueError("Cannot collate an empty batch")
    return samples[0] if len(samples) == 1 else GraphBatch(samples)


def union_graph_inputs(samples, *, layouts=None):
    """Batch the network once while retaining each cell's physical sample."""
    from data.relations import standard_layouts, normalize_layout

    declared = standard_layouts()
    for sample in samples:
        x = sample["inputs"] if isinstance(sample, dict) else sample[0]
        for key, value in x.get("_tensor_layouts", {}).items():
            value = normalize_layout(value)
            if key in declared and declared[key] != value:
                raise ValueError(f"Conflicting tensor layout for {key}")
            declared[key] = value
    for key, value in (layouts or {}).items():
        value = normalize_layout(value)
        if key in declared and declared[key] != value:
            raise ValueError(f"Conflicting tensor layout for {key}")
        declared[key] = value
    xs, atoms, boxes, offsets = {}, [], [], {}
    for graph, sample in enumerate(samples):
        x = sample["inputs"] if isinstance(sample, dict) else sample[0]
        sizes = dict(x.get("_entity_sizes", {}))
        sizes.update(
            node=len(x["node_type_idx"]),
            output=x["edge_index"].shape[1],
            bond_endpoint=2 * x["edge_index"].shape[1],
        )
        for key, layout in declared.items():
            if key in x:
                size = x[key].shape[layout.cat_dim]
                if layout.entity in sizes and sizes[layout.entity] != size:
                    raise ValueError(
                        f"Inconsistent entity count for {key}: {layout.entity}"
                    )
                sizes[layout.entity] = size
        atoms.extend(x.get("atoms", ()))
        if "positions" in x:
            box = x.get("box")
            boxes.append(box if box is not None else x["positions"].new_zeros((3, 3)))
        for key, value in x.items():
            if (
                key in {"box", "edge_keys"}
                or key.startswith("spectral_")
                or not torch.is_tensor(value)
                or value.ndim == 0
            ):
                continue
            layout = declared.get(key)
            if layout is None and not (
                value.is_floating_point()
                or value.is_complex()
                or value.dtype == torch.bool
            ):
                raise ValueError(
                    f"Integer feature {key!r} needs an explicit tensor layout"
                )
            if layout is not None:
                if layout.batch_index:
                    value = torch.full_like(value, graph)
                elif layout.references is not None:
                    if value.dtype not in {torch.int32, torch.int64}:
                        raise ValueError(
                            f"Reference tensor {key} must contain integer indices"
                        )

                    def shift(indices, entity):
                        if entity not in sizes:
                            raise ValueError(f"Unknown referenced entity set: {entity}")
                        if indices.numel() and (
                            (indices < -1).any() or (indices >= sizes[entity]).any()
                        ):
                            raise ValueError(
                                f"Out-of-range {entity} reference in {key}"
                            )
                        return torch.where(
                            indices < 0, indices, indices + offsets.get(entity, 0)
                        )

                    if isinstance(layout.references, str):
                        value = shift(value, layout.references)
                    else:
                        if value.shape[layout.index_axis] != len(layout.references):
                            raise ValueError(f"Reference axes disagree for {key}")
                        value = torch.stack(
                            [
                                shift(part, entity)
                                for part, entity in zip(
                                    value.unbind(layout.index_axis), layout.references
                                )
                            ],
                            dim=layout.index_axis,
                        )
            xs.setdefault(key, []).append(value)
        if "node_batch" not in x:
            xs.setdefault("node_batch", []).append(
                torch.full_like(x["node_type_idx"], graph)
            )
        if "edge_batch" not in x:
            xs.setdefault("edge_batch", []).append(
                torch.full_like(x["edge_type_idx"], graph)
            )
        for entity, size in sizes.items():
            offsets[entity] = offsets.get(entity, 0) + size
    if any(len(values) != len(samples) for values in xs.values()):
        raise ValueError("Batched graphs must declare the same tensor features")
    result = {
        key: torch.cat(
            values,
            dim=declared[key].cat_dim if key in declared else 0,
        )
        for key, values in xs.items()
    }
    if boxes:
        result["box"] = torch.stack(boxes)
    if atoms:
        result["atoms"] = tuple(atoms)
    result["_entity_sizes"] = offsets
    result["_tensor_layouts"] = {
        key: value.to_dict() for key, value in declared.items()
    }
    policies = [
        (sample["inputs"] if isinstance(sample, dict) else sample[0]).get(
            "_support_policies", {}
        )
        for sample in samples
    ]
    if any(policy != policies[0] for policy in policies):
        raise ValueError("Batched graphs must share support policies")
    result["_support_policies"] = policies[0]
    references = [
        (sample["inputs"] if isinstance(sample, dict) else sample[0]).get(
            "_geometry_reference"
        )
        for sample in samples
    ]
    if all(reference is not None for reference in references):
        result["_geometry_reference"] = {
            "positions": torch.cat([r["positions"] for r in references])
        }
        if all("box" in r for r in references):
            result["_geometry_reference"]["box"] = torch.stack(
                [r["box"] for r in references]
            )
    return result


def disjoint_union_samples(samples):
    """Reusable tensor union for primitive tests and masked matrix records."""
    if not samples:
        raise ValueError("Cannot collate an empty batch")
    targets = [
        sample["targets"] if isinstance(sample, dict) else sample[1]
        for sample in samples
    ]
    if any(set(target) != set(targets[0]) for target in targets):
        raise ValueError("All graphs must declare identical target schemas")
    result = {
        name: {
            key: torch.cat([target[name][key] for target in targets]) for key in record
        }
        for name, record in targets[0].items()
    }
    return union_graph_inputs(samples), result
