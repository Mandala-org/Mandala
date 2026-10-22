"""Declared periodic supports, prepared independently of differentiable geometry."""

import torch
from data.relations import support_layouts


def edge_displacements(inputs, index, shifts):
    """Differentiable geometry on an unchanged periodic relation."""
    positions = inputs["positions"]
    vectors = positions[index[1]] - positions[index[0]]
    box = inputs.get("box")
    if box is not None:
        cell = box[inputs["node_batch"][index[0]]] if box.ndim == 3 else box
        vectors = vectors + (
            torch.einsum("ei,eij->ej", shifts.T.to(vectors), cell)
            if cell.ndim == 3
            else shifts.T.to(vectors) @ cell
        )
    return vectors


def validate_support_policies(policies):
    for name, policy in policies.items():
        if not name or name == "output" or not name.isidentifier():
            raise ValueError(f"Invalid support name {name!r}")
        unknown = set(policy) - {"kind", "source", "cutoff", "exclude_self"}
        if unknown or policy.get("kind", "subset") not in {"subset", "radius"}:
            raise ValueError(f"Invalid support policy {name}: {sorted(unknown)}")
        cutoff = policy.get("cutoff")
        if cutoff is not None:
            import math

            if (
                isinstance(cutoff, bool)
                or not isinstance(cutoff, (int, float))
                or not math.isfinite(cutoff)
                or cutoff <= 0
            ):
                raise ValueError("Support cutoff must be finite and positive")
        if policy.get("kind") == "radius" and cutoff is None:
            raise ValueError("Radius support requires a cutoff")
        if "exclude_self" in policy and not isinstance(policy["exclude_self"], bool):
            raise ValueError("exclude_self must be boolean")


def prepare_graph_topology(inputs, policies=None):
    """Build missing supports once; explicitly supplied relations are authoritative.

    Topology selection is intentionally outside autograd. Geometry evaluation
    never calls this routine to reconstruct an existing relation.
    """
    previous = inputs.get("_support_policies", {})
    for name, policy in (policies or {}).items():
        if name in previous and policy != previous[name]:
            raise ValueError(
                f"Support policy {name} changed; explicitly rebuild supports before evaluation"
            )
    policies = {
        "message": {"kind": "subset", "source": "output", "exclude_self": True},
        **previous,
        **(policies or {}),
    }
    validate_support_policies(policies)
    index = inputs["edge_index"]
    shifts = inputs.setdefault("edge_shift", index.new_zeros((3, index.shape[1])))
    inputs.setdefault("node_batch", index.new_zeros(len(inputs["node_type_idx"])))
    output_lookup = {
        tuple(key): row for row, key in enumerate(torch.cat((shifts, index)).T.tolist())
    }
    sizes = dict(
        inputs.get("_entity_sizes", {}),
        node=len(inputs["node_type_idx"]),
        output=index.shape[1],
        bond_endpoint=2 * index.shape[1],
    )
    layouts = dict(inputs.get("_tensor_layouts", {}))
    pending = dict(policies)
    while pending:
        progressed = False
        for name, policy in list(pending.items()):
            source = policy.get("source", "output")
            index_slot, shift_slot = f"{name}_edge_index", f"{name}_shift"
            if index_slot not in inputs:
                if policy.get("kind", "subset") == "radius":
                    if inputs.get("box") is not None and inputs["box"].ndim == 3:
                        raise ValueError(
                            "Prepare radius supports per structure before batching"
                        )
                    from ase import Atoms
                    from ase.neighborlist import neighbor_list

                    # ASE needs species only to establish atom count here.
                    atoms = Atoms(
                        numbers=[1] * len(inputs["positions"]),
                        positions=inputs["positions"].detach().cpu().numpy(),
                        cell=inputs["box"].detach().cpu().numpy(),
                        pbc=True,
                    )
                    a, b, images = neighbor_list(
                        "ijS", atoms, policy["cutoff"], self_interaction=True
                    )
                    selected_index = torch.tensor(
                        [a.tolist(), b.tolist()], dtype=torch.long, device=index.device
                    )
                    selected_shift = torch.tensor(
                        images.T.copy(), dtype=torch.long, device=index.device
                    )
                else:
                    if source != "output" and f"{source}_edge_index" not in inputs:
                        continue
                    selected_index = (
                        index if source == "output" else inputs[f"{source}_edge_index"]
                    )
                    selected_shift = (
                        shifts if source == "output" else inputs[f"{source}_shift"]
                    )
                mask = torch.ones(
                    selected_index.shape[1], dtype=torch.bool, device=index.device
                )
                if policy.get("exclude_self", True):
                    mask &= ~(
                        (selected_index[0] == selected_index[1])
                        & (selected_shift == 0).all(0)
                    )
                if policy.get("cutoff") is not None:
                    with torch.no_grad():
                        mask &= (
                            edge_displacements(
                                inputs, selected_index, selected_shift
                            ).norm(dim=-1)
                            < policy["cutoff"]
                        )
                inputs[index_slot], inputs[shift_slot] = (
                    selected_index[:, mask],
                    selected_shift[:, mask],
                )
            elif shift_slot not in inputs:
                mapping = inputs.get(f"{name}_to_output")
                if mapping is not None and (mapping >= 0).all():
                    inputs[shift_slot] = shifts[:, mapping]
                elif not shifts.any():
                    inputs[shift_slot] = index.new_zeros(
                        (3, inputs[index_slot].shape[1])
                    )
                else:
                    # Older callers supplied just an index and cached vectors.
                    # Infer only unambiguous periodic records, never guess an image.
                    rows = []
                    for a, b in inputs[index_slot].T.tolist():
                        matches = torch.where((index[0] == a) & (index[1] == b))[0]
                        if len(matches) != 1:
                            raise ValueError(
                                f"Periodic support {name} needs explicit shifts or output mapping"
                            )
                        rows.append(int(matches[0]))
                    inputs[shift_slot] = shifts[:, rows]
            keys = torch.cat((inputs[shift_slot], inputs[index_slot])).T.tolist()
            inputs.setdefault(
                f"{name}_to_output",
                index.new_tensor([output_lookup.get(tuple(key), -1) for key in keys]),
            )
            inputs[f"{name}_batch"] = inputs["node_batch"][inputs[index_slot][0]]
            sizes[name] = len(keys)
            layouts.update(
                {key: layout.to_dict() for key, layout in support_layouts(name).items()}
            )
            del pending[name]
            progressed = True
        if not progressed:
            raise ValueError(f"Cyclic or missing source supports: {sorted(pending)}")
    inputs.update(
        _entity_sizes=sizes, _tensor_layouts=layouts, _support_policies=policies
    )
    return inputs


def rebuild_graph_supports(inputs, policies=None):
    """Explicitly select new additional supports on a fixed output graph.

    Return a new input mapping, leaving the old topology/geometry untouched.
    Changing the output graph itself belongs to structure/data preprocessing.
    """
    result = dict(inputs)
    policies = policies if policies is not None else inputs.get("_support_policies", {})
    names = {"message", *inputs.get("_support_policies", {}), *policies}
    for name in names:
        for suffix in (
            "edge_index",
            "shift",
            "to_output",
            "batch",
            "vectors",
            "lengths",
        ):
            result.pop(f"{name}_{suffix}", None)
    for key in (
        "_support_policies",
        "_entity_sizes",
        "_tensor_layouts",
        "_geometry_reference",
        "bond_frames",
        "sub_atom_idx",
        "sub_edge_idx",
        "sub_index",
        "sub_edge_ang",
    ):
        result.pop(key, None)
    return prepare_graph_topology(result, policies)
