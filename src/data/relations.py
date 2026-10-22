"""Tensor layouts for disjoint unions, independent of feature names.

References name entity sets, not architectures. A two-row edge relation has
``references='node', cat_dim=1``; a mixed node/edge relation can declare
``references=['node', 'output'], index_axis=0``. Negative sentinels survive.
"""

from dataclasses import dataclass, asdict


@dataclass(frozen=True)
class TensorLayout:
    entity: str
    cat_dim: int = 0
    references: str | tuple | list | None = None
    index_axis: int = 0
    batch_index: bool = False

    def __post_init__(self):
        if (
            not isinstance(self.entity, str)
            or not self.entity
            or isinstance(self.cat_dim, bool)
            or not isinstance(self.cat_dim, int)
        ):
            raise ValueError(
                "A tensor layout needs an entity set and concatenation axis"
            )
        if self.references is not None and self.batch_index:
            raise ValueError("A batch index cannot also reference another entity set")
        if self.references is not None and (
            not isinstance(self.references, (str, tuple, list))
            or (
                not isinstance(self.references, str)
                and any(
                    not isinstance(entity, str) or not entity
                    for entity in self.references
                )
            )
        ):
            raise ValueError("References must name entity sets")
        if isinstance(self.index_axis, bool) or not isinstance(self.index_axis, int):
            raise ValueError("index_axis must be an integer")

    def to_dict(self):
        return asdict(self)


def standard_layouts():
    """Declarations for the public graph input vocabulary."""
    layouts = {
        "edge_index": TensorLayout("output", 1, "node"),
        "edge_shift": TensorLayout("output", 1),
        "node_batch": TensorLayout("node", batch_index=True),
        "edge_batch": TensorLayout("output", batch_index=True),
        "node_type_idx": TensorLayout("node"),
        "edge_type_idx": TensorLayout("output"),
        "selfloop_edge": TensorLayout("output"),
        "sub_atom_idx": TensorLayout("bond_neighbor", 0, "node"),
        "sub_edge_idx": TensorLayout("bond_neighbor", 0, "output"),
        "sub_index": TensorLayout("bond_neighbor", 0, "bond_endpoint"),
    }
    layouts.update(support_layouts("message"))
    return layouts


def support_layouts(name):
    return {
        f"{name}_edge_index": TensorLayout(name, 1, "node"),
        f"{name}_shift": TensorLayout(name, 1),
        f"{name}_to_output": TensorLayout(name, 0, "output"),
        f"{name}_batch": TensorLayout(name, batch_index=True),
    }


def normalize_layout(layout):
    return layout if isinstance(layout, TensorLayout) else TensorLayout(**layout)
