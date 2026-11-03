"""Typed feature contracts shared by every pipeline transformation.

A :class:`FeatureSpec` describes one slot of a pipeline: which entities it is
indexed by, how its columns transform, and in which coordinate frame. Factories
use the checks below to reject invalid wiring before any module is allocated.
"""

from __future__ import annotations

from dataclasses import dataclass

from e3nn import o3

DOMAINS = ("node", "edge", "bond_neighbor", "graph", "relation")
REPRESENTATIONS = (
    "object",
    "block_vectors",
    "scalar",
    "irreps",
    "channel_irreps",
    "index",
    "geometry",
    "matrix",
    "local_components",
)


@dataclass(frozen=True)
class FeatureSpec:
    domain: str
    representation: str = "scalar"
    irreps: str | None = None
    frame: str = "global"
    width: int | None = None
    support: str | None = None
    angular_basis: str | None = None
    batch: dict | None = None
    # Names of the values of an index slot, e.g. ("onsite", "offsite").
    categories: tuple | None = None

    def __post_init__(self):
        if self.domain not in DOMAINS:
            raise ValueError(f"Unknown feature domain {self.domain}")
        if self.representation not in REPRESENTATIONS:
            raise ValueError(f"Unknown representation {self.representation}")
        if self.support is None and self.domain in {
            "edge",
            "relation",
            "bond_neighbor",
        }:
            object.__setattr__(self, "support", "output")
        if self.support is not None and (
            not isinstance(self.support, str) or not self.support
        ):
            raise ValueError("Feature support must be a nonempty index-set name")
        if self.angular_basis is None and (
            self.representation == "geometry"
            or (
                self.irreps is not None
                and any(ir.l > 0 for _, ir in o3.Irreps(self.irreps))
            )
        ):
            object.__setattr__(self, "angular_basis", "xyz")
        if self.angular_basis is not None and sorted(self.angular_basis) != [
            "x",
            "y",
            "z",
        ]:
            raise ValueError("Angular basis must name each Cartesian axis exactly once")
        if self.categories is not None:
            object.__setattr__(self, "categories", tuple(self.categories))
            if self.representation != "index" or self.width != len(self.categories):
                raise ValueError("Category labels belong to index slots, one per value")
        if self.irreps is not None:
            dimension = o3.Irreps(self.irreps).dim
            if self.representation == "irreps" and self.width not in (None, dimension):
                raise ValueError("Feature width disagrees with its irrep layout")

    @property
    def dim(self):
        return o3.Irreps(self.irreps).dim if self.irreps is not None else self.width

    @property
    def entity(self):
        """The index set the leading axis runs over (a support or a domain)."""
        return self.support or self.domain

    @property
    def invariant(self):
        """True when every channel is an even scalar."""
        return self.irreps is not None and all(
            ir.l == 0 and ir.p == 1 for _, ir in o3.Irreps(self.irreps)
        )

    def to_json(self):
        return {
            "domain": self.domain,
            "representation": self.representation,
            "irreps": self.irreps,
            "frame": self.frame,
            "width": self.dim,
            "support": self.support,
            "angular_basis": self.angular_basis,
            "invariant": self.invariant,
            "categories": list(self.categories) if self.categories else None,
        }


def default_input_specs(num_species):
    """Contracts of the input slots every graph batch provides."""
    return {
        "node_type_idx": FeatureSpec("node", "index", frame="none", width=num_species),
        "node_one_hot": FeatureSpec(
            "node", irreps=f"{num_species}x0e", width=num_species
        ),
        "edge_type_idx": FeatureSpec(
            "edge", "index", frame="none", width=num_species**2
        ),
        "edge_one_hot": FeatureSpec(
            "edge", irreps=f"{num_species**2}x0e", width=num_species**2
        ),
        "edge_lengths": FeatureSpec("edge", width=1),
        "edge_vectors": FeatureSpec("edge", "geometry", width=3),
        "edge_index": FeatureSpec("relation", "index", frame="none"),
        "edge_shift": FeatureSpec("relation", "index", frame="none"),
        "positions": FeatureSpec("node", "geometry", width=3),
        "box": FeatureSpec("graph", "geometry", width=9),
        "message_edge_index": FeatureSpec(
            "relation", "index", frame="none", support="message"
        ),
        "message_shift": FeatureSpec(
            "relation", "index", frame="none", support="message"
        ),
        "message_vectors": FeatureSpec("edge", "geometry", width=3, support="message"),
        "message_lengths": FeatureSpec("edge", width=1, support="message"),
        "message_to_output": FeatureSpec(
            "relation", "index", frame="none", support="message"
        ),
        "node_batch": FeatureSpec("node", "index", frame="none"),
        "edge_batch": FeatureSpec("edge", "index", frame="none"),
        "selfloop_edge": FeatureSpec("edge", "index", frame="none"),
        "sub_atom_idx": FeatureSpec("bond_neighbor", "index", frame="none"),
        "sub_edge_idx": FeatureSpec("bond_neighbor", "index", frame="none"),
        "sub_edge_ang": FeatureSpec("bond_neighbor", "local_components", frame="bond"),
        "sub_index": FeatureSpec("bond_neighbor", "index", frame="none"),
        "bond_frames": FeatureSpec("edge", "geometry", frame="bond", width=9),
        "pred_pair_edges_static": FeatureSpec("relation", "object"),
        "edge_partitions": FeatureSpec("relation", "object"),
    }


# Slots that are functions of the structure alone (positions, cell, species,
# graph). Parameter-free stages reading only these are evaluated once per
# structure and cached with the sample.
STRUCTURE_SLOTS = frozenset(default_input_specs(1))

# Slots that depend on atomic positions and therefore on any geometric
# derivative requested through them.
GEOMETRY_SLOTS = frozenset(
    {
        "positions",
        "box",
        "edge_lengths",
        "edge_vectors",
        "message_vectors",
        "message_lengths",
        "bond_frames",
        "sub_edge_ang",
    }
)


# Slots written by ``data.graph_features.evaluate_graph_geometry``.
DERIVED_GEOMETRY_SLOTS = GEOMETRY_SLOTS | {
    "edge_batch",
    "selfloop_edge",
    "message_edge_index",
    "message_to_output",
    "sub_index",
    "sub_atom_idx",
    "sub_edge_idx",
}


def global_irreps(spec, domain=None):
    """Irreps of a global-frame feature; rejects local or untyped features."""
    if (
        spec.frame != "global"
        or spec.irreps is None
        or spec.representation not in {"scalar", "irreps"}
    ):
        raise ValueError(
            "This transformation requires global irrep features; insert an explicit frame adapter"
        )
    if domain and spec.domain != domain:
        raise ValueError(f"Expected {domain} features, received {spec.domain}")
    return o3.Irreps(spec.irreps)


def invariant_irreps(spec, domain=None, *, what="Input"):
    """Irreps of a feature that must contain only even scalars."""
    irreps = global_irreps(spec, domain)
    if any(ir.l or ir.p != 1 for _, ir in irreps):
        raise ValueError(f"{what} must be invariant (even scalar) channels")
    return irreps


def same_support(*specs):
    """Require the same entities in the same order, not merely equal shapes."""
    supports = {spec.support for spec in specs}
    if len(supports) > 1:
        raise ValueError(
            f"Feature/index support mismatch: {sorted(map(str, supports))}"
        )
    return specs[0].support if specs else None


def index_spec(spec, domain):
    if spec.domain != domain or spec.representation != "index":
        raise ValueError(f"Expected {domain}-scoped integer index features")


def same_angular_basis(*specs):
    """Check non-scalar representations; invariant scalar channels are neutral."""
    bases = {
        spec.angular_basis
        for spec in specs
        if spec.irreps is not None and any(ir.l > 0 for _, ir in o3.Irreps(spec.irreps))
    }
    if len(bases) > 1:
        raise ValueError(
            f"Angular basis mismatch: {sorted(map(str, bases))}; insert irrep_cartesian_basis"
        )
    return next(iter(bases)) if bases else None


def require_canonical(*specs):
    """Shared components operate in the canonical xyz angular basis."""
    basis = same_angular_basis(*specs)
    if basis not in {None, "xyz"}:
        raise ValueError(
            "This component requires canonical xyz features; insert irrep_cartesian_basis"
        )
    return basis


def irrep_feature(spec, irreps, *, domain=None, support=None):
    """A global irrep feature on the entities of ``spec``."""
    irreps = o3.Irreps(irreps)
    return FeatureSpec(
        domain or spec.domain,
        "irreps",
        str(irreps),
        support=support if support is not None else spec.support,
        angular_basis=spec.angular_basis if irreps.lmax > 0 else None,
    )


def irrep_magnitudes(features, irreps):
    """Norm of every irrep copy, grouped by irrep term (for activation logging)."""
    magnitudes, start = {}, 0
    for mul, ir in o3.Irreps(irreps):
        size = mul * ir.dim
        if mul and ir.dim:
            chunk = features[:, start : start + size].reshape(-1, ir.dim)
            magnitudes[f"{mul}x{ir.l}{'e' if ir.p == 1 else 'o'}"] = chunk.norm(dim=1)
        start += size
    return magnitudes
