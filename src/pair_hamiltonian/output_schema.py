"""Canonical full-AO-block to direct-sum-irrep transformations.

This module implements Algorithm A1 of
``MANDALA_nonparametric_equivariant_descriptor_research_specification_v1``.
Unlike the legacy global reduced-tensor-product mapper, the transform is
assembled one radial-shell pair at a time.  This preserves exact provenance
for every output multiplicity while retaining a single full-block vector as
the public target.

All orbital irreps use natural parity ``(-1)**l``.  A shell pair ``(la, lb)``
therefore yields every ``L`` from ``abs(la-lb)`` through ``la+lb`` with parity
``pa*pb``; this intentionally includes pseudo-irreps such as ``1e`` from a
``p-p`` block.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import lru_cache
import hashlib
import json
from typing import Iterable

from e3nn.o3 import Irrep, Irreps, ReducedTensorProducts
import torch
from torch import nn

from core.orbital_irrep_config import OrbitalIrrepConfig

Pair = tuple[str, str]


def o3_representation_matrix(irreps: Irreps, matrix: torch.Tensor) -> torch.Tensor:
    """Evaluate an e3nn O(3) action with the input matrix's dtype and device.

    e3nn 0.6 constructs the Wigner generators on CPU.  Passing a CUDA rotation
    directly therefore mixes CPU generators with CUDA Euler angles.  The
    representation matrices are small and are not part of the model hot path,
    so evaluate them on CPU and return them to the caller's device.
    """
    evaluation_matrix = matrix.cpu() if matrix.device.type != "cpu" else matrix
    previous_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(evaluation_matrix.dtype)
        representation = irreps.D_from_matrix(evaluation_matrix)
    finally:
        torch.set_default_dtype(previous_dtype)
    return representation.to(device=matrix.device, dtype=matrix.dtype)


@dataclass(frozen=True, slots=True)
class OrbitalShell:
    """One radial AO shell occupying a contiguous magnetic-component range."""

    species: str
    shell_index: int
    radial_index: int
    l: int
    parity: int
    ao_start: int
    ao_stop: int

    @property
    def label(self) -> str:
        parity = "e" if self.parity == 1 else "o"
        return (
            f"{self.species}:{self.radial_index + 1}{self._letter()}({self.l}{parity})"
        )

    def _letter(self) -> str:
        letters = "spdfghiklm"
        return letters[self.l] if self.l < len(letters) else f"l{self.l}"


@dataclass(frozen=True, slots=True)
class IrrepCopyMetadata:
    """Provenance of one output irrep copy in the complete block vector."""

    copy_index: int
    vector_start: int
    vector_stop: int
    l: int
    parity: int
    row_shell_index: int
    column_shell_index: int
    row_ao_start: int
    row_ao_stop: int
    column_ao_start: int
    column_ao_stop: int

    @property
    def irrep_label(self) -> str:
        return f"{self.l}{'e' if self.parity == 1 else 'o'}"


@dataclass(frozen=True, slots=True)
class FullBlockSchema:
    """Immutable ordered schema for one directed species-pair AO block."""

    species_i: str
    species_j: str
    row_dimension: int
    column_dimension: int
    irreps: str
    row_shells: tuple[OrbitalShell, ...]
    column_shells: tuple[OrbitalShell, ...]
    copies: tuple[IrrepCopyMetadata, ...]
    convention: str = "mandala-shell-pair-cg-v1"

    @property
    def pair(self) -> Pair:
        return (self.species_i, self.species_j)

    @property
    def vector_dimension(self) -> int:
        return sum(copy.vector_stop - copy.vector_start for copy in self.copies)

    @property
    def content_hash(self) -> str:
        payload = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _expand_shells(species: str, irreps: Irreps) -> tuple[OrbitalShell, ...]:
    shells: list[OrbitalShell] = []
    radial_counts: dict[int, int] = {}
    ao_offset = 0
    for multiplicity, irrep in irreps:
        expected_parity = 1 if irrep.l % 2 == 0 else -1
        if irrep.p != expected_parity:
            raise ValueError(
                f"AO shell {species} {irrep} has non-natural parity; "
                "spinless OpenMX orbitals must use (-1)^l."
            )
        for _ in range(multiplicity):
            radial_index = radial_counts.get(irrep.l, 0)
            radial_counts[irrep.l] = radial_index + 1
            dimension = 2 * irrep.l + 1
            shells.append(
                OrbitalShell(
                    species=species,
                    shell_index=len(shells),
                    radial_index=radial_index,
                    l=irrep.l,
                    parity=irrep.p,
                    ao_start=ao_offset,
                    ao_stop=ao_offset + dimension,
                )
            )
            ao_offset += dimension
    return tuple(shells)


def _single_irrep(shell: OrbitalShell) -> Irreps:
    return Irreps([(1, Irrep(shell.l, shell.parity))])


def _buffer_name(prefix: str, pair: Pair) -> str:
    return f"{prefix}__{pair[0]}__{pair[1]}"


def _reduced_tensor_product_float64(
    row_shell: OrbitalShell, column_shell: OrbitalShell, dtype: torch.dtype
) -> ReducedTensorProducts:
    """Construct e3nn's fixed CG table in the requested reference precision.

    e3nn 0.5 creates ``change_of_basis`` in the process-wide default dtype and
    exposes no dtype argument on ``ReducedTensorProducts``.  Merely casting a
    float32 table to float64 cannot meet the project's 1e-10 reference gate,
    so construction is performed under a narrowly scoped default-dtype swap.
    Module construction is single-threaded; the prior setting is restored even
    if e3nn raises.
    """

    previous_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(dtype)
        return ReducedTensorProducts(
            "ij", i=_single_irrep(row_shell), j=_single_irrep(column_shell)
        )
    finally:
        torch.set_default_dtype(previous_dtype)


@lru_cache(maxsize=None)
def _accurate_irrep_intertwiner(
    l_i: int, p_i: int, l_j: int, p_j: int, l_out: int
) -> torch.Tensor:
    """Solve the real-basis CG intertwiner to float64 accuracy.

    e3nn 0.5.9's bundled Wigner constants are sufficient for float32 work but
    leave an approximately 1e-7 intertwining residual after casting to
    float64.  We solve the homogeneous intertwining equations for several
    deterministic rotations and align the unique solution with e3nn's
    convention.  This is a fixed initialization-time operation, not a learned
    or data-derived transform.
    """

    ir_i = Irreps([(1, Irrep(l_i, p_i))])
    ir_j = Irreps([(1, Irrep(l_j, p_j))])
    ir_out = Irreps([(1, Irrep(l_out, p_i * p_j))])
    dim_in = ir_i.dim * ir_j.dim
    dim_out = ir_out.dim
    identity_in = torch.eye(dim_in, dtype=torch.float64)
    identity_out = torch.eye(dim_out, dtype=torch.float64)
    angle_sets = (
        (0.37, 0.91, -0.43),
        (-1.11, 0.52, 0.79),
        (0.63, 1.27, 1.49),
        (-0.28, 2.01, -1.32),
    )
    equations: list[torch.Tensor] = []
    previous_dtype = torch.get_default_dtype()
    try:
        # e3nn <=0.5.9 builds the SO(3) generators in the process-wide
        # default dtype rather than the angles' dtype.  Keep this narrow: the
        # setting is restored before returning and construction is expected to
        # happen on the main thread.
        torch.set_default_dtype(torch.float64)
        for alpha_value, beta_value, gamma_value in angle_sets:
            alpha = torch.tensor(alpha_value, dtype=torch.float64)
            beta = torch.tensor(beta_value, dtype=torch.float64)
            gamma = torch.tensor(gamma_value, dtype=torch.float64)
            action_i = ir_i.D_from_angles(alpha, beta, gamma)
            action_j = ir_j.D_from_angles(alpha, beta, gamma)
            action_in = torch.kron(action_i, action_j)
            action_out = ir_out.D_from_angles(alpha, beta, gamma)
            equations.append(
                torch.kron(action_out, identity_in)
                - torch.kron(identity_out, action_in.T.contiguous())
            )
    finally:
        torch.set_default_dtype(previous_dtype)
    system = torch.cat(equations, dim=0)
    _u, singular_values, vh = torch.linalg.svd(system, full_matrices=False)
    candidate = vh[-1].reshape(dim_out, dim_in)

    previous_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float64)
        reference_product = ReducedTensorProducts("ij", i=ir_i, j=ir_j)
    finally:
        torch.set_default_dtype(previous_dtype)
    reference_irreps = list(reference_product.irreps_out)
    start = 0
    reference = None
    for multiplicity, irrep in reference_irreps:
        width = multiplicity * irrep.dim
        if irrep.l == l_out:
            reference = reference_product.change_of_basis.flatten(-2)[
                start : start + irrep.dim
            ].to(torch.float64)
            break
        start += width
    if reference is None:  # pragma: no cover - guarded by angular triangle rule
        raise RuntimeError(f"Missing e3nn reference sector L={l_out}")
    if torch.sum(candidate * reference) < 0:
        candidate = -candidate

    row_scale = torch.sqrt(torch.trace(candidate @ candidate.T) / dim_out)
    candidate = candidate / row_scale
    residual = max(
        torch.max(torch.abs(equation @ candidate.reshape(-1))).item()
        for equation in equations
    )
    next_singular = (
        float("inf") if singular_values.numel() == 1 else singular_values[-2].item()
    )
    if next_singular < 1.0e-8 or residual > 1.0e-11:
        raise RuntimeError(
            f"Ill-defined CG solve for ({l_i},{l_j})->{l_out}: "
            f"next singular={next_singular:.3e}, "
            f"residual={residual:.3e}"
        )
    return candidate


def _accurate_local_change_of_basis(
    row_shell: OrbitalShell,
    column_shell: OrbitalShell,
    reduced: ReducedTensorProducts,
) -> torch.Tensor:
    sectors = []
    for multiplicity, irrep in reduced.irreps_out:
        if multiplicity != 1:
            raise RuntimeError(
                "A single shell-pair CG sector must have multiplicity one"
            )
        sectors.append(
            _accurate_irrep_intertwiner(
                row_shell.l,
                row_shell.parity,
                column_shell.l,
                column_shell.parity,
                irrep.l,
            )
        )
    return torch.cat(sectors, dim=0)


class FullBlockIrrepTransform(nn.Module):
    """Exact full-block AO↔irrep and pair-reversal maps.

    For a directed species pair ``A-B``, ``blocks_to_irreps`` implements
    ``h = Q vec(H)`` and ``irreps_to_blocks`` implements ``vec(H) = Q^T h``.
    The assembled ``Q`` is orthogonal and ordered by row shell, column shell,
    then coupled ``L``.  ``reverse`` derives the irrep-space Hermiticity map
    from AO transpose rather than hard-coding exchange phases.
    """

    def __init__(
        self,
        orbital_config: OrbitalIrrepConfig,
        *,
        dtype: torch.dtype = torch.float64,
        device: torch.device | str = "cpu",
    ) -> None:
        super().__init__()
        if not isinstance(orbital_config, OrbitalIrrepConfig):
            raise TypeError("orbital_config must be an OrbitalIrrepConfig")
        self.orbital_config = orbital_config
        self.schemas: dict[Pair, FullBlockSchema] = {}

        for species_i in orbital_config.elements():
            for species_j in orbital_config.elements():
                pair = (species_i, species_j)
                schema, change_of_basis = self._build_pair(pair, dtype=dtype)
                self.schemas[pair] = schema
                self.register_buffer(
                    _buffer_name("q", pair),
                    change_of_basis.to(device=device, dtype=dtype),
                    persistent=True,
                )

        for pair in self.schemas:
            reverse_matrix = self._derive_reverse_matrix(pair)
            self.register_buffer(
                _buffer_name("reverse", pair), reverse_matrix, persistent=True
            )

    def _build_pair(
        self, pair: Pair, *, dtype: torch.dtype
    ) -> tuple[FullBlockSchema, torch.Tensor]:
        species_i, species_j = pair
        irreps_i = self.orbital_config.element_to_irreps[species_i]
        irreps_j = self.orbital_config.element_to_irreps[species_j]
        row_shells = _expand_shells(species_i, irreps_i)
        column_shells = _expand_shells(species_j, irreps_j)
        block_dimension = irreps_i.dim * irreps_j.dim

        rows: list[torch.Tensor] = []
        output_parts: list[tuple[int, Irrep]] = []
        copies: list[IrrepCopyMetadata] = []
        vector_offset = 0
        for row_shell in row_shells:
            for column_shell in column_shells:
                reduced = _reduced_tensor_product_float64(
                    row_shell, column_shell, dtype
                )
                if dtype == torch.float64:
                    local_q = _accurate_local_change_of_basis(
                        row_shell, column_shell, reduced
                    )
                else:
                    local_q = reduced.change_of_basis.flatten(-2).to(dtype=dtype)
                embedded = torch.zeros((local_q.shape[0], block_dimension), dtype=dtype)
                local_columns = []
                for row in range(row_shell.ao_start, row_shell.ao_stop):
                    for column in range(column_shell.ao_start, column_shell.ao_stop):
                        local_columns.append(row * irreps_j.dim + column)
                embedded[:, local_columns] = local_q
                rows.append(embedded)

                local_offset = 0
                for multiplicity, irrep in reduced.irreps_out:
                    for _ in range(multiplicity):
                        width = irrep.dim
                        output_parts.append((1, irrep))
                        copies.append(
                            IrrepCopyMetadata(
                                copy_index=len(copies),
                                vector_start=vector_offset + local_offset,
                                vector_stop=vector_offset + local_offset + width,
                                l=irrep.l,
                                parity=irrep.p,
                                row_shell_index=row_shell.shell_index,
                                column_shell_index=column_shell.shell_index,
                                row_ao_start=row_shell.ao_start,
                                row_ao_stop=row_shell.ao_stop,
                                column_ao_start=column_shell.ao_start,
                                column_ao_stop=column_shell.ao_stop,
                            )
                        )
                        local_offset += width
                vector_offset += local_q.shape[0]

        change_of_basis = torch.cat(rows, dim=0)
        output_irreps = Irreps(output_parts)
        schema = FullBlockSchema(
            species_i=species_i,
            species_j=species_j,
            row_dimension=irreps_i.dim,
            column_dimension=irreps_j.dim,
            irreps=str(output_irreps),
            row_shells=row_shells,
            column_shells=column_shells,
            copies=tuple(copies),
        )
        if change_of_basis.shape != (block_dimension, block_dimension):
            raise RuntimeError(
                f"Incomplete transform for {pair}: got {tuple(change_of_basis.shape)}, "
                f"expected {(block_dimension, block_dimension)}"
            )
        return schema, change_of_basis

    def _derive_reverse_matrix(self, pair: Pair) -> torch.Tensor:
        reverse_pair = (pair[1], pair[0])
        schema = self.schema(pair)
        q_forward = self.change_of_basis(pair)
        q_reverse = self.change_of_basis(reverse_pair)
        basis_vectors = torch.eye(
            schema.vector_dimension, dtype=q_forward.dtype, device=q_forward.device
        )
        blocks = (basis_vectors @ q_forward).reshape(
            schema.vector_dimension, schema.row_dimension, schema.column_dimension
        )
        return (
            blocks.transpose(-1, -2).reshape(schema.vector_dimension, -1) @ q_reverse.T
        )

    @staticmethod
    def _pair(pair: Pair | str) -> Pair:
        if isinstance(pair, str):
            fields = tuple(field.strip() for field in pair.split("-", maxsplit=1))
            if len(fields) != 2:
                raise KeyError(f"Pair must have form 'A-B', got {pair!r}")
            return fields  # type: ignore[return-value]
        if len(pair) != 2:
            raise KeyError(f"Pair must contain two species, got {pair!r}")
        return pair

    def schema(self, pair: Pair | str) -> FullBlockSchema:
        key = self._pair(pair)
        try:
            return self.schemas[key]
        except KeyError as exc:
            raise KeyError(f"Unknown species pair {key}") from exc

    def change_of_basis(self, pair: Pair | str) -> torch.Tensor:
        return getattr(self, _buffer_name("q", self._pair(pair)))

    def reverse_matrix(self, pair: Pair | str) -> torch.Tensor:
        return getattr(self, _buffer_name("reverse", self._pair(pair)))

    def irreps(self, pair: Pair | str) -> Irreps:
        return Irreps(self.schema(pair).irreps)

    def orbital_action(self, species: str, matrix: torch.Tensor) -> torch.Tensor:
        """Return the AO representation matrix for a proper/improper rotation."""
        try:
            irreps = self.orbital_config.element_to_irreps[species]
        except KeyError as exc:
            raise KeyError(f"Unknown species {species!r}") from exc
        return o3_representation_matrix(irreps, matrix)

    def output_action(self, pair: Pair | str, matrix: torch.Tensor) -> torch.Tensor:
        """Return the full target-vector representation matrix."""
        return o3_representation_matrix(self.irreps(pair), matrix)

    def blocks_to_irreps(self, pair: Pair | str, blocks: torch.Tensor) -> torch.Tensor:
        schema = self.schema(pair)
        if blocks.shape[-2:] != (schema.row_dimension, schema.column_dimension):
            raise ValueError(
                f"Block shape {tuple(blocks.shape[-2:])} does not match "
                f"{schema.pair} dimensions "
                f"{(schema.row_dimension, schema.column_dimension)}"
            )
        q = self.change_of_basis(pair)
        return blocks.to(dtype=q.dtype, device=q.device).flatten(-2) @ q.T

    def irreps_to_blocks(self, pair: Pair | str, vectors: torch.Tensor) -> torch.Tensor:
        schema = self.schema(pair)
        if vectors.shape[-1] != schema.vector_dimension:
            raise ValueError(
                f"Vector dimension {vectors.shape[-1]} does not match "
                f"{schema.pair} dimension {schema.vector_dimension}"
            )
        q = self.change_of_basis(pair)
        flat = vectors.to(dtype=q.dtype, device=q.device) @ q
        return flat.reshape(
            *vectors.shape[:-1], schema.row_dimension, schema.column_dimension
        )

    def reverse(self, pair: Pair | str, vectors: torch.Tensor) -> torch.Tensor:
        """Return the exact irrep vector for the transposed reversed block."""
        matrix = self.reverse_matrix(pair)
        if vectors.shape[-1] != matrix.shape[0]:
            raise ValueError(
                f"Vector dimension {vectors.shape[-1]} does not match reversal "
                f"input dimension {matrix.shape[0]}"
            )
        return vectors.to(dtype=matrix.dtype, device=matrix.device) @ matrix

    def schema_manifest(self) -> dict[str, dict[str, object]]:
        return {
            f"{pair[0]}-{pair[1]}": {
                **schema.to_dict(),
                "content_hash": schema.content_hash,
            }
            for pair, schema in sorted(self.schemas.items())
        }

    def iter_schemas(self) -> Iterable[FullBlockSchema]:
        for pair in sorted(self.schemas):
            yield self.schemas[pair]
