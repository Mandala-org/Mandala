"""Nonorthogonal McWeeny purification on a fixed, periodic block support.

With P the projection onto the original density edges, a step is
X=P(SD), Y=P(DX), Z=P(YX), D_next=3Y-2Z. Intermediate truncation makes
this an approximation to P(3DSD-2DSDSD), and can break Hermiticity. No
electron-count rescaling, eigenvalue clipping or hidden symmetrization occurs.
Use a global reverse-pair projection at evaluation/export when desired.
"""

from __future__ import annotations

from data.block_matrix import BlockMatrix
from core.sparse_math import build_matmul_alignment, matmul_sparse_block_matrix_aligned


class McWeenyPurifier:
    """Cache the two product topologies and reuse SD within every step.

    The density must use unit-occupation normalization (DSD=D at a projector).
    The map does not preserve Tr(DS), and convergence or improved accuracy is
    not guaranteed for fractional occupations, bad spectra or truncated support.
    """

    def __init__(
        self, density: BlockMatrix, overlap: BlockMatrix, *, chunk_size: int = 4096
    ):
        if (
            isinstance(chunk_size, bool)
            or not isinstance(chunk_size, int)
            or chunk_size <= 0
        ):
            raise ValueError("chunk_size must be a positive integer.")
        self.overlap = overlap
        self.chunk_size = chunk_size
        self.sd_alignment = build_matmul_alignment(overlap, density, density)
        self.dd_alignment = build_matmul_alignment(density, density, density)

    def step(self, density: BlockMatrix) -> BlockMatrix:
        """Return one differentiable step, retaining exactly density's edges."""
        sd = matmul_sparse_block_matrix_aligned(
            self.overlap,
            density,
            density,
            self.sd_alignment,
            chunk_size=self.chunk_size,
        )
        dsd = matmul_sparse_block_matrix_aligned(
            density, sd, density, self.dd_alignment, chunk_size=self.chunk_size
        )
        dsdsd = matmul_sparse_block_matrix_aligned(
            dsd, sd, density, self.dd_alignment, chunk_size=self.chunk_size
        )
        return density._replace_pair_blocks(
            {
                key: 3 * dsd.pair_blocks[key] - 2 * dsdsd.pair_blocks[key]
                for key in density.pair_blocks
            },
            basis=density.basis,
        )

    def __call__(self, density: BlockMatrix, *, iterations: int = 1) -> BlockMatrix:
        if (
            isinstance(iterations, bool)
            or not isinstance(iterations, int)
            or iterations < 0
        ):
            raise ValueError("iterations must be a nonnegative integer.")
        for _ in range(iterations):
            density = self.step(density)
        return density


def purify_density(
    density: BlockMatrix,
    overlap: BlockMatrix,
    *,
    iterations: int = 1,
    chunk_size: int = 4096,
) -> BlockMatrix:
    """Convenience API; reuse McWeenyPurifier directly for repeated evaluations."""
    return McWeenyPurifier(density, overlap, chunk_size=chunk_size)(
        density, iterations=iterations
    )
