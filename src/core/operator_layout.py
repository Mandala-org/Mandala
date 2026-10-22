"""Shell-coupled operator channels and explicit orbital-frame transformations.

The coupling and spin conventions follow DeepH-E3's MIT-licensed
``e3TensorDecomp`` and ``Rotate`` (Copyright (c) 2023 Xiaoxun-Gong).

Native blocks use OpenMX real-orbital order. The coupled basis uses the
real spherical harmonics in cyclic Cartesian order (y, z, x), called ``wiki``
by that reference. Coupled coefficients are NOT orthonormally normalized:
the inverse contains the reference's factor ``2*l+1``. SOC network outputs
are real: all real coefficients followed by all imaginary coefficients.
Packed SOC targets have spin components (uu, ud, du, dd), then shell entries.

MIT permission notice for the adapted algorithms:
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

from __future__ import annotations

from itertools import accumulate

import torch
from e3nn.o3 import Irreps, matrix_to_angles, wigner_3j
from torch import nn


def _pair(value):
    if isinstance(value, str):
        value = value.split()
    if len(value) != 2:
        raise ValueError("A species pair requires exactly two species.")
    return tuple(map(str, value))


def build_shell_targets(
    species_shells, *, requested=None, block_kind="all", species_pairs=None
):
    """Group matching shell slots across species into shared output channels.

    Species insertion order and ordered species-pair order are retained.
    Shell identity is (angular momentum, index relative to its first shell),
    exactly as in the reference automatic target selection. ``requested``
    supplies an explicit ordered list of mappings ``{(A,B): (shell_i,shell_j)}``;
    source strings ``"A B"`` are also accepted. ``diag``/``offdiag`` select
    shell identities, not onsite/offsite geometric edges.
    """
    shells = {str(k): tuple(map(int, v)) for k, v in species_shells.items()}
    if not shells or any(
        not ls or any(l < 0 or l > 3 for l in ls) for ls in shells.values()
    ):
        raise ValueError(
            "Provide nonempty native s/p/d/f shell lists for each species."
        )
    if block_kind not in {"all", "diag", "offdiag"}:
        raise ValueError("block_kind must be all, diag, or offdiag.")
    if requested is not None:
        targets = tuple(
            {_pair(p): tuple(map(int, ij)) for p, ij in target.items()}
            for target in requested
        )
        for target in targets:
            if not target:
                raise ValueError("An output channel cannot be empty.")
            angular = set()
            for (a, b), (i, j) in target.items():
                if (
                    a not in shells
                    or b not in shells
                    or not 0 <= i < len(shells[a])
                    or not 0 <= j < len(shells[b])
                ):
                    raise ValueError(
                        "Output channel refers to an unknown species or shell."
                    )
                angular.add((shells[a][i], shells[b][j]))
            if len(angular) != 1:
                raise ValueError(
                    "All species assignments in one channel must have identical shell angular momenta."
                )
        if not targets:
            raise ValueError("At least one output channel is required.")
        return targets
    allowed = None if species_pairs is None else {_pair(p) for p in species_pairs}
    grouped = {}
    for a, la in shells.items():
        for b, lb in shells.items():
            if allowed is not None and (a, b) not in allowed:
                continue
            for i, l1 in enumerate(la):
                for j, l2 in enumerate(lb):
                    identity = (l1, i - la.index(l1), l2, j - lb.index(l2))
                    diagonal = identity[:2] == identity[2:]
                    if (block_kind == "diag" and not diagonal) or (
                        block_kind == "offdiag" and diagonal
                    ):
                        continue
                    grouped.setdefault(identity, {})[(a, b)] = (i, j)
    if not grouped:
        raise ValueError("Shell selection produced no output channels.")
    return tuple(grouped.values())


class OrbitalFrameRotation(nn.Module):
    """Native OpenMX ↔ cyclic real-harmonic basis and active frame changes.

    ``rotate_blocks`` follows the reference row-vector rotation convention:
    its spatial R acts on positions on the right. For column-vector active
    rotations, pass ``R.T``. SOC full matrices are spin-major (all up orbitals,
    then all down orbitals). Proper/improper O(3) orbital transformations are
    supported; spin transforms use the proper part, as in the reference.
    """

    def __init__(self, dtype=torch.float32, device="cpu", spinful=False):
        super().__init__()
        self.spinful = bool(spinful)
        if dtype not in {
            torch.float32,
            torch.float64,
            torch.complex64,
            torch.complex128,
        }:
            raise ValueError("Rotation requires float32/64 or complex64/128.")
        # These permutations are real even for SOC. Keeping real buffers avoids
        # silently discarding complex spin coefficients in Module.to(dtype=...).
        if dtype.is_complex:
            dtype = torch.float64 if dtype == torch.complex128 else torch.float32
        self.register_buffer(
            "_anchor", torch.empty(0, dtype=dtype, device=device), persistent=False
        )
        for l, indices in enumerate(
            ([0], [1, 2, 0], [2, 4, 0, 3, 1], [6, 4, 2, 0, 1, 3, 5])
        ):
            self.register_buffer(
                f"permutation_{l}",
                torch.eye(2 * l + 1, dtype=dtype, device=device)[indices],
                persistent=False,
            )

    def basis_matrices(self, left_shells, right_shells, *, spinful=None):
        if isinstance(left_shells, int):
            left_shells = [left_shells]
        if isinstance(right_shells, int):
            right_shells = [right_shells]
        if any(l not in range(4) for l in [*left_shells, *right_shells]):
            raise ValueError(
                "Native orbital permutations are specified only for s/p/d/f."
            )
        left = torch.block_diag(
            *(getattr(self, f"permutation_{l}") for l in left_shells)
        )
        right = torch.block_diag(
            *(getattr(self, f"permutation_{l}") for l in right_shells)
        )
        if self.spinful if spinful is None else spinful:
            left, right = torch.block_diag(left, left), torch.block_diag(right, right)
        return left, right

    def to_coupling_basis(self, blocks, left_shells, right_shells, *, spinful=None):
        left, right = self.basis_matrices(left_shells, right_shells, spinful=spinful)
        return left.to(blocks) @ blocks @ right.to(blocks).T

    def from_coupling_basis(self, blocks, left_shells, right_shells, *, spinful=None):
        left, right = self.basis_matrices(left_shells, right_shells, spinful=spinful)
        return left.to(blocks).T @ blocks @ right.to(blocks)

    @staticmethod
    def convert_rotation(rotation):
        indices = torch.tensor([1, 2, 0], device=rotation.device)
        return rotation.index_select(-2, indices).index_select(-1, indices)

    def spin_half_rotation(self, rotation):
        if not self.spinful:
            raise ValueError("Spin rotation requires spinful=True.")
        proper = rotation * torch.det(rotation).sign()[..., None, None]
        alpha, beta, gamma = matrix_to_angles(proper)
        dtype = torch.complex128 if rotation.dtype == torch.float64 else torch.complex64
        j = (
            torch.tensor([[1, 1], [1j, -1j]], dtype=dtype, device=rotation.device)
            / 2**0.5
        )
        frequencies = torch.tensor(
            [0.5, -0.5], dtype=rotation.dtype, device=rotation.device
        )

        def z(angle):
            return torch.diag_embed(torch.exp(-1j * angle[..., None] * frequencies))

        return z(alpha) @ j @ z(beta) @ j.T.conj() @ z(gamma)

    def rotate_blocks(
        self, blocks, rotation, left_shells, right_shells, *, order_xyz=True
    ):
        if rotation.shape[-2:] != (3, 3):
            raise ValueError("Frame rotations require trailing dimensions 3x3.")
        r = self.convert_rotation(rotation) if order_xyz else rotation
        il = Irreps([(1, (int(l), (-1) ** int(l))) for l in left_shells])
        ir = Irreps([(1, (int(l), (-1) ** int(l))) for l in right_shells])
        dl, dr = il.D_from_matrix(r), ir.D_from_matrix(r)
        if self.spinful:
            spin = self.spin_half_rotation(r)

            def spin_orbital_product(orbital):
                product = torch.einsum("...ab,...ij->...aibj", spin, orbital.to(spin))
                return product.reshape(
                    *r.shape[:-2], 2 * orbital.shape[-2], 2 * orbital.shape[-1]
                )

            dl, dr = spin_orbital_product(dl), spin_orbital_product(dr)
        coupling = self.to_coupling_basis(blocks, left_shells, right_shells)
        rotated = dl.to(blocks).transpose(-1, -2).conj() @ coupling @ dr.to(blocks)
        return self.from_coupling_basis(rotated, left_shells, right_shells)

    # Explicit source conventions retained for reference-import tools.
    rotate_openmx_H_full = rotate_blocks
    wiki2openmx_H_full = from_coupling_basis
    openmx2wiki_H_full = to_coupling_basis
    D_one_half = spin_half_rotation
    rotate_matrix_convert = convert_rotation


class ShellTensorCodec(nn.Module):
    """Invertible source-compatible shell coupling, optionally shared by species.

    ``shell_pairs`` fixes the ordered output channels, with one ``(l1,l2)``
    per channel. ``from_species_shells`` additionally provides selection,
    packing, masks and reconstruction of full native blocks. Scalar shifts,
    standard deviations and Hermitian projection are separate operations.
    """

    def __init__(
        self,
        shell_pairs,
        *,
        spinful=False,
        ignore_parity=False,
        sort_irreps=False,
        dtype=torch.float32,
        device="cpu",
    ):
        super().__init__()
        if dtype not in {torch.float32, torch.float64}:
            raise ValueError("Coupled network channels require float32 or float64.")
        self.shell_pairs = tuple(tuple(map(int, pair)) for pair in shell_pairs)
        if not self.shell_pairs or any(
            len(p) != 2 or any(l not in range(4) for l in p) for p in self.shell_pairs
        ):
            raise ValueError("Provide at least one s/p/d/f shell pair.")
        self.spinful, self.ignore_parity = bool(spinful), bool(ignore_parity)
        coefficient_dtype = dtype
        self.rotation = OrbitalFrameRotation(dtype, device, spinful)
        self._records = []
        irreps, coupled_offset = Irreps(), 0
        for index, (l1, l2) in enumerate(self.shell_pairs):
            parity = 1 if ignore_parity else (-1) ** (l1 + l2)
            orbital_irreps = Irreps(
                [(1, (l, parity)) for l in range(abs(l1 - l2), l1 + l2 + 1)]
            )
            channel_irreps = orbital_irreps
            spin_slices = [0, orbital_irreps.dim]
            orbital_slices = [0]
            for _, ir in orbital_irreps:
                if spinful:
                    spin_irreps = Irreps(
                        [(1, (l, parity)) for l in range(abs(ir.l - 1), ir.l + 2)]
                    )
                    channel_irreps += spin_irreps
                    spin_slices.append(spin_slices[-1] + spin_irreps.dim)
                    orbital_slices.append(orbital_slices[-1] + ir.dim)
                    self.register_buffer(
                        f"spin_decode_{index}_{len(orbital_slices) - 2}",
                        torch.cat(
                            [
                                wigner_3j(
                                    ir.l,
                                    1,
                                    sp.l,
                                    dtype=coefficient_dtype,
                                    device=device,
                                )
                                for _, sp in spin_irreps
                            ],
                            dim=-1,
                        ),
                        persistent=False,
                    )
                    self.register_buffer(
                        f"spin_encode_{index}_{len(orbital_slices) - 2}",
                        torch.cat(
                            [
                                wigner_3j(
                                    sp.l,
                                    ir.l,
                                    1,
                                    dtype=coefficient_dtype,
                                    device=device,
                                )
                                * (2 * sp.l + 1)
                                for _, sp in spin_irreps
                            ],
                            dim=0,
                        ),
                        persistent=False,
                    )
            self.register_buffer(
                f"decode_{index}",
                torch.cat(
                    [
                        wigner_3j(l1, l2, ir.l, dtype=coefficient_dtype, device=device)
                        for _, ir in orbital_irreps
                    ],
                    dim=-1,
                ),
                persistent=False,
            )
            self.register_buffer(
                f"encode_{index}",
                torch.cat(
                    [
                        wigner_3j(ir.l, l1, l2, dtype=coefficient_dtype, device=device)
                        * (2 * ir.l + 1)
                        for _, ir in orbital_irreps
                    ],
                    dim=0,
                ),
                persistent=False,
            )
            self._records.append(
                (
                    coupled_offset,
                    coupled_offset + channel_irreps.dim,
                    spin_slices,
                    orbital_slices,
                )
            )
            coupled_offset += channel_irreps.dim
            irreps += channel_irreps
        if spinful:
            irreps += irreps
            spin_basis = (
                torch.tensor(
                    [[1, 0, 1, 0], [0, -1j, 0, 1], [0, 1j, 0, 1], [1, 0, -1, 0]],
                    dtype=(
                        torch.complex128 if dtype == torch.float64 else torch.complex64
                    ),
                    device=device,
                )
                / 2**0.5
            )
            self.register_buffer("spin_basis_real", spin_basis.real, persistent=False)
            self.register_buffer("spin_basis_imag", spin_basis.imag, persistent=False)
        sorted_metadata = irreps.sort()
        order = sorted_metadata.inv if sort_irreps else range(len(irreps))
        permutation = torch.cat(
            [torch.arange(irreps.dim, device=device)[irreps.slices()[i]] for i in order]
        )
        self.register_buffer("channel_permutation", permutation, persistent=False)
        self.register_buffer(
            "inverse_permutation", permutation.argsort(), persistent=False
        )
        self.irreps_out = sorted_metadata.irreps.simplify() if sort_irreps else irreps
        self.required_irreps_out = self.irreps_out
        self.coupled_dim = irreps.dim
        self.packed_slices = tuple(
            accumulate(
                [0, *((2 * l1 + 1) * (2 * l2 + 1) for l1, l2 in self.shell_pairs)]
            )
        )
        self.H_slices = self.packed_slices
        self.packed_dim = self.packed_slices[-1]
        self.species_shells = None
        self.channel_assignments = None

    @property
    def spin_basis(self):
        """Complex SOC mixing matrix; real buffers survive module dtype moves."""
        if not self.spinful:
            raise ValueError("Spin basis requires spinful=True.")
        return torch.complex(self.spin_basis_real, self.spin_basis_imag)

    def decode(self, channels):
        """Real coupled coefficients → packed native shell entries."""
        if channels.shape[-1] != self.coupled_dim or channels.is_complex():
            raise ValueError(f"Expected {self.coupled_dim} real coupled coefficients.")
        leading = channels.shape[:-1]
        values = channels.reshape(-1, self.coupled_dim).index_select(
            -1, self.inverse_permutation
        )
        if self.spinful:
            real, imag = values.chunk(2, dim=-1)
            values = torch.complex(real, imag)
        result = []
        for i, (begin, end, spin_slices, _) in enumerate(self._records):
            value = values[:, begin:end]
            if self.spinful:
                spin_parts = [value[:, : spin_slices[1]].unsqueeze(-1)]
                orbital_parts = [
                    torch.einsum(
                        "jkl,il->ijk",
                        getattr(self, f"spin_decode_{i}_{j}").to(value),
                        value[:, spin_slices[j + 1] : spin_slices[j + 2]],
                    )
                    for j in range(len(spin_slices) - 2)
                ]
                spin_parts.append(torch.cat(orbital_parts, dim=-2))
                value = torch.cat(spin_parts, dim=-1)
                block = torch.einsum(
                    "imn,klm,jn->ijkl",
                    value,
                    getattr(self, f"decode_{i}").to(value),
                    self.spin_basis.to(value),
                )
            else:
                block = torch.einsum(
                    "ijk,bk->bij", getattr(self, f"decode_{i}").to(value), value
                )
            block = self.rotation.from_coupling_basis(
                block, *self.shell_pairs[i], spinful=False
            )
            result.append(
                block.reshape(values.shape[0], *((4,) if self.spinful else ()), -1)
            )
        return torch.cat(result, dim=-1).reshape(
            *leading, *((4,) if self.spinful else ()), self.packed_dim
        )

    def encode(self, packed):
        """Packed native shell entries → real coupled coefficients."""
        if packed.shape[-1] != self.packed_dim or (
            self.spinful and (packed.ndim < 2 or packed.shape[-2] != 4)
        ):
            raise ValueError("Packed shell tensor has an incompatible shape.")
        if not self.spinful and packed.is_complex():
            raise ValueError("Spinless codec expects real native blocks.")
        leading = packed.shape[: -2 if self.spinful else -1]
        values = packed.reshape(-1, *((4,) if self.spinful else ()), self.packed_dim)
        result = []
        for i, (l1, l2) in enumerate(self.shell_pairs):
            block = values[
                ..., self.packed_slices[i] : self.packed_slices[i + 1]
            ].reshape(-1, *((4,) if self.spinful else ()), 2 * l1 + 1, 2 * l2 + 1)
            block = self.rotation.to_coupling_basis(block, l1, l2, spinful=False)
            if self.spinful:
                if not block.is_complex():
                    block = torch.complex(block, torch.zeros_like(block))
                value = torch.einsum(
                    "ilmn,jmn,kl->ijk",
                    block,
                    getattr(self, f"encode_{i}").to(block),
                    self.spin_basis.to(block).T.conj(),
                )
                parts = [value[:, :, 0]]
                orbital_slices = self._records[i][3]
                parts.extend(
                    torch.einsum(
                        "jlm,ilm->ij",
                        getattr(self, f"spin_encode_{i}_{j}").to(block),
                        value[:, orbital_slices[j] : orbital_slices[j + 1], 1:],
                    )
                    for j in range(len(orbital_slices) - 1)
                )
                value = torch.cat(parts, dim=-1)
            else:
                value = torch.einsum(
                    "ijk,bjk->bi", getattr(self, f"encode_{i}").to(block), block
                )
            result.append(value)
        result = torch.cat(result, dim=-1)
        if self.spinful:
            result = torch.cat([result.real, result.imag], dim=-1)
        return result.index_select(-1, self.channel_permutation).reshape(
            *leading, self.coupled_dim
        )

    get_H = decode
    get_net_out = encode
    forward = decode

    @classmethod
    def from_species_shells(
        cls,
        species_shells,
        *,
        channel_assignments=None,
        block_kind="all",
        species_pairs=None,
        **kwargs,
    ):
        shells = {str(k): tuple(map(int, v)) for k, v in species_shells.items()}
        targets = build_shell_targets(
            shells,
            requested=channel_assignments,
            block_kind=block_kind,
            species_pairs=species_pairs,
        )
        pairs = []
        for target in targets:
            (a, b), (i, j) = next(iter(target.items()))
            pairs.append((shells[a][i], shells[b][j]))
        codec = cls(pairs, **kwargs)
        codec.species_shells, codec.channel_assignments = shells, targets
        return codec

    def _block_layout(self, pair):
        if self.species_shells is None:
            raise ValueError("Full-block operations require from_species_shells().")
        pair = _pair(pair)
        if any(species not in self.species_shells for species in pair):
            raise ValueError("Unknown species in block pair.")
        left, right = (
            tuple(accumulate([0, *(2 * l + 1 for l in self.species_shells[p])]))
            for p in pair
        )
        return pair, left, right

    def packed_mask(self, pair, *, device=None):
        pair, _, _ = self._block_layout(pair)
        mask = torch.zeros(
            *((4,) if self.spinful else ()),
            self.packed_dim,
            dtype=torch.bool,
            device=device or self.channel_permutation.device,
        )
        for i, assignments in enumerate(self.channel_assignments):
            if pair in assignments:
                mask[..., self.packed_slices[i] : self.packed_slices[i + 1]] = True
        return mask

    @classmethod
    def from_signature(cls, signature, *, device="cpu", dtype=None):
        """Restore and validate the complete frozen convention without RNG effects."""
        with torch.random.fork_rng(devices=[]):
            shells = {
                name: signature["species_shells"][name]
                for name in signature["species_order"]
            }
            options = dict(
                channel_assignments=signature["channel_assignments"],
                spinful=signature["spinful"],
                ignore_parity=signature["ignore_parity"],
            )
            codec = cls.from_species_shells(shells, **options)
            if codec.layout_signature() != signature:
                codec = cls.from_species_shells(shells, sort_irreps=True, **options)
            if codec.layout_signature() != signature:
                raise ValueError("Frozen orbital layout cannot be reconstructed")
        return codec.to(device=device, dtype=dtype or torch.get_default_dtype())

    def layout_signature(self):
        """Serializable identity of native shell slots and coupled conventions."""
        if self.species_shells is None or self.channel_assignments is None:
            raise ValueError("Layout identity requires an explicit species-shell codec")
        return {
            "species_order": list(self.species_shells),
            "species_shells": {
                name: list(shells) for name, shells in self.species_shells.items()
            },
            "shell_pairs": [list(pair) for pair in self.shell_pairs],
            "channel_assignments": [
                {f"{a} {b}": list(slots) for (a, b), slots in assignment.items()}
                for assignment in self.channel_assignments
            ],
            "spinful": self.spinful,
            "ignore_parity": self.ignore_parity,
            "channel_permutation": self.channel_permutation.detach().cpu().tolist(),
            "packed_dim": self.packed_dim,
            "coupled_irreps": str(self.irreps_out),
        }

    def coupled_mask(self, pair, *, device=None):
        """Validity mask in network-channel order for shared shell selection."""
        pair, _, _ = self._block_layout(pair)
        mask = torch.zeros(
            self.coupled_dim, dtype=torch.bool, device=self.channel_permutation.device
        )
        for (begin, end, _, _), assignments in zip(
            self._records, self.channel_assignments
        ):
            if pair in assignments:
                mask[begin:end] = True
                if self.spinful:
                    half = self.coupled_dim // 2
                    mask[half + begin : half + end] = True
        return mask.index_select(-1, self.channel_permutation).to(
            device=device or mask.device
        )

    def convert_mask(self, mask):
        """Reference SOC packed-mask ordering, including real/imag duplication.

        Masks must select complete shell channels. Arbitrary entrywise masks
        cannot be converted to irreps by an index permutation alone.
        """
        if not self.spinful or mask.ndim != 3 or mask.shape[1:] != (4, self.packed_dim):
            raise ValueError("SOC mask must have shape (edges, 4, packed_dim).")
        if mask.dtype != torch.bool:
            raise ValueError("Validity masks must be boolean.")
        for begin, end in zip(self.packed_slices[:-1], self.packed_slices[1:]):
            channel = mask[..., begin:end].flatten(1)
            if not torch.all(channel == channel[:, :1]):
                raise ValueError(
                    "Coupled validity masks must select complete shell channels."
                )
        return (
            mask.permute(0, 2, 1)
            .reshape(mask.shape[0], -1)
            .repeat(1, 2)
            .index_select(-1, self.channel_permutation)
        )

    def encode_mask(self, mask):
        """Conservative observed-channel mask for possibly incomplete labels.

        Every coupled coefficient of a shell pair mixes native entries. A
        channel is usable for fitted statistics only when its entire native
        shell block is observed, including all four spin components for SOC.
        Matrix-space losses may still use the original finer entrywise mask.
        """
        trailing = 2 if self.spinful else 1
        if mask.dtype != torch.bool or mask.shape[-1] != self.packed_dim:
            raise ValueError("Expected a boolean packed operator mask.")
        if self.spinful and (mask.ndim < 2 or mask.shape[-2] != 4):
            raise ValueError("SOC validity masks require four spin components.")
        leading = mask.shape[:-trailing]
        encoded = torch.zeros(
            *leading, self.coupled_dim, dtype=torch.bool, device=mask.device
        )
        for index, (begin, end, _, _) in enumerate(self._records):
            shell = mask[..., self.packed_slices[index] : self.packed_slices[index + 1]]
            valid = shell.flatten(-trailing).all(dim=-1)
            encoded[..., begin:end] = valid[..., None]
            if self.spinful:
                half = self.coupled_dim // 2
                encoded[..., half + begin : half + end] = valid[..., None]
        return encoded.index_select(-1, self.channel_permutation)

    def pack_blocks(self, pair, blocks, *, return_mask=False):
        pair, left, right = self._block_layout(pair)
        nl, nr = left[-1], right[-1]
        spin = 2 if self.spinful else 1
        if blocks.shape[-2:] != (spin * nl, spin * nr):
            raise ValueError(
                "Native block dimensions do not match the species shell layout."
            )
        if self.spinful:
            blocks = torch.stack(
                [
                    blocks[..., :nl, :nr],
                    blocks[..., :nl, nr:],
                    blocks[..., nl:, :nr],
                    blocks[..., nl:, nr:],
                ],
                dim=-3,
            )
        result = blocks.new_zeros(*blocks.shape[:-2], self.packed_dim)
        for index, assignments in enumerate(self.channel_assignments):
            if pair in assignments:
                i, j = assignments[pair]
                result[
                    ..., self.packed_slices[index] : self.packed_slices[index + 1]
                ] = blocks[..., left[i] : left[i + 1], right[j] : right[j + 1]].flatten(
                    -2
                )
        if return_mask:
            mask = self.packed_mask(pair, device=blocks.device).expand_as(result)
            return result, mask
        return result

    def unpack_blocks(self, pair, packed):
        pair, left, right = self._block_layout(pair)
        if packed.shape[-1] != self.packed_dim or (
            self.spinful and packed.shape[-2] != 4
        ):
            raise ValueError("Packed shell tensor has an incompatible shape.")
        result = packed.new_zeros(*packed.shape[:-1], left[-1], right[-1])
        for index, assignments in enumerate(self.channel_assignments):
            if pair in assignments:
                i, j = assignments[pair]
                result[..., left[i] : left[i + 1], right[j] : right[j + 1]] = packed[
                    ..., self.packed_slices[index] : self.packed_slices[index + 1]
                ].reshape(
                    *packed.shape[:-1], left[i + 1] - left[i], right[j + 1] - right[j]
                )
        if self.spinful:
            return torch.cat(
                [
                    torch.cat([result[..., 0, :, :], result[..., 1, :, :]], dim=-1),
                    torch.cat([result[..., 2, :, :], result[..., 3, :, :]], dim=-1),
                ],
                dim=-2,
            )
        return result

    def encode_blocks(self, pair, blocks):
        return self.encode(self.pack_blocks(pair, blocks))

    def decode_blocks(self, pair, channels):
        return self.unpack_blocks(pair, self.decode(channels))
