# k-Space and Shift-Space Matrices

This note summarizes how periodic one-particle matrices such as the Hamiltonian `H`, overlap `S`, and density matrix `D` are related in:

- **k-space**: matrices indexed by crystal momentum `k`
- **shift-space**: matrices indexed by an integer lattice shift `R = (s_x, s_y, s_z)`

In this repository, the PySCF production path computes matrices in k-space and then converts them into explicit shift-resolved matrices keyed by

```text
(s_x, s_y, s_z, i, j)
```

where `(i, j)` are atom indices and `(s_x, s_y, s_z)` is the lattice-image shift.

## Objects and notation

Let:

- `a_1, a_2, a_3` be the lattice vectors collected in the cell matrix `A`
- `R = s_x a_1 + s_y a_2 + s_z a_3` be a Bravais-lattice translation
- `mu, nu` denote atomic-orbital (AO) indices
- `k` denote a sampled crystal momentum
- `N_k = k_x k_y k_z` be the number of sampled k-points

For any AO matrix `M`, we use:

- `M_{mu nu}(k)` for the **k-space** matrix
- `M_{mu nu}(R)` for the **shift-space** matrix

The same formulas apply to:

- Hamiltonian / Kohn-Sham matrix: `H`
- overlap matrix: `S`
- density matrix: `D`
- any other periodic one-body AO matrix

## Periodic AO matrix structure

Because the system is periodic, a matrix element depends only on the relative cell shift:

```math
M_{mu nu}(R) =
\langle \phi_{mu,0} \mid \hat{M} \mid \phi_{nu,R} \rangle
```

where `phi_{mu,0}` is AO `mu` in the home cell and `phi_{nu,R}` is AO `nu` translated by lattice vector `R`.

The corresponding Bloch-summed matrix at k-point `k` is the discrete Fourier transform

```math
M_{mu nu}(k) = \sum_R e^{i k \cdot R} M_{mu nu}(R).
```

This is the periodic analogue of passing from a real-space convolution kernel to its reciprocal-space representation.

## Inverse transform: k-space to shift-space

Given matrices on a uniform Monkhorst-Pack-like k-mesh, the shift-space matrices are recovered by the inverse discrete Fourier transform

```math
M_{mu nu}(R) = \frac{1}{N_k} \sum_k e^{-i k \cdot R} M_{mu nu}(k).
```

This is the key relation used in the PySCF backend.

In the implementation, PySCF's `get_phase(..., wrap_around=True)` provides the phase table

```math
\Phi_{Rk} = \frac{1}{\sqrt{N_k}} e^{i k \cdot R},
```

and the code evaluates

```math
\widetilde{M}_{RS}
= \sum_k \Phi_{Rk} M(k) \Phi_{Sk}^*
= \frac{1}{N_k} \sum_k e^{i k \cdot (R-S)} M(k).
```

Because translational invariance implies dependence only on the relative shift, this becomes

```math
\widetilde{M}_{RS} = M(R-S).
```

The exported shift-resolved matrix stack is obtained by fixing one cell index to the origin:

```math
M(R) = \widetilde{M}_{0R}.
```

In the code, this is why we:

1. build the full `R,S` transformed tensor
2. locate the zero shift `R = 0`
3. extract the slice relative to the origin

## Finite k-mesh and resolvable shifts

For a finite `kmesh = (k_x, k_y, k_z)`, only

```math
N_k = k_x k_y k_z
```

independent shifts are represented.

With wrap-around indexing, the enumerated integer shifts are

```math
s_alpha \in
\begin{cases}
\{0,1,\dots,k_\alpha-1\}, & \text{without wrap-around} \\
\{0,1,\dots,\lfloor k_\alpha/2 \rfloor, -\lfloor (k_\alpha-1)/2 \rfloor,\dots,-1\}, & \text{with wrap-around}
\end{cases}
```

for `alpha in {x,y,z}`.

So the shift-space matrix obtained from a finite k-mesh is not an arbitrary infinite-range real-space object; it is the discrete Fourier pair of the sampled k-space data. Increasing the k-mesh increases the number of explicitly represented lattice shifts.

## Central-cell matrix as the zero-shift block

The familiar "dense AO matrix for the home cell" is just the zero-shift block

```math
M(0) = M(s_x=0,s_y=0,s_z=0).
```

If only Gamma is used, `N_k = 1`, so only the zero shift is resolvable and the entire periodic matrix content collapses to the home-cell block:

```math
M(R) = 0 \text{ for all nonzero represented } R,
\qquad
M(0) = M(k=0).
```

This is why a `1 x 1 x 1` calculation cannot recover explicit nonzero lattice-image channels.

## From AO matrices to atom-pair blocks

Once `M(R)` is available in AO form, we split it into atom-pair blocks.

Let atom `i` occupy AO index range `I_i` and atom `j` occupy AO range `I_j`. Then the atom-pair block for shift `R` is

```math
M^{(ij)}(R) = M(R)[I_i, I_j].
```

Mandala stores these sparse blocks under the edge key

```math
(s_x, s_y, s_z, i, j).
```

So the block means:

```math
M^{(ij)}(R)
= \langle \phi_{i,0} \mid \hat{M} \mid \phi_{j,R} \rangle.
```

## Hermitian symmetry in shift-space

If the k-space matrix is Hermitian at each k-point,

```math
M(k)^\dagger = M(k),
```

then the shift-space matrix satisfies

```math
M(R)^\dagger = M(-R).
```

At the atom-block level this becomes

```math
M^{(ij)}(R) = \left(M^{(ji)}(-R)\right)^T
```

for real-valued matrices, or conjugate-transpose in the complex case.

This is exactly the symmetry the parser enforces when it canonicalizes the sparse edge set.

## Hamiltonian, overlap, and density

The same Fourier relation applies independently to:

### Hamiltonian / Kohn-Sham matrix

```math
H(k) = \sum_R e^{i k \cdot R} H(R),
\qquad
H(R) = \frac{1}{N_k} \sum_k e^{-i k \cdot R} H(k).
```

### Overlap matrix

```math
S(k) = \sum_R e^{i k \cdot R} S(R),
\qquad
S(R) = \frac{1}{N_k} \sum_k e^{-i k \cdot R} S(k).
```

### Density matrix

```math
D(k) = \sum_R e^{i k \cdot R} D(R),
\qquad
D(R) = \frac{1}{N_k} \sum_k e^{-i k \cdot R} D(k).
```

In practice:

- `H(k)` is the k-point Kohn-Sham matrix exported from periodic PySCF
- `S(k)` is the k-point AO overlap matrix
- `D(k)` is the k-point AO density matrix

and all three are transformed identically into shift-space.

## Observable traces in shift-space

Once we have shift-resolved blocks, important observables can be computed without reconstructing a single dense supercell matrix.

For two periodic matrices `A` and `B`,

```math
\mathrm{Tr}(AB)
= \sum_R \mathrm{tr}\!\left(A(R) B(-R)\right).
```

At the sparse block level this becomes a sum over edge-reversed atom pairs:

```math
\mathrm{Tr}(AB)
= \sum_{R,i,j}
\mathrm{tr}\!\left(
A^{(ij)}(R)\,
B^{(ji)}(-R)
\right).
```

This is the formula used by `Snapshot` for:

```math
E_{1b} = \mathrm{Tr}(D H),
\qquad
N_e = \mathrm{Tr}(D S).
```

## Practical interpretation

- **k-space matrices** are the natural output of periodic SCF/DFT codes.
- **shift-space matrices** expose the explicit lattice-image structure that Mandala uses for sparse graph construction and equivariance checks.
- A larger k-mesh yields more explicitly represented lattice shifts.
- The central-cell dense matrix is only the `R = 0` slice of the full periodic object.
- The shift-space representation preserves the physically important relation between home-cell orbitals and orbitals in translated images.

## Repository-specific convention

In the PySCF backend:

1. PySCF computes `H(k)`, `S(k)`, and `D(k)` on a uniform k-mesh.
2. These are inverse-Fourier transformed into shift-space matrices indexed by integer lattice shifts.
3. Each shift-space AO matrix is split into atom-pair blocks.
4. Blocks are stored sparsely with canonical keys

```text
(s_x, s_y, s_z, i, j).
```

That is the object loaded by `Snapshot.from_pyscf(...)` and compared against rotated snapshots in the equivariance workflow.
