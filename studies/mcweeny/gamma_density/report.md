# Hamiltonian-derived density and Γ occupations

HSD fine-tune / validation snapshot 090

Reference OpenMX density uses (D+Dᵀ)/2; legacy direct predictions are multiplied once by 0.5. All occupations and matrix errors use per-spin density.

## Ground-truth Γ spectrum

Γ matrices sum all stored periodic blocks. Occupations are eigenvalues of S^(1/2) D S^(1/2), which is similar to DS. No clipping is applied.

104 states; min -0.0026444481, max 0.91228343, sum 13.971634; 29 below zero and 0 above one. These are eigenvalues of the supplied finite-support density, not an untruncated Bloch density.

![Gamma occupation histogram](gamma_occupation_histogram.png)

## Γ density comparison

Solve HC=SC E, normalize CᵀSC=I, and form D=C_occ C_occᵀ from the lowest 16 states (32 electrons). Using all eigenvectors would give S⁻¹. This is distinct from diagonalizing HS and treating its eigenvectors as AO coefficients.

| Method | MAE | RMSE | Relative Frobenius |
|---|---:|---:|---:|
| Direct density prediction | 0.00011377567 | 0.00021532829 | 0.0099058845 |
| Reference H + reference S | 0.0020713928 | 0.0063889601 | 0.2939154 |
| Predicted H + reference S | 0.0022478173 | 0.0066778102 | 0.30720356 |
| Predicted H + predicted S | 0.0022434578 | 0.0065831695 | 0.30284974 |

## Periodic sparse density comparison

Reconstruct D(k)=C_occ(k) C_occ(k)† on a uniform Γ-centered 8³ mesh, then inverse Fourier transform onto exactly the reference edges. The k mesh is explicitly chosen here; it is not assumed to reproduce OpenMX mesh offsets. The original SCF used an 8³ mesh at 300 K; this CC† reconstruction uses integer occupations. The reference H/S control measures the combined effect of reconstruction, finite support, mesh and occupation conventions.

MAE/RMSE use all scalar entries over the union of periodic supports, matching the preceding McWeeny study. Band-energy errors use 2 Tr(D H_ref), eV/cell. Γ errors above and sparse errors below are different metrics.

| Method | MAE | RMSE | Band-energy error (eV/cell) | Electron count | Global gap (eV) |
|---|---:|---:|---:|---:|---:|
| Direct density prediction | 2.6367884e-05 | 5.6979185e-05 | 0.23125633 | 31.984106 | nan |
| Reference H + reference S | 1.3106427e-07 | 3.6949189e-07 | 2.6907889e-05 | 32 | 0.607895 |
| Predicted H + reference S | 0.00016429336 | 0.00049866058 | 0.010423269 | 32 | 0.638467 |
| Predicted H + predicted S | 0.0001656098 | 0.00049198978 | 0.08672382 | 31.981587 | 0.637633 |

All generalized-eigenproblem residuals and S-orthogonality errors are checked below 1e-10. Per-k occupation traces are checked against 16; inverse Fourier imaginary residuals are checked below 1e-10. Truncation can change the final sparse trace.

Reproduce from the repository root:

```bash
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 mandala-venv/bin/python scripts/evaluate_gamma_density.py --case-index 1 --mesh 8 --output-dir studies/mcweeny/gamma_density
```

## Findings

Hamiltonian-derived sparse density has 6.23× (reference S) or 6.28× (predicted S) the direct prediction MAE. The density-induced band-energy error nevertheless improves from 0.23126 eV/cell to 0.01042 or 0.08672 eV/cell, respectively. These energies hold reference H fixed; they are not joint H/D errors.

The reference-H/S reconstruction agrees with the stored sparse density to MAE 1.31e-7, while its full Γ projector differs substantially from the stored Γ density. This strongly supports finite real-space support as the source of the non-projector Γ spectrum, rather than fractional thermal occupation: the reference mesh gap is 0.608 eV (at 300 K, kBT is about 0.026 eV). The Γ occupation sum is not the electron count integrated over the Brillouin zone.

Additional targeted analytic checks passed for Fourier phases, the Γ block sum, inverse Fourier recovery of a known three-block scalar matrix, and a known nonorthogonal occupied projector. No broad test suite was run.
