# Finite-temperature Hamiltonian density: Si validation 090

Temperatures: 40, 50, 60, 70, 80 K. Source: Explicit CLI override.

The scale_1 generator specifies independent Gaussian Cartesian displacement sigma=0.02 Å (nominal vector RMS sqrt(3)*sigma=0.03464 Å). No temperature calibration is recorded. Displacement amplitude alone does not specify either ionic or electronic temperature. Temperatures are explicitly supplied for a sweep or taken from the recorded electronic temperature by default.

Solve HC=SC E with C†SC=I on a uniform Gamma-centered 8³ mesh. Use f=1/(1+exp((E-mu)/(kB*T))) and D(k)=C f C†. A single chemical potential per H/S model is fitted to 32 electrons across the entire mesh, not separately at each k point. At low temperature, solve electron/hole balance in log space to avoid loss of precision from subtracting nearly equal total charges. The zero-temperature baseline fills 16 states per spin and requires a positive global gap. Inverse Fourier transform onto the original periodic support.

Reference density is explicitly loaded without the checkout-dependent default spin factor and Hermitian-averaged. Legacy direct predictions are scaled by 0.5 once. MAE and RMSE use scalar entries over the union of supports. Band-energy errors hold reference H fixed and use 2 Tr(D H_ref), eV/cell.

| Method | T (K) | MAE | RMSE | Band-energy error (eV/cell) | Thermal conduction electrons/cell |
|---|---:|---:|---:|---:|---:|
| Direct density prediction | — | 2.63678842e-05 | 5.69791849e-05 | 0.231256325 | — |
| Reference H + reference S | 0 | 1.31064274e-07 | 3.69491887e-07 | 2.69078888e-05 | 0 |
| Reference H + reference S | 40 | 1.31064274e-07 | 3.69491887e-07 | 2.69078888e-05 | 3.4652e-41 |
| Reference H + reference S | 50 | 1.31064274e-07 | 3.69491887e-07 | 2.69078888e-05 | 1.66127e-33 |
| Reference H + reference S | 60 | 1.31064274e-07 | 3.69491887e-07 | 2.69078888e-05 | 2.20323e-28 |
| Reference H + reference S | 70 | 1.31064274e-07 | 3.69491887e-07 | 2.69078888e-05 | 1.00772e-24 |
| Reference H + reference S | 80 | 1.31064274e-07 | 3.69491887e-07 | 2.69078888e-05 | 5.61672e-22 |
| Predicted H + reference S | 0 | 0.000164293364 | 0.000498660582 | 0.0104232694 | 0 |
| Predicted H + reference S | 40 | 0.000164293364 | 0.000498660582 | 0.0104232694 | 4.36393e-43 |
| Predicted H + reference S | 50 | 0.000164293364 | 0.000498660582 | 0.0104232694 | 5.03082e-35 |
| Predicted H + reference S | 60 | 0.000164293364 | 0.000498660582 | 0.0104232694 | 1.1961e-29 |
| Predicted H + reference S | 70 | 0.000164293364 | 0.000498660582 | 0.0104232694 | 8.29594e-26 |
| Predicted H + reference S | 80 | 0.000164293364 | 0.000498660582 | 0.0104232694 | 6.3166e-23 |
| Predicted H + predicted S | 0 | 0.000165609796 | 0.000491989779 | 0.0867238202 | 0 |
| Predicted H + predicted S | 40 | 0.000165609796 | 0.000491989779 | 0.0867238202 | 4.90049e-43 |
| Predicted H + predicted S | 50 | 0.000165609796 | 0.000491989779 | 0.0867238202 | 5.51593e-35 |
| Predicted H + predicted S | 60 | 0.000165609796 | 0.000491989779 | 0.0867238202 | 1.29089e-29 |
| Predicted H + predicted S | 70 | 0.000165609796 | 0.000491989779 | 0.0867238202 | 8.85387e-26 |
| Predicted H + predicted S | 80 | 0.000165609796 | 0.000491989779 | 0.0867238202 | 6.68554e-23 |

All eigenproblem and S-orthogonality residuals are checked below 1e-10; mesh-integrated occupations agree with 32 electrons to 2e-9. Inverse Fourier imaginary parts are checked below 1e-10. This comparison changes occupations at fixed Hamiltonian; it is not a new self-consistent DFT calculation.

Reproduce:

```bash
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 mandala-venv/bin/python scripts/evaluate_fermi_density.py --temperatures-k 40 50 60 70 80 --mesh 8 --output-dir studies/mcweeny/fermi_density_40_80K
```

Targeted analytic solver checks passed at all five temperatures and at 300 K, including unequal band degeneracy and electron/hole balance. No broad test suite was run.
