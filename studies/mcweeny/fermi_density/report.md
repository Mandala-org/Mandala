# Finite-temperature Hamiltonian density: Si validation 090

Temperature: 300 K. Source: Si.out scf.ElectronicTemperature.

The scale_1 generator specifies independent Gaussian Cartesian displacement sigma=0.02 Å (nominal vector RMS sqrt(3)*sigma=0.03464 Å). No temperature calibration is recorded. Displacement amplitude alone does not specify either ionic or electronic temperature. Therefore this calculation uses the documented electronic temperature, not an invented scale-to-temperature conversion.

Solve HC=SC E with C†SC=I on a uniform Gamma-centered 8³ mesh. Use f=1/(1+exp((E-mu)/(kB*T))) and D(k)=C f C†. A single chemical potential per H/S model is fitted to 32 electrons across the entire mesh, not separately at each k point. The zero-temperature baseline fills 16 states per spin and requires a positive global gap. Inverse Fourier transform onto the original periodic support.

Reference density is explicitly loaded without the checkout-dependent default spin factor and Hermitian-averaged. Legacy direct predictions are scaled by 0.5 once. MAE and RMSE use scalar entries over the union of supports. Band-energy errors hold reference H fixed and use 2 Tr(D H_ref), eV/cell.

| Method | T (K) | MAE | RMSE | Band-energy error (eV/cell) | Thermal conduction electrons/cell |
|---|---:|---:|---:|---:|---:|
| Direct density prediction | — | 2.63678842e-05 | 5.69791849e-05 | 0.231256325 | — |
| Reference H + reference S | 0 | 1.31064274e-07 | 3.69491887e-07 | 2.69078888e-05 | 0 |
| Reference H + reference S | 300 | 1.3101869e-07 | 3.69364867e-07 | 2.69669375e-05 | 9.2343e-08 |
| Predicted H + reference S | 0 | 0.000164293364 | 0.000498660582 | 0.0104232694 | 0 |
| Predicted H + reference S | 300 | 0.000164293362 | 0.000498660575 | 0.0104233035 | 5.28346e-08 |
| Predicted H + predicted S | 0 | 0.000165609796 | 0.000491989779 | 0.0867238202 | 0 |
| Predicted H + predicted S | 300 | 0.000165609792 | 0.000491989775 | 0.0867238548 | 5.36437e-08 |

All eigenproblem and S-orthogonality residuals are checked below 1e-10; mesh-integrated occupations agree with 32 electrons to 2e-9. Inverse Fourier imaginary parts are checked below 1e-10. This comparison changes occupations at fixed Hamiltonian; it is not a new self-consistent DFT calculation.

Reproduce:

```bash
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 mandala-venv/bin/python scripts/evaluate_fermi_density.py --temperature-k 300 --mesh 8 --output-dir studies/mcweeny/fermi_density
```

## Interpretation and targeted validation

At 300 K the predicted-H/reference-S reconstructed density changes from its zero-temperature counterpart by MAE 1.30e-10. Its reference-density MAE remains 1.643e-4, about 6.23 times the direct prediction MAE. The 0.638 eV global gap suppresses thermal conduction occupation to 5.28e-8 electrons per cell. The finite-temperature band-energy error remains approximately 0.0104233 eV/cell. Predicted overlap gives the same qualitative conclusion.

Targeted analytic checks passed for the logistic Fermi function, one shared chemical potential across unequal k-point populations, the integer-occupation baseline, and nonorthogonal CfC† electron counting. No broad suite was executed.

Scale provenance: `/home/bartek/casus/structure_gen/gen.py` defines scale_1 sigma=0.02 Å; `/home/bartek/casus/structure_gen/perturbed_snapshots_Si/snapshot_metadata.json` records that sigma and nominal RMS displacement but no temperature calibration. These perturbations do not by themselves specify an electronic temperature. The requested scale-derived-temperature comparison remains conditional on such a calibration; 300 K is the explicitly recorded reference electronic temperature.
