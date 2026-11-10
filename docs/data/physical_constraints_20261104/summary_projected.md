Historical July inference only. Selected scenarios: silicon: `joint_predicted_reference_support`; sio2: `predicted_H_reference_SD_reference_support`.

Reference-support control and Hermitian projection only; no overlap repair, fixed-charge occupation projection, or post-cleanup aggregates exist.

These spectral values use Hermitian-projected matrices on reference support, a control requiring reference sparsity, not independent deployment. SiO2 H uses reference S/D; silicon uses predicted H/S/D. Zn reports H elements only. Missing post-cleanup results are unavailable, not replaced with raw predictions.

Means are over all contributing k points (not means of structure means); electron-count absolute errors use exact Hermitian real-space traces per structure; Fermi–Dirac residuals use one fit per structure. Worst is min for lower bounds and positive-definite success, max otherwise. Counts exclude invalid metrics; total denominators retain them. Contributing structure counts are recovered from historical per-structure summaries. Element MAEs are equally weighted structure means, with worst structure MAE; k points do not apply (0). Reference element error is zero by definition, not a physical residual.

| Material | Operator | Diagnostic | Ref mean | Ref worst | Ref n / totals | Pred mean | Pred worst | Pred n / totals |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| silicon | H | projected common-support MAE (meV) | 0 | 0 | 16 structures; 16/16 structures; 0 total kpoints | 0.435164 | 0.610003 | 16 structures; 16/16 structures; 0 total kpoints |
| silicon | S | projected common-support MAE (AO units) | 0 | 0 | 16 structures; 16/16 structures; 0 total kpoints | 1.38978e-05 | 1.67892e-05 | 16 structures; 16/16 structures; 0 total kpoints |
| silicon | D | projected common-support MAE (AO units) | 0 | 0 | 16 structures; 16/16 structures; 0 total kpoints | 5.75772e-05 | 8.24122e-05 | 16 structures; 16/16 structures; 0 total kpoints |
| silicon | S | overlap.positive_definite | 1 | 1 | 8192 kpoints; 16/16 structures; 8192 total kpoints | 0.172729 | 0 | 8192 kpoints; 16/16 structures; 8192 total kpoints |
| silicon | S | overlap.well_conditioned_positive_definite | 1 | 1 | 8192 kpoints; 16/16 structures; 8192 total kpoints | 0.172485 | 0 | 8192 kpoints; 16/16 structures; 8192 total kpoints |
| silicon | S | overlap.cholesky_failed_without_jitter | 0 | 0 | 8192 kpoints; 16/16 structures; 8192 total kpoints | 0.827271 | 1 | 8192 kpoints; 16/16 structures; 8192 total kpoints |
| silicon | S | overlap.lambda_min | 0.00118844 | 0.000872547 | 8192 kpoints; 16/16 structures; 8192 total kpoints | -0.000655239 | -0.00624094 | 8192 kpoints; 16/16 structures; 8192 total kpoints |
| silicon | S | overlap.lambda_max | 6.34486 | 6.74327 | 8192 kpoints; 16/16 structures; 8192 total kpoints | 6.34674 | 6.74962 | 8192 kpoints; 16/16 structures; 8192 total kpoints |
| silicon | S | overlap.condition_number | 5388.83 | 7156.37 | 8192 kpoints; 16/16 structures; 8192 total kpoints | 295705 | 1.02091e+08 | 1415 kpoints; 15/16 structures; 8192 total kpoints |
| silicon | S | overlap.nonpositive_eigenvalues | 0 | 0 | 8192 kpoints; 16/16 structures; 8192 total kpoints | 1.12427 | 4 | 8192 kpoints; 16/16 structures; 8192 total kpoints |
| silicon | D | density.occupation_min | -0.0334263 | -0.0418119 | 8192 kpoints; 16/16 structures; 8192 total kpoints | -0.0325261 | -0.0393333 | 1413 kpoints; 15/16 structures; 8192 total kpoints |
| silicon | D | density.occupation_max | 2.3045 | 2.49221 | 8192 kpoints; 16/16 structures; 8192 total kpoints | 2.28845 | 2.41358 | 1413 kpoints; 15/16 structures; 8192 total kpoints |
| silicon | D | density.negative_count | 37.8605 | 42 | 8192 kpoints; 16/16 structures; 8192 total kpoints | 38.54 | 44 | 1413 kpoints; 15/16 structures; 8192 total kpoints |
| silicon | D | density.overoccupied_count | 7.01294 | 12 | 8192 kpoints; 16/16 structures; 8192 total kpoints | 7.19887 | 10 | 1413 kpoints; 15/16 structures; 8192 total kpoints |
| silicon | D | electron_count_absolute_error | 9.99336e-08 | 1.952e-07 | 16 structures; 16/16 structures; 8192 total kpoints | 0.0075323 | 0.0114623 | 16 structures; 16/16 structures; 8192 total kpoints |
| silicon | D | density.zero_temperature_idempotency_relative | 0.102435 | 0.160485 | 8192 kpoints; 16/16 structures; 8192 total kpoints | 0.101953 | 0.155847 | 1413 kpoints; 15/16 structures; 8192 total kpoints |
| silicon | H | consistency.commutator_relative | 0.00235366 | 0.00352099 | 8192 kpoints; 16/16 structures; 8192 total kpoints | 0.000969553 | 0.0019418 | 1413 kpoints; 15/16 structures; 8192 total kpoints |
| silicon | H | finite_temperature_density_relative_frobenius | 0.121229 | 0.121477 | 16 structures; 16/16 structures; 8192 total kpoints | — | — | 0 valid; unavailable |
| sio2 | H | projected common-support MAE (meV) | 0 | 0 | 10 structures; 10/10 structures; 0 total kpoints | 1.99588 | 2.53462 | 10 structures; 10/10 structures; 0 total kpoints |
| sio2 | S | overlap.positive_definite | 1 | 1 | 10 kpoints; 10/10 structures; 10 total kpoints | 1 | 1 | 10 kpoints; 10/10 structures; 10 total kpoints |
| sio2 | S | overlap.well_conditioned_positive_definite | 1 | 1 | 10 kpoints; 10/10 structures; 10 total kpoints | 1 | 1 | 10 kpoints; 10/10 structures; 10 total kpoints |
| sio2 | S | overlap.cholesky_failed_without_jitter | 0 | 0 | 10 kpoints; 10/10 structures; 10 total kpoints | 0 | 0 | 10 kpoints; 10/10 structures; 10 total kpoints |
| sio2 | S | overlap.lambda_min | 0.000250166 | 0.000215888 | 10 kpoints; 10/10 structures; 10 total kpoints | 0.000250166 | 0.000215888 | 10 kpoints; 10/10 structures; 10 total kpoints |
| sio2 | S | overlap.lambda_max | 7.38574 | 7.61309 | 10 kpoints; 10/10 structures; 10 total kpoints | 7.38574 | 7.61309 | 10 kpoints; 10/10 structures; 10 total kpoints |
| sio2 | S | overlap.condition_number | 29830.9 | 34732.7 | 10 kpoints; 10/10 structures; 10 total kpoints | 29830.9 | 34732.7 | 10 kpoints; 10/10 structures; 10 total kpoints |
| sio2 | S | overlap.nonpositive_eigenvalues | 0 | 0 | 10 kpoints; 10/10 structures; 10 total kpoints | 0 | 0 | 10 kpoints; 10/10 structures; 10 total kpoints |
| sio2 | D | density.occupation_min | -0.172922 | -0.344175 | 10 kpoints; 10/10 structures; 10 total kpoints | -0.172922 | -0.344175 | 10 kpoints; 10/10 structures; 10 total kpoints |
| sio2 | D | density.occupation_max | 2.37459 | 2.56227 | 10 kpoints; 10/10 structures; 10 total kpoints | 2.37459 | 2.56227 | 10 kpoints; 10/10 structures; 10 total kpoints |
| sio2 | D | density.negative_count | 663.7 | 682 | 10 kpoints; 10/10 structures; 10 total kpoints | 663.7 | 682 | 10 kpoints; 10/10 structures; 10 total kpoints |
| sio2 | D | density.overoccupied_count | 200.9 | 203 | 10 kpoints; 10/10 structures; 10 total kpoints | 200.9 | 203 | 10 kpoints; 10/10 structures; 10 total kpoints |
| sio2 | D | electron_count_absolute_error | 2.46731e-06 | 3.1647e-06 | 10 structures; 10/10 structures; 10 total kpoints | 2.46731e-06 | 3.1647e-06 | 10 structures; 10/10 structures; 10 total kpoints |
| sio2 | D | density.zero_temperature_idempotency_relative | 0.0520496 | 0.0604673 | 10 kpoints; 10/10 structures; 10 total kpoints | 0.0520496 | 0.0604673 | 10 kpoints; 10/10 structures; 10 total kpoints |
| sio2 | H | consistency.commutator_relative | 0.000791348 | 0.000902654 | 10 kpoints; 10/10 structures; 10 total kpoints | 0.00104813 | 0.00119327 | 10 kpoints; 10/10 structures; 10 total kpoints |
| sio2 | H | finite_temperature_density_relative_frobenius | 0.0637769 | 0.0745331 | 10 structures; 10/10 structures; 10 total kpoints | 0.208385 | 0.289238 | 10 structures; 10/10 structures; 10 total kpoints |
| zncusnses | H | projected common-support MAE (meV) | 0 | 0 | 20 structures; 20/20 structures; 0 total kpoints | 3.01214 | 3.2498 | 20 structures; 20/20 structures; 0 total kpoints |

The silicon reference-support projected overlap is positive definite at only 17.27% of 8192 sampled k points; support restriction plus hermitization therefore does not establish a valid learned metric. No silicon structure has an available joint Fermi–Dirac residual. For SiO2, the mean Fermi–Dirac residual is 0.208385 versus 0.0637769 for reference H/S/D, with reference S/D retained in the prediction control. These finite-mesh consistency tests do not certify SCF convergence or Brillouin-zone-wide validity; no result after overlap or occupation repair is present. Nonzero zero-temperature idempotency residuals alone do not demonstrate a violation at 300 K.
