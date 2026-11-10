Historical July inference only. Selected scenarios: silicon: `joint_predicted_reference_support`; sio2: `predicted_H_reference_SD_reference_support`.

Reference-support control and Hermitian projection only; no overlap repair, fixed-charge occupation projection, or post-cleanup aggregates exist.

These spectral values use Hermitian-projected matrices on reference support, a control requiring reference sparsity, not independent deployment. SiO2 H uses reference S/D; silicon uses predicted H/S/D. Zn reports H elements only. Missing post-cleanup results are unavailable, not replaced with raw predictions.

Means are over all contributing k points (not means of structure means); electron-count absolute errors use exact Hermitian real-space traces per structure; Fermi–Dirac residuals use one fit per structure. Worst is min for lower bounds and positive-definite success, max otherwise. Counts exclude invalid metrics; total denominators retain them. Contributing structure counts are recovered from historical per-structure summaries. Element MAEs are equally weighted structure means, with worst structure MAE; k points do not apply (0). Reference element error is zero by definition, not a physical residual.

Each compact cell pairs reference / prediction (R / P). Mean and worst are separate columns; exact valid counts and total denominators follow below.

| Material | Operator | Element MAE (meV) mean (R / P) | Element MAE (meV) worst (R / P) |
| --- | --- | --- | --- |
| silicon | H | 0 / 0.435164 | 0 / 0.610003 |
| sio2 | H | 0 / 1.99588 | 0 / 2.53462 |
| zncusnses | H | 0 / 3.01214 | 0 / 3.2498 |

| Material | Operator | Stationarity residual mean (R / P) | Stationarity residual worst (R / P) | Fermi-Dirac residual mean (R / P) | Fermi-Dirac residual worst (R / P) |
| --- | --- | --- | --- | --- | --- |
| silicon | H | 0.00235366 / 0.000969553 | 0.00352099 / 0.0019418 | 0.121229 / — | 0.121477 / — |
| sio2 | H | 0.000791348 / 0.00104813 | 0.000902654 / 0.00119327 | 0.0637769 / 0.208385 | 0.0745331 / 0.289238 |

| Material | Operator | Element MAE (AO units) mean (R / P) | Element MAE (AO units) worst (R / P) |
| --- | --- | --- | --- |
| silicon | S | 0 / 1.38978e-05 | 0 / 1.67892e-05 |

| Material | Operator | S positive definite mean (R / P) | S positive definite worst (R / P) | S usable metric mean (R / P) | S usable metric worst (R / P) |
| --- | --- | --- | --- | --- | --- |
| silicon | S | 1 / 0.172729 | 1 / 0 | 1 / 0.172485 | 1 / 0 |
| sio2 | S (reference) | 1 / 1 | 1 / 1 | 1 / 1 | 1 / 1 |

| Material | Operator | Cholesky failure mean (R / P) | Cholesky failure worst (R / P) | S min eigenvalue mean (R / P) | S min eigenvalue worst (R / P) |
| --- | --- | --- | --- | --- | --- |
| silicon | S | 0 / 0.827271 | 0 / 1 | 0.00118844 / -0.000655239 | 0.000872547 / -0.00624094 |
| sio2 | S (reference) | 0 / 0 | 0 / 0 | 0.000250166 / 0.000250166 | 0.000215888 / 0.000215888 |

| Material | Operator | S max eigenvalue mean (R / P) | S max eigenvalue worst (R / P) | S condition number mean (R / P) | S condition number worst (R / P) |
| --- | --- | --- | --- | --- | --- |
| silicon | S | 6.34486 / 6.34674 | 6.74327 / 6.74962 | 5388.83 / 295705 | 7156.37 / 1.02091e+08 |
| sio2 | S (reference) | 7.38574 / 7.38574 | 7.61309 / 7.61309 | 29830.9 / 29830.9 | 34732.7 / 34732.7 |

| Material | Operator | S nonpositive eigenvalue count mean (R / P) | S nonpositive eigenvalue count worst (R / P) |
| --- | --- | --- | --- |
| silicon | S | 0 / 1.12427 | 0 / 4 |
| sio2 | S (reference) | 0 / 0 | 0 / 0 |

| Material | Operator | Element MAE (AO units) mean (R / P) | Element MAE (AO units) worst (R / P) |
| --- | --- | --- | --- |
| silicon | D | 0 / 5.75772e-05 | 0 / 8.24122e-05 |

| Material | Operator | Min occupation mean (R / P) | Min occupation worst (R / P) | Max occupation mean (R / P) | Max occupation worst (R / P) |
| --- | --- | --- | --- | --- | --- |
| silicon | D | -0.0334263 / -0.0325261 | -0.0418119 / -0.0393333 | 2.3045 / 2.28845 | 2.49221 / 2.41358 |
| sio2 | D (reference) | -0.172922 / -0.172922 | -0.344175 / -0.344175 | 2.37459 / 2.37459 | 2.56227 / 2.56227 |

| Material | Operator | Negative occupation count mean (R / P) | Negative occupation count worst (R / P) | Excess occupation count mean (R / P) | Excess occupation count worst (R / P) |
| --- | --- | --- | --- | --- | --- |
| silicon | D | 37.8605 / 38.54 | 42 / 44 | 7.01294 / 7.19887 | 12 / 10 |
| sio2 | D (reference) | 663.7 / 663.7 | 682 / 682 | 200.9 / 200.9 | 203 / 203 |

| Material | Operator | Electron count absolute error mean (R / P) | Electron count absolute error worst (R / P) | Zero-T idempotency residual mean (R / P) | Zero-T idempotency residual worst (R / P) |
| --- | --- | --- | --- | --- | --- |
| silicon | D | 9.99336e-08 / 0.0075323 | 1.952e-07 / 0.0114623 | 0.102435 / 0.101953 | 0.160485 / 0.155847 |
| sio2 | D (reference) | 2.46731e-06 / 2.46731e-06 | 3.1647e-06 / 3.1647e-06 | 0.0520496 / 0.0520496 | 0.0604673 / 0.0604673 |

The silicon reference-support projected overlap is positive definite at only 17.27% of 8192 sampled k points; support restriction plus hermitization therefore does not establish a valid learned metric. No silicon structure has an available joint Fermi–Dirac residual. Silicon density and stationarity means cover only 1413 usable k points, whereas reference means cover all 8192; this is not a paired whole-mesh comparison. For SiO2, the mean Fermi–Dirac residual is 0.208385 versus 0.0637769 for reference H/S/D, with reference S/D retained in the prediction control. These finite-mesh consistency tests do not certify SCF convergence or Brillouin-zone-wide validity; no result after overlap or occupation repair is present. Nonzero zero-temperature idempotency residuals alone do not demonstrate a violation at 300 K.

Detailed values and denominators:

| Material | Operator | Diagnostic | Ref mean | Ref worst | Ref n / totals | Pred mean | Pred worst | Pred n / totals |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| silicon | H | Element MAE (meV) | 0 | 0 | 16 structures; 16/16 structures; 0 total kpoints | 0.435164 | 0.610003 | 16 structures; 16/16 structures; 0 total kpoints |
| silicon | S | Element MAE (AO units) | 0 | 0 | 16 structures; 16/16 structures; 0 total kpoints | 1.38978e-05 | 1.67892e-05 | 16 structures; 16/16 structures; 0 total kpoints |
| silicon | D | Element MAE (AO units) | 0 | 0 | 16 structures; 16/16 structures; 0 total kpoints | 5.75772e-05 | 8.24122e-05 | 16 structures; 16/16 structures; 0 total kpoints |
| silicon | S | S positive definite | 1 | 1 | 8192 kpoints; 16/16 structures; 8192 total kpoints | 0.172729 | 0 | 8192 kpoints; 16/16 structures; 8192 total kpoints |
| silicon | S | S usable metric | 1 | 1 | 8192 kpoints; 16/16 structures; 8192 total kpoints | 0.172485 | 0 | 8192 kpoints; 16/16 structures; 8192 total kpoints |
| silicon | S | Cholesky failure | 0 | 0 | 8192 kpoints; 16/16 structures; 8192 total kpoints | 0.827271 | 1 | 8192 kpoints; 16/16 structures; 8192 total kpoints |
| silicon | S | S min eigenvalue | 0.00118844 | 0.000872547 | 8192 kpoints; 16/16 structures; 8192 total kpoints | -0.000655239 | -0.00624094 | 8192 kpoints; 16/16 structures; 8192 total kpoints |
| silicon | S | S max eigenvalue | 6.34486 | 6.74327 | 8192 kpoints; 16/16 structures; 8192 total kpoints | 6.34674 | 6.74962 | 8192 kpoints; 16/16 structures; 8192 total kpoints |
| silicon | S | S condition number | 5388.83 | 7156.37 | 8192 kpoints; 16/16 structures; 8192 total kpoints | 295705 | 1.02091e+08 | 1415 kpoints; 15/16 structures; 8192 total kpoints |
| silicon | S | S nonpositive eigenvalue count | 0 | 0 | 8192 kpoints; 16/16 structures; 8192 total kpoints | 1.12427 | 4 | 8192 kpoints; 16/16 structures; 8192 total kpoints |
| silicon | D | Min occupation | -0.0334263 | -0.0418119 | 8192 kpoints; 16/16 structures; 8192 total kpoints | -0.0325261 | -0.0393333 | 1413 kpoints; 15/16 structures; 8192 total kpoints |
| silicon | D | Max occupation | 2.3045 | 2.49221 | 8192 kpoints; 16/16 structures; 8192 total kpoints | 2.28845 | 2.41358 | 1413 kpoints; 15/16 structures; 8192 total kpoints |
| silicon | D | Negative occupation count | 37.8605 | 42 | 8192 kpoints; 16/16 structures; 8192 total kpoints | 38.54 | 44 | 1413 kpoints; 15/16 structures; 8192 total kpoints |
| silicon | D | Excess occupation count | 7.01294 | 12 | 8192 kpoints; 16/16 structures; 8192 total kpoints | 7.19887 | 10 | 1413 kpoints; 15/16 structures; 8192 total kpoints |
| silicon | D | Electron count absolute error | 9.99336e-08 | 1.952e-07 | 16 structures; 16/16 structures; 8192 total kpoints | 0.0075323 | 0.0114623 | 16 structures; 16/16 structures; 8192 total kpoints |
| silicon | D | Zero-T idempotency residual | 0.102435 | 0.160485 | 8192 kpoints; 16/16 structures; 8192 total kpoints | 0.101953 | 0.155847 | 1413 kpoints; 15/16 structures; 8192 total kpoints |
| silicon | H | Stationarity residual | 0.00235366 | 0.00352099 | 8192 kpoints; 16/16 structures; 8192 total kpoints | 0.000969553 | 0.0019418 | 1413 kpoints; 15/16 structures; 8192 total kpoints |
| silicon | H | Fermi-Dirac residual | 0.121229 | 0.121477 | 16 structures; 16/16 structures; 8192 total kpoints | — | — | 0 structures; 0/16 structures; 8192 total kpoints |
| sio2 | H | Element MAE (meV) | 0 | 0 | 10 structures; 10/10 structures; 0 total kpoints | 1.99588 | 2.53462 | 10 structures; 10/10 structures; 0 total kpoints |
| sio2 | S (reference) | S positive definite | 1 | 1 | 10 kpoints; 10/10 structures; 10 total kpoints | 1 | 1 | 10 kpoints; 10/10 structures; 10 total kpoints |
| sio2 | S (reference) | S usable metric | 1 | 1 | 10 kpoints; 10/10 structures; 10 total kpoints | 1 | 1 | 10 kpoints; 10/10 structures; 10 total kpoints |
| sio2 | S (reference) | Cholesky failure | 0 | 0 | 10 kpoints; 10/10 structures; 10 total kpoints | 0 | 0 | 10 kpoints; 10/10 structures; 10 total kpoints |
| sio2 | S (reference) | S min eigenvalue | 0.000250166 | 0.000215888 | 10 kpoints; 10/10 structures; 10 total kpoints | 0.000250166 | 0.000215888 | 10 kpoints; 10/10 structures; 10 total kpoints |
| sio2 | S (reference) | S max eigenvalue | 7.38574 | 7.61309 | 10 kpoints; 10/10 structures; 10 total kpoints | 7.38574 | 7.61309 | 10 kpoints; 10/10 structures; 10 total kpoints |
| sio2 | S (reference) | S condition number | 29830.9 | 34732.7 | 10 kpoints; 10/10 structures; 10 total kpoints | 29830.9 | 34732.7 | 10 kpoints; 10/10 structures; 10 total kpoints |
| sio2 | S (reference) | S nonpositive eigenvalue count | 0 | 0 | 10 kpoints; 10/10 structures; 10 total kpoints | 0 | 0 | 10 kpoints; 10/10 structures; 10 total kpoints |
| sio2 | D (reference) | Min occupation | -0.172922 | -0.344175 | 10 kpoints; 10/10 structures; 10 total kpoints | -0.172922 | -0.344175 | 10 kpoints; 10/10 structures; 10 total kpoints |
| sio2 | D (reference) | Max occupation | 2.37459 | 2.56227 | 10 kpoints; 10/10 structures; 10 total kpoints | 2.37459 | 2.56227 | 10 kpoints; 10/10 structures; 10 total kpoints |
| sio2 | D (reference) | Negative occupation count | 663.7 | 682 | 10 kpoints; 10/10 structures; 10 total kpoints | 663.7 | 682 | 10 kpoints; 10/10 structures; 10 total kpoints |
| sio2 | D (reference) | Excess occupation count | 200.9 | 203 | 10 kpoints; 10/10 structures; 10 total kpoints | 200.9 | 203 | 10 kpoints; 10/10 structures; 10 total kpoints |
| sio2 | D (reference) | Electron count absolute error | 2.46731e-06 | 3.1647e-06 | 10 structures; 10/10 structures; 10 total kpoints | 2.46731e-06 | 3.1647e-06 | 10 structures; 10/10 structures; 10 total kpoints |
| sio2 | D (reference) | Zero-T idempotency residual | 0.0520496 | 0.0604673 | 10 kpoints; 10/10 structures; 10 total kpoints | 0.0520496 | 0.0604673 | 10 kpoints; 10/10 structures; 10 total kpoints |
| sio2 | H | Stationarity residual | 0.000791348 | 0.000902654 | 10 kpoints; 10/10 structures; 10 total kpoints | 0.00104813 | 0.00119327 | 10 kpoints; 10/10 structures; 10 total kpoints |
| sio2 | H | Fermi-Dirac residual | 0.0637769 | 0.0745331 | 10 structures; 10/10 structures; 10 total kpoints | 0.208385 | 0.289238 | 10 structures; 10/10 structures; 10 total kpoints |
| zncusnses | H | Element MAE (meV) | 0 | 0 | 20 structures; 20/20 structures; 0 total kpoints | 3.01214 | 3.2498 | 20 structures; 20/20 structures; 0 total kpoints |
