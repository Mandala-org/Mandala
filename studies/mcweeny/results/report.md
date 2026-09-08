# McWeeny density purification: perturbed Silicon

Result: 0 of 18 nonzero-step variants improve density MAE over their respective unpurified baseline. See the reference-density control alongside every comparison.

Saved trained-model predictions, without re-inference or retraining. All 0/1/2/3 steps were specified before evaluation. SiOx excluded at the user's request.

Each multiplication is truncated to the original density edges, including periodic shifts. This is an approximation to the untruncated McWeeny polynomial. Raw iterates feed the next step; global reverse-pair Hermitian projection is applied only for the reported physical metrics. Legacy saved density predictions are explicitly multiplied by the manifest's saved_density_scale (0.5 for these old doubled-target checkpoints) to match the corrected OpenMX loader, 0.5(D + Dᵀ). No per-iteration normalization or electron-count correction is applied.

Density errors are element-weighted over the union of reference/prediction periodic edges, treating missing blocks as zero. Band energy is g Tr(D H), with explicit spin degeneracy g=2 for these nonmagnetic Silicon cases; reference H isolates the density contribution, while predicted H measures the joint H/D prediction. Energies in the tables are absolute errors in eV per cell, not band eigenvalue errors.

## HSD fine-tune / scale1 snapshot 078 — reference overlap

8 atoms, 696 directed edges; reference Tr(DS) = 16 (g=2). Topology setup 0.045 s, paths SD/DD 28472/28472.

| Steps | Density MAE | Density RMSE | MAE improvement | ΔE (reference H), eV | ΔE (predicted H), eV | Tr(DS) | Reference-control MAE |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 2.588636e-05 | 6.19793e-05 | 0.000% | 0.2298576 | 0.02421525 | 15.99134 | 0 |
| 1 | 0.000197588 | 0.00069837 | -663.290% | 5.340316 | 5.088388 | 15.70729 | 0.0001947809 |
| 2 | 0.0002921255 | 0.00100273 | -1028.492% | 2.461753 | 2.205507 | 15.90046 | 0.0002897115 |
| 3 | 0.0003430974 | 0.001178654 | -1225.398% | 2.267721 | 2.01071 | 15.92867 | 0.0003408269 |

Source: `model_reports/Si_perturbed/hdo_finetune/run_assets/hdo_finetune`; reference `data/small/Si_perturbed_scale1_078/HS.out`.

## HSD fine-tune / scale1 snapshot 078 — predicted overlap

8 atoms, 696 directed edges; reference Tr(DS) = 16 (g=2). Topology setup 0.049 s, paths SD/DD 28472/28472.

| Steps | Density MAE | Density RMSE | MAE improvement | ΔE (reference H), eV | ΔE (predicted H), eV | Tr(DS) | Reference-control MAE |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 2.588636e-05 | 6.19793e-05 | 0.000% | 0.2298576 | 0.02421525 | 15.99134 | 0 |
| 1 | 0.0001978354 | 0.0007000976 | -664.246% | 5.426427 | 5.174574 | 15.69819 | 0.0001947809 |
| 2 | 0.0002918036 | 0.001002832 | -1027.249% | 2.538946 | 2.282777 | 15.89179 | 0.0002897115 |
| 3 | 0.0003425922 | 0.00117801 | -1223.447% | 2.345367 | 2.088437 | 15.91952 | 0.0003408269 |

Source: `model_reports/Si_perturbed/hdo_finetune/run_assets/hdo_finetune`; reference `data/small/Si_perturbed_scale1_078/HS.out`.

## HSD fine-tune / validation snapshot 090 — reference overlap

8 atoms, 696 directed edges; reference Tr(DS) = 16 (g=2). Topology setup 0.047 s, paths SD/DD 28472/28472.

| Steps | Density MAE | Density RMSE | MAE improvement | ΔE (reference H), eV | ΔE (predicted H), eV | Tr(DS) | Reference-control MAE |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 2.636788e-05 | 5.697918e-05 | 0.000% | 0.2312563 | 0.01956261 | 15.99205 | 0 |
| 1 | 0.0001971168 | 0.0006969122 | -647.564% | 5.333843 | 5.084942 | 15.70791 | 0.0001947587 |
| 2 | 0.0002918155 | 0.001001364 | -1006.708% | 2.460754 | 2.207701 | 15.90047 | 0.0002897132 |
| 3 | 0.0003428387 | 0.001177344 | -1200.213% | 2.266225 | 2.012425 | 15.92864 | 0.0003408511 |

Source: `model_reports/paper_Si_perturbed_best_val090/run_assets/paper_Si_perturbed_best_val090`; reference `data/paper/paper_Si_perturbed_val_090/HS.out`.

## HSD fine-tune / validation snapshot 090 — predicted overlap

8 atoms, 696 directed edges; reference Tr(DS) = 16 (g=2). Topology setup 0.052 s, paths SD/DD 28472/28472.

| Steps | Density MAE | Density RMSE | MAE improvement | ΔE (reference H), eV | ΔE (predicted H), eV | Tr(DS) | Reference-control MAE |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 2.636788e-05 | 5.697918e-05 | 0.000% | 0.2312563 | 0.01956261 | 15.99205 | 0 |
| 1 | 0.000197354 | 0.0006985746 | -648.463% | 5.416552 | 5.167742 | 15.69906 | 0.0001947587 |
| 2 | 0.0002914848 | 0.001001445 | -1005.454% | 2.535002 | 2.282042 | 15.89203 | 0.0002897132 |
| 3 | 0.0003423299 | 0.001176696 | -1198.283% | 2.340887 | 2.087185 | 15.91972 | 0.0003408511 |

Source: `model_reports/paper_Si_perturbed_best_val090/run_assets/paper_Si_perturbed_best_val090`; reference `data/paper/paper_Si_perturbed_val_090/HS.out`.

## sleek-sweep-8 HSD / scale1 snapshot 078 — reference overlap

8 atoms, 696 directed edges; reference Tr(DS) = 16 (g=2). Topology setup 0.054 s, paths SD/DD 28472/28472.

| Steps | Density MAE | Density RMSE | MAE improvement | ΔE (reference H), eV | ΔE (predicted H), eV | Tr(DS) | Reference-control MAE |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 5.649261e-05 | 0.0001087639 | 0.000% | 0.1020215 | 0.01552155 | 16.00481 | 0 |
| 1 | 0.0002029468 | 0.0006998192 | -259.245% | 5.334973 | 5.421914 | 15.70636 | 0.0001947809 |
| 2 | 0.0002953876 | 0.001003026 | -422.878% | 2.446121 | 2.534685 | 15.9012 | 0.0002897115 |
| 3 | 0.0003456766 | 0.001178498 | -511.897% | 2.254379 | 2.343366 | 15.92934 | 0.0003408269 |

Source: `model_reports/silicon_perturbed_hamiltonian_sleek-sweep-8-restart-lr2em3_html_pred_ovl_078/run_assets/sleek-sweep-8-restart-lr2em3`; reference `data/small/Si_perturbed_scale1_078/HS.out`.

## sleek-sweep-8 HSD / scale1 snapshot 078 — predicted overlap

8 atoms, 696 directed edges; reference Tr(DS) = 16 (g=2). Topology setup 0.051 s, paths SD/DD 28472/28472.

| Steps | Density MAE | Density RMSE | MAE improvement | ΔE (reference H), eV | ΔE (predicted H), eV | Tr(DS) | Reference-control MAE |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 5.649261e-05 | 0.0001087639 | 0.000% | 0.1020215 | 0.01552155 | 16.00481 | 0 |
| 1 | 0.0002030273 | 0.000700637 | -259.387% | 5.393633 | 5.480548 | 15.70213 | 0.0001947809 |
| 2 | 0.0002951792 | 0.001003031 | -422.509% | 2.498773 | 2.58731 | 15.89722 | 0.0002897115 |
| 3 | 0.0003453778 | 0.001178181 | -511.368% | 2.30849 | 2.397447 | 15.9251 | 0.0003408269 |

Source: `model_reports/silicon_perturbed_hamiltonian_sleek-sweep-8-restart-lr2em3_html_pred_ovl_078/run_assets/sleek-sweep-8-restart-lr2em3`; reference `data/small/Si_perturbed_scale1_078/HS.out`.

## Interpretation and scope

The polynomial f(x)=3x²−2x³ targets occupations 0 and 1. It does not conserve Tr(DS); f(x)<0 for x>1.5. The corrected loader retains the native spin=0 density normalization, independently of the spin factor used for observable reporting. The reference-density control measures how the exact same truncated operation changes ground truth itself.

These are available local snapshot artifacts, not a complete validation/test-set benchmark. Established H/S/D checkpoints were included without selecting models or iteration counts using these results. The supplied manifest and artifact hashes identify every input. Per-iterate raw errors, Hermiticity residuals, timings, and electron-count errors are retained in results.json.
