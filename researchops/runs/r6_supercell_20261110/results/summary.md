| Structure | Atoms | Repetition | σ Å | H meV | H excl. same-parent image meV | Routed diagnostic H meV | S MAE | D MAE | Primary metric validity | Routed metric validity | Paper H meV |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | ---: |
| si-2x2x2-1 | 64 | 2×2×2 | 0.02 | 8.650596 | 1.064335 | 1.1248 | 0.0007194514 | 0.0005504245 | PD 0/64; min -1.015124; chol 0/64 | PD 0/64; min -0.4877838; chol 0/64 | 0.435164 |
| si-2x2x2-2 | 64 | 2×2×2 | 0.02 | 8.690596 | 1.085826 | 1.133893 | 0.0007212527 | 0.0005509217 | PD 0/64; min -1.003353; chol 0/64 | PD 0/64; min -0.4703414; chol 0/64 | 0.435164 |
| si-2x2x2-3 | 64 | 2×2×2 | 0.02 | 8.654309 | 1.060497 | 1.110131 | 0.000720314 | 0.0005444802 | PD 0/64; min -0.9601095; chol 0/64 | PD 0/64; min -0.3902055; chol 0/64 | 0.435164 |
| si-2x2x2-tiled | 64 | 2×2×2 | 0 | 8.136948 | 0.3947345 | 0.3836588 | 0.0007267947 | 0.0004773784 | PD 0/64; min -0.9903229; chol 0/64 | PD 0/64; min -0.3739824; chol 0/64 | 0.435164 |
| si-3x3x3-1 | 216 | 3×3×3 | 0.02 | 8.686619 | 1.113734 | 1.170933 | 0.0007189151 | 0.0005523925 | PD 0/27; min -1.004345; chol 0/27 | PD 0/27; min -0.4994761; chol 0/27 | 0.435164 |
| si-3x3x3-2 | 216 | 3×3×3 | 0.02 | 8.768735 | 1.210731 | 1.27325 | 0.000718816 | 0.0005582109 | PD 0/27; min -1.031094; chol 0/27 | PD 0/27; min -0.5506935; chol 0/27 | 0.435164 |
| si-3x3x3-3 | 216 | 3×3×3 | 0.02 | 8.717221 | 1.139948 | 1.197851 | 0.0007192382 | 0.0005506098 | PD 0/27; min -0.9890429; chol 0/27 | PD 0/27; min -0.4656319; chol 0/27 | 0.435164 |
| si-3x3x3-tiled | 216 | 3×3×3 | 0 | 8.136951 | 0.3947377 | 0.3836614 | 0.0007267949 | 0.0004773782 | PD 0/27; min -0.990323; chol 0/27 | PD 0/27; min -0.358895; chol 0/27 | 0.435164 |
| si-5x5x5-1 | 1000 | 5×5×5 | 0.02 | — | — | — | — | — | OpenMX_pending (job 837489); PD 0/8; min -0.9962599; chol 0/8 | PD 0/8; min -0.5723461; chol 0/8 | 0.435164 |
| si-5x5x5-tiled | 1000 | 5×5×5 | 0 | 8.136951 | 0.3947371 | 0.3836612 | 0.0007267949 | 0.0004773784 | PD 0/8; min -0.9903229; chol 0/8 | PD 0/8; min -0.3739824; chol 0/8 | 0.435164 |
| sio2-2x1x1-1 | 300 | 2×1×1 | 0.02 | 2.855742 | 2.855742 | — | — | — | H only; S not predicted | — | 1.99588 |
| sio2-2x1x1-tiled | 300 | 2×1×1 | 0 | 2.534627 | 2.534627 | — | — | — | H only; S not predicted | — | 1.99588 |
| sio2-2x1x1-2 | 300 | 2×1×1 | 0.02 | 1.729488 | 1.729488 | — | — | — | H only; S not predicted | — | 1.99588 |
| sio2-2x2x1-1 | 600 | 2×2×1 | 0.02 | — | — | — | — | — | OpenMX_pending (job 837830) | — | 1.99588 |
| sio2-2x2x1-tiled | 600 | 2×2×1 | 0 | 2.534622 | 2.534622 | — | — | — | H only; S not predicted | — | 1.99588 |

Silicon perturbed-cell H MAE is 8.6506–8.76874 meV versus 0.435164 meV on paper validation, with similar errors at 64 and 216 atoms. Excluding same-parent-image blocks gives 1.0605–1.21073 meV. Routing with known parent labels gives 1.11013–1.27325 meV; this diagnostic does not establish a deployable correction. SiO2 perturbed-cell H MAE is 2.85574, 1.72949 meV versus 1.99588 meV on validation. Primary overlap is positive definite at 0/273 sampled perturbed-cell k-points; its lowest eigenvalue across those cells is -1.03109. Parent-routed diagnostic overlap is positive definite at 0/273 sampled perturbed-cell k-points; its lowest eigenvalue across those cells is -0.550694. The metric uses full geometry support without repair, whereas element errors use common reference support.

Each row is one structure; repetition expands the source cell, and σ is the independent Gaussian displacement standard deviation (zero for tiled controls). H, S and D are scalar-entry-weighted MAEs on matching directed periodic support after Hermitian projection of the prediction, against OpenMX (D multiplied by two); H is in meV, S dimensionless, D in AO-kernel units. H excl. removes every non-onsite pair sharing a source-atom identity, including nonzero lattice shifts; onsite plus different-parent blocks remain. Routed columns use unchanged silicon heads with construction parent labels and never replace primary predictions. Metric columns give positive-definite k-point counts, global minimum overlap eigenvalue, and Cholesky successes without jitter on full geometry support, using 4×4×4 (2× Si) 3×3×3 (3× Si), or 2×2×2 (5× Si) Gamma-centered meshes. SiO2 predicts H only; — means unavailable or not predicted. Paper H is the material validation mean, not a supercell target. Category fractions below use common support, separately counting directed blocks and scalar entries; JSON also retains prediction/reference full-support fractions and raw errors. Tiled references fold source operators; the SiO2 control requires converged source Gamma sampling and the 3× and 5× Si controls fold the original 8-mesh calculation.

| Structure | Operator | Category | Blocks fraction | Scalar fraction | MAE (H meV) |
| --- | --- | --- | ---: | ---: | ---: |
| si-2x2x2-1 | density | onsite | 0.01149425 | 0.01149425 | 0.0004744509 |
| si-2x2x2-1 | density | different_parent | 0.9195402 | 0.9195402 | 0.00014216 |
| si-2x2x2-1 | density | same_parent_image | 0.06896552 | 0.06896552 | 0.006006615 |
| si-2x2x2-1 | hamiltonian | onsite | 0.01149425 | 0.01149425 | 12.81284 |
| si-2x2x2-1 | hamiltonian | different_parent | 0.9195402 | 0.9195402 | 0.917479 |
| si-2x2x2-1 | hamiltonian | same_parent_image | 0.06896552 | 0.06896552 | 111.0651 |
| si-2x2x2-1 | overlap | onsite | 0.01149425 | 0.01149425 | 3.283509e-05 |
| si-2x2x2-1 | overlap | different_parent | 0.9195402 | 0.9195402 | 1.451667e-05 |
| si-2x2x2-1 | overlap | same_parent_image | 0.06896552 | 0.06896552 | 0.01023302 |
| si-2x2x2-2 | density | onsite | 0.01149425 | 0.01149425 | 0.0004546187 |
| si-2x2x2-2 | density | different_parent | 0.9195402 | 0.9195402 | 0.0001409125 |
| si-2x2x2-2 | density | same_parent_image | 0.06896552 | 0.06896552 | 0.006033762 |
| si-2x2x2-2 | hamiltonian | onsite | 0.01149425 | 0.01149425 | 12.89343 |
| si-2x2x2-2 | hamiltonian | different_parent | 0.9195402 | 0.9195402 | 0.9382309 |
| si-2x2x2-2 | hamiltonian | same_parent_image | 0.06896552 | 0.06896552 | 111.355 |
| si-2x2x2-2 | overlap | onsite | 0.01149425 | 0.01149425 | 3.283428e-05 |
| si-2x2x2-2 | overlap | different_parent | 0.9195402 | 0.9195402 | 1.436907e-05 |
| si-2x2x2-2 | overlap | same_parent_image | 0.06896552 | 0.06896552 | 0.0102611 |
| si-2x2x2-3 | density | onsite | 0.01149425 | 0.01149425 | 0.0004218587 |
| si-2x2x2-3 | density | different_parent | 0.9195402 | 0.9195402 | 0.0001329279 |
| si-2x2x2-3 | density | same_parent_image | 0.06896552 | 0.06896552 | 0.006052281 |
| si-2x2x2-3 | hamiltonian | onsite | 0.01149425 | 0.01149425 | 12.1681 |
| si-2x2x2-3 | hamiltonian | different_parent | 0.9195402 | 0.9195402 | 0.9216524 |
| si-2x2x2-3 | hamiltonian | same_parent_image | 0.06896552 | 0.06896552 | 111.1708 |
| si-2x2x2-3 | overlap | onsite | 0.01149425 | 0.01149425 | 3.283444e-05 |
| si-2x2x2-3 | overlap | different_parent | 0.9195402 | 0.9195402 | 1.42428e-05 |
| si-2x2x2-3 | overlap | same_parent_image | 0.06896552 | 0.06896552 | 0.01024918 |
| si-2x2x2-tiled | density | onsite | 0.01149425 | 0.01149425 | 0.0002981839 |
| si-2x2x2-tiled | density | different_parent | 0.9195402 | 0.9195402 | 4.677707e-05 |
| si-2x2x2-tiled | density | same_parent_image | 0.06896552 | 0.06896552 | 0.006248595 |
| si-2x2x2-tiled | hamiltonian | onsite | 0.01149425 | 0.01149425 | 6.639364 |
| si-2x2x2-tiled | hamiltonian | different_parent | 0.9195402 | 0.9195402 | 0.3166766 |
| si-2x2x2-tiled | hamiltonian | same_parent_image | 0.06896552 | 0.06896552 | 112.6568 |
| si-2x2x2-tiled | overlap | onsite | 0.01149425 | 0.01149425 | 3.283208e-05 |
| si-2x2x2-tiled | overlap | different_parent | 0.9195402 | 0.9195402 | 1.3991e-05 |
| si-2x2x2-tiled | overlap | same_parent_image | 0.06896552 | 0.06896552 | 0.0103465 |
| si-3x3x3-1 | density | onsite | 0.01160043 | 0.01160043 | 0.0004879842 |
| si-3x3x3-1 | density | different_parent | 0.9194415 | 0.9194415 | 0.0001433443 |
| si-3x3x3-1 | density | same_parent_image | 0.06895811 | 0.06895811 | 0.006017204 |
| si-3x3x3-1 | hamiltonian | onsite | 0.01160043 | 0.01160043 | 13.19037 |
| si-3x3x3-1 | hamiltonian | different_parent | 0.9194415 | 0.9194415 | 0.9613651 |
| si-3x3x3-1 | hamiltonian | same_parent_image | 0.06895811 | 0.06895811 | 110.9324 |
| si-3x3x3-1 | overlap | onsite | 0.01160043 | 0.01160043 | 3.283455e-05 |
| si-3x3x3-1 | overlap | different_parent | 0.9194415 | 0.9194415 | 1.462286e-05 |
| si-3x3x3-1 | overlap | same_parent_image | 0.06895811 | 0.06895811 | 0.01022489 |
| si-3x3x3-2 | density | onsite | 0.01160043 | 0.01160043 | 0.0004931575 |
| si-3x3x3-2 | density | different_parent | 0.9194415 | 0.9194415 | 0.0001510672 |
| si-3x3x3-2 | density | same_parent_image | 0.06895811 | 0.06895811 | 0.005997737 |
| si-3x3x3-2 | hamiltonian | onsite | 0.01160043 | 0.01160043 | 14.18049 |
| si-3x3x3-2 | hamiltonian | different_parent | 0.9194415 | 0.9194415 | 1.047094 |
| si-3x3x3-2 | hamiltonian | same_parent_image | 0.06895811 | 0.06895811 | 110.8136 |
| si-3x3x3-2 | overlap | onsite | 0.01160043 | 0.01160043 | 3.283366e-05 |
| si-3x3x3-2 | overlap | different_parent | 0.9194415 | 0.9194415 | 1.50296e-05 |
| si-3x3x3-2 | overlap | same_parent_image | 0.06895811 | 0.06895811 | 0.01021803 |
| si-3x3x3-3 | density | onsite | 0.01149425 | 0.01149425 | 0.0004644922 |
| si-3x3x3-3 | density | different_parent | 0.9195402 | 0.9195402 | 0.0001428042 |
| si-3x3x3-3 | density | same_parent_image | 0.06896552 | 0.06896552 | 0.006002371 |
| si-3x3x3-3 | hamiltonian | onsite | 0.01149425 | 0.01149425 | 13.48387 |
| si-3x3x3-3 | hamiltonian | different_parent | 0.9195402 | 0.9195402 | 0.9856491 |
| si-3x3x3-3 | hamiltonian | same_parent_image | 0.06896552 | 0.06896552 | 111.0104 |
| si-3x3x3-3 | overlap | onsite | 0.01149425 | 0.01149425 | 3.283408e-05 |
| si-3x3x3-3 | overlap | different_parent | 0.9195402 | 0.9195402 | 1.472933e-05 |
| si-3x3x3-3 | overlap | same_parent_image | 0.06896552 | 0.06896552 | 0.01022709 |
| si-3x3x3-tiled | density | onsite | 0.01149425 | 0.01149425 | 0.0002981838 |
| si-3x3x3-tiled | density | different_parent | 0.9195402 | 0.9195402 | 4.677692e-05 |
| si-3x3x3-tiled | density | same_parent_image | 0.06896552 | 0.06896552 | 0.006248595 |
| si-3x3x3-tiled | hamiltonian | onsite | 0.01149425 | 0.01149425 | 6.639309 |
| si-3x3x3-tiled | hamiltonian | different_parent | 0.9195402 | 0.9195402 | 0.3166805 |
| si-3x3x3-tiled | hamiltonian | same_parent_image | 0.06896552 | 0.06896552 | 112.6568 |
| si-3x3x3-tiled | overlap | onsite | 0.01149425 | 0.01149425 | 3.283209e-05 |
| si-3x3x3-tiled | overlap | different_parent | 0.9195402 | 0.9195402 | 1.399125e-05 |
| si-3x3x3-tiled | overlap | same_parent_image | 0.06896552 | 0.06896552 | 0.0103465 |
| si-5x5x5-tiled | density | onsite | 0.01149425 | 0.01149425 | 0.000298185 |
| si-5x5x5-tiled | density | different_parent | 0.9195402 | 0.9195402 | 4.677705e-05 |
| si-5x5x5-tiled | density | same_parent_image | 0.06896552 | 0.06896552 | 0.006248596 |
| si-5x5x5-tiled | hamiltonian | onsite | 0.01149425 | 0.01149425 | 6.639277 |
| si-5x5x5-tiled | hamiltonian | different_parent | 0.9195402 | 0.9195402 | 0.3166803 |
| si-5x5x5-tiled | hamiltonian | same_parent_image | 0.06896552 | 0.06896552 | 112.6568 |
| si-5x5x5-tiled | overlap | onsite | 0.01149425 | 0.01149425 | 3.283127e-05 |
| si-5x5x5-tiled | overlap | different_parent | 0.9195402 | 0.9195402 | 1.399117e-05 |
| si-5x5x5-tiled | overlap | same_parent_image | 0.06896552 | 0.06896552 | 0.0103465 |
| sio2-2x1x1-1 | hamiltonian | onsite | 0.006689262 | 0.006689262 | 34.4194 |
| sio2-2x1x1-1 | hamiltonian | different_parent | 0.9933107 | 0.9933107 | 2.643183 |
| sio2-2x1x1-1 | hamiltonian | same_parent_image | 0 | 0 | — |
| sio2-2x1x1-tiled | hamiltonian | onsite | 0.00669344 | 0.00669344 | 29.23546 |
| sio2-2x1x1-tiled | hamiltonian | different_parent | 0.9933066 | 0.9933066 | 2.354702 |
| sio2-2x1x1-tiled | hamiltonian | same_parent_image | 0 | 0 | — |
| sio2-2x1x1-2 | hamiltonian | onsite | 0.007076473 | 0.007076473 | 13.3402 |
| sio2-2x1x1-2 | hamiltonian | different_parent | 0.9929235 | 0.9929235 | 1.646739 |
| sio2-2x1x1-2 | hamiltonian | same_parent_image | 0 | 0 | — |
| sio2-2x2x1-tiled | hamiltonian | onsite | 0.00669344 | 0.00669344 | 29.2354 |
| sio2-2x2x1-tiled | hamiltonian | different_parent | 0.9933066 | 0.9933066 | 2.354698 |
| sio2-2x2x1-tiled | hamiltonian | same_parent_image | 0 | 0 | — |

Parent-routing diagnostic (requires known parent labels):

| Structure | H meV | S MAE | D MAE | PD fraction | Minimum S eigenvalue | Cholesky successes |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| si-2x2x2-1 | 1.1248 | 2.214668e-05 | 0.0001421522 | 0 | -0.4877838 | 0/64 |
| si-2x2x2-2 | 1.133893 | 2.124676e-05 | 0.0001404747 | 0 | -0.4703414 | 0/64 |
| si-2x2x2-3 | 1.110131 | 2.131743e-05 | 0.0001322225 | 0 | -0.3902055 | 0/64 |
| si-2x2x2-tiled | 0.3836588 | 1.351525e-05 | 4.953477e-05 | 0 | -0.3739824 | 0/64 |
| si-3x3x3-1 | 1.170933 | 2.227854e-05 | 0.0001432348 | 0 | -0.4994761 | 0/27 |
| si-3x3x3-2 | 1.27325 | 2.323706e-05 | 0.0001507699 | 0 | -0.5506935 | 0/27 |
| si-3x3x3-3 | 1.197851 | 2.250987e-05 | 0.0001426043 | 0 | -0.4656319 | 0/27 |
| si-3x3x3-tiled | 0.3836614 | 1.351549e-05 | 4.953464e-05 | 0 | -0.358895 | 0/27 |
| si-5x5x5-1 | — | — | — | 0 | -0.5723461 | 0/8 |
| si-5x5x5-tiled | 0.3836612 | 1.351541e-05 | 4.953476e-05 | 0 | -0.3739824 | 0/8 |

1000-atom follow-up: after OpenMX completes, run:

```bash
ssh rosi4 "cd /data/home2/brzoza73/casus/mandala_worktrees/r6 && sbatch --parsable researchops/runs/r6_supercell_20261110/final_compare_5x.sh"
```

600-atom SiO2 follow-up: after OpenMX completes, run:

```bash
ssh rosi4 "cd /data/home2/brzoza73/casus/mandala_worktrees/r6 && sbatch --parsable researchops/runs/r6_supercell_20261110/final_compare_sio2_4x.sh"
```
