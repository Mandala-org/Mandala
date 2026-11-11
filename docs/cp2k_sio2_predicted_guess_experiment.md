# SiO2 paper predictions as native CP2K density guesses

Registered 11 November 2026 as `sio2-predicted-cp2k`. The four independent
scientific arms cross two 13-AO-per-atom bases (standard DZVP-MOLOPT-SR-GTH
and the previously frozen 24-primitive OpenMX-fitted Gaussians) with two
maps (cross-overlap projection and coefficient/angular permutation).

The fixed paper SiO2 checkpoint is Hamiltonian-only. Reuse its strict-load
historical inference export for the 150-atom `13_67164000` structure, verify
the checkpoint and operator hashes and split identity, and solve the
generalized source H/S problem using reference OpenMX overlap. Fill its
lowest 400 orbitals twice, yielding a common source density for all arms.
Check reproduction of the previous frozen model-H density to 1e-7 per entry.
This is not a density-head evaluation, new training, or O(3) certification of
the historical parity-breaking FiLM model.

For target overlap S and target/source cross overlap B, project with
T = S_retained^-1 B (threshold 1e-5) and P_target = T P_source T^T.
The coefficient map instead uses the supplied supervisor BasisLayout
permutation U and P_target = U P_source U^T. Both receive only the same
scalar correction 800/Tr(P_target S); no target occupied-space projector,
occupation rounding, or PSD repair is applied. MOLOPT coefficient transfer
is explicitly exploratory: radial slots are identified by ordinal position
inside each angular channel although those radial functions were not fitted
to the OpenMX PAOs. Equal multiplicity permits this mapping but does not
make it a metric-preserving change of basis.

Both maps are written with our native real-Gamma `.kp` density serializer,
not the supervisor's LS_SCF serializer or natural-orbital WFN route.
Target atomic restart metadata and whole-atom image masks are reused;
template density values enter only the exact roundtrip control. Before
releasing full calculations, the preparation allocation executes the five
focused native-density tests and one exact-path CP2K import step for each
of the four arms. Failure leaves explicit readiness guards unsatisfied.

Physics and SCF settings are unchanged from the ground-truth controls:
PBE/GTH-q4 Si/q6 O, Gamma, cutoff600/relative80 Ry, integral thresholds1e-12,
EPS_SCF1e-7, EPS_EIGVAL1e-5, Broyden0.15/history16, standard diagonalization,
integer filling and at most200 steps. Compare against the already converged
native-Gamma atomic controls (MOLOPT87 steps; fitted172 steps). Require
successful RESTART import, 800 electrons, convergence and endpoint energy
agreement within1e-5Ha. No optimal-settings or total-speedup claim is made.

All allocations use cpu-genoa,32 CPUs in one rank, OMP32/BLAS1 and explicit
affinity checks. MOLOPT gets96GiB/2.5h and fitted192GiB/36.5h; the prior
atomic runs took approximately17min and13h respectively. Common preparation
gets192GiB/4h and gates the four otherwise parallel scientific jobs.

| Purpose | Job |
| --- | ---: |
| Common provenance, mapping and import preflight | 840830 |
| MOLOPT / cross-overlap projection | 840831 |
| MOLOPT / exploratory coefficient transfer | 840832 |
| Fitted / cross-overlap projection | 840833 |
| Fitted / coefficient transfer | 840834 |

Frozen source, spec, launchers and outputs:
`/bigdata/casus/wdm/BB/mandala/cp2k_initial_guess/sio2_predicted_20261111_v1`.
Small local records are in this worktree's
`researchops/runs/sio2_predicted_cp2k_20261111`; large predictions and restarts
remain on ROSI. The user's primary checkout is not edited.
