# Paper-model CP2K transfer comparison

Registered and submitted on ROSI on 5 November 2026. This is a new matched
comparison, separate from the completed Si030 benchmark described in
[the feasibility report](cp2k_initial_guess_feasibility.md) and from the
[supervisor converter review](cp2k_supervisor_converter_comparison.md).
Scheduler records and numerical outcomes are stored separately. Submission
does not establish SCF convergence or a speedup.

## Materials and frozen paper predictors

| Material | Paper specimen | Composition | Native PAOs | Paper model |
| --- | --- | --- | --- | --- |
| Si | `paper_Si_perturbed_val_090`, source `Si_scale_1_090` | 8 Si; 104 AOs; 32 valence electrons | Si7.0-s2p2d1 | `92sqde98`, H/D/S and energy-trained checkpoint |
| SiO₂ | `paper_SiO2_val_13_67164000` | 50 Si + 100 O; 1950 AOs; 800 valence electrons | Si9.0-s2p2d1, O7.0-s2p2d1 | `ebg4g3h3`, Hamiltonian checkpoint |

The exact specimens are the paper validation/DOS examples, not a new selection
based on CP2K convergence. Checkpoint SHA256 values are
`e2b2f922b10c092e068a67ef879b615b267ba3fcc41e7431043069691c05ef39`
(Si) and
`18c9376825637077d013c20344ef30b8cbb01466a7005b7f7cd2525cf7961e3f`
(SiO₂). Preparation verifies the checkpoint bytes, publication validation-index
membership, source HS/info/CIF hashes, exact PAO hashes, and raw prediction
hashes against the archived strict historical inference contracts. It reuses
those predictions; it does not retrain or migrate the historical weights.

This experiment uses real Γ-point WFN restarts, the scope implemented by the
supplied converter. SiO₂ was generated at Γ. Si was generated with an 8×8×8
mesh: here its real-space H and S are Fourier-summed at Γ and filled anew.
This is explicitly a different Brillouin-zone sampling, not a reproduction of
the paper's Si sampling or a collapse of a k-integrated density into Γ.

## Common initial electronic state

The primary learned state is obtained from `H_model C = S_source C epsilon`,
using the reference OpenMX overlap and the lowest `Ne/2` generalized
eigenvectors. Its restricted total density is `P_source = 2 C_occ C_occ^T`.
The reference-H control uses the identical filling recipe. This supplies the
same initial state to both transfer methods and accommodates the SiO₂ model,
which has no density head. It is a zero-temperature integer-occupation test.

Si also has a separately labelled density-head control. Its raw historical
prediction is already restricted total density. After Fourier summation, the
top `Ne/2` eigendirections of `S_source^(1/2) P S_source^(1/2)` define an
explicit rank-`Ne/2` projector. Raw charge, occupation extrema and the change
made by this operation are recorded. This control is not labelled as an
unchanged predicted density. It differs from the fractional occupation repair
used in the earlier Si030 experiment.

## Transfer methods and controls

For numerical OpenMX functions `phi` and target Gaussian functions `chi`, let
`B = <chi | phi>`, including periodic images, and `S_t = <chi | chi>`.
Our route projects the occupied orbitals using the retained inverse metric,
then orthonormalizes the projected occupied subspace in `S_t`. Equivalently,
with `S_t = U Lambda U^T` on eigenvalues above `1e-5`,

```
W = U Lambda^(-1/2)
A = W^T B C_occ
C_target = W A (A^T A)^(-1/2)
P_target = 2 C_target C_target^T
```

This preserves electron count and occupied rank after the stated metric
repair. It does not preserve every entry of the original density matrix.
The cross overlaps use the original numerical PAOs, analytic angular
integration, and one-dimensional radial quadrature; no radial Gaussian fit
of the source PAO enters this projection integral.

The supervisor route uses the actual supplied `openmx_to_cp2k.py` converter
with dense `P_source`, the reference source overlap, explicit restricted-total
spin, and WFN occupation-preservation mode. Source-metric natural orbitals
factor P, and its angular permutation transfers their coefficients to fitted
Gaussian counterparts. No target-metric correction is requested in the
converter. The target-metric electron trace before CP2K is recorded; CP2K can
subsequently orthonormalize the imported orbitals. This tests the supervisor's
WFN route, not the DBCSR/LS_SCF route. Keeping conventional diagonalization in
every arm avoids confounding the transfer with a solver change.

Both routes run in the **same fitted target basis**, alongside an atomic
baseline. A second family measures our projection into standard
DZVP-MOLOPT-SR-GTH, with its own atomic baseline. MOLOPT headers receive unique
element-specific aliases because the supervisor's layout reader indexes names
without an element; the Gaussian numerical parameters are unchanged.

| Family | Common cases in each material | Additional Si cases |
| --- | --- | --- |
| Fitted PAOs | ATOMIC, reference-H projection, reference-H coefficient transfer, model-H projection, model-H coefficient transfer | density-head projector through each transfer |
| Standard MOLOPT | ATOMIC, reference-H projection, model-H projection | density-head projector through projection |

There are 11 Si cases and 8 SiO₂ cases, 19 total, organized as four independent
SCF family jobs after two preparation jobs. Compare steps saved against the
atomic control within each family. Absolute energies in different basis
families are not comparable convergence checks.

The fitted library is generated with the supplied fixed shared-exponent recipe
and 24 primitives per angular channel. Diagnostic radial/kinetic acceptance
flags remain visible; `allow-inaccurate` permits this approximation experiment,
not a claim that the fit meets the supervisor's production gates. No fit is
selected or changed using its SCF outcome. The existing accepted BASIS_OMX
library is not overwritten.

## Angular conventions, periodicity and checks

The upstream cached operators are decoded from e3nn to native OpenMX order.
The cross-overlap target rows use native CP2K real harmonics, while the
supervisor's basis-layout permutation handles coefficient transfer. Full
angular multiplets share each radial map. Consequently the transfer algebra
is covariant under consistent source/target AO rotations. Focused tests check
covariance, occupied-space charge, retained-rank behavior, Gaussian integrals
against an independent backend, and exact preservation under basis relabeling.

This is separate from model equivariance. The paper checkpoints retain their
historical FiLM semantics, which are **not generally O(3)-certified**; the newer
physical-constraints investigation identifies this limitation. Passing a
transfer test does not certify the historical neural network or quantify GPW
rotation/grid errors. No numerical physical-rotation benchmark is claimed
from this unrotated batch.

Periodic H and S are summed over their real-space image blocks at Γ. The
supervisor's repeated-Gamma-density folding rule is not applied to the
native-k-mesh Si density blocks. Projection sums PAO/Gaussian image overlaps
with conservative per-image tail bounds. Preparation checks radial quadrature
against piecewise integration and requires consistent joint overlap Gram
matrices. Before any imported guess runs, each family's atomic calculation
prints native CP2K S, which must match the independent Gaussian overlap to
`2e-6` maximum absolute error. A failed interface gate stops that family.

All SCF arms use CP2K 2023.1, PBE/GTH-q4 Si and GTH-q6 O, cutoff 600 Ry,
relative cutoff 80 Ry, integral thresholds `1e-12`, `EPS_EIGVAL 1e-5`,
`EPS_SCF 1e-7`, Broyden mixing 0.15/history 16, and at most 200 iterations.
Each starts in a fresh directory. Output checks require the intended restart
without atomic fallback, the correct electron count, explicit convergence,
and final energy agreement with the same-basis atomic result. The frozen
energy tolerances are `1e-5` Ha for Si and `1.5e-4` Ha for SiO₂. These are SCF
endpoint checks, not convergence of operator errors with respect to grids or
basis fits. CP2K timing and preparation timing are reported separately;
archived inference reuse is not evidence of end-to-end prediction speedup.

## Results checked on 6 November 2026

Three families completed, comprising 14 converged SCF cases. All completed
families passed native-overlap, restart, electron-count and same-basis final
energy checks. No completed learned-guess case reduced the SCF count below its
matched atomic baseline.

| Material/basis | ATOMIC | Reference-H projection | Reference-H coefficient | Learned-H projection | Learned-H coefficient |
| --- | ---: | ---: | ---: | ---: | ---: |
| Si MOLOPT | 16 | 16 | — | 21 | — |
| Si fitted PAOs | 18 | 21 | 20 | 53 | 32 |
| SiO₂ MOLOPT | 90 | 84 | — | 111 | — |

Si's separately labelled density-head projector took 16 steps in MOLOPT and
22 (projection) / 25 (coefficient transfer) in the fitted basis. This is a
repaired occupied-space control, not an unchanged predicted density.

For Si in the same fitted basis, the supervisor's learned-H transfer was
faster than our projection (32 versus 53 steps), but both lost to ATOMIC (18).
All seven fitted-basis Si endpoint energies agree within `2.57e-8` Ha.
Completed MOLOPT endpoints agree within `1.18e-6` Ha for Si and `1.85e-7` Ha
for SiO₂. The six-step reference-H reduction for SiO₂ did not reduce measured
CP2K subprocess wall time (514 s versus 477 s atomic); the learned guess took
677 s. Fitted Si ATOMIC took 328 s versus 5.9 s in MOLOPT, so basis-dependent
cost remains material even before adding preparation overhead.

Both materials' fits and all 19 initializations were prepared successfully.
The fitted-basis SiO₂ SCF job hit its **2700-second per-calculation timeout**
during ATOMIC, after nine SCF iterations at approximately 229 s per later
iteration. That timeout is a resource limit, not a demonstrated scientific
nonconvergence at the 200-step limit. None of its four imported guesses ran.
There is therefore no completed SiO₂ projection-versus-coefficient result.

## Final outcomes checked on 7 November 2026

All 19 cases have executed: 14 passed the frozen convergence and interface
checks, and all five fitted-basis SiO₂ cases reached the 200-iteration ceiling
without satisfying `EPS_SCF=1e-7`. No comparison jobs remain running or queued.
All five CP2K processes returned zero, ended normally and retained 800
electrons; the imported cases read RESTART without atomic fallback. The
wrapper/aggregate failures represent scientific acceptance failures, not
scheduler timeouts or CP2K crashes.

| Fitted SiO₂ guess | Iterations executed | Final printed SCF residual | CP2K wall time, hours |
| --- | ---: | ---: | ---: |
| ATOMIC | 200, not converged | 1.5e-7 | 12.95 |
| Reference-H projection | 200, not converged | 2.0e-7 | 10.60 |
| Reference-H coefficient transfer | 200, not converged | 1.8e-7 | 11.06 |
| Learned-H projection | 200, not converged | 2.0e-7 | 12.51 |
| Learned-H coefficient transfer | 200, not converged | 2.1e-7 | 12.49 |

The native fitted-basis overlap matched the independent analytic overlap to
`4.53e-9` maximum absolute error. Endpoint energies agree with the atomic
endpoint within `5.63e-9` Ha, but this is not a converged-energy or SCF-step
saving result: every case explicitly printed `SCF run NOT converged`. Residuals
in the final iterations fluctuate above the threshold; their cause has not
been isolated. We have not relaxed the registered criterion after seeing these
outputs or launched another fit/SCF parameter search.

The overall registered comparison therefore provides **no demonstrated
learned-guess acceleration**. On converged Si fitted-basis runs, supervisor
coefficient transfer is faster in steps than our H-derived projection, but
both are slower than atomic. Fitted SiO₂ cannot provide a converged ranking.
This conclusion covers these two specimens and the registered Γ-point
protocol, not a general failure of learned initial guesses. Original paper Si
sampling was 8×8×8, a limitation already registered above.

## Recovery diagnosis and corrected scheduling, 6 November

The four-rank runtime pilot 828275 completed with initial energies matching
within `2.40e-9` Ha, but its second step took **251.9 s**, slower than the
original singleton's 228.9 s. The eight-rank pilot exceeded 96 GiB. There was
no demonstrated MPI speedup. Gate 828283 therefore failed its registered
150-second performance criterion, which had been intended to fit 200 steps
into the short partition's nine-hour case budget. This was a scheduling and
budget failure, not a numerical converter failure.

Atomic job 828285 acquired `DependencyNeverSatisfied`; its four guesses and
aggregation job remained downstream. All six stale jobs (828285–828289,
828291) were cancelled explicitly. The v6 snapshots and failed gate remain
unchanged, rather than retroactively lowering that registered threshold.

The replacement uses `cpu-genoa`, whose wall-time limit is five days, rather
than `cpu-genoa-int`, whose limit is ten hours. It restores the already tested
single MPI rank × 32 OpenMP threads. Each case receives 128 GiB, a 16-hour
subprocess limit and a 16.5-hour allocation. At the observed 229 s per step,
200 iterations require approximately 12.7 hours plus setup. The original
singleton peaked near 92 GiB, motivating the extra memory headroom.

Fresh, byte-verified case inputs and immutable references to the original v4
fitted basis and WFN guesses are in
`/bigdata/casus/wdm/hamiltonian_learning/cp2k_initial_guess/paper_method_comparison_20261106_sio2_v7`.
A first launch there failed at NumPy import before invoking CP2K: the CP2K
module injected Python 3.11 SciPy paths into the profile's Python 3.10
runtime. The separately registered execution amendment
`paper_method_comparison_20261106_sio2_execution_v8` restores the complete
module executable PATH and explicitly replaces PYTHONPATH with the frozen
source paths, as in the successful earlier v5 calculations. No scientific
input, fit, checkpoint, guess or SCF parameter changed.

The new ATOMIC job **828500** was verified running actual CP2K with the
correct Python 3.10 environment. The four imported guesses wait for its
completion and must pass the runner's native-overlap/interface guard before
invoking CP2K. Using an `afterany` scheduling dependency with this explicit
pre-execution guard makes a failed prerequisite produce an immediate,
recorded guard failure rather than a permanently pending dependency chain.
Scientific atomic nonconvergence remains a recorded outcome and does not
suppress independent guesses if its runtime and interface checks passed.
The per-case runner's focused synthetic integration check passed for the
interface guard, separate case records, baseline comparison and overwrite
refusal. Numerical outcomes are separate from submission records.

| Active long-partition recovery | Job |
| --- | ---: |
| ATOMIC | 828500 |
| Reference-H projection | 828509 |
| Reference-H coefficient transfer | 828510 |
| Learned-H projection | 828511 |
| Learned-H coefficient transfer | 828512 |

The four guesses are normally dependency-pending behind the running atomic
job. No `DependencyNeverSatisfied` entries remained in the user's queue after
the stalled chain was cancelled and these replacements were submitted.

## Execution and reproducibility

The immutable prepared-artifact ROSI directory is
`/bigdata/casus/wdm/hamiltonian_learning/cp2k_initial_guess/paper_method_comparison_20261105_v4`.
The SCF launch registration and scheduler logs are in
`/bigdata/casus/wdm/hamiltonian_learning/cp2k_initial_guess/paper_method_comparison_20261105_execution_v5`.
It reuses the v4 prepared artifacts and case directories. Large matrices,
PAOs and restart files stay on ROSI. Local small operational
records are in `researchops/runs/cp2k_projection_vs_coefficient_20261105`:
specification, source hashes, frozen submission scripts and workflow ledger.
Those ignored records must accompany any later public result package.

| Material | Preparation job | MOLOPT SCF job | Fitted-basis SCF job |
| --- | --- | --- | --- |
| Si | 827201 | 827212 | 827213 |
| SiO₂ | 827202 | 827214 | 827215 |

Si preparation completed before its active SCF jobs were submitted. SiO₂
SCF jobs have `afterok` dependencies on preparation job 827202.
Twelve focused local tests passed, supplemented by a compatibility check using
the actual installed MOLOPT text and the supplied supervisor WFN metadata
reader. The table records the original submission identities. Completion and timeout
status are reported above; recovery records are separate.

The first revision exposed a reproducible ROSI SciPy/OpenBLAS segmentation
fault in the eight-thread generalized eigensolver. The identical single-thread
reference/model solves passed an allocated-node diagnostic. Preparation now
uses one BLAS thread, with two/eight cross-overlap worker processes for
Si/SiO₂. The second revision exposed the MOLOPT alias ambiguity; the fourth
contains tested relabeling and selected-entry extraction, avoiding unrelated
installed-library formats unsupported by the supplied reader. Earlier runs
remain immutable and their
dependent jobs were cancelled. Scientific configurations were not selected
using SCF results. SCF jobs use 8 CPU threads for Si and 32 for SiO₂, on
allocated CPU nodes, without a GPU request.

The first SCF launchers failed before executing CP2K: reactivating the Python
virtual environment reset PATH and removed the CP2K module bin directory.
The separately frozen v5 execution amendment restores that bin directory
after activation. It reuses unchanged v4 guesses and inputs; no expensive
preparation was repeated for this launcher-only correction. Native CP2K
execution and the MOLOPT overlap calibration have since passed for Si.

Implementation entry points are `scripts/prepare_cp2k_method_comparison.py`,
`scripts/run_cp2k_method_comparison.py`, and
`src/analysis/cp2k_gamma_compare.py`, alongside the accelerated analytic
cross-overlap kernel in `src/analysis/cp2k_projection.py`.
