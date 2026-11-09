# Supervisor converter review and comparison with our CP2K work

Reviewed 5 November 2026. This reviews the email, the supplied
`OpenMX_MANDALA_CP2K_converter.zip`, and our completed silicon benchmark.
Attachment instructions were treated as documentation, not as authorization to
install dependencies, overwrite restarts, or launch calculations.

The package provides a working design for importing the original six-atom
Gamma density using fitted OpenMX-like Gaussians. It complements our k-point
projection adapter. Its reported water improvement does not contradict our
negative learned-guess result for silicon, and is not yet a separately
identified trained-MANDALA benchmark.

## What changed since the first attachment

The earlier package supplied an accepted fitted-basis library and a 14-AO,
three-atom smoke input, with no density restart converter. We established that
our periodic H/H/H/H/O/O example has 80 AOs and cannot use that input.

The new package identifies H7.0-s3p2 (9 AOs per H) and O7.0-s3p3d2 (22 AOs
per O), preserving blocks 9/9/9/9/22/22 and 16 valence electrons. It includes
numerical PAO identification evidence, converter sources, a separate diagnostic
80-AO basis, an original WFN, an original native DM, a separately named final DM,
three independent CP2K inputs, and JSON result/provenance records.

The numerical identification compares 5/6/7-bohr PAOs, same-center overlaps,
periodic self-image overlaps including p/d blocks, and all atoms. H7/O7
identification residuals are around 1e-7, substantially smaller than rejected
cutoff candidates. This establishes identity within the pinned catalog, not an
exact Gaussian representation. Geometry comes from the supplied original
extxyz, not the approximate position-operator reconstruction; malformed
position blocks were excluded from that reconstruction.

The accepted `data/BASIS_OMX` is byte-identical to the previously audited library
(SHA256 c74b097d57b4386cea45482ad70054c0c429bdda91410e664be9910e315f5294).
The new `example80/BASIS_OMX_MANDALA80_COMPACT` is a separate 24-primitive-per-
angular-channel test basis and explicitly fails those unchanged quality gates.
The earlier conclusion that the rich H/O selections were not accepted remains
correct; the new example explicitly uses an approximate diagnostic alternative.

## Code and restart routes

| File | Role |
| --- | --- |
| `read_openmx_matrix.py` | Preserve repeated section labels, parse atom/image blocks, reject malformed/nonfinite data; distinguish real/imaginary density occurrences. |
| `openmx_to_cp2k.py` | Explicit spin and Gamma policy, actual basis-layout checks, angular permutation, charge checks, DM/WFN output and provenance. |
| `cp2k_wfn.py` | Version-specific real Gamma WFN metadata and MO records; explicit byte order and Fortran record-marker width. |
| `dm_to_dbcsr.c` and CMake file | Use the target DBCSR C bindings to serialize/read back atom-block real symmetric density matrices; no CP2K source changes. |
| `convert_basis.py` | Numerical PAO interpolation, normalized Gaussian fits, shared and opt-in segmented exponent optimization, radial/kinetic diagnostics. |
| Database tools and tests | Pinned source identities, library generation/acceptance, parsing, synthetic numerical checks, and optional native CP2K tests. |

The converter consumes OpenMX text or real dense ASCII/NumPy matrices, not a
MANDALA checkpoint or an e3nn tensor file. Model inference and correct decoding
into native OpenMX order remain upstream responsibilities.

For the direct route, with U the block-diagonal angular permutation:

    P_fit = g U P_input U^T

Here g=2 for a restricted per-spin source, and g=1 for restricted total density.
The C program writes exactly this already-transformed matrix using DBCSR, then
reads it back and checks equality. Python does no eigensolve, metric repair or
cross-overlap projection by default. An optional occupation audit is cubic.
The initial matrix is preserved, but its density function and electron trace
can change when numerical PAOs are replaced by fitted Gaussians. Dense Python
assembly and serializer buffers use O(N^2) storage; LS_SCF support does not make
this exporter a linear-scaling-memory implementation.

The LS input uses QS/LS_SCF, RESTART_READ, atomic clustering and S_PRECONDITIONER
NONE. The serializer must be built against a compatible target DBCSR library;
its native format should not be treated as a universal restart standard.

For WFN output, source-metric natural orbitals factor the same density. With
S_source=L L^T, diagonalize L^T P L = V f V^T and form C=L^(-T)V. Then
C^T S_source C=I and C f C^T reconstructs P. This costs O(N^3) but does not solve
the KS Hamiltonian. CP2K may subsequently orthonormalize C in the fitted target
metric, changing P. Eigenvalues written to WFN are zero placeholders, not a KS
spectrum. Preserved fractional occupations require a fixed LIST throughout SCF;
this is a meaningful constraint for finite-temperature/metallic applications.
The bundled 80-AO example instead has eight occupied orbitals with occupation 2.

For the checked repeated-Gamma dump, P(i,j) repeats for each image: retain one
copy, while summing overlap image blocks to obtain S(Gamma). The code checks
repetition and rejects nonzero imaginary parts. This rule does not apply to our
8x8x8 silicon real-space density, whose blocks genuinely depend on image R.
General k points, complex/SOC matrices and arbitrary periodic folding are not
implemented in this package.

## Supplied SCF results and interpretation

These are supervisor-reported CP2K results, read from `results_compact.json`.
The archive includes no full SCF output logs or timings. We did not rerun the
three physical CP2K calculations or compile/run DBCSR in this review.

| Six-atom water case | Solver | SCF iterations | Final energy, Ha |
| --- | --- | ---: | ---: |
| ATOMIC | Conventional SCF, diagonalization | 22 | -33.818519262798 |
| WFN restart | Conventional SCF, diagonalization | 17 | -33.818519448946 |
| Direct DM restart | LS_SCF | 17 | -33.818519948593 |

Inputs specify CP2K 2026.2, CPU grid backend, four MPI ranks and eight OpenMP
threads, PBE/GTH-q1 H and GTH-q6 O, cutoff 600 Ry, relative cutoff 80 Ry,
EPS_DEFAULT/EPS_PGF_ORB/EPS_PPNL 1e-12, CORE_PPL GRID and CANONICALIZE FALSE.
The conventional SCF uses EPS_SCF 1e-7 and Broyden alpha 0.15. LS_SCF has its
own mixing/purification path and criterion.

WFN versus ATOMIC is the directly matched initial-guess contrast: 5 fewer
steps, or about 23%, on one structure. Direct DM also converges and demonstrates
an import route, but its 17 steps versus the conventional 22-step baseline
confounds solver and guess. An LS_SCF atomic baseline is required to isolate
its initialization benefit. Equal numerical EPS_SCF values across these
solvers need not define the same stopping criterion. Neither comparison
establishes end-to-end speedup without timing.

All final energies differ by less than 7e-7 Ha in this target Hamiltonian.
That demonstrates a consistent converged endpoint, not agreement with OpenMX
energies or an accurate approximation to the source operator.

Both restart reports identify the exact local
`data/small/H2O/original/H2O.matrix` hash
10f3a1f16430513ff1f7a1c309abd9905d4342f3a1f4e8c2fe01533aba12c154.
The ZIP provides no trained checkpoint/run identity or separately identified
learned prediction. Consequently the 22-to-17 evidence establishes transfer
of that original source matrix; a trained-MANDALA acceleration claim needs a
separate prediction provenance record and matched calculation.

## Remaining radial approximation

| Quantity | Supplied diagnostic |
| --- | ---: |
| Source Tr(P S_source) | 16.00000000010 |
| Same transferred P in fitted CP2K metric | 15.99647482700 |
| Charge difference | -0.003525173 electrons (~0.022%) |
| Maximum source/target overlap entry difference | 0.0831740 |
| Relative overlap Frobenius difference | 2.296% |
| H third-s radial L2 / kinetic expectation error | 3.900% / 15.595 mHa |
| O third-s radial L2 / kinetic expectation error | 3.030% / 16.016 mHa |
| O third-p radial L2 / kinetic expectation error | 4.085% / 18.649 mHa |
| Target overlap condition number | ~1.15e5 |

The original library bounds are 1% radial L2, 1 mHa kinetic expectation and
1e-3 same-center overlap error. The diagnostic 80-AO basis fails the first two.
A small total charge error does not imply small operator errors; different AO
metric errors can cancel in the scalar trace. The reported 1e-7 PAO identity
checks concern numerical source orbitals, not the inaccurate Gaussian fits.

The new segmented fitter independently optimizes radial exponents, uses
regularized variable projection and a reduced-residual Jacobian, and retains
radial and kinetic checks. The email reports testing fixed fits on equilibrium,
stretched and bent water and a dimer; those multi-geometry benchmark results
are not bundled as a separate result set. No reliably improved replacement
library is shipped. The accepted BASIS_OMX remains unchanged.

## Relationship to our implementation and results

| Aspect | Supervisor package | Our completed silicon work |
| --- | --- | --- |
| Physical example | Six-atom H2O cell, 80 AOs, 16 electrons | Eight Si, 104 AOs, 32 electrons |
| Periodicity | Checked real repeated-Gamma | Full complex 8x8x8 Bloch mesh |
| Target basis | Gaussian approximations to source H/O PAOs | Standard DZVP-MOLOPT-SR-GTH |
| Density transformation | Preserve coefficients after spin/angular conversion | Actual PAO/Gaussian cross overlap, T=solve(S_target,S_cross), P_target=T P_source T^dagger |
| Physical repair | None in direct DM; optional explicit WFN projector | Fixed covariant charge/occupation repair, 8.24% metric change for learned Si |
| Import interface | Gamma WFN or native LS_SCF DBCSR DM | CP2K 2023.1 k-point .kp AO-image density |
| Main approximation | Fitted PAO shapes/metric and different pseudopotential | Projection loss, finite-range/prediction errors, repair and different pseudopotential |
| Model provenance | Original source matrix hash; no separately identified prediction | Paper-selected Si checkpoint, strict frozen historical replay |
| SCF outcome | Reported 22 atomic, 17 WFN, 17 DM with solver caveat | Measured 14 atomic, 12 projected OpenMX reference, 14 projected MANDALA; native controls 2 |

Both routes preserve the complete source AO inventory, handle angular/spin
conventions explicitly, and avoid an external density-grid import. CP2K still
uses its internal GPW grid for Hartree/XC. The supplied Gamma WFN/LS_SCF paths
add interfaces we did not implement in our k-point adapter. Our initial Gamma
feasibility discussion should therefore be updated to include native LS_SCF
DM import as an available route, rather than implying WFN or a new CP2K hook
is the only alternative to an external grid.

The mathematical approaches serve different goals. If fitted orbitals are
sufficiently close to the source PAOs, unchanged coefficients make an efficient
guess. If using a genuinely different CP2K basis, cross-overlap projection is
the appropriate orbital least-squares map. Coefficient preservation alone
cannot establish physical density preservation across changed radial functions.
Our projection also does not eliminate pseudopotential differences or guarantee
fewer SCF steps.

The example's per-spin factor must not be blindly reused for historical learned
models: our paper checkpoint already predicts restricted total density.
Likewise the earlier checkpoint replay compatibility discrepancy concerns model
inference, while this package begins with an already supplied matrix. Neither
serializer can fix an incorrectly decoded or normalized model prediction.

The positive water result and negative learned Si result are compatible:
materials, source/prediction provenance, bases, periodic sampling, solver,
CP2K version and representation changes differ. In particular our prior
three-atom/14-AO water smoke result of 29 steps is not this six-atom baseline.

## Equivariance and remaining validation

A radial Gaussian fit can preserve the analytic real-harmonic rotation law even
when its radial accuracy is poor. Shared radial treatment across all m in a
shell is essential. With U converting source angular order, the consistent
rotations satisfy D_CP2K=U D_OpenMX U^T. Rotate atoms and cell together and use
P_R=D P D^T; rotate the full d shell, not merely xyz coordinates or p orbitals.
The converter accepts native OpenMX/CP2K order, not an unconverted e3nn matrix.

The supervisor has not quantified the reference/-30-degree CP2K operator error,
and the archive contains no such result. We tested analytic projection and
model rotation/reflection covariance, but did not establish full GPW KS-matrix
rotation accuracy for this water case either. CANONICALIZE FALSE retains the
intended cell orientation. Compare S, kinetic T and H after undoing the AO
rotation. For KS H and energies, separately converge the GPW grid and screening
tolerances; converged SCF by itself does not bound their numerical operator
error. Keep fitted-basis error separate from numerical-grid rotation error.

A minimal next matched validation would retain the supplied fitted basis:
conventional ATOMIC/reference-WFN/learned-WFN, and a separate
LS_SCF-atomic/reference-DM/learned-DM comparison. Freeze prediction/checkpoint
identity, spin convention, geometry, electron count, solver and occupation
policy. Then compare reference and rotated cells plus an independently specified
grid/tolerance sequence. No new cluster jobs were submitted for this review.

## Independent checks performed in this review

Seventeen targeted offline tests passed, including parsing/repeated-Gamma
semantics, angular ordering, binary WFN records, spin/dense-input handling,
nonphysical-occupation rejection, and a direct-export test that forbids spectral
operations. The private 80-AO source test was enabled rather than skipped.

Additionally, independent readback of the packaged 80-AO WFN reconstructed the
permuted spin-summed original density with relative difference
7.0035e-11; source trace was 16.00000000010 and WFN trace 16.0. The WFN has
8 occupied orbitals at occupation 2 and correct 9/9/9/9/22/22 basis metadata.
This verifies its content against our exact source file, not a physical CP2K
SCF rerun or native DBCSR binary compatibility with ROSI.

Small provenance/results are under
`researchops/runs/cp2k_supervisor_converter_review_20261105/`.
