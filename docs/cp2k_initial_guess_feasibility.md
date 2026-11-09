# MANDALA initial guesses for CP2K: basis audit and silicon integration

**Latest result:** the completed matched silicon benchmark took 14 SCF steps
with ATOMIC, 12 with projected reference OpenMX density, and 14 with the
projected MANDALA guess. The adapter works; MANDALA acceleration was not observed
for this snapshot. See the final benchmark section for settings and limitations.

The attachment audit was performed on 3 November 2026. ROSI integration
continued on 4 November. The supplied files do **not** implement a MANDALA
initial guess. Native CP2K controls work; the completed learned-guess comparison showed no
SCF-step reduction on this snapshot. The silicon section below records the subsequent integration.

## What was supplied and what passed

The reviewed attachments were `BASIS_OMX`, `OpenMX_basis_tools_H2O.zip`, and
`H2O_basis_smoke.inp` from the user's Downloads directory. The standalone basis
and the archive's basis are byte-identical (SHA256
`c74b097d57b4386cea45482ad70054c0c429bdda91410e664be9910e315f5294`).
The archive contains radial fitting tools, a provenance/quality catalogue,
GTH potentials, offline tests, and an optional CP2K integral/SCF test. Its README
explicitly says there is no density-matrix or wavefunction-restart converter.
The cited PR/source commit was not independently retrievable during this audit;
the attachment's provenance claim is not evidence of merge/release status.

Executed checks:

* All 279 exported selections across 81 elements parse and satisfy the supplied
  catalogue thresholds. Largest independently recomputed contraction-norm
  discrepancy: **3.50e-10**. This is an offline numerical check, not execution of
  CP2K's basis reader.
* Four targeted supplied tests passed: angular signs/order through g, Gaussian
  normalization/derivatives, synthetic fit/export normalization, and shipped
  library quality/provenance. The native CP2K test was initially **skipped** locally;
  the subsequent ROSI smoke calculation and integral checks are reported below.
* Independently re-evaluated the smoke-test H7.0/O7.0 radial contractions against
  checksum-matched local PAOs, using order-8 radial quadrature without refitting.

| Smoke selection/channel | Largest normalized radial L2 error | Largest kinetic expectation error, Ha |
| --- | ---: | ---: |
| H7.0 s2 | 0.0060063 | 0.00044976 |
| H7.0 p1 | 0.0039236 | 0.00022980 |
| O7.0 s1 | 0.00019525 | 0.0000012991 |
| O7.0 p1 | 0.00085535 | 0.0000092877 |

These pass the attachment's radial/kinetic thresholds. They are not bounds on
molecular energy, forces, density errors, or SCF convergence. Finite-support
PAOs and Gaussian tails do not define identical subspaces.

Of the 230 PAO source identities used by exported entries, 213 match the local
OpenMX installation, 14 are absent locally, and 3 Ge files have different hashes.
The H7.0 and O7.0 files used above match exactly. Other local PAOs must not be
silently substituted for a checksum-pinned source.

## The selected repository example differs from the smoke test

| Property | `data/small/H2O/original` | Attached CP2K smoke input |
| --- | --- | --- |
| Atoms/order | H H H H O O | O H H |
| Boundary conditions | Periodic six-atom cell | Isolated water, 12 Angstrom box |
| H shells / orbitals per atom | s3p2 / 9 | H7.0-s2p1 / 5 |
| O shells / orbitals per atom | s3p3d2 / 22 | O7.0-s1p1 / 4 |
| Total orbitals | **80** | **14** |
| Neutral valence electrons | **16** | **8** |

The original information file contains Cartesian and fractional coordinates.
Their least-squares cell reconstruction is approximately
`diag(3.34810, 3.34810, 3.23996)` Angstrom, with a maximum coordinate residual
of 3.90e-6 Angstrom from rounded printed coordinates. Use the original input
cell when available; this reconstruction should carry that precision limitation.
The file establishes shell multiplicities, but does **not** identify the exact
PAO stems/cutoffs, pseudopotential files, or original k-point mesh.

Every H s3p2 selection in the supplied catalogue (5/6/7 bohr) and every O s3p3d2
selection (5/6/7 bohr) is **omitted** from the exported library for failed radial
L2 and kinetic thresholds; O6.0 additionally fails same-center overlap. Therefore
the exported library cannot reproduce the original model's full AO inventory.
Reducing the shells requires a physical subspace projection; deleting matrix
rows/columns or padding them does not provide one.

The smoke input uses `SCF_GUESS ATOMIC` and disables restart output. Even a
successful execution would certify only that particular fitted-basis CP2K
calculation, with no comparison to MANDALA. `BASIS_SET_FILE_NAME data/BASIS_OMX`
and `POTENTIAL_FILE_NAME data/GTH_POTENTIALS` resolve relative to the execution
directory. The archive supplies those paths; copying only the input does not.

## Equivariance and basis transformations

The supplied angular permutation U_l agrees with the inspected CP2K 2026.2
Cartesian-to-spherical formula and local OpenMX `Get_Orbitals.c`. For p shells,
OpenMX `(px,py,pz)` maps to CP2K `(py,pz,px)`; for d shells the zero-based order
is `[2,4,0,3,1]`. The same permutation is repeated for each radial shell.
CP2K's source orders each shell by m=-l,...,+l. See
[CP2K's angular transformation source](https://github.com/cp2k/cp2k/blob/v2026.2/src/aobasis/orbital_transformation_matrices.F).

Numerical angular checks used s through g, 100 sphere directions and 16 proper
or improper transformations per l. The largest residual in the intertwiner
identity `U_l D_OMX(R) = D_CP2K(R) U_l` was **2.67e-15**. Radial maps of the form
`T_l = A_l (tensor product) U_l`, with shared radial coefficients across every
m, satisfied the corresponding identity to **5.11e-15**. Inversion parity is
`(-1)^l`. This verifies the analytic conversion principle, not a loaded model
or CP2K grid implementation.

MANDALA's OpenMX/e3nn boundary also changes Cartesian coordinates from xyz to
yzx. Use the existing native conversion/export boundary (including geometry and
cell), then map native OpenMX orbitals into the target CP2K order. The identical
numbers in two angular permutation tables do not imply identical Cartesian
frames. Apply each axis/order conversion exactly once. A trained checkpoint
still needs an end-to-end rotation/reflection test after decoding and transfer.

For an **exact**, square, invertible basis relation `Phi_t = Phi_s A`, the
operator and density matrices transform differently:

```text
H_t = A^dagger H_s A       S_t = A^dagger S_s A
C_t = A^-1 C_s            P_t = A^-1 P_s A^-dagger
N = Tr(P_t S_t) = Tr(P_s S_s)
```

Orthogonal permutations hide this distinction. It becomes essential when
normalizations, radial functions or dimensions differ. The synthetic complex
nonorthogonal check preserved electron count to 1.96e-14; incorrectly applying
the Hamiltonian formula to the density changed the electron count by 21.2 in
that synthetic example.

For **different** numerical/ Gaussian spaces, compute actual target and
cross overlaps on the intended structure, with periodic sums and phases:

```text
S_ts = <chi_t | phi_s>     T = solve(S_tt, S_ts)
C_t_raw = T C_s           P_t_raw = T P_s T^dagger
```

This is a least-squares projection, not an exact similarity transform. For a
closed-shell integer-occupied subspace, inspect rank loss and orthonormalize:

```text
G = C_t_raw^dagger S_tt C_t_raw
C_t = C_t_raw G^(-1/2)    P_t = 2 C_t C_t^dagger
```

Reject insufficient retained rank instead of inverting tiny eigenvalues.
Fractional occupations require a separately defined occupation-preserving or
refilling procedure; do not silently mix differently occupied orbitals.
An overlap solve must use the retained, well-conditioned target space, following
the target solver's linear-dependence policy. Check `Tr(P S)`, Hermiticity and
the eigenvalues of `S^(1/2) P S^(1/2)` (in [0,2] for a spin-summed restricted
density). For periodic systems these checks use the k weights and per-k
matrices, not just the zero-shift block. A generally geometry-dependent
multicenter projection must also satisfy rotational covariance; a constant
atom-local radial map alone does not certify it.

CP2K normalizes primitives and contractions in its default orbital basis path;
the attached coefficients were prepared for that convention. Changing
normalization changes the AO representation. See
[CP2K's basis normalization implementation](https://github.com/cp2k/cp2k/blob/v2026.2/src/aobasis/basis_set_types.F).

Before use, assemble and globally Hermitian-project physical periodic
predictions, pairing `(i,j,L)` with `(j,i,-L)` using conjugate transpose.
Self-symmetrizing an offsite block is not this projection. For a Hamiltonian-only
checkpoint, solve `H_s(k) C = S_s(k) C epsilon`, set occupations using the correct
electron count, then transfer the resulting occupied subspace or density.
Use a trustworthy source overlap and verify positivity of its retained metric.

The current OpenMX parser reads only `Density matrix spin=0`, with **no spin
sum**. Its density-grid reconstruction also explicitly preserves that channel.
For this nonmagnetic cell, the channel count is 8 electrons and the restricted
total-density count is 16. Multiply by two only after confirming that the
checkpoint retains this source convention. A density constructed from orbitals
with occupations two is already spin-summed. Spin-polarized/SOC checkpoints
need a different, explicit spin contract.

## Practical CP2K entry routes

**Scalar grid route, conditional on executable support:** evaluate
the predicted density with the **original numerical PAOs** on the exact CP2K
finest grid and pass a scalar electron-density cube. This avoids Gaussian fitting
of the original s3p2/s3p3d2 source basis and lets CP2K retain its intended Gaussian
basis/pseudopotentials. MANDALA already has PAO density reconstruction in
`src/analysis/openmx_density_grid.py`, but its present CLI targets OpenMX grid
metadata and preserves one spin channel; CP2K grid discovery, spin summation and
import verification still need implementation.

The [current development SCF manual](https://manual.cp2k.org/trunk/CP2K_INPUT/FORCE_EVAL/DFT/SCF.html)
documents `SCF_GUESS EXTERNAL_DENSITY` and `EXTERNAL_DENSITY_FILE_NAME`. The cube
must match CP2K's finest real-space grid and supplies the density for the first
KS Hamiltonian only. This capability is absent from the inspected
[2026.2 SCF interface](https://manual.cp2k.org/cp2k-2026_2-branch/CP2K_INPUT/FORCE_EVAL/DFT/SCF.html).
Treat it as version-dependent, and verify its supported method/spin/grid modes
with the actual build before choosing this route. A dense periodic scalar
density includes all required image contributions and must integrate to 16 here.
Re-evaluate a rotated density on the rotated grid; scalar density transforms as
`rho_R(r) = rho(R^-1 r)`, rather than behaving like an invariant array of voxels.
Finite-grid interpolation and GPW discretization add numerical rotation errors.

**Orbital restart route for a build without external-density support:** project source occupied
orbitals into the chosen target Gaussian basis, metric-orthonormalize, and write
a version-tested CP2K wavefunction restart or implement a small CP2K import hook.
The attachment supplies neither adapter. `SCF_GUESS RESTART` does not read an
OpenMX text matrix or an arbitrary NumPy density. CP2K's
[MO restart implementation](https://github.com/cp2k/cp2k/blob/v2026.2/src/qs_mo_io.F)
uses binary records with basis metadata, orbitals and occupations. Gamma `.wfn`
and k-sampled `.kp` need their respective adapters and a CP2K round-trip check.
Check the log: a missing restart falls back to ATOMIC and invalidates the
comparison. A density cube also does not by itself supply occupied MO coefficients
for an orbital-based restart.

Both routes retain CP2K's own self-consistent Hamiltonian after initialization.
The OpenMX and GTH pseudopotentials are different. Matching valence counts is
necessary for this neutral transfer but does not establish transferability.
An OpenMX-trained guess can help without reproducing the CP2K Hamiltonian;
whether it helps must be measured.

## Frozen comparison needed to claim acceleration

First use the selected original structure as an integration case, preserving all
six atoms, periodic images, original cell precision and a declared k mesh. Freeze
one target CP2K calculation: executable/build, PBE functional, GTH potentials,
target basis, charge/spin, k mesh, grid, convergence tolerance, mixing and solver.
Changing target basis between guess arms would confound the result.

1. Run an atomic baseline and a native CP2K converged-density/MO import round trip.
   The latter checks that the import is actually used and the final solution is
   preserved. A CP2K oracle includes an initial SCF cost and is only a diagnostic.
2. Transfer the **reference OpenMX** density/orbitals with the same adapter.
   This tests whether even an accurate source-code guess helps across the code/
   pseudopotential change. It is a transfer diagnostic, not a MANDALA result.
3. After those checks, compare the **trained MANDALA** guess to ATOMIC with the
   identical CP2K settings, fresh projects and empty histories. The directory
   supplied by the user is reference data, not a checkpoint. Identify a trained
   checkpoint with the matching H/O basis, output heads and spin normalization.
4. Use the original/rotated pair for equivariance QA. Rotate positions, full cell
   and reciprocal mesh consistently; check density/subspace covariance before
   judging grid-level SCF behavior. A periodic rotation must not rotate atoms
   while leaving their cell fixed.
5. Report SCF updates/KS builds, convergence failures, initial residual, final
   energy/density (forces if relevant), and total wall time including inference,
   density evaluation/projection and IO. Freeze tolerances before comparison.
   Fewer SCF steps with a different final state or failed import is not a success.
   One integration geometry cannot establish acceleration on held-out structures.

The original offline audit made no CP2K or model-acceleration claim. Subsequent
ROSI calculations and the selected silicon checkpoint are recorded below. The
six-atom H2O model comparison remains separate from the three-atom smoke test.

## Audit artifacts

Operational state and bounded scientific evidence are kept separately under
`researchops/runs/cp2k_guess_feasibility_20261103/` (locally ignored by Git):
`task.json`, `basis_validation.json`, `verify.py`,
`attachment_audit.contract.json`, `attachment_audit.json`, and `targeted_tests.log`.
The contract records source bytes, runtime, seed and checks before the analytic
calculation. Reproduction requires access to the original attachment and pinned
PAO files; this document is not a standalone data package. No training split,
checkpoint-selection result or SCF result was generated in that initial offline audit.


## Silicon integration on ROSI (4 November 2026)

The requested eight-atom snapshot is `Si_scale_1_030`, from
`perturbed_snapshots_Si/scale_1`. Its exact `conf.dat` geometry is used, rather
than the rounded coordinates in the information file. Source settings are
PBE, nonmagnetic, 300 K, `Si7.0-s2p2d1`, `Si_PBE19`, and an 8x8x8 shifted
Monkhorst-Pack grid. There are 13 orbitals per Si and 32 valence electrons.

The official [Si7.0 PAO](https://www.openmx-square.org/vps_pao2019/Si/Si7.0.pao)
is available from the OpenMX 2019 library. Its SHA256 is
`24c45d323ad0e4f652776791c0dd9a4c1357d7e63b91333503634401352f843f`;
the download matches the local OpenMX installation byte for byte. The exact
private library copy named in the original input is not readable. Consequently,
the public file must additionally be checked against the recorded HS overlaps;
filename agreement alone does not certify the source basis.

### Completed controls

CP2K 2023.1 (`git:b888bd8`) was run on allocated ROSI CPU nodes.

| Case | SCF steps | Final energy, Ha | Wall time, s |
| --- | ---: | ---: | ---: |
| Eight-atom Si, ATOMIC | 14 | -31.489604395925515 | 102.69 |
| Three-atom attached H2O smoke, ATOMIC | 29 | -16.936296597821134 | 46.17 |
| Si, serialized native converged-density restart | 2 | -31.489608942108223 | diagnostic |

The Si target uses PBE, `DZVP-MOLOPT-SR-GTH`, `GTH-PBE-q4`, 400 Ry cutoff,
60 Ry relative cutoff, full complex 8x8x8 k sampling, 300 K smearing,
`EPS_SCF=1e-7`, standard diagonalization, and Broyden mixing alpha 0.1 with
16 history vectors. All guess arms must use these same scientific settings.
The water smoke integral check also passed: `Tr(P S)=7.999999999997084`,
minimum overlap eigenvalue 0.0196569; Gaussian normalization and kinetic
self-integrals agreed to less than 7e-11.

The native Si restart read/write is byte-exact. A separate original-native
versus Python-serialized replay control subsequently passed: both took two
steps, with an energy difference of 7.1e-15 Ha (job 823497). The first replay wrapper was
marked failed because it searched for a log message from a different CP2K
version and demanded energy agreement at 1e-7 Ha. CP2K 2023.1 actually reports
`Density guess: RESTART`. The atomic and restarted runs stopped at different
points within the SCF density criterion, differing by 4.55e-6 Ha. Preserve this
failed validation record. The fresh original-native versus Python-serialized
native control compares steps and energies directly (1e-10 Ha), while the
registered cross-guess endpoint comparison uses 1e-5 Ha. The native converged
guess is an import diagnostic and does not demonstrate MANDALA acceleration.

### Direct AO density import

For this k-sampled CP2K 2023.1 calculation, the `.kp` restart contains **AO
real-space image density matrices**, not MO coefficients. This follows the
version-1 format in [kpoint_io.F](https://github.com/cp2k/cp2k/blob/v2023.1/src/kpoint_io.F)
and [cp_fm_types.F](https://github.com/cp2k/cp2k/blob/v2023.1/src/fm/cp_fm_types.F).
The new `src/data/cp2k_restart.py` reads and writes the ROSI GCC/x86 Fortran
record format. It uses a native template for atom/basis/image metadata.
It is not a general Gamma `.wfn` writer or a portable CP2K restart standard.

```text
&DFT
  WFN_RESTART_FILE_NAME /absolute/path/projected_density.kp
  &SCF
    SCF_GUESS RESTART
    ... identical baseline settings ...
  &END SCF
&END DFT
```

CP2K stores 160 image matrices in this case, using its symmetric neighbor-list
packing. The image label does not mean every atom block represents the same
physical +L direction. The adapter certifies each nonzero stored overlap block
against analytic Gaussian overlaps at +L and -L, then reconstructs directed
physical images and their Hermitian reverses. A Fourier/native-packing round
trip is required before any projected guess is exported. Simply copying a
104x104 matrix into each record is scientifically invalid. In addition, the
AO text printer mirrors the upper triangle inside each diagonal atom block.
For nonzero images this conceals nonsymmetric physical entries. Certification
uses the printed upper entries and restores the full independently integrated
Gaussian block; the binary density payload must retain both triangles. A
periodic native `Tr(P S)=32` gate additionally checks the inferred convention.

### Basis projection and equivariance

Three frozen Gaussian fits to the richer Si numerical PAOs failed the original
radial/kinetic thresholds for the second s and p channels (roughly 1.7-1.8%
radial L2 errors). No fitted Si basis was exported and thresholds were retained.
Instead, `src/analysis/cp2k_projection.py` evaluates the **numerical PAO to
Gaussian cross overlap directly**: angular integrals are analytic, radial
integrals use piecewise quadrature, and target p/d integrals are analytic center
derivatives. This avoids approximating the PAO by Gaussians while keeping
CP2K's standard target basis.

The implementation checks independent Gaussian integrals, including coincident
centers, and full rotation/reflection covariance. The physical projection is
performed for each k point:

```text
T(k) = solve(S_CP2K(k), S_CP2K,OpenMX(k))
P_raw(k) = T(k) P_OpenMX,total(k) T(k)^dagger
```

The cross overlap has the actual orbital ordering, radial normalization,
periodic images and Bloch phases. CP2K builds neighbor lists from central-cell
wrapped positions, whereas the source OpenMX matrices refer to input positions.
The source and target centers must remain distinct in the cross integrals.
Otherwise the associated atom-dependent Bloch phases are wrong; a first native
overlap check caught precisely this mismatch before any guess was exported. Its joint Gram residual
`S_OpenMX - S_cross^dagger solve(S_CP2K,S_cross)` must be positive semidefinite
within the frozen tolerance. Target overlap eigenvalues below 1e-5 are rejected
by this initial implementation; rank-aware transfer is not implemented.

For a directly predicted density, first apply global directed-pair Hermitian
projection. The source closed-shell parser convention is one spin channel, so
multiply by two **once**. Projection can lose charge and predictions can have
unphysical occupations. The registered repair diagonalizes
`S_CP2K^(1/2) P_raw S_CP2K^(1/2)`, shifts all k-point occupation eigenvalues by a
shared scalar, clips to [0,2], and fixes the k-averaged electron count to 32.
Report the raw charge, raw occupation extrema and repair norm. This is a
covariant nearest-density repair, not proof that the learned density is accurate.

The selected checkpoint is the paper's silicon run **92sqde98**, named by
`paper/data/validation_relative_metrics/silicon_perturbed.json` and
`scripts/download_paper_model_configs.py`. It is
`silicon_hdo_energy_stage2_47h25/leafy-sweep-15-half_gt-coef3em3-lr3em4-47h25/best_model.pt`.
Selection is inherited from the paper, not optimized on this SCF result. The
script requires strict checkpoint loading and decoded-density covariance under
a proper rotation and inversion. It applies native xyz to e3nn yzx geometry
conversion once, then returns the density to OpenMX's native orbital convention.
A geometry-only radius graph is used; reference matrices are not model inputs.

The planned five cases are ATOMIC, original native CP2K restart, byte-exact
serialized native restart, projected reference OpenMX density, and projected
MANDALA density. Preparation is gated on the native layout, PAO source overlap,
quadrature, joint Gram, and model covariance checks. Failed gates must be
resolved before submission. The matched calculation runner is
`scripts/run_cp2k_guess_comparison.py`.

### OpenMX grid evaluation

The original log gives a 64x64x64 grid with origin in bohr
`(0.080367979406, 0.080367977044, -1.205519688735)` and diagonal steps
`0.160735958813`. A scalar density is formed from the source PAOs as
`rho(r) = sum P(i,j,L) phi_i(r) phi_j(r-L)` with periodic images and the correct
spin sum. `src/analysis/openmx_density_grid.py` already implements numerical PAO
reconstruction. Its radial interpolation is currently linear and is not
bit-identical to native OpenMX interpolation.

For native OpenMX evaluation, insert the predicted directed density blocks into
OpenMX's `CDM`, respecting its atom/neighbor/spin indexing, then call the native
`Set_Density_Grid` and cube-output path before an SCF update can replace the
guess. `OutData.c` doubles the nonmagnetic spin-0 grid for total density.
This requires an import/export hook; copying a predicted matrix into a restart
filename is insufficient. Native `Si_rst/*.crst*` files contain density minus
the atomic contribution; `*.adrst*` contains the atomic contribution. Their
per-rank decomposition must be reconstructed before comparison. They are not
standalone total-density cubes.

The installed CP2K 2023.1 SCF interface does not provide the development
`EXTERNAL_DENSITY` cube route. Thus exporting a source-grid cube alone would
not initialize this executable. A future cube workflow must evaluate on CP2K's
exact finest grid (origin, cell, dimensions, units), verify integrated charge,
and explicitly test the importing version. The direct `.kp` AO route is the
current silicon integration target.

Small ledgers and summaries are retained locally under
`researchops/runs/cp2k_atomic_baselines_20261103` and
`researchops/runs/cp2k_si_density_transfer_20261103`; large matrices, source
snapshots and outputs remain in the ROSI `cp2k_initial_guess` namespace.


### Current validation state and historical checkpoint normalization

The ROSI numerical projection job 824398 completed successfully. The native
Gaussian overlap maximum discrepancy was 2.71e-9, Fourier/image repacking
error 2.33e-16, and native `Tr(P S)` was 32.00000000049346. The official PAO
onsite overlap discrepancy was 7.07e-8; order-4 versus order-6 mixed radial
quadrature differed by 2.64e-13. The joint Gram residual minimum eigenvalue
was positive (0.0002913), with a mean AO representation loss of 0.00942.
Decoded-model covariance relative error was 2.48e-6 for the tested rotation
and zero for inversion. These certify numerical conventions; they do not
establish model accuracy or SCF acceleration.

The nine focused implementation tests passed locally and then directly on
Terok (no Slurm), including the independent PySCF integral check. The Terok
run used four CPU cores and recorded its container/runtime separately.

**The first learned export is held pending a normalization correction.** It
applied the present parser's spin-sum factor 2 to the historical paper model,
yielding 58.36 raw source electrons and a 51% metric repair. Repository history
shows that before commit `a0ceeb95558dcfb8fe8f6efe67167471ecd31d56`
(30 September 2026), the parser returned `D + D.T`; it now returns
`0.5*(D + D.T)`. The selected checkpoint predates that change. Thus the model's
training density convention must be established independently of the current
parser. A closed-shell historical total-density model must not be doubled
again; current native spin-0 reference matrices still require factor 2.
This is being checked against reference blocks and the canonical evaluation
path on Terok. Do not label the factor-2 learned export a validated guess or
claim its repaired density demonstrates model accuracy.


The Terok normalization diagnostic completed. Geometry-only and DatasetFactory
inference agree (raw source trace 29.1818), ruling out the new structure-only
input builder as the source of the discrepancy. Against the correctly doubled
historical reference density, relative MAE is 57.1%, substantially above the
paper's aggregate validation metric of 1.95%. This single structure need not
match the aggregate metric, but the difference warrants a historical-code
replay before claiming a faithful paper-model SCF benchmark. Mapper constants
are not stored in this checkpoint; no saved mapper buffer was discarded.
A strict replay using source commit `e78a7c04e18d005d47def9eb7c9bd0f68b168d99`
(chosen by checkpoint date) is being run separately on Terok.


### Faithful historical replay established on Terok

The same checkpoint under the frozen July source commit
`e78a7c04e18d005d47def9eb7c9bd0f68b168d99` gives relative density MAE
**1.8804%** on this structure and a source electron trace of **31.97614**.
This agrees in scale with the paper's 1.952% aggregate validation error.
The exact-geometry replay uses the CIF equivalent to `conf.dat`, loads all
learned weights strictly, and passes rotation/inversion checks (maximum decoded
rotation error 4.57e-6; inversion error zero). Historical density targets are
restricted total density, so no extra factor 2 is applied to this prediction.
The current parser's reference spin-0 density still needs factor 2.

This establishes a numerical compatibility discrepancy in the current model
restoration path, independently of the PAO/CP2K projection. Its precise framework
fix is not part of the adapter yet. The maintained preparation script refuses
to infer this exact paper-checkpoint hash through the known unfaithful path.
Use the frozen historical prediction with `scripts/project_cp2k_density_guess.py`;
this exporter requires strict-load and proper/improper transformation evidence,
an explicit density convention, and checks charge after finite native-image
packing. It records all input/output hashes. This avoids a misleading learned
SCF result based on a distorted checkpoint replay.


### Corrected AO export and prepared comparison

The faithful historical density was projected and exported directly on Terok.
Its raw target-basis electron count was 31.95597. The fixed metric occupation
repair changed the density by 8.24% in metric Frobenius norm, yielding 32
spin-summed electrons. Importantly, the electron-count check **after finite
CP2K native image packing** also gives exactly 32. The projected restart SHA256
is `8d14aba145aded242b5df2f0e81ad832629588a4c23fa65ce34da47cf956a007`.
The repair and basis projection remain part of the guess's definition and must
be reported when assessing model utility.

A fresh five-case job is prepared at ROSI
`/bigdata/casus/wdm/hamiltonian_learning/cp2k_initial_guess/si_guess_comparison_20261104_v4`.
`run_comparison.sh` is the frozen zero-argument Slurm launcher; it runs ATOMIC,
original-native restart, serialized-native restart, projected OpenMX reference,
and corrected historical-MANDALA restart with identical CP2K 2023.1 settings.
`prepared/READY.json` pins the corrected density hash. The job has **not** been
submitted, following the latest instruction to perform current tests on Terok.
The learned initial guess's SCF iteration count and net acceleration are still
unmeasured. A source-code compatibility regression has been isolated by replay;
its general framework fix remains separate from this pinned benchmark path.


## Mapping used in the executable silicon benchmark

The map transfers the density operator between two different finite AO spaces.
The source is numerical `Si7.0-s2p2d1`; the target is CP2K's normalized Gaussian
`DZVP-MOLOPT-SR-GTH`. Both have 13 AOs per Si, but equal dimensions do not make
them equivalent bases. The source PAO and target GTH pseudopotentials are not
converted into each other; CP2K retains its target Hamiltonian during SCF.

For each source orbital, `phi = R_nl(r) Y_lm(real)`. Each target radial function
is a contraction of normalized primitives `r^l exp(-alpha r^2)` multiplying
its real spherical harmonic. Primitive/contraction normalization and the
appropriate angular factors are included before overlap integration.

The native angular orders for this case are:

| shell | OpenMX | CP2K |
| --- | --- | --- |
| p | px, py, pz | py, pz, px |
| d | z2, x2-y2, xy, xz, yz | xy, yz, z2, xz, x2-y2 |

Repeat the permutation for each radial shell. This angular permutation alone
cannot convert the numerical PAO radial functions to the target Gaussians.
The MANDALA/e3nn Cartesian xyz-to-yzx conversion is a separate boundary and
is applied once, then undone when decoding back to OpenMX orbitals.

For every source/target center pair and periodic lattice image L, integrate
`S_ts(L)[mu,nu] = <chi_mu,0 | phi_nu,L>`. The implementation performs the angular
integral analytically. Expanding the displaced Gaussian yields modified
spherical Bessel functions and one-dimensional radial integrals; Gaussian p
and d integrals are analytic center derivatives of the s-Gaussian integral.
The PAO radial interpolant is sampled with piecewise Gauss quadrature. No
Gaussian fit of the PAO and no density-grid interpolation enter this route.
The independent Gaussian-limit check uses PySCF integrals.

Fourier transform the directed image blocks with one consistent convention,
`S_ts(k) = sum_L exp(2*pi*i*k.L) S_ts(L)`, and likewise for source P and target S.
Source and target centers use their own image gauges: OpenMX input positions
versus CP2K central-cell neighbor-list positions. Cross integrals explicitly
retain that distinction. Dropping it introduces atom-dependent phase errors.

In each Bloch sector the least-squares fit is:

```text
min_Ct || Phi_s Cs - Chi_t Ct ||^2
S_tt Ct = S_ts Cs
T = solve(S_tt, S_ts)
P_t_raw = T P_s T^dagger
```

Thus this is a multicenter orbital projection, rather than an exact similarity
transform. In particular, using the covariant Hamiltonian matrix transformation
for a contravariant AO coefficient density would be incorrect. Projection uses
the actual nonorthogonal AO metric and retains no target eigenvalue below 1e-5;
this case has minimum eigenvalue 0.00706 and needs no rank truncation.

Equivariance follows from `S_ts' = D_t S_ts D_s^dagger` and
`S_tt' = D_t S_tt D_t^dagger`, which imply `T' = D_t T D_s^dagger`.
Complete angular multiplets, shared radial treatment across magnetic components,
and parity `(-1)^l` are retained. Cell vectors rotate with atom coordinates,
while fractional k points and image labels remain consistent. Explicit proper
and improper transformations are tested. The geometry-dependent multicenter
map need not be diagonal in l; its full transformation law is the requirement.

Before transfer, globally Hermitian-project paired image blocks:
`P(i,j,L) = 0.5*(P_raw(i,j,L) + P_raw(j,i,-L)^dagger)`.
Never replace this with self-symmetrization of an offsite block. The historical
paper density is already restricted spin-summed; the current native spin-0
reference is doubled once. Both use 32 total electrons for this neutral Si cell.

Finite-range and learned density blocks need not yield admissible occupation
eigenvalues. Diagonalize the metric density `X_k = S_tt^(1/2) P_t_raw S_tt^(1/2)`.
Apply one shared scalar occupation shift over all k points, clip eigenvalues
to [0,2], and choose the shift so the equal-weight k-averaged trace is 32.
Transform back with `S_tt^(-1/2)`. This is the fixed nearest metric-density
repair, not an energy minimization. It preserves covariance because it acts
spectrally and treats degenerate eigenvalues identically. Its 8.24% norm change
for the faithful MANDALA guess is part of the benchmark definition.

Finally inverse-Fourier transform and pack into the certified 160 native CP2K
image records. Nonzero-image matrices are generally nonsymmetric individually;
Hermiticity pairs opposite images. The symmetric sparse neighbor-list and text
printing conventions must be handled separately. Replace only the template's
density payload, retaining verified basis metadata, and use `SCF_GUESS RESTART`
with the explicit `.kp` filename. Re-read the packed density and check charge;
the learned export gives exactly 32. Native restart original-versus-serialized
controls check that CP2K actually consumes the intended format.

The native template contributes basis/image metadata and an overlap metric;
its converged density values are not used in the learned guess. The original
converged restart is included only as an import-control arm. Initialization
metadata/overlaps should ultimately be extracted before an SCF loop rather
than requiring a converged preliminary calculation. The current integration
work is not an end-to-end acceleration measurement.

On 4 November the user explicitly authorized execution of the frozen five-case
comparison. ROSI job **825130** was submitted with four CPUs and a 15-minute
limit. Results are recorded separately from submission state.


Execution update: job 825130 was observed running on cge054, with the atomic
arm at SCF step 10. Subsequent result retrieval was interrupted by SSH public-key
authentication rejection on rosi4 and rosi5, including from the UTS gateway.
The scheduler submission and running observation are confirmed; final SCF counts
are not yet retrieved. Preserve this distinction rather than inferring outcomes
from submission success or earlier controls. The user has been asked to check
ROSI login access. The submitted job was not cancelled or resubmitted.


## Completed matched SCF comparison (ROSI job 825130)

The five frozen calculations completed successfully on cge054 with CP2K 2023.1,
four CPU threads, identical input scientific settings, and fresh directories.
SSH access was restored and the complete scientific summary retrieved.

| Initial guess | SCF steps | CP2K subprocess wall time, s | Final energy, Ha |
| --- | ---: | ---: | ---: |
| ATOMIC | 14 | 86.27 | -31.489604395926 |
| Original native converged CP2K density (control) | 2 | 12.65 | -31.489608942108 |
| Serialized native CP2K density (control) | 2 | 12.85 | -31.489608942108 |
| Projected reference OpenMX density | 12 | 68.36 | -31.489608872271 |
| Projected historical-paper MANDALA density + fixed metric repair | 14 | 78.98 | -31.489608760596 |

**The MANDALA guess did not reduce the number of SCF steps for this structure:**
both ATOMIC and MANDALA required 14. The projected reference OpenMX density
saved two steps (12 versus 14), demonstrating a modest cross-code initialization
benefit for the reference density. The native controls needed two steps and
produced identical final energies, validating the native restart serialization.

Every case converged, reported 32 electrons, ended normally, and passed the
registered endpoint energy tolerance of 1e-5 Ha relative to ATOMIC. Every
restart arm reported RESTART without missing-file fallback. The ATOMIC stopping
energy differs from the native-control energy by 4.546e-6 Ha; the MANDALA energy
differs from that native-control endpoint by 1.815e-7 Ha. Equal EPS_SCF does not
mean the stopping energies are numerically identical across initial guesses.

CP2K subprocess times were 86.27 s (ATOMIC) and 78.98 s (MANDALA), but these are
single timings and exclude inference, integration, occupation repair and IO.
The prototype's overlap/projection preparation itself took minutes. These
numbers do not establish end-to-end computational acceleration. The result is
an implemented and verified density-transfer route with a negative SCF-step
acceleration result on this one snapshot. It does not establish that other
structures or models cannot benefit.

The reference-density advantage is small enough that learned operator error,
finite-range density representation, and its physical occupation repair could
remove it. The present experiment does not isolate their individual effects;
it would be incorrect to attribute the negative result to a single cause.
Any subsequent solver/repair changes or wider benchmark should be registered
as a new comparison rather than used to replace this frozen result.

Compact results and the acceptance contract are retained at
`researchops/runs/cp2k_si_density_transfer_20261103/comparison_v4/`.
Large outputs remain in the ROSI run directory named above.
