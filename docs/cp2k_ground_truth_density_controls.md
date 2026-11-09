# Actual OpenMX density controls

Registered on 8 November 2026, extending
[the paper-model comparison](cp2k_paper_method_comparison.md). This experiment
uses the stored OpenMX density blocks, rather than replacing them with a
zero-temperature density reconstructed from the reference Hamiltonian.

## Source and interpretation

The specimens remain eight-atom Si090 and 150-atom SiO₂ 13_67164000. Both
source calculations use 300 K electronic occupations. The archived Si density
was integrated on an 8×8×8 mesh and must not be folded as repeated Gamma data.
A fresh OpenMX Gamma calculation at the same Si geometry/basis/PP was therefore
generated with the pinned OpenMX 3.9 source and DFT_DATA19 Si7.0-s2p2d1 /
Si_PBE19 files. Job 834158 finished normally; its last dUele was 3.25e-10 Ha
and printed NormRD 7.87e-9. This regenerated Gamma reference is distinct from
the paper's original native-k Si density. SiO₂ uses its archived native Gamma
density, Si9.0-s2p2d1 / O7.0-s2p2d1, unchanged.

At Gamma, D(i,j) repeats across the recorded images of a pair. We select one
copy from **any** recorded image and verify all others agree; image zero need
not exist. Overlap blocks, in contrast, are summed over images. Native spin-0
density is multiplied by two, with the source trace checked against 32 or 800
valence electrons. Imaginary densities and non-Gamma source meshes are rejected.

For SiO₂, 18,260 of 22,500 atom pairs occur in some image, but only 9,860 occur
at image zero. Unrecorded pairs are explicitly zero, matching the finite-block
supervised representation. Such pairs do not contribute to the density with
confined source PAOs. This specifies the archived **density function**, but does
not recover every nonlocal coherence of the full KS density operator. Basis
replacement/operator projection can depend on that zero-extension convention.
This limitation is part of the experiment, not a claim of unique exact
operator conversion.

An earlier attempted H/Fermi completion differed from recorded blocks by
1.08e-6 and failed its registered 1e-7 consistency gate. It is not used here;
that threshold has not been relaxed. No source occupations are rounded and no
source occupied-space projector is substituted for the stored matrix.

## Matched transfer experiment

Let P be the finite-block source density after the verified spin conversion,
B the numerical-PAO/Gaussian cross overlap, and S the native target overlap.
The two raw maps are

```
T = S_retained^(-1) B
P_projection = T P T^T
P_coefficient = U P U^T
P_guess = Ne / Tr(P_raw S) * P_raw
```

The retained target directions have overlap eigenvalues greater than 1e-5,
matching the SCF basis threshold. U is the supplied supervisor converter's
actual basis-layout angular permutation. Both guesses receive the same
explicit scalar charge correction. Raw trace, normalization factor and target
metric spectrum are reported. There is no occupation rounding, clipping,
PSD repair, H-completion or target occupied-space orthonormalization. The
metric spectrum of a finite-block representation is diagnostic; it is not
silently treated as a complete physical occupation spectrum.

Projection uses the previously certified cross overlaps. The coefficient
route uses the same fixed 24-primitive shared fitted library as the earlier
comparison. This tests the supervisor's **coefficient-map mathematics** through
our native-density serializer, rather than running his LS_SCF serializer or
his natural-orbital WFN route. No basis fit, model or SCF parameters are
selected using these outcomes. The fitted library's diagnostic radial/kinetic
quality limitations remain unchanged.

Both maps run in that fitted basis. Projection into standard MOLOPT has a
separate atomic baseline. There are four atomic baselines and six stored-D
guesses: three per material. A matched baseline is necessary because this
uses CP2K's native AO-density entry point, rather than its ordinary Gamma MO
restart path. Comparisons with the earlier WFN results must retain that
protocol distinction.

## Why native AO-density restarts

With the ordinary diagonalization settings used in the earlier benchmark,
Gamma WFN import orthonormalizes the orbitals and reapplies occupations before
constructing density. A persistent occupation LIST would instead constrain
the subsequent SCF problem. We use real, one-point Gamma k-point density
restarts (`.kp`) to supply the density itself while leaving subsequent SCF
occupation updates free. See the pinned
[CP2K 2023.1 initialization implementation](https://raw.githubusercontent.com/cp2k/cp2k/support/v2023.1/src/qs_initial_guess.F)
and [k-point input documentation](https://manual.cp2k.org/cp2k-2023_1-branch/CP2K_INPUT/FORCE_EVAL/DFT/KPOINTS.html).

The writer uses target-generated basis metadata and whole-atom image storage
masks. No template density values enter a transferred GT guess. Whole-atom
Frobenius masks preserve complete angular multiplets. The native template's
values are used only for an identity/reimport control. The small interface
probe, job 834151, passed: 370 image payloads were repacked with zero numerical
change, the trace was exactly 32, binary readback was identical, and CP2K read
RESTART with 32 electrons and no atomic fallback. Five focused tests check
basis covariance, fractional values, missing-origin Gamma handling, implied
Hermitian reverse blocks and absence of silent spectral repair.

Each group repeats native repacking/import checks before releasing GT guesses.
Endpoint agreement with the extra-two-step control is required when its
atomic baseline converged. An unconverged baseline can legitimately evolve
on extra steps, so its interface checks remain payload equality, native Gamma
structure, charge and successful RESTART import. Such a baseline does not
suppress independent guesses. These adapter checks do not certify a neural
network's equivariance or quantify GPW grid errors under physical rotations.

SCF physics/settings remain PBE/GTH-q4 Si and q6 O, cutoff 600 Ry, relative
cutoff 80 Ry, integral thresholds 1e-12, EPS_SCF 1e-7, at most 200 steps,
Broyden mixing 0.15/history 16 and standard diagonalization. Final-energy
checks are within each material/basis group. Nonconvergence at 200 is not a
200-step converged result. Source temperature is an initial-guess property;
the target SCF retains the preceding experiment's integer-filling policy.

## Execution and provenance

Prepared cases and scientific results are under
`/bigdata/casus/wdm/hamiltonian_learning/cp2k_initial_guess/ground_truth_native_density_20261108_v2`.
Source verification/import preparation has a separately frozen
`ground_truth_native_density_prepare_20261108_v3` source snapshot: it releases
dense template payloads before CP2K and handles unconverged control endpoints
correctly. Running baselines and density-map mathematics were not changed.
Small workflow/source records are retained locally in
`researchops/runs/cp2k_ground_truth_density_20261108`.

| Group | Atomic | Source/import preparation | GT projection | GT coefficient |
| --- | ---: | ---: | ---: | ---: |
| Si MOLOPT | 834183 | 834199 | 834213 | — |
| Si fitted | 834184 | 834200 | 834214 | 834215 |
| SiO₂ MOLOPT | 834185 | 834201 | 834216 | — |
| SiO₂ fitted | 834186 | 834202 | 834217 | 834218 |

Dependencies use `afterany` with explicit file/interface guards, avoiding
permanent `DependencyNeverSatisfied` chains. Native images and guesses stay on
ROSI; only bounded summaries are collected. Dense image serialization and
diagnostic eigensolves are prototype overhead. This experiment measures SCF
steps, not an end-to-end speedup from a production adapter that avoids an
atomic template calculation. SCF convergence alone does not establish radial
fit or finite-grid operator accuracy.

## Partial results, 8 November 2026

| Material / target basis | Atomic steps | Stored-D projection | Stored-D coefficient |
| --- | ---: | ---: | ---: |
| Si090 / MOLOPT | 16 | 14 | — |
| Si090 / fitted PAOs | 17 | 13 | 14 |
| SiO₂ / MOLOPT | 87 | 61 | — |
| SiO₂ / fitted PAOs | Running | Waiting | Waiting |

All four completed stored-D guesses converged, imported RESTART without atomic
fallback, retained the correct electron count and finished normally. Their
absolute energy differences from their matched atomic endpoints were
5.30e-7, 8.22e-8, 5.81e-8 and 1.05e-7 Ha for Si MOLOPT projection, Si fitted
projection, Si fitted coefficient and SiO₂ MOLOPT projection respectively.
These are distinct from the earlier reference-H reconstructed-density results.
Si fitted projection saves four steps (23.5%), the coefficient map saves three
(17.6%), and SiO₂ MOLOPT projection saves 26 (29.9%). These controls demonstrate
that a source reference density can help; they do not establish a trained-model
speedup or isolate angular mapping from radial and Hamiltonian differences.

The raw mapped charges before the explicit scalar correction were 31.980205
(Si MOLOPT projection), 31.999802 (Si fitted projection), 32.003245
(Si fitted coefficient) and 798.697995 (SiO₂ MOLOPT projection). The SiO₂
zero-extended density has a target-metric minimum eigenvalue of -0.01627 and
maximum 2.16480 after normalization; no clipping was applied. Thus this is a
working initialization from the archived finite-block representation, not a
claim of an exactly representable positive full density operator.

The remaining fitted SiO₂ atomic job is 834186; preparation 834202 and guesses
834217/834218 remain guarded behind it. Collector 834222 will assemble all ten
case summaries after the jobs finish. The local bounded evidence is
`researchops/runs/cp2k_ground_truth_density_20261108/partial_results_20261108.json`.
