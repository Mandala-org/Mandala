# Test-suite audit and dated verification records

The inventory section records the 22 October 2026 audit, not the current suite.
Scientific follow-up and consolidation records below have their own dates,
scopes and limitations. These are executed-check records, not measured line or
branch coverage and not claims that an automated CI service exists.

## Inventory audit — 22 October 2026

The baseline on `develop` (`bc64e04`) contained **433 test definitions in 93
nonempty test modules, expanding to 752 pytest cases**. Three Luna agents at
medium reasoning reviewed disjoint areas; a fourth review assignment covered
scripts and analysis. The parent reviewed their recommendations and changes.
Every original definition has an individual entry in
[`test_inventory.json`](test_inventory.json), including all expanded case IDs,
reviewer, rationale, final decision, and replacement IDs. The inventory also
lists new tests and test-source checksums. It is a dated snapshot, not an
assertion that future edits have been audited.

Review included fixture data, delegated assertion helpers, parametrizations,
production call paths, and overlap with surviving tests. A helper comparing
outputs, parameter/input gradients and Adam updates is substantive even when its
caller contains no `assert` statement. Constructor, rejection and dispatch tests
were retained where they establish a distinct contract; they are not counted as
numerical evidence for a model's mathematics.

## Inventory

| Area | Original definitions |
| --- | ---: |
| Network components, model, objectives and controllers | 207 |
| Data, parsers, geometry, caches and serialization | 82 |
| Training and experiment scripts | 49 |
| Core representations and sparse algebra | 24 |
| Training workflows | 21 |
| Integration | 16 |
| Physics | 12 |
| Analysis units and numerical analysis | 12 |
| Equivariance | 7 |
| GPU | 2 |
| Naming utilities | 1 |
| **Total** | **433** |

The revised inventory contains **432 definitions in 94 nonempty modules,
expanding to 722 cases**. Fifty-one original definitions were strengthened or
consolidated, fourteen redundant definitions were removed, and fourteen new
definitions were added. The former standalone S2 factory test is incorporated
into the four-activation factory grid. The empty H2O workflow file was removed.
The smaller expanded case count primarily comes from removing a 48-case MLP
shape-only grid whose complete variant/depth/final-activation grid remains in
the equivariance test, now also asserting output shape.

## Findings and changes

| Weakness | Discriminating coverage now present |
| --- | --- |
| Equivariance test without a transformed-output comparison; scalar-only rotation fixture | Actual scalar representation action; a literal nontrivial p-orbital quarter-turn, nonsymmetric s/p matrices and preserved periodic reverse records |
| Paired encoder/decoder or basis roundtrips can cancel matching errors | Fixed s–p coupled coefficients, fixed p–d forward basis coefficients and asymmetric dense-matrix block slices, alongside general roundtrips |
| Encoder, normalization and aggregation shape checks | Deterministic species routing and VJPs, receiver scatter with unequal degrees, explicit graph normalization and activation formulas |
| Force and stress checks only required finite/nonzero outputs | Float64 energy finite differences; active diagonal and shear stress components in a skew cell; actual force-loss parameter-gradient contribution |
| Training that merely completed without an exception | Real optimizer-step count, changed parameters, finite train/validation losses; all three recipes' s/p objective values, label dependence and trainable readout gradients |
| Expected training loss copied production slicing loops | Closed-form per-operator label offsets with known MSEs, plus explicit weighted loss values and gradients |
| Identity-only irrep errors permit a hardcoded zero result | Nonzero independently calculated per-irrep absolute and relative errors |
| Parser and DOS checks stopped at keys/shapes | Literal reference energy/block values, directed periodic edge identities, exact grid and analytic two-level Gaussian DOS |
| Serialization checked only energy/electron count | Full matrices, edges, lookup, geometry, basis, orbital layout and distinct partial-coverage masks for every operator; literal decoder payload independent of the serializer |
| Heatmap scaling test only checked an image existed | Captured displayed arrays and exact common color limits, with the extreme value independently placed in each panel |
| Reporting ignored complex errors | Imaginary-only SOC error regression with hand-calculated MAE/MSE; invalid matrix supervision cases |
| Weak continuation and search dispatch checks | Three checkpoint modes, changed-input hashes, every split-overlap pair, immutable continuation inputs, Optuna pruning decisions/guards, objective return and trial metadata |
| Duplicate launcher lifecycle tests | One four-model lifecycle grid checks exact recovery paths/status and both epoch-limit and minimum-LR completion |
| Cache tests conflated format migration and configured preprocessing | Historical version invalidation remains; actual density symmetrization and distance normalization independently check the affected cache layers |
| Missing mapping could silently skip a test | Every ordered species pair in the factory union must map successfully; exact type labels and bounds replace a tautological membership check |

Confirmed redundant encoder/update shape tests, tensor-product constructor/shape
checks, an inverse-only converter repeat, sparse trace repeat, orbital substring
checks, a force smoke, and a pure PyTorch GPU allocation check were removed.
The inventory names the substantive survivor for each removal. Independent
parameter grids, error contracts and optional source comparisons were preserved.

## Production bugs exposed by the new tests

1. Shift-resolved reporting cast complex matrices to real numbers, discarding
   imaginary SOC errors. It now computes complex128 differences and
   magnitude-based MAE/MSE.
2. `Snapshot` reconstruction omitted persisted orbital coverage masks. Loading
   now restores each matrix's masks on the requested device.
3. Euclidean norms of identically zero irrep channels produced NaNs when force
   training differentiated gradients again. The affected shared activations now
   use an explicit zero first/second derivative convention at exact zero.
   Forward values and nonzero-vector derivatives remain unchanged. This is a
   point convention: the Euclidean norm has no ordinary derivative at the
   origin, and its Hessian still grows near zero.
4. Explicit `objective.kind` and legacy `loss_l1_fraction` could disagree about
   which loss retained gradients. Aggregation could select a detached MSE/MAE
   and silently train with zero gradients. One resolved mixture now controls
   gradient retention, aggregation and pair/irrep diagnostics.

The complex-error and mask regressions failed before their fixes. Higher-order
and explicit-objective gradient regressions exposed the corresponding training
failures. Fixed-positive diagnostic readout initializations were removed after
tracing the objective bug; the workflow checks use normal seeded initialization.

Four temporary fault injections independently confirmed test sensitivity:
identity/no-op block rotation, discarded coverage masks, raw zero-vector norms,
and discarded imaginary matrix errors each failed the intended regression.
These checks changed no repository source files. They are focused fault
injections, not a whole-program mutation score.

## Verification and limits

Only test files directly relevant to the edits, plus frozen reference and hybrid
regressions, were executed together. Full-suite **collection** verifies the
inventory but does not execute the suite. The combined run across 43 affected/regression files passed **413 tests**,
with **one GPU skip**, in 242.51 seconds (46 warnings). After strengthening the
W&B forwarding assertion, that file was verified again: **20 passed** in 1.44
seconds. Black, Ruff and the whitespace check passed. All **722 final cases**
collected successfully; the inventory accounts for every original and current
case.

The five committed upstream numerical fixtures and their tolerances were not
changed. Optional live differential tests remain available with
`MANDALA_DEEPH_REFERENCE`, `MANDALA_DEEPHE3_REFERENCE` and
`MANDALA_MACEH_REFERENCE_ROOT`; native preprocessing also accepts
`MANDALA_MACEH_REFERENCE`. Their fixtures verify pinned-source integrity before
comparison. Missing checkouts remain an explicit coverage limitation, not a
passing source comparison.

CUDA is unavailable on this machine. The remaining GPU test exercises actual
E3GNN training gradients and strict reload, and now copies its dataset to avoid
mutating a session fixture. It is gated by `MANDALA_RUN_GPU_TESTS=1` and fails if
that opt-in is set without CUDA. GPU execution and multi-version CUDA/PyTorch
compatibility remain unverified. The global large-data gate uses
`MANDALA_LARGE_DATA_ROOT`.

Tiny deterministic fixtures establish mathematical and workflow contracts.
They do not establish publication accuracy, large-system scaling, convergence,
or generalization. No long training or cluster workload was run for this audit.
The timing utility's saved records are keyed only by node ID; its `--resume`
results can be stale after source edits and must not be treated as fresh audit
results.

## Scientific contract follow-up — 23 October 2026

This follow-up adds the ten high-priority scientific test families below. The
inventory above remains the dated 22 October snapshot. The new tests use tiny
complete model graphs, raw native records, independently calculated physical
quantities, and real optimization. No new case depends on a source checkout,
CUDA, a cluster, or downloaded datasets.

| Family | Evidence and scope |
| --- | --- |
| O(3), mixed parity, and SOC | Unsorted even/odd scalars, vectors and tensors; proper rotations, improper rotations and inversion before and after asymmetric Adam updates. Native s/p blocks use independent Cartesian orbital actions and Pauli spin rotations; full global models are checked again after training. |
| Atom and edge ordering | Raw geometry through all three reference presets and all four hybrid examples; every directed physical block, scalar objective, position gradient and parameter gradient must agree after relabeling. |
| Raw-source reproduction | Pinned upstream native-file graph builders, s/p labels and nonzero periodic images; exact initial parameters, outputs, every parameter gradient and three fresh Adam updates. DeepH exercises both computed and stored frames. |
| Composition | Configured endpoint tensor-product primitives versus the composite, sum/mean reducers, optional species connections, normalization and residuals; every used parameter/input gradient and two Adam updates. Attention uses an independent per-receiver/per-head softmax loop, unequal degrees and an empty receiver. |
| Periodicity | Atom wrapping, unimodular lattice changes, and a replicated supercell preserve every mapped directed block. The extensive scalar objective scales with the cell. |
| Heterogeneous batching | Unequal cells, mixed s/p/d species, SOC, periodic images, partial masks, and empty message graphs; individual versus batched outputs, element-weighted losses, parameter gradients and position gradients. Pooled batch normalization has its own explicit statistics oracle. |
| Physical operators | Complex, noncommuting 2×2 Hamiltonian/overlap operators; literal Fourier sums, independent SciPy generalized eigensolves, occupation-built density, electron count, energy, native complex export/reload, and global reversed-pair adjoint projection. Spectral losses and first/second derivatives agree with independent finite differences. Invalid overlap fails unless cleanup is selected explicitly. |
| Geometry and derivatives | Near-collinear local environments, signed-zero onsite harmonics, mixed force/parameter gradients, and periodic strain first/second finite differences. A smooth cutoff message map approaches the rebuilt-neighbor boundary; an ordinary Gaussian map explicitly does not. |
| Supervision isolation | Real dataset construction and the training loss remain unchanged when held-out raw labels and masked training entries are poisoned. Fitted buffers, gradients and the first Adam update agree exactly; changing an unmasked training label is a positive control. |
| Learning | All three reference models and one mixed attention model train all parameters and three independent operator heads for 500 Adam steps on three raw geometries. Each Hamiltonian, density and overlap error must fall below 20% of initialization and 60% of a constant-per-entry/edge-index baseline. Targets depend on distance, species and angular direction. |

The source-only generator is
[`generate_raw_training_fixtures.py`](../scripts/generate_raw_training_fixtures.py).
It records pinned commits, hashes of all upstream Python sources, package
versions, precision, seeds, raw native records and complete parameter
coordinates. The three compressed fixtures are about 4.4 MiB together. DeepH
uses source float32 initialization followed by double-precision preprocessing
and evaluation; the other two use double precision throughout. Regenerate with
the three explicit source roots, for example:

```bash
PYTHONPATH=src:. python scripts/generate_raw_training_fixtures.py \
  --deeph-root /path/to/pinned/DeepH \
  --deeph-e3-root /path/to/pinned/DeepH-E3 \
  --mace-h-root /path/to/pinned/MACE-H
```

MACE-H's raw-source comparison deliberately uses identity output statistics to
isolate model/training mathematics; train-only fitted statistics and label
isolation have separate end-to-end tests. It does not reproduce upstream
full-dataset leakage. The small SOC attention hybrid explicitly completes its
fixed decoder layout with an irrep linear readout: structurally unreachable
channels are zero, rather than being silently removed or called spinless.

The probes exposed and motivated these production fixes:

- FiLM's odd scalar conditioning and shifts now respect inversion; learned
  residual/bilinear scales share one parameter across every irrep's magnetic
  components. Scalar gating preserves arbitrary irrep order, and fitted means
  cannot introduce a pseudoscalar bias.
- Zero channels and onsite displacements have explicit finite derivative
  conventions in magnitude gates, norms, harmonics and graph normalization.
  DeepH's differentiable and ordinary harmonic paths share the upstream
  north-pole zero-displacement convention.
- SOC frame rotation promotes real input blocks to a complex dtype instead of
  dropping the spin unitary's imaginary entries.

Old coordinate-wise residual-scale checkpoint tensors are accepted only when
their magnetic components already agree. Symmetry-breaking old tensors are
rejected with an explicit migration message; projecting them changes the
model. Old optimizer states with coordinate-sized scales require a separate
migration or a fresh optimizer. Odd-channel FiLM behavior is intentionally
corrected. Ordinary nondegenerate reference mathematics remains covered by the
five earlier immutable source fixtures, whose data and tolerances are unchanged.

An independent critic rejected superficial shared-helper expectations and
verified that faults in odd FiLM conditioning, trained residual scales,
periodic shifts, label masks, and attention normalization fail the intended
tests. A further Fourier-sign fault fails the independent operator oracles.
These are focused fault injections, not a comprehensive mutation score.

The tests also expose two **limits**, rather than treating them as guarantees:
DeepH's source frame selection depends on record order at exact distance ties,
and deleting a neighbor with a nonzero Gaussian message causes a discontinuity.
Those cases are named characterizations, not skipped failures. Collinear or
isolated bond-frame environments are rejected explicitly. Full O(3) model
claims apply to the global equivariant presets; local-frame reference behavior
has its separate geometry and source-parity contracts. Arbitrary externally
imported anisotropic statistics can also violate covariance.

Only targeted tests relevant to these changes were executed. CPU examples
establish numerical contracts and small learning capability; publication
accuracy, large-system generalization, GPU behavior and distributed execution
remain outside this validation.

All **85 new cases** (26 test definitions) passed across focused runs: 21
symmetry/checkpoint cases, 32 geometry cases, 11 composition/operator cases,
17 preprocessing/batching/isolation cases, and four short learning experiments.
The learning experiments took 374.76 seconds together on this CPU. The targeted
regression run passed 141 cases with 42 optional source-dependent skips;
enabling the pinned checkouts then passed all 78 cases in the four relevant
live-oracle/data modules with no skips. These overlapping run counts must not
be added as distinct tests. Ruff, Black formatting and whitespace checks passed.


## Repository consolidation — 2 November 2026

This verification covers the pending scientific contracts, shared source/basis
loading, physical-entry metrics, optional matrix-loss/radial gates, pilot and
ablation tooling, checkpoint precision, and continuation. It selects 37 relevant
files explicitly; unrelated suites and scientific datasets were not executed.
The four 500-step learning cases were run separately from the remaining cases.

A new float64 regression trains through the public CLI with only
`training.dtype`, saves a real best checkpoint, resets the ambient dtype, and
compares reloaded validation errors with training's recorded errors. It exposed
an evaluator that read only the root dtype and a configuration round-trip that
compared `torch.float64` with the string `float64`. Both now resolve precision
semantically; conflicting precisions remain errors.

Documentation now distinguishes historical inventories from current checks,
execution records from scientific evidence, and original registrations from
later resource amendments. Obsolete handoff/job-state prose was removed.
Frozen scientific sources, registered configurations and checkpoints were not
changed by consolidation. The original five supplied-feature oracles and the
three pending compressed raw-native oracles retain their data and tolerances.

The remaining selected cases passed **516 tests**, with **10 skips** and four
learning cases deselected, in 427.15 seconds. The separate learning run passed
all **four cases** in 252.03 seconds: **520 passed and 10 skipped** across the
37-file selection. Skips comprise four optional native-source comparisons,
four optional MACE-H executable comparisons, and two GPU continuation cases.
The pinned DeepH-E3 checkout was enabled; frozen offline oracles all ran.
A separate overlapping precision/configuration check passed 17 cases and is
not added to those totals.

The offline SOC lab verifier passed **96 spin/orbital cases** against MANDALA
and independent NumPy equations. Twelve launchers passed `bash -n`; 50 selected
Python files passed Ruff and pinned Black 24.3.0 checks. Documentation links and
Git whitespace checks passed. These CPU checks do not establish GPU/runtime
compatibility, publication accuracy, convergence, or generalization. No cluster
workload was launched for this consolidation.
