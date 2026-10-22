# Test-suite audit — 22 October 2026

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
