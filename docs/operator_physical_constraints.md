# Physical validity of the paper operators

This protocol addresses the reviewer question about physical constraints using
the original three publication checkpoints. The diagnostic implementation is
on `develop`. It does not retrain, select checkpoints, repair spectra, or use
test results to choose a model. Small workflow records are in
`researchops/runs/paper_physical_constraints_20261104`; large predictions and
source snapshots use the ROSI project namespace `physical_constraints/`.

## Code audit and scope

The ZnCu2Sn(SeS)2 and SiO2 models predict H only. They cannot establish validity
of **learned** S or D. Assess their H against reference S and D, explicitly
labelling the reference operators. The silicon model predicts H, S and D.
All three paper configurations disable spectral loss and density rescaling.
Silicon's band-energy guidance contracts predicted H with reference D; it
does not constrain the joint occupation spectrum or stationarity.

`E3GNN.predict_matrices(physical=True)` restores the fitted radial envelope;
the word `physical` does not imply a physical-constraint projection. Validation
can Hermitian-average outputs (`symmetrize_validation_output`). Optional
scalar density rescaling enforces one trace only. Neither mechanism guarantees
positive S, bounded density occupations, idempotency, or H/D consistency.
The spectral eigensolver can clean up S or retry with jitter when requested;
the diagnostic below deliberately applies neither, and counts failures.

Current OpenMX parsing returns Hermitian-averaged **spin=0** D. Historical paper
targets used D+D.T, i.e. twice this channel in these nonmagnetic examples.
Reference D is therefore explicitly multiplied by two; the paper model's
predicted D is already restricted total. Occupations are bounded by 0 and 2.
The physical electron target comes from the OpenMX input populations minus
system charge, independently of either reference or predicted matrix traces.

## Frozen calculation plan

1. Use strict checkpoint loading and record checkpoint, source, runtime,
   envelope and input hashes before interpreting predictions. Reconstruct
   the documented validation slices (80/20, seed 44 for chalcogenide; 90/10,
   seed 42 for silica; 80/16 from 96 structures, seed 43 for silicon). Save the full discovery
   list and split. A current population with missing files is explicitly an
   incomplete reconstruction, not proof of historical membership. These are
   validation diagnostics, not an independent test benchmark.
2. Measure all available validation structures at a frozen Gamma-centered
   3x3x3 mesh for the chalcogenide, native 1x1x1 for silica, and native 8x8x8
   for silicon. Mesh results establish behavior on sampled k points only;
   the chalcogenide's native input mesh is 11x11x11. No Brillouin-zone-wide
   certification is implied. Preserve structure and k-point failure rates.
3. Include full available reference operators and references truncated at the
   model cutoff. This quantifies support/smearing/discretization effects on
   the reference residuals before attributing them to the model.
4. Evaluate the complete raw deployment support. Also evaluate an explicitly
   labelled reference-support control: predictions on graph edges outside the
   reference's finite PAO support can be unobserved during training. Report
   their number and magnitude; never silently discard them to improve results.
   This control uses reference sparsity, so it is not independent deployment.
5. For silicon, evaluate joint H/S/D, predicted H/S with reference D, and
   reference H/S with predicted D. This separates overlap errors from density
   errors. Measure raw Hermiticity before Hermitian-averaging each Bloch
   matrix, which is the global reverse-pair projection, not individual
   offsite-block self-symmetrization.

Strict loading alone does not certify historical forward semantics. An existing
silicon investigation and local reproduction show large prediction differences
between current `develop` and the July implementation despite identical learned
weights. Run current `develop` and an explicit frozen historical inference
control separately. The historical source is commit
`e78a7c04e18d005d47def9eb7c9bd0f68b168d99`, fixed from checkpoint history;
the metric implementation remains `develop` in both cases. Do not describe
current-restoration failures as properties of the original trained model.
The historical control is an audit route, not a silent production fallback.
The discrepancy has been isolated on silicon to the FiLM treatment of odd
scalars: signed versus squared conditioning inputs and additive versus
multiplicative odd-scalar biases. `set_checkpoint_film_semantics` now exposes
an explicit evaluation-only `legacy_2026_07` replay through `develop`; the
corrected `o3` behavior remains the default. A representative silicon replay
matches the independently frozen historical H/D/S errors exactly. This is a
replay of the historical function, not a claim that its nonlinearities enforce
O(3) symmetry. Its semantics report must accompany results. The CLI selects it
with `--film-semantics legacy_2026_07`.

## Definitions

For each fractional k point, sum all available periodic directed blocks with
exp(+2 pi i k.L). Do not analyze an isolated onsite block or only L=0.
With a^h=(a+a^dagger)/2, report ||a-a^dagger||_F/||a||_F and its maximum
entry before using a^h in the following tests. All diagnostic algebra uses
complex128. Float32 inference and printed input precision remain in the data.

For S^h, report minimum/maximum eigenvalues, condition number, counts of
nonpositive eigenvalues, and unjittered Cholesky failure. A usable metric
requires lambda_min > max(1e-8, 1e-8*max|lambda|). Do not compute occupations
in an invalid metric; mark those cases unavailable and retain denominators.

For a valid S, define P=S^(1/2) D^h S^(1/2) and
F=S^(-1/2) H^h S^(-1/2). The eigenvalues of P, **not ordinary eigenvalues of
AO D**, are the occupation numbers. Report minimum/maximum, counts outside
[-1e-5, 2+1e-5], negative and excess occupation mass, and Tr(P). Report the
exact sparse real-space trace Tr(DS) as well as the normalized k-mesh average.
Their difference diagnoses aliasing/truncation; a small analysis mesh need not
integrate products of finite-support Fourier polynomials exactly.

Report ||P^2-2P||_F/(2||P||_F) as a **zero-temperature diagnostic**. The inputs
use 300 K electronic smearing, so nonzero idempotency residuals alone are not
a violation. Reference finite-support density kernels can also violate strict
occupation bounds after Fourier reconstruction.

Stationarity requires [F,P]=0, equivalently HDS-SDH=0. Report
||[F,P]||_F/(||F-Tr(F)I/n||_F ||P||_F). Centering F removes dependence on a
constant energy gauge. Commutation alone permits wrong or inverted filling.
Therefore fit **one common chemical potential per structure**, at the declared
temperature and independent electron target, over the sampled k points, and
compare P with 2*f_FD(F). The relative Frobenius residual includes both
offdiagonal density and wrong occupations. It is an algebraic consistency
test, not a recomputation of the DFT functional or an SCF convergence test.

See the [OpenMX density-operator definition](https://www.openmx-square.org/tech_notes/tech2-1_0/tech2-1_0.html)
and [PySCF AO density and SCF implementation](https://pyscf.org/_modules/pyscf/scf/hf.html)
for occupation/density conventions. The metric formulas above follow from
orthogonalizing the generalized AO eigenproblem.

## Implementation and validation

- `src/analysis/physical_constraints.py`: streaming Bloch assembly, metric
  spectra, occupation diagnostics, stationarity and global finite-T filling.
- `scripts/report/evaluate_physical_constraints.py`: original-checkpoint
  restoration, frozen input identities, explicit operator scenarios and JSON.
- `scripts/report/export_paper_operators.py`: isolated strict historical
  inference control; raw operators stay in the large-data namespace.
- `scripts/report/evaluate_density_support_control.py`: verify immutable raw
  exports against their producer manifest, then test density on native block
  support with trusted reference H/S. This result-dependent control separates
  density errors from overlap failure and from predictions on unobserved edges.
- `net.checkpoint_compat.set_checkpoint_film_semantics`: explicit historical
  nonlinear semantics for evaluation, preserving learned weights and the
  corrected default for new models.
- `tests/unit/analysis/test_physical_constraints.py`: exact nonorthogonal
  projectors, wrong/inverted populations, fractional occupations, invalid S,
  complex basis covariance, energy-gauge invariance and periodic phases.

Targeted regression checks also cover the SiO2 envelope family without an r0
parameter, the historical metadata whitelist, historical FiLM replay and the
corrected FiLM inversion contract. Existing source/data and unrelated work are
preserved.

## Result-dependent improvements

After the registered diagnostics, prioritize measured failures:

* Extend the explicit historical FiLM replay with saved per-material forward
  tensors and serialized semantics metadata before treating it as a general
  checkpoint migration. Preserve corrected O(3) behavior for newly trained
  models; changing old nonlinearities is not an innocuous weight translation.
* Define output support from actual basis/PAO support, or supervise all graph
  output edges including known zeros. Validate using no reference matrix
  values and report the change as a separate inference contract.
* Prefer the cheap analytical reference overlap where available. If learned
  S is required, examine a positive-definite global parameterization or an
  explicitly reported overlap correction with correction norm and spectrum.
* If occupations fail, evaluate an S-metric occupation projection with fixed
  charge and [0,2] bounds as a separate postprocessed result. Scalar charge
  rescaling alone is insufficient. Report how much D and observables change.
* If H/D consistency fails, compare independent D prediction with D built
  from H and trustworthy S using fixed finite-T filling. A training penalty
  on [F,P] can help stationarity but does not fix filling; use occupation or
  finite-T density consistency as well, after validation-only selection.
* A genuine self-consistency claim needs the backend's H[D] residual or an
  SCF restart comparison, not just the commutator and band-energy trace.
* For strict density representability, obtain an untruncated reference kernel
  or test a physical completion of omitted blocks. A reference-overlap
  eigensolve provides one finite-temperature completion; compare it on observed
  blocks as well as comparing its full spectrum with the zero-extended kernel.

Train or tune such changes only in a newly registered validation campaign;
freeze choices before any confirmatory test evaluation. No speculative
retraining grid is part of this diagnostic batch.
