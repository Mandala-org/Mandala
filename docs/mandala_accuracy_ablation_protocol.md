# Graphene/MoS₂ accuracy ablation protocol

The next recipe comparison follows the separate
[fixed-recipe common-split benchmark plan](mandala_common_split_benchmark_plan.md).
Architecture ablations are deferred; the benchmark uses terminal checkpoints
and no new validation tuning.

This is the current scientific contract, reconciled on **2 November 2026**.
Historical registrations remain immutable. The 29 October long grid was amended
on 2 November by holding unstarted repeats and bounding pending continuations;
it is no longer an executable instruction to run every arm for two 48-hour
allocations. The historical registration and its amendment must be interpreted
together. Dated observations are in the [investigation summary](mandala_accuracy_investigation.md).

Future batches use one matched initialization seed at explicit 48/200-epoch
milestones, a small set of informative late confirmations with their baseline
controls, and replication seeds after that review. The next optimization batch
must include a separately registered earlier-learning-rate-decay control under
matched update and wall-time budgets. A shortened schedule is a hypothesis,
not an established equivalent of late optimization.

## Preserve the pilots and the shared interfaces

Keep the immutable `operator_benchmark_pilots_v1` source/configuration and the
two-segment allocation unchanged. A pilot may receive no more than its original
2 × 48 h. Do not edit its running checkout, replace its checkpoints, reuse its
output directories, or reinterpret its original validation selection rule.
The frozen final pilot launcher requests a test evaluation; that historical
evaluation is outside the new validation-only selection process. Never inspect
its test metrics to choose new configurations. If test results were already
used to steer a decision, mark that test set exposed and obtain independent
confirmatory data before claiming an independent final test.

Use `DatasetFactory → E3GNNDataset → E3GNN → scripts.train` for every model.
Architecture changes are configuration changes or reusable registered
transformations. No model-specific dataset, trainer, optimizer, evaluator,
forward implementation or source-name dispatch is allowed. Paper-specific
pair-descriptor protocols do not impose separate onsite/offsite training on
this repository-wide MANDALA investigation.

## Gate 0: certify what is comparable

Before attributing differences to architecture, write a machine-readable
comparison ledger with one row for every literature value and every checkpoint.

| Required field | Acceptance/action |
|---|---|
| Dataset identity | Archive/version/checksum, material, structure count, orbital basis and spin convention match or the difference is explicit. |
| Split identity | Retain exact path/content-hash membership, not only a seed/ratio. Pilot sizes are Graphene 270/90/90 and MoS₂ 300/100/100. Seed 42 does not establish equality to unpublished reference indices. |
| Physical entries | All stored observed directed Hamiltonian entries; padding/missing entries excluded; no label-derived distance filtering. Audit onsite, periodic shifted-self and reverse-pair coverage. |
| Units | Archived eV → internal Hartree exactly once → reported eV/meV exactly once. Check marker values through raw/native reader paths, including metadata precedence. |
| Basis/axes | OpenMX orbital order and xyz geometry converted exactly once at their declared representation boundaries. Already-converted input must carry explicit metadata. Record s/p/d/f permutations and round-trip/oracle evidence. |
| Frame | Reconstruct the same global physical orbital frame. Elementwise MAE is not invariant under a generic bond-frame rotation. A local-frame training-log MAE is not a global-frame benchmark. |
| Projection | Raw and full reverse-pair Hermitian-projected predictions are separate metrics. Never infer projection from a filename or use offsite self-symmetrization. |
| Reduction | Specify per-entry, per-edge, species-pair mean/sum and per-structure/dataset reduction. A sum or unweighted mean of Mo–Mo/Mo–S/S–Mo/S–S means is not an entrywise mean. |
| Selection | Checkpoint rule, epoch, initialization seed, actual updates/LR history, stopping reason and wall time are explicit. Latest and validation-best checkpoints answer different questions. |

Recompute validation summaries from a fixed checkpoint only on an allocated
Slurm node. Login-node work may read small JSON/CSV/log files, hashes and Slurm
state; it may not load checkpoints or process datasets. Put large predictions,
caches and checkpoints under the documented bigdata roots; retain manifests,
bounded aggregate results and logs under `researchops/runs/<run-id>`.

For structure `s`, let `n_s` be its number of observed directed matrix entries,
`a_s = sum |H_pred-H_target|` and `q_s = sum |H_pred-H_target|²` in eV.
Report both `mean_s(a_s/n_s)` and `sum_s(a_s)/sum_s(n_s)`, named explicitly.
Similarly distinguish `mean_s(sqrt(q_s/n_s))` from
`sqrt(sum_s(q_s)/sum_s(n_s))`. The shared `*_elementwise` epoch metrics use
the mean of structure reductions; their label alone does not identify a
dataset-entry-weighted metric. Report raw/projected versions, target/raw
Hermiticity residuals and sample/entry counts. Species-pair, onsite/offsite and
distance-bin errors are diagnostic partitions of these same entries, never
replacements for the declared primary metric.

Use two comparison tiers. An **exact reproduction** requires verified matching
data, split indices, evaluation and training protocol. A **common-split internal
comparison** requires identical frozen splits and common physical evaluation
within MANDALA, even when the published indices are unavailable. Literature
numbers with unresolved reductions/frames/splits remain contextual; do not
compute a definitive reproduction ratio from them. Software oracle tests
certify selected computations, not published convergence or test accuracy.

Audit the pilot's `symmetrize_hamiltonian_targets: true` against the raw archived
targets. Report the actual entrywise/max changes and target Hermiticity residual
without silently correcting the dataset. Target symmetrization is either a
measured preprocessing change or an algebraic no-op; that decision follows the
training-data audit, not which validation score improves. The source-basis and
energy audit must pass independently of the training results.

## Common experiment contract

Freeze source hash, resolved configuration, runtime/package/GPU identity,
training and validation membership, raw-data hashes, graph and output supports,
all fitted training statistics, initialization/ordering seeds, budget, metrics,
failure criteria and planned contrasts before starting a batch. Test labels and
test metrics are excluded from ablation training/evaluation/selection; retain
reserved membership separately. Do not fit normalization, envelopes, means,
species statistics or graph cutoffs on validation/test targets.

Select checkpoints by the same raw global structure-mean entrywise validation
MAE in every causal contrast. Report the projected metric at that checkpoint
as a diagnostic. A later physical projected-selection study must be registered
separately with its own controls. Keep both raw directions in supervision and
evaluate the actual shared forward computation. Representation conversions,
symmetry projection and evaluation must not erase the intended treatment.

Within each contrast hold constant data/support, target preprocessing, model
parameters where compatible, initialization, shuffle order, effective batch,
dtype, Adam hyperparameters, clipping, schedule and budget except for the named
treatment. Same integer seeds do not guarantee matched initialization after
changing construction order: record initial tensor hashes and pair parameters
by semantic name when feasible. The historical long registration fixes initialization seeds
42, 43 and 44 and split seed 42. Future replication seeds must be declared in
their own stage; keep split membership fixed across them. Do not change both data split and
initialization seed and call that a seed replication.

There are two budget questions. Matched update/epoch comparisons test learning
at comparable exposure; matched GPU time comparisons test attainable accuracy
at a resource budget. Report both and never interpret equal wall time as equal
optimization. Record parameter count, forward/backward time, memory, epochs,
actual optimizer updates, examples seen and rollback/repeated-update counts.
Width/degree/depth changes are capacity and compute changes; a subsequent
parameter-matched and, separately, compute-matched control is needed before
attributing their gain solely to a mathematical component.

## Future staged screens: independent audit and controlled probes

The following work does not depend on interpreting earlier validation scores
and may be prepared as one fail-fast ordered batch. Validation-dependent
promotions remain behind the next gate. Existing pilots continue unchanged.

1. On Slurm, audit source conventions, training support, target asymmetry,
   loss denominators and explicit split identity; evaluate frozen pilot best
   checkpoints on validation only when a safe checkpoint is available. Save
   compact raw/projected global-entry totals and diagnostic partitions.
2. Run a fresh pilot-architecture control and preregistered changes of matrix
   loss kind (MAE/MSE), species-pair versus entry reduction, rollback and LR
   policy. A full factorial identifies interactions; an explicitly listed
   one-factor-at-a-time screen estimates effects only at its fixed baseline.
   Do not call a joint change in rollback, decay, clipping and learning rate an
   isolated optimizer effect. In particular, keep the MAE/MSE comparison's
   LR/Adam epsilon and scale fixed and call it a fixed-optimizer effect; a
   separately matched LR grid is needed for an objective-at-best-optimization
   comparison because the gradients change scale and units.
3. If required infrastructure has passed focused checks, run a frozen
   term-versus-copy radial-weight control on exactly the same architecture,
   initial state, data, objective and optimization. First inspect forward and
   gradient utilization analytically. A code correctness finding alone does
   not establish its contribution to benchmark error.

The concrete job list in each immutable stage manifest is the registration;
unlisted combinations are not implicitly authorized. Start with one matched
initialization seed per material/hardware/runtime block. Declare both the
48- and 200-epoch exposure endpoints and wall-time caps before execution;
report an unreached endpoint as resource-incomplete. Early ranking cannot
establish convergence or statistical superiority.

Keep the baseline and informative contrasts through late optimization before
adding seeds. A short 48-epoch screen cannot exercise a plateau scheduler whose
patience is sixty epochs. Test earlier decay as an explicit separate contrast,
with the same architecture and declared epoch/update and wall-time comparisons.

The original 14-candidate screen and expanded replication grid are historical
registrations described below, not default budgets for future batches. Existing
preparation tools require an explicit new registration for promoted subsets;
there is no automatic validation-based promotion or implicit resubmission.

The shared orbital-block control is `matrix_loss_reduction`:
legacy `pair_sum` must stay the default; `pair_mean` averages nonempty pair
means; `elements` divides summed observed entry errors across species pairs by
their total observed count. With complete fixed blocks, pair-sum to pair-mean
isolates global loss scale,
while pair-mean to elements isolates relative weighting at a mean-error scale;
their finite values and gradient norms can still differ. Changing
`objective.kind`/`loss_l1_fraction` changes MAE/MSE, while
`objective.epoch_weight` in the packed-shell path does not change the legacy
orbital-block pair-sum objective. The Graphene single-species case is an
important negative control: pair-sum and entry-mean objectives should coincide
when the single pair has uniform full blocks. Preserve the historical
`hamiltonian_mae` diagnostic meaning and report the independent entry metric.
New `pair_mean`/`elements` policies include target coverage masks and the
configured `partial_train` edge selection. They reject irrep objectives,
packed-shell supervision and edge-weighting policies instead of silently
changing their meaning. The legacy default retains its existing behavior.

## Gate 1 and Batch B: confirm effects and bridge recipes

Review all results from the registered stage together. Numerical/support/convention failures require a code
fix and a newly frozen control; they do not justify changing scientific
parameters during a running comparison. After the early/intermediate review, register informative contrasts and their
baseline controls through late optimization, retaining uncertain contrasts
where justified. Add paired replication seeds only after that review. Every
promotion, nominated contrast and budget must be frozen before execution;
validation-informed amendments must remain explicit. Report all registered results, paired
per-seed differences, range and mean; three seeds remain a small sample.

Then construct two explicitly different recipe comparisons:

* **Source-default recipes** retain each pinned DeepH/DeepH-E3/MACE-H architecture,
  objective and optimizer, with the common train-only-statistics policy and
  frozen common split. They test achievable recipe performance and retain
  deviations from published preprocessing/training as caveats.
* **Controlled architecture recipes** share data/support, physical target,
  objective reduction, optimizer, clipping, schedule, selection and budget.
  They isolate a whole architecture bundle, subject to parameter/compute
  confounding. They do not by themselves identify an individual layer.

First certify the representation bridge between `orbital_blocks`/e3nn and
`packed_shells`/native targets using decoded matrices, masks, losses and
parameter gradients under the same scalar objective. Different packed order,
local coordinates or energy-unit scaling can preserve a forward matrix while
changing the Adam training trajectory. Check these explicitly. For original
DeepH, its local-frame objective is a distinct treatment even if evaluation is
global. Its five-layer topology, batch size 3, clipping, and milestone schedule
must not be silently compared as an isolated equivariance effect against a
two-layer MAE/rollback model.

Compare normalized and unnormalized affine outputs separately. MACE-H
`species_irrep_affine` uses training-only per-pair/per-irrep statistics and
`gradient_mode: identity`; `identity` versus `physical` preserves the forward
affine map but changes the backward map. It is a gradient-preconditioning
contrast. Removing the affine changes both forward initialization and
parameter coordinates; preserve/measure those changes or label it a bundle.
Never reproduce full-dataset statistic fitting to match an upstream leak.

## Gate 2: isolate architecture only after the bridge works

Choose follow-ups from validation and diagnostics, then preregister complete
matched controls before execution. Useful hypotheses, not established causes:

| Suspected limitation | Controlled follow-up |
|---|---|
| Ineffective radial conditioning | Term/copy weight utilization, fixed shared radial output and count, then capacity/compute controls where needed. |
| Species imbalance/objective scale | Pair-sum/entry reduction × MAE/MSE factorial; inspect Mo–Mo/Mo–S/S–Mo/S–S counts/errors and gradients. |
| Insufficient environment propagation | Fixed-width 2/3/5 stages with identical residual/normalization rules; then parameter- and compute-matched controls. |
| Attention or normalization | Replace one aggregation or normalization transformation at a time, retaining irrep multiplicities/readout/support. |
| Missing higher-order local correlations | Toggle neighbor-density/symmetric-product correlation while fixing surrounding edge/readout stages; do not replace the whole MACE-H recipe and call it a body-order effect. |
| Readout/initialization or scaling | Hold upstream features fixed and swap explicit shared readout/affine operations; report shell coverage, gradients and initial prediction magnitudes. |
| Radial support versus encoding | Keep target support fixed; separately vary message support and radial encoding range. A distance-filtered target set changes the task. |
| Plateau or rollback prevents convergence | Fixed-architecture controller contrasts with complete LR, rollback and update histories and a declared long budget. |

An improvement shared by Graphene and MoS₂ supports a general component
hypothesis; a material-specific result remains material-specific. Interactions
require a registered factorial follow-up, not adding independent gains. A
useful component can be optimization-mediated: call it so instead of inferring
representational expressivity solely from validation MAE.

## Stop rules, selection and final reporting

Stop a job for nonfinite loss/gradient, target/support mismatch, unit/frame
failure, checkpoint/runtime mismatch, data leakage, OOM or unsafe continuation.
Record failure, last safe epoch and actual consumed resources; do not replace a
failed seed or silently shrink the model/batch. A retry after an engineering
fix belongs to a new source identity and repeats affected controls. Reaching a
registered epoch/time/LR stop is a budget outcome, not evidence of convergence.

For short screens report all validation trajectories and named endpoints.
For longer confirmation register one common epoch/update horizon and equal
allocation cap before execution; any extended convergence study is a new batch.
Use paired structure-level errors to quantify validation uncertainty, but do
not treat matrix elements or checkpoints as independent replications. Temporal
correlation between MD structures can invalidate naive structure bootstrap
intervals; use trajectory/block resampling only when grouping provenance is
available and otherwise report that limitation. Seed variability is distinct
from sample variability and from repeated model selection on the validation set.

Freeze final configuration, checkpoint rule, seed ensemble policy and metric
before a single confirmatory test campaign. No test results choose components,
seeds, projection policy, clipping, cutoff, objective, stopping rule or reporting
denominator. A material whose test labels/results have already informed these
choices requires independent test data for a confirmatory claim.

## Executable interfaces

- `scripts.prepare_operator_accuracy` freezes source, configurations, split and
  input identities, contrasts and budgets, and verifies their hashes.
- `scripts.audit_operator_accuracy` produces train/validation convention,
  support, target and fixed-checkpoint diagnostics.
- `scripts.probe_operator_runtime` checks ABI/runtime identity, finite model
  forward/backward behavior and memory in an allocation; it refuses test data.
- `scripts.submit_operator_accuracy` and `scripts.submit_operator_grid` submit
  manifest-defined work. `scripts.release_operator_accuracy` releases held work
  only after the registered independent certifications pass.
- `scripts.run_training_segment` uses shared safe epoch-boundary continuation;
  final evaluation is validation-only unless explicitly requested otherwise.
- `scripts.evaluate_operator_checkpoint --split val` evaluates a selected
  checkpoint under the declared precision and common physical-entry metrics.
- `scripts.analyze_operator_accuracy` reads bounded validation CSVs, preserves
  missing fixed-horizon endpoints and hardware/seed blocks, and rejects test
  metric columns.

A helper's existence does not register a new batch. New launchers must omit
`--evaluate-final-test`, use new record/checkpoint namespaces and freeze their
actual scientific contract. Do not repurpose the original pilot launchers,
whose separately registered final test evaluation must not guide ablations.
All dataset/checkpoint work, including `scripts.train --dry-run`, belongs on
an allocated compute node for real scientific data.

## Historical campaign and applied amendments

### 29 October registration

Before new ablation scores were observed, the authorized long campaign
registered 14 material/treatment combinations at initialization seeds 42/43/44,
fixed split seed 42, up to 500 epochs, and two 48-hour allocations per arm.
Its 42 arms and 84 allocations had an upper cap of 4,032 GPU-hours. Hardware
assignments were frozen within material/seed blocks. The separate V100 screen
registered the same 14 combinations at seed 45, 48 epochs and six hours each,
with 12/24/48-epoch endpoints and a different runtime (Torch 2.5.1/CUDA 12.1).
It is analyzed separately, not pooled into the long hardware/runtime blocks.
Deterministic certifications passed and the registered first segments were
released; the original hold/release description is not current queue state.

### 2 November resource-only amendment

The original pilots and active training runs were preserved. Eleven still-
unstarted repeats and their eleven continuations were held. Two other proposed
repeats had already started and were preserved. The two distinct MoS₂ seed-44
objective controls remained eligible.

Twenty-six pending 48-hour continuations were replaced with resource-only
2–10-hour dispatchers (combined upper cap 205 rather than 1,248 GPU-hours).
They use the frozen runner and preserve source, configurations, splits,
checkpoint, optimizer, scheduler and RNG state. Parent completion metadata
controls eligibility: already-complete models are skipped; a remainder above
75 epochs or an estimated ten-hour cap is deferred before scientific data or
checkpoints are loaded. Two redundant completed-model resumes were cancelled
without replacement. The final analysis dependencies were updated accordingly.

This amendment was informed by validation/runtime evidence. Held, deferred or
resource-incomplete arms must remain missing in the original 42-arm analysis;
the amended output is not a completed balanced three-seed grid. Allocation
success and allocated-hour ceilings are not measures of convergence or consumed
GPU time. Exact decisions/source hashes remain in the local operational ledger
`researchops/runs/mandala_accuracy_v1/workflow/staging_20261102/`.

The [research-workflow policy](RESEARCHOPS.md) governs later staged
experiments. Neither historical registrations nor old pending-job lists grant
authorization to launch a new campaign.
