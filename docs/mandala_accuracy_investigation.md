# MANDALA accuracy investigation: evidence snapshot

For the completed allocations and final findings, see the
[6 November campaign report](mandala_accuracy_final_report.md).

Consolidated **2 November 2026**. The numerical snapshot below was captured
10:49–10:51 Europe/Warsaw; the operational amendment was verified at 12:00.
These are dated observations, not live scheduler state. All accuracy analysis
here is validation-only. It does not establish a ranking against published
held-out test scores or a final causal explanation of an accuracy gap.

## Scientific state

The original pilot architecture improved substantially with additional training
and learning-rate reductions. The pilot raw validation errors initially captured
at epochs 298/202 were 2.097/3.353 meV for Graphene/MoS₂. The later snapshot is:

| Original pilot | Latest validation epoch (zero-based) | Raw MAE (meV) | Projected MAE (meV) |
| --- | ---: | ---: | ---: |
| Graphene | 952 | 0.222 | 0.202 |
| MoS₂ | 639 | 0.572 | 0.500 |

MAE is the mean over structures of each structure's observed directed matrix-
entry errors, in the global physical frame. Projection is full reverse-pair
Hermitian projection. Latest, minimum-observed-through-horizon and independently
reloaded checkpoint results are distinct quantities. The table does not replace
validation-best selection, prove convergence or isolate a schedule effect from
additional training.

SiOx is a separate dataset and selection history. Its first continuation ended
at the time budget with a latest epoch-3012 result of about 2.005 raw / 1.949
projected meV. Historical species-pair-sum MAE is not the entrywise metric.

## Certified preprocessing and comparison limits

Full train/validation certificates passed for Graphene (270/90 structures) and
MoS₂ (300/100). Their declared reserved test counts are 90/100. Raw input hashes,
source identities, graph support, orbital/Cartesian conversion and target
cleanup were checked independently of validation ranking. Within the declared
tolerances, no duplicate unit, orbital or Cartesian conversion was detected.

- OpenMX/e3nn orbital and Cartesian round trips were exact in these certificates.
- Maximum eV/Hartree round-trip difference was 9.54×10⁻⁷ eV.
- Target cleanup changed an entry by at most 5.07×10⁻⁸ eV.
- Raw target Hermiticity residual was at most 1.01×10⁻⁷ eV.

These certify the specific frozen paths, not every optional configuration.
Later development commits do not rewrite their execution sources.

Published readers depend on directory discovery order before seeded shuffling;
matching a seed and split proportions does not prove identical membership.
DeepH local-frame MAE, global-frame entrywise MAE, species-pair averages,
structure-mean errors and dataset-entry-weighted errors are not interchangeable.
The [literature audit](mandala_accuracy_literature_audit.md) records these limits.
A common-split internal comparison can be valid without being an exact
reproduction of a published test evaluation.

## What the registered contrasts show so far

- At 200 epochs, fixed-optimizer MSE is worse than the MAE baseline on Graphene
  for the three observed initialization seeds. At 500 epochs, the seed-43
  endpoint is 1.764 versus 0.272 meV. This is a loss/configuration contrast,
  including gradient scale, clipping, optimizer epsilon and controller response;
  it does not prove MSE is intrinsically inferior after retuning optimization.
- Copy-wise radial gates improved early V100 best-through-horizon errors on both
  materials, but longer results are mixed. At 200 epochs Graphene seed 42 gives
  2.140 versus 2.131 meV, while seed 44 gives 1.171 versus 1.246 meV. An early
  improvement is not a demonstrated final architecture benefit.
- Source audit confirmed implicit AdamW decay, historical grouped-term radial
  gates, different objective denominators and different reference recipe bundles.
  Their isolated accuracy effects remain unresolved. See the
  [component audit](mandala_accuracy_component_audit.md).

These observations favor investigating optimization before attributing the
initial gap to architecture. They do not establish that late decay alone caused
the improvement or that all registered controls have converged.

## Compute state and applied staging amendment

At the pre-amendment review, the long grid had allocated 939.44 GPU-hours; 27 of
42 models had started and three had reached the registered 500-epoch endpoint.
The separate 14-arm V100 screen completed successfully using 75.60 allocated
GPU-hours, but stopped at 20–41 epochs: all arms reached 12, twelve reached 24,
and none reached 48. Missing matched endpoints remain resource-incomplete.
Allocated hours are not a measurement of productive GPU utilization.

The user-approved amendment was applied on 2 November:

- Eleven still-unstarted repeats and their eleven dependent continuations were
  held. Two other proposed repeats had started and were preserved.
- The two distinct MoS₂ seed-44 objective controls remained eligible.
- Twenty-six pending 48-hour continuations were replaced with 2–10-hour
  resource-only dispatchers, with a combined 205-hour ceiling. Already-complete
  models are skipped; expensive remainders are deferred before loading data or
  checkpoints. Two redundant resumes were cancelled without replacement.
- Original pilot continuations and active runs retained their source,
  configurations, splits and checkpoint/optimizer/scheduler/RNG histories.
- Final analysis dependencies were updated. Held/deferred arms must remain
  missing in the original 42-arm output; it is not a completed balanced grid.

This is a validation-informed amendment. The original frozen registrations are
retained and are not a standing instruction to repeat the full long grid. Future
work follows the staged policy and explicit earlier-decay control in the
[ablation protocol](mandala_accuracy_ablation_protocol.md).

## Evidence custody and reproducibility

Operational records live in the local, unversioned
`researchops/runs/mandala_accuracy_v1/` workspace. Relevant paths are:

| Record | Relative path within that workspace |
| --- | --- |
| Full train/validation certificates | `revision_2/results/{graphene,mos2}_audit.json` |
| Long and V100 registrations | `{long_v1,v100_v1}/batch/registration.json` |
| Latest numerical/compute review | `evidence/status_20261102T094956Z/` |
| Applied amendment and verification | `workflow/staging_20261102/{operations,after}.json` |
| Earlier snapshot chronology | dated files under `evidence/` and `workflow/` |

The pilot source hash is
`78c56ac89602ea2d6771a82b55e74f6ad81e8d2742a8d0421f447d2ff988c4df`;
the expanded execution source is
`0670074724c76c60f9871cad0dff28560a09a0bea313ee94b4c657e2bfca0623`.
Each manifest records actual source bytes, not just the dirty tree's base commit.

These private operational paths are not downloadable evidence in a fresh clone.
A public results claim needs the corresponding bounded manifests, data-access
instructions, metrics and checkpoint identities packaged alongside it. Full
scheduler dumps and transient handoffs are not maintained as repository guides.
See [research workflows](RESEARCHOPS.md) for storage and execution rules.
