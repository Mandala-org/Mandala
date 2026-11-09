# Fixed-recipe benchmark: MANDALA, DeepH-E3 and MACE-H

Compare the three complete recipes on spinless monolayer Graphene and MoS₂.
DeepH and architecture swaps are outside this batch. Use one prespecified
initialization seed per material for the first comparison. No new validation
tuning, sweeps, validation-best reporting, or test-driven decisions.

## Freeze the split first

Prefer the MACE-H Table S4 memberships. Recover the actual train, validation,
and test IDs from the authors' split records or reconstruct them with the
original ordered graph index and split implementation, and verify the result
against authoritative membership evidence. The [paper](https://arxiv.org/html/2508.15108v2)
reports 270/90/90 Graphene and 300/100/100 MoS₂ structures. The
[released configurations](https://zenodo.org/records/15223696) specify seed 42
and 60/20/20, but point to graph caches on the authors' scratch filesystem.
They do not establish exact membership. Do not certify a split from seed
and proportions, or assume rebuilt directory order matches their cache.

If exact membership cannot be verified, use the existing explicit MANDALA
pilot memberships for all three methods and label the result a new common-split
benchmark. Freeze and hash the structure IDs and OpenMX target identities.
Retrain on the chosen memberships; do not transfer pilot weights onto a
different split. The already reported pilot test results are historical and
must not guide the recipe, stopping rule or evaluator.

## Freeze the recipes

The target is the **MACE-H paper's monolayer comparison**, not the original
DeepH-E3 headline result or the repository reference defaults. Use
[Table S3 and the accompanying methods](https://arxiv.org/html/2508.15108v2)
for reported settings. Both reference models use Gaussian node radial
features, batch size 1 and three stages; use the angular limits in the
table and the reported MSE training objective.

| Recipe | Graphene initial LR | MoS₂ initial LR | Other fixed choices |
|---|---:|---:|---|
| DeepH-E3 as run in the MACE-H paper | 0.003 | 0.005 | Node and edge angular limits 5 |
| MACE-H | 0.006 | 0.006 | Correlation order 3, MACE/hidden/edge angular limits 5, shift-and-scale disabled |
| MANDALA | 0.0027 | 0.0027 | Original material-specific pilot recipe, fixed from prior validation evidence: two stages, MAE, original optimizer and controller |

Use the actual frozen MANDALA pilot configurations from
`operator_benchmark_pilots_v1` (source identity `78c56ac89602…`), not a
new combination of favorable ablation settings. Keep the reported reference
settings authoritative. The inspected Zenodo config archive
(`configs.tar.gz`, MD5 `34e80384fb931595282a3c520a46038d`) contains
MACE-H monolayer INIs, but no separate DeepH-E3 monolayer INIs. The MACE-H
MoS₂ INI uses 0.01 where S3 says 0.006. Recover original run manifests if
available to resolve scheduler details, epoch limits, widths, and any
paper/config discrepancies. For parameters not reported in the paper or
verified in original run records, freeze a source-backed value and document
its provenance and any deviations before training; do not tune it on
validation. An unresolved discrepancy prevents a claim of exact historical
reproduction, not the common-split benchmark.

## Terminal checkpoints and shared scoring

1. Register each recipe's stopping rule and budget before launching, using
   documented recipe limits where verified and an explicit fixed endpoint
   where not. Keep the recipe's specified LR and rollback mechanics where
   applicable: validation signals prescribed by a controller are part of
   training, not a search over recipes or reported checkpoints. Record
   completed updates, wall time, GPU class and any budget-limited endpoint.
2. Score only the terminal training state. Use the explicit checkpoint
   argument of `scripts/evaluate_operator_checkpoint.py` with the verified
   terminal checkpoint, never a validation-best artifact. The existing
   automatic best-checkpoint test paths in the trainer and segment runner
   are unsuitable; prepare a terminal-only launch/evaluation path and
   verify it on a targeted fixture before the benchmark. Keep recovery
   checkpoints for Slurm continuation.
3. Use one evaluator, global OpenMX orbital convention, meV, and identical
   observed periodic target keys and entry masks. Verify support before
   scoring and flag missing predictions rather than silently shrinking the
   comparison set. Report structure-mean entrywise MAE raw and after the
   same complete reverse-pair Hermitian projection, plus directed MoS₂
   species-pair MAEs and counts. Test only after recipe, split, evaluator and
   endpoint rule are frozen.

Use the shared DatasetFactory/E3GNN interfaces and the
[audited reference components](reference_architecture_coverage.md).
The repository presets are starting points: their defaults do not match
Table S3 (for example the LR, angular limits and MACE-H affine readout).
Check the resolved monolayer settings and scheduler implementation, and
verify prediction coverage and terminal-checkpoint handling on a small
fixture before starting the six model/material runs. Use a common GPU class
where possible, four CPUs per one-GPU allocation and explicit memory.
This batch compares the three recipes; causal architecture tests come later.
