# Research operations and immutable runs

All reference and custom models use `DatasetFactory → E3GNNDataset → E3GNN →
scripts.train`. Storage formats select data readers, not model classes or
separate trainers. See [model configuration](modular_operator_models.md) for
the shared interfaces.

## Separate source, workflow and scientific evidence

- Commit implementation, configurations, tests and reusable documentation.
- Before calculation, freeze the actual source bytes, resolved configuration,
  runtime, split membership, input identities, fitted-statistics provenance,
  checkpoint-selection rule and budget. A Git commit alone cannot identify a
  dirty execution tree; its immutable source manifest can.
- Keep scheduler state and execution decisions separate from scientific
  results. Successful allocation/exit is not proof of convergence or reaching
  a registered epoch endpoint.
- Keep existing execution snapshots and checkpoints immutable when the
  development branch changes. Resume against the original scientific contract;
  a changed source/runtime/configuration requires an explicitly registered run.

Small logs, frozen specifications, provenance and bounded summaries belong in
`researchops/runs/<run-id>`. This is a local operational workspace, not part of
the public repository checkout. Its records must accompany any result claimed
for a publication; references to private records alone are not a reproducible
public result package. Tasks record an executor, research/implement/calculate
kind, description and state in the research-operations control plane. If that
service is unavailable, retain an explicit local task/workflow ledger rather
than treating a Markdown handoff as live execution state.

Follow [the storage policy](RESEARCH_FILE_STORAGE.md) for datasets, predictions,
caches and checkpoints. Do not commit source archives, checkpoints, copied
datasets or transient scheduler dumps as documentation.

## Cluster execution

Use an isolated GNN checkout/source snapshot and the site's MANDALA runtime
profile. Never switch another active checkout. Put scientific computation,
dataset processing and checkpoint loading on allocated compute nodes. Login
nodes may inspect small manifests, CSV/log summaries, hashes and scheduler
metadata. The local repository documentation does not assert which jobs are
running now or which GPU partition is currently available.

Segmented training uses the shared epoch-boundary recovery machinery, preserving
optimizer, scheduler, RNG and selection state. A best-model checkpoint is for
evaluation, not recovery. Request sufficient shutdown/I/O headroom, and inspect
failures rather than retrying indefinitely.

## Accuracy-investigation policy, amended 2 November 2026

For future batches, start with one matched initialization seed per material,
hardware and runtime block, with explicit early/intermediate milestones
(48/200 epochs in the present protocol). Advance informative contrasts and
their baseline controls through late optimization before adding replication
seeds. Freeze every stage before execution; label validation-informed changes.

The next optimization batch includes an explicit earlier-learning-rate-decay
control under matched exposure and wall-time budgets. Do not assume a shorter
schedule gives equivalent final accuracy or silently alter a frozen run.

The existing campaign has a resource-only staging amendment: deferred repeats,
bounded continuations and missing endpoints remain explicit. Its historical
42-arm registration is not a mandate for another blanket multi-seed grid.
See the [ablation protocol](mandala_accuracy_ablation_protocol.md) and
[dated evidence summary](mandala_accuracy_investigation.md).

The specialized pair-descriptor Hamiltonian subproject has additional rules;
those do not impose separate onsite/offsite fitting on general MANDALA models.
