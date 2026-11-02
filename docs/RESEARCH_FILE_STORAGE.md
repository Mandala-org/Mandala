# Research file storage

Keep small execution records under `researchops/runs/<run-id>`: task/workflow
ledgers, submission scripts, immutable specifications, source/data identities,
bounded metric summaries and small logs. This local operational workspace is
not included in a fresh repository clone. Public results require a separately
packaged, bounded reproducibility record and documented data access.

Large datasets, checkpoints, per-example predictions and caches stay under the
ROSI project root `/bigdata/casus/wdm/hamiltonian_learning`, in distinct
namespaces. Preserve existing data and split provenance; do not invent an
alternate shared data root or overwrite another run's outputs.

Version-control source, reusable configurations, documentation, tests and
bounded synthetic/frozen test fixtures. Do not add copied scientific datasets,
checkpoint files, frozen source archives or complete scheduler dumps to Git.
The pair-descriptor experimental subproject uses its own separate run/data
namespace; its additional scientific rules do not apply to general GNN work.

See [research operations](RESEARCHOPS.md) for execution and experiment policy.
