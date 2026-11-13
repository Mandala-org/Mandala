# Physical-operator audit evidence

These bounded JSON records support the physical-validity analysis of the
paper operators (`summary_projected.md`). They contain diagnostics, input identities, resolved
configurations and source manifests; they do not contain operator datasets or
trained checkpoints. Workflow state and scheduler logs are not included.

`historical_summary.json` and each material's aggregate describe the frozen
July inference control. `develop_summary.json` describes restoration with the
corrected current nonlinearities. These are separate functions even when
their learned state dictionaries load strictly. The later explicit
`--film-semantics legacy_2026_07` mode reproduces historical FiLM evaluation
through develop without changing its default for newly trained models.

The methods, tolerances, native-support controls and finite-density-support
caveats are specified in the parent protocol. Scalar matrix errors here use
common support; separate fields quantify prediction-only blocks. The physical
audits separately retain those blocks or restrict to reference support.
Never interpret a common-support matrix MAE as a bound on the full output.

## Access and reproduction

The original OpenMX files, CIF geometries and the three selected checkpoints
currently reside in the MANDALA ROSI project namespace
`/bigdata/casus/wdm/hamiltonian_learning`. Access requires permission to that
project. No public download location for these exact paper inputs was supplied
for this audit. Their paths and SHA256 identities are in the contracts. An
external publication package must provide the corresponding data/checkpoint
access or an archive; these summaries alone do not reproduce the inputs.

Immutable execution snapshots and raw historical predictions remain under
`physical_constraints/paper_20261104_v{1,2,3}` in that project namespace. The
manifests here identify the actual source bytes, including uncommitted
diagnostic code, rather than relying on the Git head alone. The historical
source is pinned to commit `e78a7c04e18d005d47def9eb7c9bd0f68b168d99`.

Run inference and dense algebra on allocated compute nodes, with one GPU and
four CPUs for GPU jobs, using the MANDALA runtime profile. Freeze a fresh
output directory and actual source manifest for a new execution. The entry
point is `scripts/report/evaluate_physical_constraints.py`; use
`--film-semantics legacy_2026_07` for the original paper function, and preserve
the checkpoint and split settings from the contracts. The frozen independent
control instead uses `--prediction-source-root` and a large-data prediction
cache. `scripts/report/summarize_physical_constraints.py` aggregates complete
structure files without discarding invalid overlap cases.

The Zn envelope copy frozen for this audit differs slightly from the active
ROSI artifact at the same path. `envelope_audit.json` records both identities
and a fixed-grid comparison: relative Frobenius difference 5.58e-7 and maximum
relative difference 2.90e-5 where the envelope exceeds 1e-8. The audit uses one
fixed artifact for both inference versions. SiO2's envelope is byte-identical
to the active ROSI artifact. No artifact was selected using physical-test
results; bitwise historical provenance of the Zn fitted artifact is not
asserted.
