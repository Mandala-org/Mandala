# Validation-only error-distribution evaluation, 2026-11-10

Branch: `referee-r5-metrics-fast`. Outcome: partial; silicon failed the mandatory
exact-equivalence gate, so no Zn replacement was submitted or registered.
No files under `paper/` or README.md were edited.

## Implementation

- `scripts/report/evaluate_validation_error_distribution.py`: optional
  `--load-split validation` (default `all`). Reuse the existing dataset builders'
  ordered discovery and seeded shuffle, selecting memberships before matrices
  load. Preserve the requested training count in output and record load mode.
- `src/data/factory.py`: optional `load_splits` limits dataset construction while
  retaining all original training metadata, shared mapper construction,
  configuration resolution, and validation vocabulary/spin checks. Unloaded
  splits return None. Cache keys, cache readers/writers, and density conventions
  are unchanged.
- `tests/analysis/test_validation_error_distribution.py`: synthetic two-scale
  directory listing, 105 complete snapshots per scale plus incomplete entries;
  compare full/validation-only ordered membership for Zn seed 44 and silicon
  seed 43. Check only validation datasets are constructed and training metadata
  still defines the mapper. A later strengthening places oxygen only in the
  training metadata, ensuring this species survives in the mapper.

Implementation commits after rebase: `23b27c0` and `5ab947c`; factory type
annotation: `5dc47a2`. Early ROSI checkpoint `92afa26` was retained as an ancestor
by `8c81404`, allowing a normal push after the required develop rebase.
The evaluator base commit was replayed as `e37480d`.

## Verification

On Terok, passed:

```
python3.13 -m compileall -q scripts/report/evaluate_validation_error_distribution.py src/data/factory.py tests/analysis/test_validation_error_distribution.py
git diff --check
```

On ROSI at implementation checkpoint `92afa26`, passed:

```
source ~/scripts/profiles/mandala.sh
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONPATH=src:. python -m pytest -q tests/analysis/test_validation_error_distribution.py
```

Result: **9 passed, 1 warning in 7.02 s** (login-node NVML warning).
The subsequent training-only-species strengthening was compiled but not rerun;
work stopped after the silicon gate failed. No metrics were evaluated on the
rebased final branch.

Silicon was submitted using the original launcher with the new option and
updated source hashes, saving outside paper/:

```
sbatch researchops/runs/r5_error_distribution_20261110/silicon_perturbed_validation_only.sbatch
scontrol update JobId=837022 TimeLimit=00:15:00
```

Job **837022 COMPLETED, 00:01:13, exit 0:0**, H100. The launcher used the same
checkpoint, convention, metadata policy, seed 43, density convention and GPU
settings as the original. Output:
`researchops/runs/r5_error_distribution_20261110/silicon_perturbed_validation_only.json`.
Both paths report 80 training and 16 validation structures (only 96 canonical
silicon snapshots exist), with 1,881,984 elements per operator.

The exact comparison was Python dictionary equality of `before['metrics']`
and `after['metrics']`, with JSON loaded from the original paper result and the
new result. **Failed: 21 of 36 fields differ**; the complete exact differences
are in `silicon_comparison.json`. The separate 1% paper-MAE gate passes, but
that does not meet this task's stricter exact-equivalence requirement.
For example Hamiltonian MAE changed from `0.00043516440060726235` to
`0.00043516437820002`. Reference centered sums of squares and element counts
agree. The cause of these last-digit differences has not been diagnosed;
no tolerance relaxation or rerun was attempted.

## Cache investigation

The canonical `paper_2026_07` metadata selection agrees with current discovery
for all 100 Zn snapshots. Both runs use `e3nn`; this is not a convention or
metadata-path mismatch. Metadata policy only changes discovery, not the cache
implementation.

Historical cache keys used only version, matrix/info stats, geometry,
convention and dtype. All 100 Zn snapshots have matching old-schema `v4`
entries with `allow_openmx_positions_box_from_out=false`. The current cache
uses `v5` plus loader policy, density symmetrization and preprocessing
signature. This deliberately invalidates old doubled-density snapshots.
Old v2/v3/v4 files therefore cannot satisfy the current key. No cache semantics
were changed and no old entries were promoted or relabeled.

A metadata-only probe through the actual factory (matrix dataset construction
replaced with path recording) found **55/100 current-key files** during this
investigation. Thus the cache is partially populated; the earlier observation
of only two new files does not establish that all accesses were misses. A real
current-key load succeeded in **0.124 s**:

```
Snapshot.load('/bigdata/casus/wdm/hamiltonian_learning/data/ZnCuSnSeS/snapshot_cache/HS_ebc8fe7793be5ca4897fe3fc4a288936.pt')
```

For first training snapshot `_055`, the old v4 key is
`HS_72f4def8c26834a5bf6d2c98e8bbc3a8.pt`, and the actual current key is
`HS_ebc8fe7793be5ca4897fe3fc4a288936.pt`. These both exist; the current file
was modified at 2026-11-10 04:32:12 +0100, after job 836817 began at 04:12:26.
The factory probe and historical-key evidence are saved on ROSI under
`researchops/runs/r5_error_distribution_20261110/cache_diagnostic_validation_only.json`.
The separate hand-reconstructed key-count probe omitted the resolved loader
policy and is not a valid current-cache hit count; its historical v4 counts
remain valid. Current-key hit counts above came from the real dataset method.

## Jobs and operational choices

- Original Zn **836817 FAILED, 01:58:37, exit 124:0**; left alone as requested.
- Backup **837021 FAILED, 00:00:09, exit 1:0** before cancellation. It was
  queued when work began. A reversible hold was attempted while silicon was
  queued, but Slurm reported it already finished. `scancel 837021` was still
  issued; no active backup remains. The existing launcher's frozen commit
  guard is a possible failure cause, not verified from its log.
- Silicon **837022 COMPLETED**; exact metric equality failed.
- **No new Zn job, no queue state, no attempt registration**: the explicit
  silicon gate forbids proceeding. The Zn launcher was not changed.

Only a separate silicon launcher was prepared on ROSI. Its inherited
`sb question` failure trap was removed and frozen hashes refreshed; paper/
output was redirected to researchops/. Its two-hour scheduler request was
shortened to 15 minutes to permit backfill. Before any future Zn submission,
its internal `timeout 7100s` must also be extended alongside a four-hour
scheduler limit. These launcher changes are operational files outside Git.

Choices to review/revert: the scoped factory-method interception in the
report evaluator; retaining all training metadata to guarantee mapper
identity; the ancestry-only merge preserving the early push; and the
shortened silicon time limit. No scientific definition was relaxed.
