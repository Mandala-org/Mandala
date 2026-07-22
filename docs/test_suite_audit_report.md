# Test Suite Audit Report

## Verified state

- 467 collected cases: 452 active and passing, 15 explicitly deferred.
- Unit: 377 passed, 7 skipped in 176 s serially; slowest test 18.8 s.
- Integration: 18 passed, 5 skipped; slowest 2.5 s.
- Equivariance: 39 passed; slowest 4.5 s.
- Physics: 12 passed; slowest 14.6 s.
- Analysis: 2 passed.
- Workflow: 4 passed, 3 skipped; slowest 2.0 s.
- All active tests observed in serial domain runs complete in under one minute.
- Ruff passes for the changed tests and timing-audit utility.

The optional full branch-coverage run was stopped at the user's request. Each
domain was nevertheless run and verified independently.

## Main repairs

- Repaired stale APIs for seven-value graph features, matrix-space irrep metrics,
  head-output wrapping, cache keys, and k-space complex dtypes.
- Updated deleted `Si_DM`/`info.txt` paths where real snapshot coverage remains
  useful.
- Replaced repeated 81 MB OpenMX parsing in algebra-only tests with deterministic
  scalar/vector and multi-species fixtures.
- Preserved real OpenMX parser, rotation-reference, spatial, and historical
  atom-image relabeling regressions.
- Re-enabled and accelerated precomputed/on-the-fly feature equivalence testing.
- Replaced random non-reverse-closed equivariance graphs with deterministic valid
  graphs.
- Added a genuine one-batch Lightning workflow test and explicit workflow markers.
- Added `scripts/audit_test_timings.py` for isolated subprocess timing with hard
  per-test timeouts.
- Enabled strict pytest configuration/markers and stopped silently ignoring the
  workflow directory.
- Limited default xdist parallelism to two workers because many workers loading the
  81 MB OpenMX fixtures simultaneously caused severe resource contention.

## Explicitly deferred cases

1. Seven tests for removed `scripts/train_silicon.py` are retained but skipped.
   They encode resume, split, worker, callback, and GPU-dataset contracts that need
   migration to `scripts/train.py` and `scripts/dataset.py`.
2. Five integration cases and one workflow case for `train_target="irreps"` are
   retained but skipped. `Config` documents matrix-only training and the current
   shared step converts predictions to matrices and requires `BlockMatrix` targets.
3. Two workflow tests that dynamically rewrite the former Hydra entry point are
   retained but skipped. The current entry point is argparse-based
   `scripts/wandb_run.py`; these should be rewritten around `run_training()`.

## Follow-up issues

- `Snapshot` cannot currently be safely copied with `copy.deepcopy` because its
  dynamic attribute handling recurses. Tests avoid depending on copying, but the
  production behavior deserves a focused fix.
- Force/stress prediction requires Hamiltonian, overlap, and density heads. With a
  Hamiltonian-only model the current failure is an indirect `NoneType` error rather
  than a clean precondition error.
- Real OpenMX fixtures are individually fast enough but parallel execution can
  become memory-bound. Keep these scientifically valuable regressions and avoid
  aggressive automatic worker counts.
- Coverage measurement and systematic identification of untested production
  branches remain for the next audit round.
