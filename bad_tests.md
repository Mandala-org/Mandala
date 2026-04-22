# Bad Tests

## Tests encountered and corrected during this merge batch

- `tests/unit/net/test_net_common.py::test_hyperparams_defaults`
  - Problem: stale expectation set still allowed unsupported `Config.nonlin_kind` values such as `gate` and `id`.
  - Action taken: updated the assertion to match the actual supported activation kinds and the new `e3mlp_variant` default.

- `tests/unit/net/test_e3mlp.py`
  - Problem: imported helpers from another test module via `from tests.equivariance...`, which breaks standalone execution depending on how `pytest` resolves the `tests` package.
  - Action taken: inlined the tiny helper functions so the file runs independently.

- `tests/workflow/test_train_script.py`
  - Problem: hard-codes the old Hydra-based `scripts/train.py` contract and rewrites the decorator source at runtime. After the training-entrypoint redesign, that file no longer reflects the actual training workflow shape.
  - Action taken: not updated in this batch. It should be replaced with workflow coverage for `scripts/wandb_run.py` plus direct tests of `scripts/train.py::run_training`.
