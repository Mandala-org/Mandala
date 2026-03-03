# Planned Comparison Scripts

These scripts are the implementation targets for the execution plan.

## 1. `phase1_basis_parity.py`

Purpose:

- Compare OpenMX->e3nn block conversion and block<->irrep mapping between:
  - `external/DeepH-E3/deephe3/e3modules.py` (`Rotate`, `e3TensorDecomp`),
  - `src/core/basis_converter.py` and `src/core/block_irrep_mapper.py`.

Outputs:

- `reports/phase1_basis_parity.yaml`
- optional parity scatter plots.

## 2. `phase2_graph_sh_parity.py`

Purpose:

- Compare graph edges, displacements, SH, and radial features between:
  - DeepH-E3 graph generation (`deephe3/graph.py`),
  - main src (`src/data/graph_features.py`),
  - minimal study build path (`overfit_water_minimal.py`).

Outputs:

- `reports/phase2_graph_sh_parity.yaml`
- mismatch examples JSON.

## 3. `phase3_loss_parity.py`

Purpose:

- Compute and compare different objective formulations on same frozen predictions:
  - masked global MSE,
  - per-key summed MSE,
  - optional per-irrep decomposed variants.

Outputs:

- scalar loss comparison + gradient-norm/cosine comparison report.

## 4. `phase4_minimal_ablation_runner.py`

Purpose:

- Launch controlled ablations for the minimal study:
  - SH mode,
  - loss weighting,
  - scheduler order,
  - symmetrization policy.

Outputs:

- `reports/phase4_ablation_table.csv`
- best/worst config summaries.

## 5. `phase5_training_parity_eval.py`

Purpose:

- Evaluate end-to-end parity on a matched dataset split and matched metrics.

Outputs:

- final parity report and trajectory plots.

## Notes

- Keep all scripts deterministic and write explicit run manifests.
- Use strict edge matching by default in all comparison scripts.
