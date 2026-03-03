# DeepH-E3 Comparison Study

This folder contains a repository-grounded comparison between:

1. `external/DeepH-E3` (reference implementation),
2. `studies/minimal_overfit_study` (explicit overfit study code),
3. main implementation under `src/` and `scripts/`.

## Goal

Explain where training and evaluation behavior can diverge (including convention handling), identify likely simple bugs, and define an execution plan for a strict apples-to-apples study.

## Contents

- `findings_repository_survey.md`
  - Complete source inventory and what each file contributes.
- `comparison_three_way.md`
  - Detailed three-way comparison: data, graph, model, loss, optimizer, metrics, eval.
- `spherical_harmonics_conventions.md`
  - Deep convention analysis (OpenMX and e3nn basis/order handling).
- `bug_hypotheses.md`
  - Candidate bugs and high-probability regressions to check first.
- `execution_plan.md`
  - Step-by-step plan for a rigorous comparison study.
- `scripts/README.md`
  - Planned comparison scripts and expected outputs.

## Scope of Sources

Main sources inspected include:

- `external/DeepH-E3/README.md`
- `external/DeepH-E3/deephe3/default_configs/*.ini`
- `external/DeepH-E3/deephe3/{model.py,e3modules.py,graph.py,data.py,kernel.py,utils.py,parse_configs.py,analyzer.py}`
- `studies/minimal_overfit_study/{overfit_water_minimal.py,common.py,strict_checks.py,analyze_model.py,detailed_logging.py}`
- `src/{data,core,net}` key modules: snapshot/graph_features/basis_converter/block_irrep_mapper/e3gnn/heads/layers/encoders
- `scripts/train_silicon.py`
- Prior local notes in `studies/deeph-e3_comparison/*`
