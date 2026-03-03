# Repository Survey Findings

This document lists what was found in-repo that documents or implements DeepH-E3-relevant algorithms, then maps equivalent locations in the minimal study and in `src/`.

## 1. DeepH-E3 documentation and configs

### Primary documentation

- Project overview and paper pointer:
  - `external/DeepH-E3/README.md:3`
- Usage pipeline (`preprocess -> train -> eval`):
  - `external/DeepH-E3/README.md:47`
  - `external/DeepH-E3/README.md:61`
  - `external/DeepH-E3/README.md:71`
- Explicit note to set `local_coordinate = False` during preprocess:
  - `external/DeepH-E3/README.md:49`

### Default configs (strong signal for intended algorithm)

- Data ingestion modes and expected files:
  - `external/DeepH-E3/deephe3/default_configs/base_default.ini:10`
  - `external/DeepH-E3/deephe3/default_configs/train_default.ini:34`
  - `external/DeepH-E3/deephe3/default_configs/eval_default.ini:29`
- Training defaults and scheduler/revert policy:
  - `external/DeepH-E3/deephe3/default_configs/train_default.ini:85`
  - `external/DeepH-E3/deephe3/default_configs/train_default.ini:106`
  - `external/DeepH-E3/deephe3/default_configs/train_default.ini:115`
- Target selection semantics (`target_blocks_type`, `selected_element_pairs`):
  - `external/DeepH-E3/deephe3/default_configs/train_default.ini:135`
  - `external/DeepH-E3/deephe3/default_configs/train_default.ini:142`
- Network defaults and warnings:
  - `external/DeepH-E3/deephe3/default_configs/train_default.ini:168`
  - `external/DeepH-E3/deephe3/default_configs/train_default.ini:159`

## 2. DeepH-E3 core implementation files

### Configuration parsing and hard constraints

- `only_ij` and `convert_net_out` asserted off in parser:
  - `external/DeepH-E3/deephe3/parse_configs.py:154`
  - `external/DeepH-E3/deephe3/parse_configs.py:160`
- SH/SBF mode selection and parity handling:
  - `external/DeepH-E3/deephe3/parse_configs.py:163`
  - `external/DeepH-E3/deephe3/parse_configs.py:168`

### Graph/data

- Edge feature definition `[dist, dx, dy, dz]`:
  - `external/DeepH-E3/deephe3/graph.py:68`
- Reverse edge key utility:
  - `external/DeepH-E3/deephe3/graph.py:60`
- Graph generation with periodic images and key construction `(sx,sy,sz,i,j)`:
  - `external/DeepH-E3/deephe3/graph.py:266`
  - `external/DeepH-E3/deephe3/graph.py:300`
- Aij mask/label assembly:
  - `external/DeepH-E3/deephe3/graph.py:316`
  - `external/DeepH-E3/deephe3/data.py:206`

### Model and irreps output conversion

- SH setup uses `normalize=True`, `normalization='component'`:
  - `external/DeepH-E3/deephe3/model.py:353`
- SH input vector reorder to `(y,z,x)` from edge_attr:
  - `external/DeepH-E3/deephe3/model.py:401`
- OpenMX/wiki basis transforms and rotation conversion:
  - `external/DeepH-E3/deephe3/e3modules.py:44`
  - `external/DeepH-E3/deephe3/e3modules.py:128`
- Wigner 3j based `get_H` / `get_net_out` decomposition:
  - `external/DeepH-E3/deephe3/e3modules.py:315`
  - `external/DeepH-E3/deephe3/e3modules.py:348`

### Training and evaluation

- Loss: masked MSE over selected elements:
  - `external/DeepH-E3/deephe3/utils.py:37`
  - `external/DeepH-E3/deephe3/utils.py:40`
- Train loop and scheduler step:
  - `external/DeepH-E3/deephe3/kernel.py:269`
  - `external/DeepH-E3/deephe3/kernel.py:299`
- Data split logic:
  - `external/DeepH-E3/deephe3/kernel.py:807`
  - `external/DeepH-E3/deephe3/kernel.py:828`
- Eval/test export and analyzer:
  - `external/DeepH-E3/deephe3/kernel.py:615`
  - `external/DeepH-E3/deephe3/analyzer.py:161`

## 3. Minimal overfit study files

### Main training script

- Core script and options:
  - `studies/minimal_overfit_study/overfit_water_minimal.py:82`
- Convention controls (`xyz-permutation`, `change-box`, `box-convention`):
  - `studies/minimal_overfit_study/overfit_water_minimal.py:285`
  - `studies/minimal_overfit_study/overfit_water_minimal.py:291`
  - `studies/minimal_overfit_study/overfit_water_minimal.py:298`
- Target canonicalization + symmetrization:
  - `studies/minimal_overfit_study/overfit_water_minimal.py:534`
  - `studies/minimal_overfit_study/overfit_water_minimal.py:559`
- Graph canonicalization and SH call:
  - `studies/minimal_overfit_study/overfit_water_minimal.py:612`
  - `studies/minimal_overfit_study/overfit_water_minimal.py:670`
- Strict edge checks:
  - `studies/minimal_overfit_study/overfit_water_minimal.py:726`

### Shared utilities

- Minimal model classes and metrics:
  - `studies/minimal_overfit_study/common.py:178`
  - `studies/minimal_overfit_study/common.py:477`
  - `studies/minimal_overfit_study/common.py:959`
- Edge canonicalization helper:
  - `studies/minimal_overfit_study/common.py:1523`
- DOS/eigen diagnostics:
  - `studies/minimal_overfit_study/common.py:1364`
  - `studies/minimal_overfit_study/common.py:1427`

### Comparison/eval helper scripts

- Analysis script with custom metrics and visualizations:
  - `studies/minimal_overfit_study/analyze_model.py:63`
  - `studies/minimal_overfit_study/analyze_model.py:113`

## 4. Main code in src/

### Data conventions and canonicalization

- OpenMX<->e3nn basis conversion matrices:
  - `src/core/basis_converter.py:32`
- Coordinate basis transforms in `Snapshot._change_basis`:
  - `src/data/snapshot.py:366`
  - `src/data/snapshot.py:376`
- Canonical edge ordering in snapshot and graph features:
  - `src/data/snapshot.py:142`
  - `src/data/graph_features.py:118`

### Graph feature generation

- Neighbor list and displacement formula:
  - `src/data/graph_features.py:60`
  - `src/data/graph_features.py:86`
- SH mode in main code:
  - `src/data/graph_features.py:193`

### Network/loss

- Lightning model and loss assembly:
  - `src/net/e3gnn.py:38`
  - `src/net/e3gnn.py:324`
  - `src/net/e3gnn.py:531`
- Head implementation:
  - `src/net/heads.py:21`

## 5. Existing prior local comparison artifacts

An older study folder exists:

- `studies/deeph-e3_comparison/study_plan.md:1`
- `studies/deeph-e3_comparison/run_study.py:1`
- `studies/deeph-e3_comparison/basis_equivalence_report.yaml`

Observed status:

- Basis-equivalence report says zero diff for its tested path:
  - `studies/deeph-e3_comparison/basis_equivalence_report.yaml`
- Equivariance part was intentionally skipped as incomplete:
  - `studies/deeph-e3_comparison/equivariance_report.yaml`
  - `studies/deeph-e3_comparison/run_equivariance_study.py:10`

## 6. What is documented versus what is still implicit

Explicitly documented in code/config:

- DeepH-E3 expects SH convention adaptation (`(y,z,x)` reorder).
- OpenMX<->wiki/e3nn block basis transforms are hard-coded and central.
- DeepH-E3 trains with masked global MSE.

Still implicit / easy to misalign:

- Exact weighting differences between per-key MSE accumulation and global masked MSE.
- Whether all edge sets are identical during loss and metrics (prefix truncation vs exact).
- Symmetrization policy differences (target symmetrization and prediction symmetrization timing).
