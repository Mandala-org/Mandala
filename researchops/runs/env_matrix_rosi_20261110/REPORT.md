Environment matrix follow-up, 2026-11-10

Branch: referee-r8-r9-v2. Both H100 Python 3.13 and 3.14 GPU runs passed.

Files and latest commits at report preparation:

- `.github/workflows/tests.yml`: `04e7262`
- `docs/data/tested_environments.json`: `6d4a2f0`
- `docs/getting_started/installation.rst`: `20a327b`
- `docs/getting_started/tested_environments_table.rst.inc`: `6d4a2f0`
- `pyproject.toml`: `04e7262`
- `researchops/runs/env_matrix_rosi_20261110/README.txt`: `0914d6b`
- `researchops/runs/env_matrix_rosi_20261110/attempt1/py313-latest-cu13x-rosi.json`: `a9d5673`
- `researchops/runs/env_matrix_rosi_20261110/attempt1/py314-latest-cu13x-rosi.json`: `a9d5673`
- `researchops/runs/env_matrix_rosi_20261110/attempt1/slurm-837254.log`: `a9d5673`
- `researchops/runs/env_matrix_rosi_20261110/ci_results.json`: `9d25d64`
- `researchops/runs/env_matrix_rosi_20261110/env_cu13x_smoke.sbatch`: `d2fe08c`
- `researchops/runs/env_matrix_rosi_20261110/install_cuda_compat.py`: `18b36a6`
- `researchops/runs/env_matrix_rosi_20261110/prepare_hdf5_terok.sh`: `d2fe08c`
- `researchops/runs/env_matrix_rosi_20261110/results/py313-latest-cu13x-rosi.json`: `6d4a2f0`
- `researchops/runs/env_matrix_rosi_20261110/results/py313-latest-cu13x.json`: `6d4a2f0`
- `researchops/runs/env_matrix_rosi_20261110/results/py314-latest-cu13x.json`: `6d4a2f0`
- `researchops/runs/env_matrix_rosi_20261110/results/py314-t2xx-cu128.json`: `6d4a2f0`
- `researchops/runs/env_matrix_rosi_20261110/results/py315-latest-cu13x.json`: `6d4a2f0`
- `researchops/runs/env_matrix_rosi_20261110/submissions.json`: `a9d5673`
- `researchops/runs/env_matrix_rosi_20261110/verify_environment.py`: `d2fe08c`
- `researchops/runs/env_matrix_rosi_20261110/wheel_selection.json`: `9d25d64`
- `scripts/report/render_tested_environments.py`: `e424219`
- `tests/unit/scripts/test_reproduce_examples.py`: `3f175ae`

Verification commands

```bash
export PATH="$HOME/.local/bin:$PATH"
export PYTHONPATH=src:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 WANDB_MODE=offline MPLBACKEND=Agg
CUDA_VISIBLE_DEVICES="" python -m pytest -n 16 -q tests/unit tests/physics tests/equivariance -m "(unit or physics or equivariance) and not (gpu or slow or large_data or network or integration)" --ignore=tests/unit/data/test_snapshot_spatial.py --ignore=tests/physics/test_pyscf_rotation_reference_files.py
MANDALA_RUN_GPU_TESTS=1 python -m pytest -q tests/gpu/test_cuda_smoke.py
MANDALA_RUN_GPU_TESTS=1 python -m pytest -n 4 -q tests -m gpu
python3 scripts/report/render_tested_environments.py --check
```

The CPU command runs in each tested interpreter. JSON records include absolute
interpreter paths, exact installation commands, JUnit counts and wall times.
The four current CPU runs each pass 593 unit + 69 physics + 39 equivariance
tests, with 68 optional-reference skips and 52 successful subtests.

Choices and limits

- No production model code changed. The CPU reproduction tests now mock cuDNN
  version metadata alongside CUDA availability/device metadata.
- Python 3.15 needs source h5py 3.16.0 and local HDF5 1.14.5 headers/libraries.
  No Python 3.15 GPU run was requested in Part B; metadata remains <3.15.
- ROSI H100 actually has kernel driver 570.158.01. The retry uses the NVIDIA
  cuda-compat-13-4 userspace libraries, driver 615.71.10, in this worktree only.
  The package is pinned, SHA256 checked, and activated via LD_LIBRARY_PATH.
- CUDA CRT and NVVM are pinned to 13.4.92 with nvcc 13.4.92. Unpinned 13.5
  dependencies produce PTX 9.5 that nvcc 13.4 cannot assemble.
- All 10 original records are unchanged. Initial ROSI failures are retained.
- Terok Sphinx HTML build passes with one warning: Graphviz dot is missing.
  The GitHub documentation job installs graphviz explicitly.
- No paper/ or README.md files changed.

Final measured follow-up matrix (subprocess wall time):

| Environment (Python) | torch/tag | host / driver | CPU passed / skipped (s) | GPU smoke | GPU subset |
|---|---|---|---|---|---|
| py314-t2xx-cu128 (3.14.10) | 2.11.0+cu128 / cu128 | terok / 550.163.01 | 701 passed, 0 failed, 68 skipped (27.481 s) | 2 passed, 0 failed, 0 skipped (8.005 s) | not run |
| py315-t2xx-cu128 (3.15.2) | none / cu128 | terok / 550.163.01 | not run | not run | not run |
| py314-latest-cu13x (3.14.10) | 2.15.0+cu134 / cu134 | terok / 550.163.01 | 701 passed, 0 failed, 68 skipped (30.660 s) | not run | not run |
| py315-latest-cu13x (3.15.2) | 2.15.0+cu134 / cu134 | terok / 550.163.01 | 701 passed, 0 failed, 68 skipped (42.402 s) | not run | not run |
| py313-latest-cu13x (3.13.18) | 2.15.0+cu134 / cu134 | terok / 550.163.01 | 701 passed, 0 failed, 68 skipped (43.253 s) | not run | not run |
| py313-latest-cu13x-rosi-attempt1 (3.13.18) | 2.15.0+cu134 / cu134 | gh003 / 570.158.01 | not run | 0 passed, 2 failed, 0 skipped (13.146 s) | 0 passed, 4 failed, 0 skipped (15.695 s) |
| py314-latest-cu13x-rosi-attempt1 (3.14.10) | 2.15.0+cu134 / cu134 | gh003 / 570.158.01 | not run | 0 passed, 2 failed, 0 skipped (13.777 s) | 0 passed, 4 failed, 0 skipped (15.684 s) |
| py313-latest-cu13x-rosi (3.13.18) | 2.15.0+cu134 / cu134 | gh003 / 570.158.01 | not run | 2 passed, 0 failed, 0 skipped (29.850 s) | 4 passed, 0 failed, 0 skipped (20.433 s) |
| py314-latest-cu13x-rosi (3.14.10) | 2.15.0+cu134 / cu134 | gh003 / 570.158.01 | not run | 2 passed, 0 failed, 0 skipped (8.370 s) | 4 passed, 0 failed, 0 skipped (13.757 s) |

The GPU subset includes the smoke tests; these counts are not disjoint.

Newest supported: Python 3.14.10, torch 2.15.0+cu134, CUDA source
torch-scatter 2.1.2, torch_geometric 2.8.1, H100 kernel driver
570.158.01 plus isolated cuda-compat-13-4 driver libraries 615.71.10.

ROSI job 837275 completed successfully (0:0) in 00:26:39, 4 CPUs, 32 GB.
First job 837254 retained failed GPU checks; its exit propagation bug was
fixed before the retry. Both attempts are registered in the second brain.

The final assembly commit adds the Python 3.14 H100 JSON, retry Slurm log,
validation.json, and this report; its hash is supplied in the final response.
