# Mandala Demo Repository

This folder contains a small, notebook-friendly demo sequence for Mandala.

The demos are designed to be run as plain `.py` files first and later converted
to `.ipynb` notebooks.

## Order

1. `demo_00_repo_overview.py`
2. `demo_01_load_snapshot_and_basis.py`
3. `demo_02_build_graph_and_align_targets.py`
4. `demo_03_block_irrep_roundtrip.py`
5. `demo_04_forward_pass_and_head_outputs.py`
6. `demo_05_observable_losses.py`
7. `demo_06_tiny_overfit_training.py`
8. `demo_07_multi_backend_comparison.py`
9. `demo_08_artifacts_and_diagnostics.py`

## Usage

Run any file directly:

```bash
python demos/demo_00_repo_overview.py
```

Each script adds the repo root and `src/` directory to `sys.path`, so it works
from a checkout without extra environment setup beyond the project
dependencies.

## Design Notes

- The demos prefer the bundled H2O OpenMX example.
- The backend-comparison demo degrades gracefully if a local PySCF artifact is
  not available.
- The training demo uses a tiny, local overfit loop so it stays fast.
- Every script is structured with notebook conversion in mind, using `# %%`
  cell markers where the flow benefits from it.
