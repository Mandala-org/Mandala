# Mandala Demos

These scripts are meant to be run directly and later converted to notebooks.

Principles:
- use Mandala classes and functions directly
- keep the flow explicit
- use the stateless evaluation helpers for checkpoint inference
- keep prints short and focused

Suggested order:
1. `demo_01_load_and_convert_snapshots.py`
2. `demo_02_dataset_and_alignment.py`
3. `demo_03_irrep_roundtrip.py`
4. `demo_04_model_forward_and_knobs.py`
5. `demo_05_tiny_training_observables_and_diagnostics.py`
6. `demo_06_silicon_snapshot_density_blockmatrix.py`
7. `demo_07_checkpoint_evaluation.py`

`demo_07` requires a trained checkpoint. It prints an actionable message when
`best_model.pt` is absent rather than fabricating predictions.
