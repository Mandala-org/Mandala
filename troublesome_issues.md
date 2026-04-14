# Troublesome Issues

## Real-data H2O dataset/model setup can stall in tight test loops

- Symptom: standalone runs of real-data H2O-at-5A setup paths such as `DatasetFactory.create()` and `tests/unit/net/test_e3gnn.py::test_forward_h2o_cutoff5_with_split_head` did not complete within a reasonable tight-loop debugging window in this environment.
- What was tried:
  - direct `pytest` of the H2O forward test
  - direct timed Python probe around `DatasetFactory.create()`
- Current status:
  - synthetic regression tests for the new irrep-part path and the new E3MLP variants pass
  - the existing real-data H2O forward regression is still worth keeping, but it is too expensive or currently hanging for fast iteration
- Likely next debugging target:
  - isolate whether the time sink is in `Snapshot.from_openmx(...)`, `E3GNNDataset(...)` preprocessing, or downstream graph-feature construction
