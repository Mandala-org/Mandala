# Operator benchmark pilots

The production path is `DatasetFactory → E3GNNDataset → E3GNN → scripts.train`.
File formats do not select an architecture or an independent training loop.
`DatasetFactory.add_snapshot(directory, purpose="train")` accepts raw OpenMX,
periodic HDF5, or PySCF inputs; explicit `(matrix_path, info_path)` pairs remain
supported. Raw OpenMX metadata can use material-specific names such as
`graphene.out`. Ambiguous directories require explicit filenames.

## Basis and units

The public monolayer benchmarks come from
[Zenodo 7553640](https://doi.org/10.5281/zenodo.7553640). Archive MD5 checksums and
extracted-file CRCs are verified by `scripts/download_operator_benchmarks.py`.
The archived Hamiltonians are in eV, lengths in Å, and matrix orbitals in the
authors' OpenMX order. DeepH-E3's `Rotate.openmx2wiki_H_full` uses exactly the
same s/p/d/f permutations as MANDALA; its model also orders displacement
coordinates `(y,z,x)`. HDF5 preprocessing does **not** mean these matrices have
already been converted to e3nn.

The shared reader converts energy eV → Hartree once, then `Snapshot` applies
the orbital permutation and matching Cartesian-axis change once. Repeated
`snapshot.to_e3nn()` is idempotent. The pilot YAMLs explicitly declare:

```yaml
data:
  representation: orbital_blocks
  graph_source: stored
  basis: e3nn
  source_basis: openmx
  source_cartesian_axes: xyz
```

Already-converted HDF5 can declare `orbital_basis: e3nn` and
`cartesian_axes: yzx` in each `info.json`, or use the equivalent
`data.source_basis` / `data.source_cartesian_axes` configuration. For e3nn
matrices accompanied by original xyz geometry, declare `cartesian_axes: xyz`:
only geometry is permuted. Conflicting metadata/configuration is rejected.
Untagged published HDF5 retains the documented OpenMX/xyz defaults. Marker
tests independently check all s/p/d/f permutations and conversion round trips.

## Comparable errors

`{train,val,test}/hamiltonian_mae_elementwise`, `mse_elementwise`, and
`rmse_elementwise` describe raw predictions in the global physical matrix
frame, in eV, eV², and eV respectively. Each observed directed element counts
once; padding and unobserved shell entries do not count. Every structure first
reduces over all its entries, then the epoch averages structures. Species-pair
sizes and diagnostic flags cannot change this reduction. Optional projected
validation errors use the suffix `_projected` and the full reverse-pair
Hermitian projection, never offsite-block self-symmetrization.

Historical `hamiltonian_mae` is retained for compatibility with saved runs and
their monitors. In the old paper configuration it sums species-pair means;
it must not be presented as the elementwise metric. Training objectives and
old restart controllers are unchanged by the additional diagnostics.

For frozen SiOx training, a diagnostics-only callback in
`researchops/runs/operator_benchmark_pilots_v1` uses the same metric helper,
evaluates the previous best checkpoint before continuing, and logs raw and
projected errors separately. The original source/runtime contract and saved
optimizer/scheduler state remain in use. The evaluation also reports a second
dataset-entry-weighted reduction explicitly, rather than conflating it with
the mean of structure-level errors.

## Frozen pilot protocol

| Setting | Graphene | MoS₂ |
|---|---:|---:|
| Structures | 450 | 500 |
| Train / validation / test | 270 / 90 / 90 | 300 / 100 / 100 |
| Split and initialization seed | 42 | 42 |
| Maximum GPU allocation | 2 × 48 h | 2 × 48 h |

Split membership uses sorted snapshot paths and a frozen NumPy permutation.
These match the published split proportions and the upstream default seed;
published split membership is not established by that seed. No claim is made
that unpublished reference split indices are identical. Full file hashes and exact
membership are retained in each run manifest. No validation/test labels fit
statistics or select pilot configurations.

`configs/pilots/*_mandala.yaml` retain the paper MANDALA architecture, learning
rate 0.0027, batch size 1, and Hamiltonian-only MAE objective. The SiOx-specific
radial envelope is disabled. Stored graph support is retained without target
distance filtering; the 10 Å setting controls radial encoding. Energy,
electron-count, force and stress objectives are disabled because these public
training archives do not supply their required labels. Selection and plateau
scheduling use raw elementwise validation MAE. A single seed per dataset is a
pilot, not a converged or statistically repeated benchmark claim.

Each zero-argument `scripts/train_operator_pilots/{graphene,mos2}_{1,2}.sh`
calls the model-independent `scripts.run_training_segment` CLI. Segment 2 has
an `afterok` dependency and restores only a safe epoch-boundary checkpoint.
Preparation is counted in the 45 h internal session budget, leaving time within
each 48 h allocation for shutdown and final evaluation. There are exactly two
segments; a failure requires inspection rather than an unbounded retry loop.
After segment 2, the selected best checkpoint is evaluated on validation and
held-out test data once using `scripts.evaluate_operator_checkpoint`.

All scripts, source hashes, specifications, scheduler records and bounded
metrics reside in `researchops/runs/operator_benchmark_pilots_v1`. The immutable
source snapshot was captured from a dirty `develop` tree, including the
scientific correctness fixes. Its manifest records the base commit and every
actual source hash. Later commits to `develop` do not alter that frozen source
or the SiOx checkout. Run IDs and scheduler state are dated operational records;
see the [investigation summary](mandala_accuracy_investigation.md).
Datasets and checkpoints remain under the documented bigdata root.

## Focused verification

```bash
PYTHONPATH=src:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 mandala-venv/bin/pytest -q \
  tests/unit/net/test_operator_metrics.py \
  tests/unit/data/test_source_conventions.py \
  tests/unit/data/test_deeph_e3_units.py \
  tests/workflow/test_operator_pilots.py
```

Additional targeted checks exercise raw/HDF5 equivalence through all three
reference configurations, original block supervision, cache behavior, and
epoch-boundary continuation. Broad unrelated suites are not required.
