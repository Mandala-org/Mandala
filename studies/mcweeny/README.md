# Fixed-support McWeeny study — perturbed Silicon

## Finding

No accuracy improvement on the available saved predictions: all 1/2/3-step
variants worsen density MAE and both density-only and joint H/D band-energy
errors, with either reference or predicted overlap. SiOx is excluded as requested.
This is a **snapshot-level study**, not a full-dataset test score: the existing
best H/S/D fine-tune is evaluated on snapshots 078 and validation 090, with the
older `sleek-sweep-8` H/S/D model as a comparison on 078. We did not retrain,
reselect checkpoints, or choose an iteration count using these results.

Best-model validation snapshot 090, reference overlap:

| Applications | Density MAE | Density RMSE | Density-only band-energy error, eV/cell | Joint H/D band-energy error, eV/cell |
|---:|---:|---:|---:|---:|
| 0 | 2.63679e-5 | 5.69792e-5 | 0.23126 | 0.01956 |
| 1 | 1.97117e-4 | 6.96912e-4 | 5.33384 | 5.08494 |
| 2 | 2.91815e-4 | 1.00136e-3 | 2.46075 | 2.20770 |
| 3 | 3.42839e-4 | 1.17734e-3 | 2.26622 | 2.01243 |

See [the complete generated report](results/report.md) and
[machine-readable metrics and input hashes](results/results.json) for all six
model/snapshot/overlap cases, including raw-direction errors, Hermiticity
residuals, relative Frobenius errors, timings, and electron-count diagnostics.
The reference-density control itself acquires approximately `1.95e-4` density
MAE after one step. The bias of the fixed-support purification operation is
already larger than the model's original prediction error. This control does
not separately identify finite-occupation effects versus sparse truncation.

## Density normalization correction

Both text and processed-HDF5 OpenMX loaders now use `(D + Dᵀ)/2`, preserving
native spin=0 normalization. The writer no longer halves the density, and both
dataset cache versions are bumped. Existing cache files and checkpoints are
not rewritten. Manually saved snapshots and exported predictions are not
automatically migrated; old checkpoints still predict their original targets.

These saved predictions were trained against the previous doubled targets.
The frozen manifest therefore explicitly applies `saved_density_scale=0.5`
**once**, before the baseline and all purification steps. Fresh predictions
trained with the corrected loader should instead specify `1.0`. This is a
target-convention conversion, not a fitted correction or purification step.
Reference data are freshly parsed using the corrected loader.

For these nonmagnetic Silicon cases the reference per-spin `Tr(DS)` is 16 for
8 atoms. The report explicitly uses spin degeneracy 2 for total electron
counts and band energies: `N=2 Tr(DS)`, `E_band=2 Tr(DH)`. Holding reference H
fixed isolates density-induced energy error; predicted H measures joint error.
No existing Snapshot energy/force API is redefined by this study. Density-grid
utilities now label their output according to input normalization; no implicit
spin sum occurs there either.

## Implementation and mathematical conventions

`core.sparse_math.build_matmul_alignment(A, B, output)` precomputes

`(AB)[i,j,L] = Σ_(k,L1) A[i,k,L1] B[k,j,L−L1]`.

Different periodic images remain distinct; this is not a Gamma-point dense
product. Compatible orbital-block shapes are grouped for batched matrix
multiplication and indexed accumulation, with bounded-size chunks. Only
integer topology is prepared on CPU; numerical products retain PyTorch device,
dtype and autograd. Alignment is reused and rejects stale edge order/support.
Output edges, order, and even zero-valued blocks are retained exactly.

With P the original density support projection, every step computes

`X=P(SD); Y=P(DX); Z=P(YX); D_next=3Y−2Z`.

There are exactly three block products per step, with SD reused. **Intermediate
products also use the original support**, so this differs from truncating only
the final exact polynomial. It can break Hermiticity. Raw iterates feed the next
step; a global reverse-pair Hermitian projection is applied for physical metrics
only, with raw metrics reported alongside. No clipping, trace normalization, or
projection is hidden in the purification API. A fixed sparsity pattern preserves
orbital O(3) covariance; graph shifts are convolved rather than folded.

```python
from core.density_purification import McWeenyPurifier

matrices = model.predict_matrices(x)
density, overlap = matrices["density"], matrices["overlap"]
# Use unit-occupation density; explicitly convert legacy doubled targets first.
purifier = McWeenyPurifier(density, overlap)
one = purifier(density, iterations=1)
two = purifier(density, iterations=2)
three = purifier(density, iterations=3)
physical_three = 0.5 * (three + three.transpose())
```

Purification remains opt-in, not a new model default. The untruncated map
targets occupations 0/1 and is not guaranteed to improve finite-temperature
reference densities or preserve electron number.

## Reproduce

From the repository root, using the active environment and locally available
artifacts named in `silicon_cases.json`:

```bash
bash scripts/run_mcweeny_silicon.sh
```

This frozen zero-argument CPU launcher runs all six cases and all four iterates
with 64 threads. It does not submit a cluster job. Its output goes to
`analysis_outputs/mcweeny_silicon`; the checked-in results were generated locally
with the same CLI and 4 threads:

```bash
mandala-venv/bin/python scripts/evaluate_density_purification.py --manifest studies/mcweeny/silicon_cases.json --output-dir studies/mcweeny/results --threads 4 --chunk-size 4096
```

Run only the directly relevant regression tests (see `AGENTS.md`):

```bash
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 mandala-venv/bin/python -m pytest -q -p no:cacheprovider \
  tests/unit/core/test_density_purification.py \
  tests/unit/analysis/test_purification_evaluation.py \
  tests/unit/data/test_openmx_parser.py::test_dataset_factory_fixture_restores_the_real_openmx_loader \
  tests/unit/data/test_openmx_parser.py::test_density_symmetrization_averages_periodic_reverse_pairs \
  tests/unit/data/test_openmx_writer.py \
  tests/unit/data/test_deeph_e3_units.py \
  tests/unit/data/test_gnn_dataset_cache.py::test_corrected_density_normalization_invalidates_both_cache_layers
```

Final verification (2026-09-07): the targeted command above passed all 23
checks, including the fixture-restoration regression. No broad suite was rerun.
All 22 recorded source, manifest, and input hashes match the saved evaluation;
`git diff --check` is clean.

Regression coverage includes dense nonorthogonal and noncommuting polynomial
oracles, periodic convolution and truncation, mixed block sizes, all 0/1/2/3
iterations, rotation/inversion/reflection covariance, gradients through both D
and S, zero-path products, stale topology rejection, reference-aligned accuracy
and energy metrics, and loader/writer normalization.

## Changed files

- `src/core/sparse_math.py`: reusable periodic block-product alignments and
  chunked, differentiable batched multiplication on a specified output support.
- `src/core/density_purification.py`: opt-in `McWeenyPurifier` and
  `purify_density`, with the prescribed three-product order.
- `src/analysis/density_purification.py`: reference-aligned raw/physical density
  errors, band energies, occupation traces and reference controls.
- `src/data/openmx_parser.py`, `src/data/snapshot.py`: density Hermitian average
  in both text and processed-HDF5 OpenMX ingestion.
- `src/data/openmx_writer.py`: native-scale density export, without halving.
- `src/data/gnn_dataset.py`: invalidate both old density-target cache layers.
- `src/analysis/openmx_density_grid.py`: correct normalization documentation
  and output labels; no implicit spin sum.
- `scripts/evaluate_density_purification.py`, `scripts/run_mcweeny_silicon.sh`:
  manifest-driven evaluation, frozen launcher and reproducible reports.
- `tests/unit/core/test_density_purification.py`,
  `tests/unit/analysis/test_purification_evaluation.py`: new algebra and accuracy
  regression suites; OpenMX parser/writer, DeepH-E3 and cache tests are updated.
- `tests/conftest.py`: restore the real loader immediately after constructing
  the synthetic factory fixture. Its old session-long mock caused an unrelated
  rotation-file test to load synthetic H/O/Si data during the combined suite;
  a loader-restoration regression test is included.
