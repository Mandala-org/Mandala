# MANDALA accuracy component audit — 29 October 2026

This audit covers the local `develop` working tree, including pre-existing
uncommitted work. The remote immutable pilot source and actual run manifests
remain authoritative for claims about running experiments. Check matching file
hashes before attributing any running-pilot behavior to a local source finding.
No dataset, checkpoint or cluster computation was used for this source audit. Line references identify the inspected pre-ablation source; later optional-control insertions shift current line numbers. The source manifests and named functions provide exact provenance.

The source establishes several concrete differences from the repository's
pinned DeepH, DeepH-E3 and MACE-H recipes. It does **not** establish how much of
the Graphene/MoS₂ error gap each difference causes. The initial, single-seed pilot snapshot could not distinguish architecture
from optimization or objective effects. This is the dated source audit;
subsequent certificates, observations and compute amendments are summarized in
[the investigation status](mandala_accuracy_investigation.md).

## Confirmed findings and limits

| Component | Confirmed behavior and evidence | Interpretation |
|---|---|---|
| Radial channel dependence | Historical `EquiConv` emits `irreps_out.num_irreps` radial values but applies one shared value to every multiplicity in a grouped irrep term. See `src/net/layers.py:278`, and the preserved grouped-loop oracle in `tests/unit/net/test_architecture.py:143`. | A concrete capacity restriction and unused radial outputs; its effect on accuracy is unmeasured. It is historical behavior explicitly preserved by the existing test, so it must not be silently changed in the frozen pilots. |
| Optimizer decay | A flat configuration without `optimization` selects AdamW with only its learning rate specified (`src/net/e3gnn.py`, `configure_optimizers`, approximately line 2085). `make_optimizer` forwards unspecified options (`src/net/training_policies.py:200`). A local parameter-only probe confirms `weight_decay=0.01`; pilot `l2_reg_coef=0` does not disable AdamW decay. | Strong optimization confound against reference Adam with explicit zero decay. It is not evidence that decay is harmful on these datasets. |
| Supervision reduction | Legacy orbital-block supervision sums one mean loss per directed species pair (`src/net/e3gnn.py`, matrix branch beginning at approximately line 1357). The explicit elementwise diagnostics pool observed entries (`src/net/operator_metrics.py:13`). | For one-species Graphene these reductions coincide. For MoS₂, different species-pair sizes/frequencies change their relative gradient weights. The new explicit reduction knob enables this ablation without altering the legacy default. |
| Objective and units | The pilots use physical-block MAE in internal Hartree (`configs/pilots/graphene_mandala.yaml:78`, `:90`). Reference packed-shell objectives use native eV MSE (`configs/reference/deeph_e3.yaml:56`; `src/data/preprocessing.py:66`). | A reference-versus-pilot comparison changes loss shape, scale and parameterization together. A clean MAE→MSE ablation must retain the same orbital-block representation and units. |
| Depth and aggregation | Both pilots have two node/edge message steps, softmax attention, no update residuals and no species self-connection (`configs/pilots/graphene_mandala.yaml:10`). DeepH-E3 has three coupled steps (`configs/reference/deeph_e3.yaml:82`); original DeepH has five scalar steps and explicit residual additions (`configs/reference/deeph.yaml:96`, `:150`). | Plausible depth/conditioning effects. A full recipe swap cannot isolate them. Attention normalizes incoming weights; a sum reducer preserves coordination magnitude, so their difference is also an inductive-bias change (`src/net/layers.py:29`). |
| Many-body products | MACE-H explicitly builds three neighbor-density/symmetric-product steps with correlation 3 (`configs/reference/mace_h.yaml:25`, `:98`). Original MANDALA lowers to endpoint node and edge transforms (`src/net/component_defaults.py:38`). | A real architecture difference; MANDALA is already nonlinear and multi-hop, so it should not be described as lacking all many-body information. |
| Output calibration | MACE-H fits train-only pair/irrep affine statistics and uses identity backward scaling (`configs/reference/mace_h.yaml:179`; `src/net/training_policies.py:62`). Original pilot readout has separate onsite/shifted-self/offsite and pair heads but no fitted affine calibration (`src/net/heads.py:66`, `:116`). | Calibration and surrogate gradients can affect convergence independently of the MACE products. Pair-specific learned heads already exist; enabling affine statistics is not the same as merely adding species awareness. |
| Dataset-specific guidance | Public pilots explicitly disable energy/electron/force/stress/spectral objectives (`configs/pilots/graphene_mandala.yaml:74`, `:90`, `:102`). | These are Hamiltonian-only MANDALA pilots. They do not test a claimed benefit from observable guidance. Additional objectives require the actual overlap/density/observable labels; they cannot be enabled as a nominal architecture switch. |

Graphene and MoS₂ use the same pilot architecture/objective, while MoS₂ adds
multiple species, unequal orbital-block sizes and a more heterogeneous target
distribution. This makes species-pair weighting/calibration useful discriminating
hypotheses. It does not prove either explains the larger MoS₂ MAE.

## Preprocessing and metric contract

The existing path does not show a confirmed double conversion:

- `load_deeph_e3_block_matrix` divides archived Hamiltonians by
  `HARTREE_TO_EV` once (`src/data/deeph_e3_parser.py:318`). It explicitly retains
  reverse periodic keys and validates their Hermitian relation (`:327`).
- Source basis and geometry axes are separate declarations, with conflicts
  rejected (`src/data/deeph_e3_parser.py:201`). Geometry is normalized to the
  source orbital convention before `Snapshot` changes the combined
  representation (`:217`). The pilot declares OpenMX/xyz source and e3nn
  destination (`configs/pilots/graphene_mandala.yaml:173`).
- `Snapshot._change_basis` returns immediately when already in the requested
  basis (`src/data/snapshot.py:442`); the OpenMX→e3nn branch converts orbitals and
  geometry together (`:453`). Repeated `to_e3nn()` does not repeat the operation.
- Packed-shell preprocessing intentionally converts physical Hartree blocks
  back to native eV supervision (`src/data/preprocessing.py:66`).
  `IrrepsBlockData.pack_matrix` and `to_blocks` explicitly apply the corresponding
  forward/inverse unit and frame conversions (`src/data/block_matrix.py:980`,
  `:1091`). This representation boundary is different from double conversion
  of a stored target.
- Pipeline angular conventions are explicit, and global irrep features are
  normalized at their representation boundary (`src/net/pipeline.py:614`).
  A Cartesian convention change and a local bond-frame rotation are different
  operations; neither should be inferred from a filename.

Independent marker/round-trip tests cover these source conventions. Full
training/validation certificates were subsequently completed for the frozen
pilot source; see the dated investigation summary. Every new execution source
still requires its own certificate, reporting archive/source hashes, observed units, shell order,
support cardinality, maximum reverse residual, conversion round-trip error and
target-entry checksums. It should perform no fitted validation/test statistics.
Do not train a deliberately double-converted control: that tests a corrupt input
pipeline, not a scientifically valid preprocessing alternative.

The pilots retain stored Hamiltonian support and disable target-distance
filtering (`configs/pilots/graphene_mandala.yaml:154`;
`src/data/gnn_dataset.py:682`). The 10 Å setting is a radial encoding range,
not a confirmed graph cutoff. Gaussian features use `cutoff=False`
(`src/data/graph_features.py:84`), and no SiOx envelope is active. Measure
train-only distance quantiles and Gaussian feature norms before deciding whether
support outside the radial range is substantial. Changing the radial range on
fixed support is distinct from changing the message graph or removing targets.

All comparisons must report global physical-frame raw MAE and full reverse-pair
projected MAE separately. The diagnostic helper counts every observed directed
entry once and excludes entries outside coverage masks (`src/net/operator_metrics.py:13`). Epoch
entrywise metrics average per-structure values (`src/net/e3gnn.py`,
`_shared_step`, entry-metric logging). Also retain dataset-entry-weighted totals
when structure sizes differ. Historical pair-sum `hamiltonian_mae`, local-frame
MAE, global-frame MAE and irrep MAE are not interchangeable. General orbital
rotations preserve Frobenius MSE but need not preserve entrywise MAE.

Target Hermitian cleanup is enabled in the pilots but disabled in the reference
recipes (`configs/pilots/graphene_mandala.yaml:88`;
`configs/reference/deeph_e3.yaml:3`). Validate the raw source residual before
classifying this as harmless numerical cleanup. Training predictions are raw;
pilot validation projection is separately enabled (`:87`, `:185`), while the
selection/scheduling metric is explicitly raw elementwise MAE (`:54`, `:60`).

## Controlled validation-only experiments

Use the same archived inputs, exact split membership, initialization seeds,
training shuffle, source/runtime identity and output support for every paired
comparison. Report raw/projected MAE, RMSE, onsite/offsite contributions,
species-pair counts/errors, Hermiticity residual, completed updates/epochs,
learning-rate history and elapsed GPU time. Freeze the primary criterion as raw
global entrywise validation MAE; reserve the test labels for a later frozen
confirmation. Do not equate literature split proportions/seed with published
exact membership when indices are unavailable.

Minimal first-factor controls, all started fresh:

| Contrast | Only intended change | Why it is informative |
|---|---|---|
| Radial gate mapping | `radial_weight_granularity: term → copy` | Same radial MLP parameter shapes; tests whether independent radial dependence per irrep copy helps. |
| Weight decay | Explicit AdamW `weight_decay: 0.01 → 0.0` | Isolates default decay; retain LR, scheduler and rollback. Adam at zero decay is a later optimizer-family control. |
| Pair reduction | `matrix_loss_reduction: pair_sum → elements` | Tests MoS₂ pair weighting; Graphene is a one-species negative control. |
| Loss shape | `loss_l1_fraction: 1.0 → 0.0` | Tests MAE versus MSE on the same Hartree block targets; retain reduction. Interpret changed gradient scale explicitly. |
| Aggregation | `node_update_message_agg: attention → average` | Both normalize incoming contributions; then a separately registered sum arm can test coordination dependence. |
| Depth | `num_layers_gnn: 2 → 3` | Tests receptive-field/composition depth. Record parameter count and throughput; equal wall time and equal updates answer different questions. |

The factors above are independent definitions that can be preregistered together;
do not promote their combinations after inspecting the first few curves. A small
predeclared interaction grid for radial mapping × loss reduction, or reduction ×
loss shape, is preferable to assigning an interaction effect to one component.
Only launch an additional interaction after a result gate if it was not frozen
in advance. Repeat selected contrasts with additional initialization seeds on
the identical split before claiming a stable component effect.

After that gate, useful decompositions remain within the common interfaces:

1. Isolate the scheduler from the optimizer: raw plateau versus EMA plateau at
   identical monitor/factor/patience, then rollback on versus off. The reference
   recipes bundle different patience, cooldown, LR floor/termination, smoothing
   and rollback behavior; copying the full policy measures a bundle.
2. Add train-only affine output calibration to the same backbone, then compare
   `gradient_mode: physical` versus `identity`. Use explicit packing/codec stages
   to connect the existing partitioned head; compare calibrated versus
   uncalibrated controls with identical representation and loss. This needs a
   recipe adapter, not a MACE-H-specific training path. An onsite scalar species
   mean is a separate interpretable residual-baseline experiment; it must respect
   equivariance and be fitted only on training data.
3. Compare endpoint tensor-product transforms and MACE symmetric products on
   matched shared objective/data policies. Full reference recipes remain useful
   end-to-end controls but cannot identify a unique architecture cause.
4. Choose a radial range or pair-distance normalization using a frozen
   training-only descriptor certificate. `pair_distance_normalization: pair_r0`
   currently requires an envelope/r0 artifact (`src/net/e3gnn.py:686`), so it is
   not a self-contained boolean normalization switch.

Do not retune an architecture using test MAE, mix early and converged learning
curves, reuse pilot optimizer state for a fresh-factor comparison, fit output
statistics on all structures, or crop difficult long-range targets to improve
the reported metric.

## Shared implementation and focused verification

The optional `Config.radial_weight_granularity` lives beside the shared radial
options (`src/net/common.py:196`) and is copied into ordinary endpoint transforms
by `default_graph` (`src/net/component_defaults.py:12`). It has two values:

- `term` is the default and retains historical output/gradient/checkpoint
  behavior, including unused final radial outputs.
- `copy` assigns one radial value to every irreducible copy and shares it only
  across the copy's `2l+1` components. It is an accuracy ablation, not a validated
  accuracy correction.

Both modes preserve the radial MLP widths, initialization and checkpoint tensor
shapes. Strict loading of the same state is supported, while the chosen mode is
serialized in configuration. Existing running pilots are untouched. Frozen new
runs must explicitly record the mode and an updated immutable source manifest.

A local analytic probe with output `4x0e+2x1o` found six emitted gates but only
two used in `term`, with component map `[0,0,0,0,1,1,1,1,1,1]`. In `copy` all six
are used, with map `[0,1,2,3,4,4,4,5,5,5]`. Focused tests compare forward values
and every input/parameter gradient against explicit irrep-copy multiplication,
check active radial gradients, rotations/reflections, configuration lowering and
historical strict checkpoint loading. The old grouped-loop preservation test is
also retained.

```bash
PYTHONPATH=src:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  mandala-venv/bin/python -m pytest -q \
  tests/unit/net/test_radial_weight_granularity.py \
  tests/unit/net/test_architecture.py::test_equiconv_vectorized_radial_scaling_matches_pathwise_output_and_gradients
```

Result: **10 passed**. Ruff and scoped Git whitespace checks passed for these
changes. These checks certify the optional component behavior; they do not
measure benchmark accuracy or certify the remote run's frozen source identity.
