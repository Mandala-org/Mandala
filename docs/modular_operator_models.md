# Modular operator models

MANDALA assembles reference architectures and custom mixtures through one
`E3GNN`. A model recipe specifies meaningful transformations, their ordered
inputs and outputs, preprocessing, objective and optimization. Flat `Config` settings compile into registered transformations too. There is
one dataset factory, model forward contract, objective implementation, optimizer
factory, training loop, checkpoint workflow and prediction interface.

## Start a run

Run commands from the repository root in the installed MANDALA environment:

```bash
PYTHONPATH=src:. python -m scripts.train \
  --config configs/reference/mace_h.yaml \
  --set data.root=/path/to/native_snapshots \
  --output /path/to/new_run --dry-run
```

Dry-run prepares the split, resolves orbital channels, fits training statistics,
constructs `E3GNN`, and runs a real forward/backward step. It writes no run
directory. Remove `--dry-run` to train. `--accelerator cpu` selects CPU explicitly.
`--max-wall-clock-seconds 170000` requests a stop at the next epoch boundary.

Explicit splits avoid dependence on directory discovery. Supply these in a copy
of a preset, retaining its other sections:

```yaml
data:
  splits:
    train: [/data/snapshot_001, /data/snapshot_002]
    val: [/data/snapshot_003]
    test: [/data/snapshot_004]
```

OpenMX `HS.out`/metadata pairs and native HDF5/text snapshots both enter
`DatasetFactory` and `E3GNNDataset`; conversion is optional. Alternatively
`data.root` discovers snapshot directories and `training.split`
sets ratios or explicit sizes. `training.data_split_seed` defaults to the
training seed. Splits must be disjoint; unknown species or changed orbital
layouts in held-out structures are rejected. Species and shell vocabularies
come from training metadata, never held-out labels.

The output directory contains `run_manifest.json`, `latest_checkpoint.pt`,
and `best_model.pt`. The immutable specification records resolved configuration,
input hashes, splits, runtime and fitted-statistics provenance. Restart with the
same configuration and output directory:

```bash
PYTHONPATH=src:. python -m scripts.train \
  --config my_run.yaml --output /path/to/new_run \
  --resume /path/to/new_run/latest_checkpoint.pt
```

Recovery uses the existing MANDALA epoch-boundary machinery, including optimizer,
scheduler, RNG and model-selection state. Fitted statistics are restored rather
than refitted. `best_model.pt` is for selection, not recovery. Source-specific
rollback and incomplete restart behavior are intentionally not reproduced.

## Configure transformations

The root sections are `schema_version`, `reference`, `parameters`, `species`,
`layout`, `layout_rules`, `codec`, `data`, `training`, `objective`,
`optimization`, and `pipeline`. `reference` is provenance, not dispatch.
`parameters` holds interpolation variables; `species` and `layout` are resolved
against the training orbital vocabulary. Tensor-product completion and
channel-harmonic layout rules are reusable mathematical shape operations.
Resolution also embeds `pipeline.operator_layout`: the exact frozen species
order, shell-channel assignments, spin convention and coupled permutation.
Inference validates this identity before reconstructing physical blocks.

Each stage has a unique `id`, a `transform`, positional `inputs`, fresh
`outputs`, and constructor `options`:

```yaml
pipeline:
  stages:
    - id: radial_features
      transform: gaussian_basis
      inputs: [edge_lengths]
      outputs: [radial]
      options: {start: 0.0, stop: 7.2, n_gaussians: 128, trainable: false}
  outputs: {hamiltonian: final_operator}
```

This illustrates stage syntax; a complete recipe must produce `final_operator`.
See the standalone reference and hybrid YAML files for executable compositions.
Saved outputs cannot be overwritten, so residual links and separate feature
streams remain explicit and reproducible. `pipeline.outputs` selects the final
operator slots. A forward trace is available with
`model.pipeline(inputs, return_trace=True)`.

The validator checks entity domain, representation, exact irrep layout, frame,
Cartesian angular convention, and support/index set. Message edges and output
edges are distinct supports. Equal tensor sizes do not make them interchangeable.
All shared global irrep features use canonical `xyz` coordinates. External
irreps are normalized at the representation boundary. Harmonic encodings are
constructed directly in that convention; native operator decoding converts
coordinates at the reconstruction boundary. Preset cyclic-axis declarations
record upstream provenance rather than spreading conversions through updates.
`irrep_cartesian_basis` remains a low-level representation adapter. Bond-frame orbital components are not global
scalar irreps. Invalid combinations fail during model construction.

| Transform family | Examples and purpose |
|---|---|
| Encodings | `embedding`, `one_hot`, `gaussian_basis`, `spherical_harmonics`, `cutoff_radial_expansion` |
| Equivariant messages | `endpoint_tensor_product_node`, `endpoint_tensor_product_edge`, `graph_irrep_norm` |
| Many-body aggregation | `neighbor_density`, then `symmetric_product`; `angular_degree_expansion` consumes explicitly saved features |
| Invariant/local messages | `scalar_gated_node`, `endpoint_scalar_edge`, `local_coordinate_message` |
| Existing MANDALA features | `endpoint_message_aggregation` reuses configurable attention; `partitioned_irrep_readout` reuses `DeepHead` |
| Explicit adapters | `irrep_cartesian_basis`, `irrep_to_bond`, `bond_local_linear`, `shell_tensor_decode`, `local_orbital_output` |
| Independent message operations | `endpoint_gather`, `radial_tensor_product`, `indexed_aggregation`, `projected_attention` |
| Geometry on fixed supports | `periodic_displacement`, `vector_lengths`, `bond_frames`, `bond_harmonics` |
| Independent feature operations | `gated_scalar_message`, `gaussian_envelope`, `scale_features`, `irrep_activation`, `species_tensor_product`, `irrep_residual` |
| Composition/scaling | `irrep_linear`, `scalar_linear`, `add`, `concat`, `pair_distance_affine`, `species_irrep_affine` |

`FeaturePipeline` is the assembler, not another architecture implementation.
Component modules are in `net.transforms`; shell/spin codecs are in
`core.operator_layout`. Paper-named model classes are not used.

## Native operator data and export

The shared loader accepts OpenMX files and the native interchange:
`element.dat`, `orbital_types.dat`, `site_positions.dat`, `lat.dat`, `info.json`,
and explicit HDF5 support/targets. Positions are stored as 3×N, lattice vectors
as columns, distances in Å, Hamiltonians in eV, and orbitals in native OpenMX
order. HDF5 keys are `[Rx,Ry,Rz,i,j]` with one-based atom indices. SOC uses
complex spin-major full matrices. Unit/convention declarations are checked.

Both reverse directions, onsite and shifted-self edges remain distinct. The
reference presets disable target symmetrization and output projection; these
remain explicit policies in the common interface. Shell
selection produces explicit masks; missing/nonfinite entries do not enter the
loss or fitted statistics. Disjoint-union batching offsets node, message-edge,
output-edge and local-neighborhood relations independently.

Reference equivariant presets use stored Hamiltonian support. The local-frame
preset derives frames from overlap support; `support_file: rc.h5` and
`frame_source: stored` reproduce a supplied original DeepH preprocessing result.
Derivation uses shortest independent bonds, preserving the declared input order
for ties. Collinear environments fail explicitly. Raw OpenMX inputs use the existing parser without an intermediate conversion
job. The loader does not run DFT.

For inference without operator labels, choose an explicit radius graph. This is
a different graph policy from reference stored support:

```bash
PYTHONPATH=src:. python -m scripts.export_prediction \
  --checkpoint /path/to/new_run/best_model.pt \
  --format native --structure /path/to/native_geometry \
  --graph-radius 7.2 --output /path/to/predictions
```

Omit `--graph-radius` to use the frozen support policy. Export writes native
HDF5 matrices, global-frame values, per-entry coverage masks, geometry and
checkpoint provenance. Partial shell coverage stays explicit. Output blocks
remain raw directed predictions. The Python entry point is
`model.predict_matrices(inputs)`. It returns ordinary `BlockMatrix` objects in
canonical Hartree units for Hamiltonians, with `coverage_masks`. The native
writer converts Hamiltonians to eV. `model(inputs)` always returns
`IrrepsBlockData`, including for explicit transformation graphs.

## Objectives and scaling

Reference presets use global masked matrix-element MSE and Adam with explicit hyperparameters.
Epoch weighting is explicit: matrix elements for original DeepH, edges for
DeepH-E3/MACE-H. Complex reduction is also explicit: separate real components for
the original local readout, complex magnitudes for the coupled SOC readouts.
Original DeepH clips the global gradient norm at 4.2 and multiplies the LR
by 0.4, 0.5 and 0.5 after zero-based epochs 500, 2000 and 3000. The declared
`milestone_decay` policy includes its epoch position in scheduler state. The
equivariant presets monitor an EMA-smoothed validation loss before plateau
scheduling. Reference presets explicitly disable inherited spike rollback and
select checkpoints using their declared validation loss. `training.min_lr` is a termination threshold, not an optimizer floor.

`species_irrep_affine` fits per-species-pair, per-irrep statistics on the exact
training manifest. Scalar channels receive a mean shift; scales pool components
within each irrep field with the reference standard-deviation convention.
Empty/constant fields safely keep unit scale. Fitted buffers and the training
manifest hash are checkpointed. Validation and test data cannot be passed to
the fitting API. This deliberately fixes the upstream full-dataset fit.

MACE-H's identity-backward scaling is intentional, documented with ablations in
[the paper and supplementary S7](https://arxiv.org/html/2508.15108v2#S7).
For coefficient `x`, mean `mu` and scale `sigma`, it computes
`x + stop_gradient(x*(sigma-1)) + mu`. Forward values equal `sigma*x+mu`,
while the coefficient Jacobian is one. This avoids multiplying coefficient
gradients by very large/small scales. The reported convergence benefit comes
from the source experiments; MANDALA has not repeated that scientific ablation.

Choose `gradient_mode: identity` to use this surrogate gradient, or `physical`
for the actual affine derivative. `model(inputs, physical_derivatives=True)`
forces the physical affine rule through the configured pipeline. The common forward refreshes differentiable geometry/features and local bond
frames when physical derivatives are requested. Force, stress, energy, electron
count and spectral objectives use the same training step for every graph.
Force and stress supervision requires corresponding labels and energy operators.

## Verification and extension

Focused offline verification requires no upstream installation:

```bash
PYTHONPATH=src:. python -m pytest -q \
  tests/unit/net/test_reference_fixtures.py \
  tests/unit/net/test_hybrid_pipelines.py \
  tests/unit/data/test_operator_dataset.py \
  tests/workflow/test_modular_training.py
```

Source differential tests accept `MANDALA_DEEPH_REFERENCE`,
`MANDALA_DEEPHE3_REFERENCE`, and `MANDALA_MACEH_REFERENCE_ROOT`. They compare
parameter initialization, intermediate primitives, native outputs, input and
parameter gradients, and optimizer steps under the same numerical backend.
See the coverage matrix for pins, scope and isolated oracle adapters.

To add a transformation, implement an ordinary reusable `nn.Module`, declare its
typed factory contract, add a focused analytic/source comparison, and demonstrate
a configuration using it. Initialization and parameter coordinates matter for
training fidelity. A forward-equivalent change of basis does not generally
preserve Adam's coordinate-wise moments.


## Shared Python interface and representation policies

Load either flat settings or a transformation recipe with
`Config.from_mapping(document)`, collect paths with `DatasetFactory(cfg)`, and
construct `E3GNN(mapper, cfg)` from the returned mapper. The factory resolves
shape variables against training metadata and updates that same configuration.
`model.fit_training_data(train_dataset)` prepares any declared fitted
transformations. The shared trainer calls it for fresh runs; continuation restores
checkpointed statistics. Do not fit on held-out samples.

`data.representation` chooses `orbital_blocks` or `packed_shells`. These are
supervision adapters, not model implementations. The original defaults use
orbital blocks and E3NN orbital basis. Reference presets explicitly request
packed shell entries and native OpenMX basis. `data.basis`, `geometry_dtype`,
`validate_hermitian`, `validate_support`, `canonicalize_edges`, `matrix_files`
and `support_operator` are data policies. Neither a file format nor an
architecture name changes the selected graph, optimizer or training loop.
Use explicit basis/frame transformations when connecting representations.

`species.ordering` selects alphabetical or atomic-number indexing; an explicit
`species.order` overrides it. `species.shells` can declare a vocabulary in
advance, including species absent from training. Otherwise only training
metadata establish the vocabulary. All splits must agree with that fixed shell
and spin layout. Shell order is preserved, including interleaved shells.

`GraphSample` contains inputs, targets and small metadata, while preserving
`x, y = sample`. The shared collator returns one sample or a `GraphBatch`.
For a batch the network runs once on a disjoint graph union, preserving batch
normalization and graph-local relations. Predictions are separated again before
observable and spectral calculations, each using its own cell and references.
Masked matrix objectives reduce across supervised entries for each operator;
observable objectives reduce across cells. The independent disjoint-union
utility is also available for testing tensor primitives.

The common step composes matrix objectives, missing-entry masks, partial-edge
selection, output symmetrization, envelope transforms, loss weighting,
regularization, observable guidance, spectral loss and derivative objectives.
`symmetrize_validation_output` explicitly controls
validation projection; reference presets disable it to retain their default
raw validation objectives. Target symmetrization is independently configurable
for Hamiltonian, overlap and density. Complex projection uses the Hermitian
adjoint. Irrep-part objectives require complete shell blocks; partial shells
need a masked matrix objective. Force/stress supervision requires labels,
differentiable geometry, and the operator outputs needed to construct energy.
Missing scientific prerequisites raise errors rather than silently disabling
requested training terms.

Stages can independently specify `initialization: {seed: 17, kind: normal,
std: 0.01, gain: 1.0}`. The default `kind: component` retains each mathematical
component's initialization, including the reference RNG draw order. Explicit
stage seeds isolate that stage's RNG stream. `uniform` and `zeros` are also
available. `role: readout` selects trainable readout stages when freezing the
backbone; it is independent of stage/model names.

Original node and edge updates are separate `endpoint_irrep_node` and
`endpoint_irrep_edge` transformations. `partitioned_irrep_readout` and
`partitioned_blocks_to_shells` separate learned readout from orbital packing.
These can be interchanged with scalar/local-coordinate, tensor-product,
neighbor-density, correlation and angular-expansion components. Composite
mathematical transformations remain reusable library components, with typed
shape, support, angular-basis and frame checks.

A native geometry directory works with the shared exporter for both original
and reference graphs. Structure-only OpenMX export uses the same geometry
providers. When a recipe used stored DFT support, choose an explicit
`--graph-radius` for structure-only inference; geometry cannot recover the
missing stored support. Native export retains coverage and provenance.

Fresh/continued runs share a single manifest containing configuration, input
hashes, source and dependency fingerprints, relevant environment settings,
external envelope/Fermi-cache hashes and fitted-statistics provenance. An
execution change requires a new run; do not bypass a mismatch to continue an
existing scientific trajectory. Historical original parameter names are
translated only at checkpoint loading.


## Interface migration

| Removed separate entry point | Shared replacement |
|---|---|
| `scripts.train_modular` | `scripts.train` and `run_training` |
| `scripts.predict_modular` | `scripts.export_prediction` and `model.predict_matrices` |
| `OperatorDataset` | `DatasetFactory` → `E3GNNDataset` with declared supervision adapters |
| `net.operator_codecs` | `BlockIrrepMapper` with `core.operator_layout` math utilities |
| Early-return model branch and tensor-only forward | One component graph and `IrrepsBlockData` output contract |
| Separate controller, collator, objective and statistics orchestration | Shared training step, graph batching, optimizer factories and train-only fitting lifecycle |
| Mandatory SiOx conversion and model-specific training dispatch | Raw input manifest and one training command; names select YAML presets |
| Separate specification/selection files | One run manifest and one dataset manifest |

Existing mathematical modules remain configurable transformations. They are not
standalone architecture implementations. Original flat settings compile into the
same typed graph as explicit recipes; reference and hybrid YAMLs remain useful
examples. The library retains historical parameter-name translation at checkpoint
loading, while removed production entry points have no duplicate implementations.

Public prediction serialization preserves the frozen layout, frame, units and
packed values. Physical block serialization retains coverage masks, including
when rows are reordered. Structure-only inputs honor the frozen species order
and require explicit radius support when stored DFT support is unavailable.


## Independent supports, geometry and relation batching

Support selection is a preprocessing policy, separate from differentiable
geometry evaluation. The output support remains the selected stored or radius
graph. Additional supports can be selected independently:

```yaml
data:
  supports:
    message: {kind: subset, source: output, exclude_self: true, cutoff: 7.2}
    local: {kind: subset, source: output, exclude_self: false, cutoff: 5.0}
  neighborhood_support: local
```

`kind: radius` with `cutoff` constructs an independent periodic support, which
may include edges absent from the output support. Its `*_to_output` mapping uses
-1 for absent edges. Without an explicit policy, message support is the output
support minus zero-shift onsite edges, matching the reference preset convention.
Bond-local neighborhoods must select edges with output-edge features; attempting
to reference missing output features fails explicitly. Support names and graph
membership are retained during position/cell differentiation. Topology must be
prepared per structure before batching; graph rebuilding after structural motion
is an explicit preprocessing action, not a derivative operation.
`rebuild_graph_supports` returns a newly selected support mapping without
mutating the old inputs; conflicting policies on already prepared supports
raise an error. Rebuilding the output graph uses the structure/data builder.

`prepare_graph_topology` builds missing relations. `evaluate_graph_geometry`
calculates vectors, lengths, local harmonics and frames on those fixed relations.
The compatibility helper `add_graph_views` delegates to these two functions;
its old `refresh` argument only recalculates geometry. Dataset geometry caches
include position/cell references; changed coordinates or requested derivatives
trigger recalculation. Forward-local geometry does not mutate dataset samples
or leave a consumed autograd graph in a subsequent training step.

Batching uses tensor layouts describing entity sets, concatenation axes and
references. For a configured external relation:

```yaml
pipeline:
  inputs:
    short_edges:
      domain: relation
      representation: index
      support: short
      batch: {cat_dim: 1, references: node}
```

The support name supplies the entity set. A mixed relation can declare
`references: [node, output]` and `index_axis: 0`. Standard fields have the same
layout declarations in `data.relations`; the collator contains no per-name
index-offset branches. Negative mapping sentinels remain negative, and untyped
integer fields or out-of-range indices are rejected.

`configs/hybrids/tensor_messages_projected_attention.yaml` demonstrates a
factorized radial tensor-product message map followed by the original projected
attention reducer, activation and residual stages. Explicit differentiable
periodic-displacement and length stages consume prepared relations and cannot
rebuild topology. The attention reduction is shared with the original MANDALA
node update, preserving its computations and parameter registration. Reference composites remain convenience operations;
the independent operations permit configuration-only alternatives. Scalar
composites also expose envelope width, activation choices, normalization,
aggregation and residual policies while retaining their reference defaults.

Fitted affine statistics retain their declared target-coordinate interpretation.
Their affine operator is normalized into canonical feature coordinates at the
representation boundary, including non-isotropic imported statistics. Training
statistics remain restricted to the training split.
