# Modular operator models

MANDALA assembles reference architectures and custom mixtures through one
`E3GNN`. A model recipe specifies meaningful transformations, their ordered
inputs and outputs, preprocessing, objective and optimization. The existing
flat `Config` path remains available for existing experiments and checkpoints.

## Start a run

Run commands from the repository root in the installed MANDALA environment:

```bash
PYTHONPATH=src:. python -m scripts.train_modular \
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

Alternatively `data.root` discovers snapshot directories and `training.split`
sets ratios or explicit sizes. `training.data_split_seed` defaults to the
training seed. Splits must be disjoint; unknown species or changed orbital
layouts in held-out structures are rejected. Species and shell vocabularies
come from training metadata, never held-out labels.

The output directory contains `specification.json`, `latest_checkpoint.pt`,
and `best_model.pt`. The immutable specification records resolved configuration,
input hashes, splits, runtime and fitted-statistics provenance. Restart with the
same configuration and output directory:

```bash
PYTHONPATH=src:. python -m scripts.train_modular \
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
Global irreps in `xyz` and cyclic `yzx` coordinates need an explicit
`irrep_cartesian_basis` adapter. Bond-frame orbital components are not global
scalar irreps. Invalid combinations fail during model construction.

| Transform family | Examples and purpose |
|---|---|
| Encodings | `embedding`, `one_hot`, `gaussian_basis`, `spherical_harmonics`, `cutoff_radial_expansion` |
| Equivariant messages | `endpoint_tensor_product_node`, `endpoint_tensor_product_edge`, `graph_irrep_norm` |
| Many-body aggregation | `neighbor_density`, then `symmetric_product`; `angular_degree_expansion` consumes explicitly saved features |
| Invariant/local messages | `scalar_gated_node`, `endpoint_scalar_edge`, `local_coordinate_message` |
| Existing MANDALA features | `endpoint_message_aggregation` reuses configurable attention; `partitioned_operator_readout` reuses `DeepHead` |
| Explicit adapters | `irrep_cartesian_basis`, `irrep_to_bond`, `bond_local_linear`, `shell_tensor_decode`, `local_orbital_output` |
| Composition/scaling | `irrep_linear`, `scalar_linear`, `add`, `concat`, `pair_distance_affine`, `species_irrep_affine` |

`FeaturePipeline` is the assembler, not another architecture implementation.
Component modules are in `net.transforms`; shell/spin codecs are in
`net.operator_codecs`. Paper-named model classes are not used.

## Native operator data and export

The modular loader consumes the common preprocessed native interchange:
`element.dat`, `orbital_types.dat`, `site_positions.dat`, `lat.dat`, `info.json`,
and explicit HDF5 support/targets. Positions are stored as 3×N, lattice vectors
as columns, distances in Å, Hamiltonians in eV, and orbitals in native OpenMX
order. HDF5 keys are `[Rx,Ry,Rz,i,j]` with one-based atom indices. SOC uses
complex spin-major full matrices. Unit/convention declarations are checked.

Both reverse directions, onsite and shifted-self edges remain distinct. No
automatic target symmetrization or Hermitian projection is applied. Shell
selection produces explicit masks; missing/nonfinite entries do not enter the
loss or fitted statistics. Disjoint-union batching offsets node, message-edge,
output-edge and local-neighborhood relations independently.

Reference equivariant presets use stored Hamiltonian support. The local-frame
preset derives frames from overlap support; `support_file: rc.h5` and
`frame_source: stored` reproduce a supplied original DeepH preprocessing result.
Derivation uses shortest independent bonds, preserving the declared input order
for ties. Collinear environments fail explicitly. Native raw-DFT conversion
remains with the existing parsers/converters; the modular loader does not run DFT.

For inference without operator labels, choose an explicit radius graph. This is
a different graph policy from reference stored support:

```bash
PYTHONPATH=src:. python -m scripts.predict_modular \
  --checkpoint /path/to/new_run/best_model.pt \
  --snapshot /path/to/native_geometry \
  --graph-radius 7.2 --output /path/to/predictions
```

Omit `--graph-radius` to use the frozen support policy. Export writes native
HDF5 matrices, global-frame values, per-entry coverage masks, geometry and
checkpoint provenance. Partial shell coverage stays explicit. Output blocks
remain raw directed predictions. The Python entry point is
`model.predict_operator_blocks(sample, codec, return_masks=True)`; it prepares
input device/dtype and validates orbital/frame compatibility.

## Objectives and scaling

Presets use global masked matrix-element MSE and Adam with explicit hyperparameters.
Epoch weighting is explicit: matrix elements for original DeepH, edges for
DeepH-E3/MACE-H. Complex reduction is also explicit: separate real components for
the original local readout, complex magnitudes for the coupled SOC readouts.
The equivariant presets monitor an EMA-smoothed validation loss before plateau
scheduling. `training.min_lr` is a termination threshold, not an optimizer floor.

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
forces the physical affine rule through the configured pipeline. Geometry
derivatives additionally require differentiable geometry/features and, for
local methods, differentiable bond frames. Cached preprocessing does not supply
those derivatives automatically. The legacy force/stress observable workflow
continues to use its existing representation; modular tensor objectives do not
silently enable legacy observable-loss flags.

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
