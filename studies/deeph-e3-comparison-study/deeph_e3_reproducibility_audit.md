# DeepH-E3 Reproducibility Audit Against Mandala

## Scope

This note summarizes whether Mandala can reproduce the behavior of DeepH-E3 when using DeepH-E3's default training settings from `deephe3/default_configs/train_default.ini`.

The comparison was based on the following DeepH-E3 files:

- `deephe3/default_configs/train_default.ini`
- `deephe3/parse_configs.py`
- `deephe3/kernel.py`
- `deephe3/model.py`
- `deephe3/data.py`
- `deephe3/graph.py`
- `deephe3/e3modules.py`
- `deephe3/utils.py`

The comparison was based on the following Mandala files:

- `src/net/common.py`
- `src/net/e3gnn.py`
- `src/net/layers.py`
- `src/net/encoders.py`
- `src/net/heads.py`
- `src/net/layer_norm.py`
- `src/net/activations.py`
- `src/data/gnn_dataset.py`
- `src/data/graph_features.py`
- `src/data/openmx_parser.py`
- `src/data/edge_alignment.py`
- `src/core/block_irrep_mapper.py`
- `scripts/dataset.py`
- `scripts/train.py`

## Verdict

Mandala cannot reproduce DeepH-E3's default training behavior exactly.

The closest match is a narrow regime:

- non-spinful data
- Hamiltonian-only targets
- batch size `1`
- no energy, force, stress, or electron-count losses
- ideally a single-element dataset such as silicon

Within that regime, Mandala can approximate the broad architecture and training setup, but it still diverges in data ingestion, edge construction, readout parameterization, normalization semantics, loss aggregation, and optimizer/scheduler behavior.

## Closest Matching Mandala Settings

```yaml
cutoff_radius: 7.2
l_max: 4
hidden_irreps: "64x0e+32x1o+16x2e+8x3o+8x4e"
hidden_base_dim: 64
edge_encoder_style: "deeph_e3"
tp_type: "separate_weight"
num_layers_gnn: 3
use_self_connection: true
e3layernorm: true
head_use_node_embeddings_for_self_edges: false
separate_shifted_self: false
head_e3mlp_layers: 1
neck_depth: 1
internal_e3mlp_layers: 0
head_use_tensor_square: false
head_use_mlp_log_scale: false
matrix_targets: ["hamiltonian"]
train_target: "matrix"
symmetrize_output: false
symmetrize_hamiltonian_targets: false
train_on_energy: false
enable_energy: false
train_on_num_electrons: false
enable_num_electrons: false
train_on_forces: false
enable_forces: false
train_on_stress: false
enable_stress: false
loss_l1_fraction: 0.0
lr: 0.002
use_lr_scheduler: true
lr_scheduler_factor: 0.5
lr_scheduler_patience: 120
lr_scheduler_min_lr: 3e-5
lr_scheduler_target: "val/loss_total"
revert_on_spike: true
revert_decay_patience: 20
revert_decay_rate: 0.8
revert_spike_factor: 2.0
grad_clip_val: 0.0
batch_size: 1
max_epochs: 3000
seed: 42
require_exact_edge_match: true
```

This is the closest available approximation, not an exact behavior match.

## Detailed Misalignment Inventory

### Hard blockers

1. DeepH-E3 trains directly from saved graphs or DeepH-preprocessed folders and HDF5 files. Mandala does not provide a native DeepH-E3 graph or HDF5 training path and instead expects snapshot-style inputs.
2. DeepH-E3 can derive graph edges directly from the hopping keys stored in HDF5 when `radius < 0`. Mandala builds an ASE neighbor graph first and only later aligns or repairs edge sets against targets.
3. DeepH-E3 groups outputs by orbital-shell target block identity through `target_blocks_type=all`. Mandala predicts full per-pair blocks through its `ReducedTensorProducts` mapping and per-pair heads.
4. DeepH-E3 uses one shared edge-output layer plus `e3TensorDecomp` reconstruction. Mandala uses split diagonal, off-diagonal, and shifted-self heads with separate per-pair projections.
5. DeepH-E3 computes one global masked MSE over matrix elements. Mandala averages losses within each pair key and then sums those pair losses, which changes effective weighting whenever multiple pair keys are present.
6. DeepH-E3 uses `Adam` with the custom `RevertDecayLR` controller. Mandala uses `AdamW` with Lightning's `ReduceLROnPlateau`.
7. DeepH-E3's main training loop does not clip gradients. Mandala clips gradients by default through Lightning trainer settings.

### Important architectural and numerical mismatches

1. DeepH-E3 `e3LayerNorm` subtracts means on all irreps and normalizes only scalars by default. Mandala subtracts means only for scalars and normalizes all irreps.
2. DeepH-E3's radial weighting acts per multiplicity copy through `e3ElementWise`. Mandala's `EquiConv` learns one radial weight per irrep entry.
3. DeepH-E3 initializes edge distance features with a fixed Gaussian basis and `stop=6.0`. Mandala's closest `deeph_e3` encoder uses `soft_one_hot_linspace` features followed by a learned linear projection.
4. Mandala always applies an extra post-convolution equivariant nonlinearity. DeepH-E3's default path does not include that extra stage in the same place.
5. DeepH-E3 encodes node species with an `e3nn.o3.Linear` from one-hot species features. Mandala uses a learned `nn.Embedding`.
6. Mandala symmetrizes Hamiltonian targets by default unless explicitly disabled. DeepH-E3 default training does not symmetrize labels before loss evaluation.

### Training pipeline mismatches

1. DeepH-E3's default split logic is ratio-based with NumPy shuffling. Mandala's dataset builders use different snapshot selection logic.
2. Mandala's `apply_cutoff_to_targets=False` path mutates dataset behavior in ways that do not correspond to DeepH-E3's target handling.
3. DeepH-E3's plateau scheduler is wrapped by a custom controller with validation-loss smoothing, cooldown handling, checkpoint reversion, and explicit decay after a sustained spike. Mandala originally lacked the revert-to-best spike recovery logic and only used the plain Lightning scheduler path.

### Output and observables scope mismatches

1. Mandala can train on Hamiltonian, overlap, density, energy, electron count, forces, and stress within one stack. DeepH-E3's default comparison target is centered around matrix prediction and the downstream evaluation path is organized differently.
2. For spinful datasets the answer becomes a strict "no": DeepH-E3 supports complex spinful labels, while Mandala's OpenMX path is explicitly `spin=0` only.

## Bottom Line

Mandala can imitate the broad shape of a DeepH-E3 Hamiltonian experiment, but it cannot currently reproduce DeepH-E3 exactly.

The main reasons are:

- different data ingress assumptions
- different edge construction semantics
- different readout and block-assembly parameterization
- different matrix loss aggregation
- different optimizer and scheduler stack
- unsupported spinful path

The newly added revert-on-spike callback narrows one scheduler mismatch, but it does not remove the larger architectural and data-path differences listed above.
