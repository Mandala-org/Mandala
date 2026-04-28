# Mandala Appendix Planning and Exhaustive Config Inventory

This document is a source-of-truth draft for the paper appendix describing the Mandala configuration surface.
It mirrors the live appendix in `paper/appendix.tex` and is based primarily on the current `src/` implementation.

## Scope

This inventory covers the stable `Config` dataclass in `src/net/common.py`

This inventory does not treat the older study scripts as the primary API.
Where a feature still exists only in studies or only as a compatibility placeholder, that is marked explicitly.

## Proposed Appendix Structure

The cleanest appendix structure for the paper would be:

1. Backend and data-loading options
2. Geometry, graph construction, and preprocessing
3. Hidden representations and edge/node encoders
4. Message passing and tensor-product variants
5. Readout heads and equivariant MLP variants
6. Prediction targets, observable guidance, and physics-aware postprocessing
7. Optimization, regularization, and training stability
8. Logging, artifacts, benchmarking, and reproducibility
9. Constraints and appendix notes

## Primary Source Files

- `src/net/common.py`
- `src/net/e3gnn.py`
- `src/net/encoders.py`
- `src/net/layers.py`
- `src/net/heads.py`
- `src/net/e3mlp_variants.py`
- `src/net/activations.py`
- `src/net/observable_metrics.py`
- `src/data/gnn_dataset.py`
- `src/data/factory.py`
- `src/data/graph_features.py`
- `src/net/artifacts.py`

## 0. Top-Level Runtime Settings Outside `Config`

These are not fields on `Config`, but they are part of the effective Mandala setup and should likely appear in the appendix.

| Setting | Location | Allowed values | Meaning | Notes |
| --- | --- | --- | --- | --- |
| `convention` | `DatasetFactory(..., convention=...)`, `E3GNNDataset(..., convention=...)` | In practice `e3nn` is the default stable training convention; backend-specific basis conventions also exist (`openmx`, `fhi-aims`, `pyscf`) through `Snapshot.change_basis(...)` and parser paths. | Chooses the orbital / real-spherical-harmonics convention used after loading. | Important appendix item even though it is not a `Config` field. |
| snapshot registration | `DatasetFactory.add_snapshot(matrix_path, info_path, purpose=...)` | `purpose in {"train", "val"}` | Registers matrix/info pairs for dataset creation. | Backend type is inferred from file suffixes such as OpenMX (`.scfout` + info file) and PySCF (`.npz` + `.json`). |

## 1. Geometry, Basis, and Graph-Construction Options

These options control cutoff behavior, hidden irreps construction, geometric embeddings, and dataset-side graph preparation.

| Option | Type / default | Allowed values | Meaning | Notes and interactions |
| --- | --- | --- | --- | --- |
| `cutoff_radius` | `float = 7.0` | positive float | Global neighbor cutoff used for graph construction and, optionally, target filtering. | Used by `E3GNNDataset`, `graph_features`, and force/stress recomputation paths. |
| `l_max` | `int = 4` | nonnegative integer | Maximum angular momentum used for spherical harmonics and hidden irreps auto-construction. | Controls `Irreps.spherical_harmonics(l_max)` and `build_hidden_irreps(...)`. |
| `hidden_base_dim` | `int = 64` | positive integer | Base multiplicity for the auto-built hidden irreps at `l = 0`. | Higher `l` multiplicities are halved geometrically by `build_hidden_irreps(...)`. |
| `hidden_irreps` | `str \| None = None` | any valid `e3nn.o3.Irreps` string or `None` | Explicit hidden representation override. | If set, this bypasses `build_hidden_irreps(...)`. |
| `emb_use_odd_features` | `bool = True` | `True`, `False` | Whether auto-built hidden irreps include odd-parity channels. | Only affects the auto-built case when `hidden_irreps is None`. |
| `edge_type_emb_dim` | `int = 32` | positive integer | Dimension of the learned scalar embedding for edge types. | Used only by `edge_encoder_style="rich"`. |
| `edge_encoder_style` | `str = "rich"` | `"rich"`, `"distance"` | Selects the edge encoder family. | `"rich"` uses edge-type embedding + radial features + spherical harmonics. `"distance"` uses a distance-only scalar projection. |
| `edge_encoder_use_sh_tensor_square` | `bool = False` | `True`, `False` | Whether to apply `TensorSquare` to the spherical-harmonic features before combining them with edge-type and radial features. | Only relevant for `edge_encoder_style="rich"`. |
| `e3layernorm` | `bool = True` | `True`, `False` | Enables equivariant layer normalization after encoder and update blocks. | Used in `EdgeEncoder`, `EdgeUpdateBlock`, `NodeUpdateBlock`. |
| `n_radial` | `int = 64` | positive integer | Number of radial basis channels in `soft_one_hot_linspace(...)`. | Affects edge geometry embedding dimensionality. |
| `radial_layers` | `Sequence[int] = (128,)` | sequence of hidden widths | Hidden layer widths of the radial MLP used inside `EquiConv`. | Final output width is appended automatically by `RadialMLP`. |
| `radial_embedding_scale` | `str = "none"` | `"none"`, `"sqrt_n_radial"` | Optional scale applied to the radial basis embedding. | Implemented in `compute_edge_geometry_from_static_edges(...)`. |
| `apply_cutoff_to_targets` | `bool = True` | `True`, `False` | Whether target sparse matrices are filtered to the same cutoff as the graph inputs. | Dataset preprocessing option. Important for exact graph-target alignment semantics. |
| `require_exact_edge_match` | `bool = True` | `True`, `False` | Whether graph-target reconciliation must be exact. | Used in dataset alignment and preprocessing consistency checks. |
| `precompute_edge_features` | `bool = True` | `True`, `False` | Whether edge radial embeddings and spherical harmonics are cached in the sample payload. | If positions or box require gradients, features are recomputed at forward time regardless. |

## 2. Message Passing, Tensor Products, and Update Topology

These settings control the core equivariant graph network.

| Option | Type / default | Allowed values | Meaning | Notes and interactions |
| --- | --- | --- | --- | --- |
| `num_layers_gnn` | `int = 2` | positive integer | Number of message-passing blocks. | Each block performs one node update and one edge update. |
| `tp_type` | `str = "separate_weight"` | `"separate_weight"`, `"fully_connected"` | Tensor-product implementation used inside `EquiConv`. | `"separate_weight"` uses `SeparateWeightTensorProduct`, taken from the DeepH-E3 implementation. `"fully_connected"` uses e3nn's `FullyConnectedTensorProduct`. |
| `use_self_connection` | `bool = True` | `True`, `False` | Enables learned scalar-conditioned skip/self-connection tensor products in node and edge updates. | For nodes the conditioning basis is species one-hot; for edges it is edge-type one-hot. |
| `edge_update_residual` | `bool = True` | `True`, `False` | Adds residual connection to the edge update output when dimensions match. | Active in `EdgeUpdateBlock`. |
| `node_update_residual` | `bool = True` | `True`, `False` | Adds residual connection to the node update output when dimensions match. | Active in `NodeUpdateBlock`. |
| `edge_update_node_combine` | `str = "concat"` | `"concat"`, `"sum"`, `"tensor_product"` | Chooses how source and destination node features are combined before the edge update convolution. | `concat` uses a direct-sum representation, `sum` sums node irreps, and `tensor_product` uses a learned equivariant tensor product back into node-irrep space. |
| `node_update_message_agg` | `str = "sum"` | `"sum"`, `"average"`, `"attention"` | Aggregation rule for edge messages into node updates. | `attention` uses scalar multi-head weights from node queries and edge keys, with per-head equivariant value projections from edge features. |
| `node_update_attention_scalar_dim` | `int = 64` | positive integer | Total scalar query/key width used by attention aggregation. | Must be divisible by `node_update_attention_heads`. |
| `node_update_attention_heads` | `int = 4` | positive integer | Number of attention heads used when `node_update_message_agg="attention"`. | Per-head weighted message sums are averaged to produce the final node-space update. |

## 3. Readout Heads and Output Structure

These settings control the transition from the shared latent representation to operator-valued outputs.

| Option | Type / default | Allowed values | Meaning | Notes and interactions |
| --- | --- | --- | --- | --- |
| `neck_depth` | `int = 1` | positive integer | Depth of the shared per-class trunk (`diag`, `offdiag`, optional `shifted_self`) before matrix-specific projections. | Used in `DeepHead`. |
| `internal_e3mlp_layers` | `int = 0` | nonnegative integer | Additional equivariant refinement depth inside node and edge update blocks after the main convolution/update. | Used in both `NodeUpdateBlock` and `EdgeUpdateBlock`. |
| `head_e3mlp_layers` | `int = 1` | positive integer | Depth of the final matrix-specific equivariant projection MLPs in the heads. | Active head-depth knob in `DeepHead`. |
| `head_use_node_embeddings_for_self_edges` | `bool = True` | `True`, `False` | For zero-shift diagonal self edges, choose node features instead of edge features as head input. | Changes the diagonal branch input type in `DeepHead`. |
| `separate_shifted_self` | `bool = False` | `True`, `False` | Splits shifted-self edges (`i=j`, nonzero lattice shift) into their own readout branch. | Also affects `partial_train="offdiag"` semantics. |
| `head_use_tensor_square` | `bool = False` | `True`, `False` | Applies `TensorSquare` before the branch-specific projection head. | Enabled independently for diagonal, off-diagonal, and shifted-self branches. |
| `head_diag_output_scale` | `float = 1.0` | real scalar | Post-scale applied to diagonal and shifted-self head outputs. | Implemented via `post_scale` in `E3MLP`. |
| `head_offdiag_output_scale` | `float = 1.0` | real scalar | Post-scale applied to off-diagonal head outputs. | Implemented via `post_scale` in `E3MLP`. |
| `head_use_mlp_log_scale` | `bool = False` | `True`, `False` | Enables branch-specific learned log-amplitude scaling MLPs. | Each branch predicts a scalar log scale which is exponentiated and multiplied onto the vector output. |
| `head_log_scale_mlp_n_layers` | `int = 1` | positive integer | Depth of the scalar log-scale MLPs used when `head_use_mlp_log_scale=True`. | These log-scale MLPs are scalar (invariant) networks. |

## 4. E(3)-MLP Variants, Activations, and Normalization

These settings control the internal equivariant MLP family used in refinement blocks and heads.

### 4.1 Main variant selectors

| Option | Type / default | Allowed values | Meaning | Notes |
| --- | --- | --- | --- | --- |
| `e3mlp_variant` | `str = "basic"` | `"basic"`, `"normact"`, `"gate"`, `"gatemagnitudes"`, `"film"`, `"resnormact"`, `"resgatemagnitudes"`, `"bilinear"` | Global default E3MLP block family. | `"basic"` uses plain `Linear + make_nonlinearity`; all others use `src/net/e3mlp_variants.py`. |
| `internal_e3mlp_variant` | `str \| None = None` | any valid E3MLP variant or `None` | Override for internal message-passing refinement MLPs. | Falls back to `e3mlp_variant` when `None`. |
| `head_e3mlp_variant` | `str \| None = None` | any valid E3MLP variant or `None` | Override for readout-head projection MLPs. | Falls back to `e3mlp_variant` when `None`. |

### 4.2 Variant block parameters

| Option | Type / default | Allowed values | Meaning | Notes |
| --- | --- | --- | --- | --- |
| `e3mlp_output_scale` | `float = 1.0` | real scalar | Output scale passed into scaled linear layers used by non-basic E3MLP variants. | Affects all variant blocks unless overridden per call site. |
| `e3mlp_weight_init_scale` | `float = 1.0` | real scalar | Multiplicative scale on initial weights for variant blocks. | Used by `ScaledLinear`. |
| `e3mlp_residual_scale` | `float = 0.25` | real scalar | Initial layer-scale / residual strength for residual-style variant blocks. | Used by `ResidualBlock` and `BilinearSelfTPBlock`. |
| `e3mlp_pre_norm` | `bool = False` | `True`, `False` | Whether variant blocks use pre-normalization by `EquivariantRMSNorm`. | Only affects non-basic variant mode. |
| `e3mlp_norm_eps` | `float = 1e-8` | positive float | Numerical epsilon for `EquivariantRMSNorm`. | Only relevant for pre-norm variant paths. |
| `e3mlp_film_hidden_dim` | `int = 128` | positive integer | Hidden width of the invariant FiLM MLP. | Only used by the `"film"` variant. |

### 4.3 Activation and equivariant nonlinearity family

| Option | Type / default | Allowed values | Meaning | Notes |
| --- | --- | --- | --- | --- |
| `nonlin_kind` | `str = "normact"` | `"normact"`, `"s2act"`, `"gate_scalars_mlp"`, `"gate_magnitudes"` | Equivariant nonlinearity used by the `"basic"` E3MLP path and update blocks. | Implemented in `src/net/activations.py`. |
| `activation_scalar` | `str = "leakyrelu"` | `"silu"`, `"relu"`, `"gelu"`, `"tanh"`, `"sigmoid"`, `"leakyrelu"`, `"softplus"`, `"softsign"` | Scalar activation used in activation modules and variant blocks. | Shared across several activation factories. |
| `activation_gate` | `str = "softplus"` | `"silu"`, `"relu"`, `"gelu"`, `"tanh"`, `"sigmoid"`, `"leakyrelu"`, `"softplus"`, `"softsign"` | Gate activation used in gate-style activations. | Used by `make_nonlinearity(...)` and some E3MLP variants. |
| `activation_odd_scalar` | `str = "tanh"` | effectively `"tanh"` or `"identity"` | Activation applied to odd-parity scalar channels in variant blocks. | The helper `odd_safe_activation(...)` explicitly supports `"tanh"` and `"identity"`; unknown names fall back to `tanh`. |
| `activation_odd_gate` | `str = "tanh"` | effectively `"tanh"` or `"identity"` | Activation applied to odd-parity gate scalars in variant blocks. | Same caveat as `activation_odd_scalar`. |
| `s2act_res` | `int = 128` | positive integer | Resolution parameter for e3nn `S2Activation`. | Only used when `nonlin_kind="s2act"`. |
| `norm_kind` | `str = "component"` | `"component"`, `"norm"` | Normalization mode for `NormActivation`. | Only used when `nonlin_kind="normact"`. |

## 5. Prediction Targets, Observable Guidance, and Physics-Aware Options

These settings determine what the model predicts, how supervision is defined, and what observable-level guidance is enabled.

| Option | Type / default | Allowed values | Meaning | Notes and interactions |
| --- | --- | --- | --- | --- |
| `matrix_targets` | `list = ["hamiltonian", "overlap", "density"]` | currently `hamiltonian`, `overlap`, `density` | The set of sparse operators predicted by the model. | Observable training has hard requirements on which subsets must be present. |
| `train_target` | `str = "matrix"` | `"matrix"`, `"irreps"` | Whether direct supervision is applied in block-matrix space or irrep-vector space. | Governs both dataset target preparation and forward/loss logic. |
| `partial_train` | `str \| None = None` | `None`, `"diag"`, `"shifted_self"`, `"offdiag"` | Restricts loss computation to a subset of edge classes. | In `offdiag` mode, shifted-self edges are excluded only if `separate_shifted_self=True`. |
| `train_on_irrep_parts` | `bool = False` | `True`, `False` | Computes matrix-space loss after projecting predictions and targets into separate irrep components. | Cannot be combined with `train_target="irreps"`. |
| `enable_energy` | `bool = True` | `True`, `False` | Enable energy observable evaluation and logging. | Requires energy targets in the dataset if you want metrics. |
| `enable_num_electrons` | `bool = True` | `True`, `False` | Enable electron-count observable evaluation and logging. | Uses `Tr(DS)` with trace-aligned sparse matrices. |
| `enable_forces` | `bool = False` | `True`, `False` | Enable force evaluation and logging. | Forces are computed by autodiff through `Snapshot.get_energy()`. |
| `enable_stress` | `bool = False` | `True`, `False` | Enable stress staging and prediction support. | Stress is computed by autodiff through `Snapshot.get_stress()`. |
| `train_on_energy` | `bool = True` | `True`, `False` | Add energy loss term. | Requires `loss_coef_observables != 0`. Also requires `hamiltonian` and `density` prediction unless `train_observables_on_gt=True`. |
| `train_on_num_electrons` | `bool = True` | `True`, `False` | Add electron-count loss term. | Requires `loss_coef_observables != 0`. Also requires `overlap` and `density` prediction unless `train_observables_on_gt=True`. |
| `train_on_forces` | `bool = False` | `True`, `False` | Add force loss term. | Requires `loss_coef_forces != 0`. |
| `train_on_stress` | `bool = False` | `True`, `False` | Add stress loss term. | Requires `loss_coef_stress != 0` |
| `train_observables_on_gt` | `bool = False` | `True`, `False` | For observable losses, combine one predicted operator with the complementary ground-truth operator instead of using only predictions. | Energy supports `H_pred D_gt` and `H_gt D_pred`; electron count supports `D_pred S_gt` and `D_gt S_pred`. |
| `log_partial_gt_observables` | `bool = False` | `True`, `False` | Log hybrid observable diagnostics even when not training on them. | Useful for ablation and interpretability. |
| `symmetrize_output` | `bool = True` | `True`, `False` | Symmetrize predicted block matrices before matrix-space loss computation. | Only affects `train_target="matrix"`. |
| `symmetrize_hamiltonian_targets` | `bool = True` | `True`, `False` | Symmetrize Hamiltonian targets during dataset preprocessing. | Applied when building matrix targets from snapshots. |
| `rescale_density_to_num_electrons` | `bool = False` | `True`, `False` | Rescale predicted density matrices to match the target number of electrons before artifact evaluation. | This is used in artifact/metric evaluation rather than the training loss path. |

## 6. Optimization, Regularization, and Training Stability

| Option | Type / default | Allowed values | Meaning | Notes |
| --- | --- | --- | --- | --- |
| `lr` | `float = 3e-4` | positive float | Learning rate for `AdamW`. | Used in `configure_optimizers()`. |
| `max_epochs` | `int = 100` | positive integer | Intended training horizon. | Usually consumed by the Lightning trainer setup around the model. |
| `batch_size` | `int = 1` | positive integer | Batch size for dataset loaders. | Current logging assumes batch-size 1 semantics in several places. |
| `accumulate_grad_batches` | `int = 1` | positive integer | Gradient accumulation factor. | Typically consumed by trainer setup rather than the model itself. |
| `dropout` | `float = 0.0` | `0 <= p < 1` | Equivariant dropout on all irrep coefficients. | Used in node and edge update blocks. |
| `l1_reg_coef` | `float = 0.0` | nonnegative float | Global L1 regularization coefficient. | Added directly to total loss in `_shared_step(...)`. |
| `l2_reg_coef` | `float = 0.0` | nonnegative float | Global L2 regularization coefficient. | Added directly to total loss in `_shared_step(...)`. |
| `init_weights_factor` | `float = 1.0` | positive float | Global post-construction multiplicative scale applied to all floating-point parameters. | Implemented in `_apply_init_weights_factor()`. |
| `grad_clip_val` | `float \| None = 0.5` | nonnegative float or `None` | Gradient clipping threshold. | Intended for trainer setup. |
| `loss_l1_fraction` | `float = 0.0` | between `0` and `1` | Blend between L2 and L1 block losses. | `0` means pure MSE, `1` means pure MAE, intermediate values interpolate. |
| `loss_coef_observables` | `float = 1e-5` | nonnegative float | Shared weight for energy and electron-count loss terms. | Must be nonzero when training on energy or electron count. |
| `loss_coef_forces` | `float = 0.0` | nonnegative float | Weight for force loss. | Must be nonzero when `train_on_forces=True`. |
| `loss_coef_stress` | `float = 0.0` | nonnegative float | Intended weight for stress loss. | Validated when `train_on_stress=True`, but the current training loop does not yet consume it. |
| `use_lr_scheduler` | `bool = True` | `True`, `False` | Enable `ReduceLROnPlateau`. | If false, optimizer is returned without scheduler. |
| `lr_scheduler_factor` | `float = 0.5` | positive float less than 1 | Multiplicative decay factor for `ReduceLROnPlateau`. | Used when `use_lr_scheduler=True`. |
| `lr_scheduler_patience` | `int = 60` | nonnegative integer | Plateau patience in epochs. | Used when `use_lr_scheduler=True`. |
| `lr_scheduler_min_lr` | `float = 1e-8` | nonnegative float | Minimum LR floor. | Used when `use_lr_scheduler=True`. |
| `lr_scheduler_target` | `str = "val/loss_total"` | logged metric name | Metric monitored by `ReduceLROnPlateau`. | Must match a logged metric key. |
| `revert_on_spike` | `bool = True` | `True`, `False` | Enable the custom "revert to best checkpoint after sustained spikes" callback behavior. | Callback-side option; not consumed directly by the model. |
| `revert_monitor` | `str \| None = None` | metric name or `None` | Metric monitored by the revert-on-spike callback. | Callback-side option. |
| `revert_decay_patience` | `int = 20` | nonnegative integer | Number of bad epochs before triggering revert/decay. | Callback-side option. |
| `revert_decay_rate` | `float = 0.8` | positive float less than 1 | LR decay applied after revert. | Callback-side option. |
| `revert_spike_factor` | `float = 2.0` | positive float | Defines the spike threshold relative to the best score. | Callback-side option. |

## 7. Logging, Evaluation Artifacts, and Benchmarking

| Option | Type / default | Allowed values | Meaning | Notes |
| --- | --- | --- | --- | --- |
| `run_name` | `str = "mandala-run"` | arbitrary string | Human-readable run name. | Typically propagated to checkpointing/logging wrappers. |
| `verbosity` | `int = 1` | integer | Controls summary-print verbosity. | Used by dataset and model summary helpers. |
| `wandb_project` | `str \| None = None` | string or `None` | Weights and Biases project name. | Used by outer training harness / logger setup. |
| `log_on_step` | `bool = False` | `True`, `False` | Log metrics on each training step. | Passed to `self.log_dict(...)`. |
| `log_on_epoch` | `bool = True` | `True`, `False` | Log metrics on epoch aggregation. | Passed to `self.log_dict(...)`. |
| `log_partial_gt_observables` | `bool = False` | `True`, `False` | Log hybrid observable diagnostics. | See Section 5. |
| `log_per_irrep_metrics` | `bool = False` | `True`, `False` | Compute and log per-irrep metrics during training/evaluation. | Can be expensive. |
| `print_per_irrep_metrics` | `bool = False` | `True`, `False` | Print per-irrep metrics to the console during evaluation callbacks. | Artifact/logging side. |
| `log_per_irrep_images` | `bool = False` | `True`, `False` | Save per-irrep matrix comparison images in artifact callbacks. | Used by `ArtifactCheckpointCallback`. |
| `log_activation_mag` | `bool = False` | `True`, `False` | Track and log per-irrep activation magnitudes. | Used in encoders and message-passing blocks. |
| `log_data` | `bool = False` | `True`, `False` | Emit detailed dataset-side logging. | Convenience/logging flag. |
| `log_forward` | `bool = False` | `True`, `False` | Emit detailed forward-pass logging. | Convenience/logging flag. |
| `benchmark` | `bool = True` | `True`, `False` | Enable timing/benchmark reporting paths. | Used by benchmarking / artifact-side tooling. |
| `log_interval` | `int = 1` | positive integer | Epoch interval for detailed artifact logging. | Used by `ArtifactCheckpointCallback` through `should_log_epoch(...)`. |
| `adaptive_log_interval` | `bool = False` | `True`, `False` | Use adaptive epoch logging cadence rather than a fixed interval. | Implemented in `src/net/silicon_study_logging.py`. |
| `video_max_atoms` | `int \| None = 6` | positive integer or `None` | Max atom count to include when rendering matrix/video artifacts. | Consumed by artifact plotting/video generation. |
| `log_model` | `bool = False` | `True`, `False` | Whether to log the full model artifact to WandB. | Outer logger / trainer setup concern. |

## 8. Runtime, Device, Caching, and Reproducibility

| Option | Type / default | Allowed values | Meaning | Notes |
| --- | --- | --- | --- | --- |
| `safety_checks` | `bool = False` | `True`, `False` | Enable expensive validation checks on graph alignment, shape matching, and geometry consistency. | Useful for debugging and appendix reproducibility notes. |
| `dtype` | `torch.dtype = torch.float32` | `torch.float32`, `torch.float64`, etc. | Default dtype for network modules and tensors where the config is consulted directly. | `OmegaConf` resolvers in `common.py` support config serialization. |
| `device` | `str = "cpu"` | any valid torch device string | Default device target. | Outer trainer setup usually supersedes this. |
| `gpus` | `int = 0` | nonnegative integer | Requested GPU count. | Trainer-side convenience knob. |
| `num_workers` | `int \| None = None` | nonnegative integer or `None` | DataLoader worker count. | `None` means automatic or harness-defined behavior. |
| `save_dir` | `str = "checkpoints"` | filesystem path | Root directory for checkpoints and artifacts. | Used by training harness / callbacks. |
| `snapshot_cache_dir` | `str \| None = None` | filesystem path or `None` | Location of raw `Snapshot` cache and preprocessed sample cache. | Used by `E3GNNDataset` caching path construction. |
| `dataset_device` | `str \| None = None` | torch device string or `None` | Optional target device for processed dataset payloads. | Dataset-side placement knob. |
| `seed` | `int = 42` | integer | Global random seed for reproducibility. | Typically consumed by the outer training script. |
| `tune` | `str \| None = None` | `None`, `"ray"`, `"wandb"`, or other harness-defined names | Hyperparameter tuning backend selector. | Harness-side option. |
## 9. Constraints and Recommended Appendix Notes

These constraints are important enough that the appendix should likely state them explicitly rather than leaving them implicit.

1. `train_on_energy=True` with `train_observables_on_gt=False` requires `matrix_targets` to include both `hamiltonian` and `density`.
2. `train_on_num_electrons=True` with `train_observables_on_gt=False` requires `matrix_targets` to include both `density` and `overlap`.
3. `train_on_energy=True` or `train_on_num_electrons=True` requires `loss_coef_observables != 0`.
4. `train_on_forces=True` requires `loss_coef_forces != 0`.
5. `train_on_stress=True` requires `loss_coef_stress != 0`.
6. `train_on_irrep_parts=True` cannot be combined with `train_target="irreps"`.
7. `partial_train="offdiag"` behaves differently depending on `separate_shifted_self`.
8. `hidden_irreps` overrides the automatic `build_hidden_irreps(l_max, hidden_base_dim, emb_use_odd_features)` logic completely.
