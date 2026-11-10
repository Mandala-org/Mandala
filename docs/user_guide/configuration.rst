Configuration
=============

A run is described by one YAML document. It has two complementary layers:

* **flat fields** of :class:`~net.common.Config` (``cutoff_radius``,
  ``hidden_irreps``, ``node_update_message_agg``…), which configure the
  default MANDALA network and the training loop, and
* **sections** (``data``, ``training``, ``species``, ``parameters``,
  ``layout_rules``, ``codec``, ``objective``, ``optimization``, ``pipeline``,
  ``reference``), which describe data policies, explicit architectures and
  optimization.

Without a ``pipeline`` section the flat fields compile into the default
graph; with one, the stages are taken verbatim. Both produce the same
:class:`~net.e3gnn.E3GNN` class and use the same trainer.

A flat MANDALA configuration
----------------------------

.. literalinclude:: ../../examples/first_model/silicon.yaml
   :language: yaml

The most important flat fields:

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - Field
     - Effect
   * - ``cutoff_radius``
     - Graph and target cutoff in Å.
   * - ``l_max``, ``hidden_irreps``
     - Degree of the spherical harmonics and the hidden irrep layout. If
       ``hidden_irreps`` is null, ``hidden_base_dim`` channels at
       :math:`\ell=0` are halved for every degree up to ``l_max``.
   * - ``num_layers_gnn``
     - Number of node/edge message-passing layers.
   * - ``tp_type``
     - ``separate_weight`` or ``fully_connected`` convolution products.
   * - ``node_update_message_agg``
     - ``sum``, ``average`` or ``attention``.
   * - ``edge_encoder_style``, ``edge_encoder_use_sh_tensor_square``
     - ``rich`` or ``distance`` edge encoder; harmonic tensor square.
   * - ``use_self_connection``, ``*_update_residual``
     - Species self-connections and residual paths.
   * - ``e3mlp_variant``, ``internal_e3mlp_variant``, ``head_e3mlp_variant``
     - Equivariant MLP block types (:doc:`../theory/architecture`).
   * - ``nonlin_kind``, ``activation_scalar``, ``activation_gate``
     - Equivariant nonlinearity and its scalar functions.
   * - ``neck_depth``, ``head_e3mlp_layers``, ``head_pair_mode``,
       ``separate_shifted_self``
     - Readout structure; ``separate_shifted_self`` defaults to ``false`` and
       ``true`` is reserved for legacy checkpoint inference.
   * - ``matrix_targets``
     - Subset of ``hamiltonian``, ``overlap``, ``density``.
   * - ``hamiltonian_envelope_mode``, ``hamiltonian_envelope_path``
     - Fitted distance envelope (:doc:`../theory/architecture`).
   * - ``loss_l1_fraction``, ``matrix_loss_reduction``
     - Matrix objective (:doc:`../theory/objectives`).
   * - ``train_on_energy``, ``train_on_num_electrons``,
       ``loss_coef_observables``, ``train_observables_on_gt``
     - Observable guidance.
   * - ``spectral_loss_enabled``, ``spectral_loss_coef``, ``spectral_loss_*``
     - Spectral guidance.
   * - ``lr``, ``max_epochs``, ``lr_scheduler_*``, ``revert_on_spike``,
       ``checkpoint_monitor``
     - Optimization and model selection.

Every field with its default is listed in :doc:`../reference/configuration`.

The ``data`` section
--------------------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Key
     - Meaning
   * - ``root``
     - Directory searched recursively for snapshots; used with
       ``training.split``.
   * - ``splits``
     - Explicit ``train``/``val``/``test`` lists of snapshot directories.
   * - ``representation``
     - ``orbital_blocks`` (per-pair blocks in the e3nn convention) or
       ``packed_shells`` (shell codec in native order).
   * - ``basis``
     - Orbital convention of the targets: ``e3nn`` or ``openmx``.
   * - ``graph_source``, ``graph_radius``
     - ``stored`` DFT support or a ``radius`` graph.
   * - ``supports``, ``neighborhood_support``
     - Additional message and local supports (:doc:`data`).
   * - ``validate_hermitian``, ``validate_support``, ``canonicalize_edges``
     - Input checks and canonical edge ordering.
   * - ``matrix_files``, ``support_file``, ``frame_source``
     - File names of periodic HDF5 inputs and frame policy for local-frame
       architectures.

With ``root``, the snapshot directories are shuffled with
``training.data_split_seed`` (defaulting to ``training.seed``) and divided by
``training.split``: ``{train_ratio, val_ratio, test_ratio}`` or explicit
``{train_size, val_size, test_size}``.

The ``training`` section
------------------------

``seed``, ``data_split_seed``, ``dtype`` (``float32``/``float64``),
``batch_size``, ``max_epochs``, ``min_lr``, ``grad_clip_val``,
``grad_clip_algorithm`` and ``split``. Training keys that are also flat
fields must agree with them when both are given.

Recipes: explicit architectures
-------------------------------

A recipe adds ``pipeline`` (stages and outputs) and usually ``objective`` and
``optimization``. The reference presets are complete examples; their
``parameters`` section collects the numbers worth tuning:

.. code-block:: yaml

   parameters:
     cutoff_radius: 7.2
     node_irreps: 64x0e+32x1o+16x2e+8x3o+8x4e
   pipeline:
     stages:
       - id: radial_basis
         transform: gaussian_basis
         inputs: [edge_lengths]
         outputs: [radial]
         options: {start: 0.0, stop: '${parameters.cutoff_radius}', n_gaussians: 128}

.. code-block:: yaml

   objective: {kind: mse, epoch_weight: edges, weights: {hamiltonian: 1.0}}
   optimization:
     optimizer: {kind: adam, lr: 0.002, betas: [0.9, 0.999]}
     scheduler: {kind: plateau, factor: 0.5, patience: 120, monitor: val/loss, ema_alpha: 0.1}

``objective.terms`` adds registered per-structure scalar losses to the matrix
objective. Each entry has a ``name``, a nonnegative ``weight`` (default 1),
and optional ``options`` passed to the function. Import the registration module
before constructing the model; :doc:`../extending/new_observable` gives a
complete example and a two-epoch training command.

``codec`` configures the shell codec (``spinful``, ``ignore_parity``,
``sort_irreps``), ``species`` the ordering (``alphabetical`` or
``atomic_number``, or an explicit ``order``) and an optional fixed shell
vocabulary, and ``layout_rules`` derived shapes (:doc:`../architecture/pipeline`).

Overrides on the command line
-----------------------------

Any key can be overridden with ``--set`` using dotted paths and YAML values:

.. code-block:: bash

   PYTHONPATH=src:. python -m scripts.train --config configs/reference/deeph_e3.yaml \
     --set data.graph_source=radius --set data.graph_radius=6.0 \
     --set 'data.splits.train=[data/small/Si]' \
     --set 'data.splits.val=[data/small/Si_perturbed_scale1_078]' \
     --set parameters.node_irreps=16x0e+8x1o+4x2e \
     --output runs/deeph_e3_si --dry-run

``--dry-run`` resolves the configuration, builds the datasets and the model,
runs one forward and backward pass and prints the parameter count, the loss
and the split sizes without writing anything.

Python
------

.. code-block:: python

   from net.common import Config
   from net.configuration import resolve_configuration

   cfg = Config(cutoff_radius=6.0, num_layers_gnn=3)        # flat
   cfg = Config.from_mapping(yaml_document)                  # flat or sectioned
   resolved, codec = resolve_configuration("recipe.yaml",
                                           species_shells={"Si": [0, 0, 1, 1, 2]})
   cfg = Config.from_mapping(resolved)                       # fully resolved recipe

The former ``separate_shifted_self`` head handled blocks between an atom and
its periodic images separately from ordinary pairs. It was retired because
those same blocks become ordinary pairs when the unit cell is tiled, causing
predictions to depend on the cell representation. New training uses the pair
head for every block with ``i != j`` or a nonzero lattice shift and rejects
``separate_shifted_self: true`` and ``partial_train: shifted_self``. Historical
checkpoints retain their saved architecture and exact routing through
``E3GNN.load_from_checkpoint``; callers restoring a saved config and state dict
separately use ``E3GNN.from_checkpoint_config`` and then load the weights strictly.
