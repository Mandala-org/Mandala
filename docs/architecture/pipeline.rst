The typed feature pipeline
==========================

Every :class:`~net.e3gnn.E3GNN` — the default MANDALA network, the DeepH,
DeepH-E3 and MACE-H presets and any custom model — is a
:class:`~net.pipeline.FeaturePipeline`: an ordered list of *stages* that read
named feature *slots* and write new ones. Architecture names never select
code paths; they select YAML files.

Anatomy of a stage
------------------

.. code-block:: yaml

   pipeline:
     inputs:                     # optional extra input contracts
       message_batch: {domain: edge, representation: index, frame: none, support: message}
     stages:
       - id: radial              # unique module name
         transform: gaussian_basis
         inputs: [edge_lengths]  # positional, never inferred by name
         outputs: [radial]       # fresh slots; existing slots cannot be overwritten
         options: {start: 0.0, stop: 6.0, n_gaussians: 32}
         role: feature           # feature | encoder | message | readout
         initialization: {seed: 17, kind: component}
     outputs: {hamiltonian: hamiltonian_native}

* **Slots are write-once.** A stage cannot overwrite an input or a previously
  written slot, so residual links, parallel streams and long skip connections
  are explicit and the forward pass is a pure dataflow graph.
* **Inputs are positional.** The meaning of each input is defined by its
  position in the transformation's signature.
* **Outputs** map operator names (``hamiltonian``, ``overlap``, ``density``)
  to slots holding matrix-valued features on the output-edge support.
* **Roles** label stages. ``role: readout`` selects the trainable stages when
  ``freeze_backbone_train_heads_only`` freezes everything else.
* **Initialization** can override a stage's parameters with ``normal``,
  ``uniform`` or ``zeros`` (with ``std``, ``bound``, ``gain``) and isolate its
  random stream with ``seed``; ``component`` keeps the layer's own scheme.

Typed feature contracts
-----------------------

Every slot carries a :class:`~net.pipeline.FeatureSpec`:

.. list-table::
   :header-rows: 1
   :widths: 18 30 52

   * - Field
     - Values
     - Meaning
   * - ``domain``
     - ``node``, ``edge``, ``bond_neighbor``, ``graph``, ``relation``
     - The kind of entity each row belongs to.
   * - ``representation``
     - ``scalar``, ``irreps``, ``channel_irreps``, ``index``, ``geometry``,
       ``matrix``, ``local_components``, ``block_vectors``, ``object``
     - How the numbers are to be interpreted.
   * - ``irreps``
     - e3nn string, e.g. ``16x0e+8x1o``
     - The exact irrep layout of irrep-valued features.
   * - ``frame``
     - ``global``, ``bond``, ``none``
     - Coordinate frame. Bond-frame components are not global irreps.
   * - ``support``
     - ``output``, ``message``, custom names
     - The index set of edge-like entities. Two edge features with equal
       shapes but different supports are not interchangeable.
   * - ``angular_basis``
     - permutation of ``xyz``
     - Cartesian axis order of non-scalar irreps; global features use ``xyz``.
   * - ``width``
     - integer
     - Width of non-irrep features.
   * - ``batch``
     - layout mapping
     - How a custom relation is offset when graphs are batched.

When the pipeline is compiled, every stage's factory receives the specs of its
inputs, validates them and returns the specs of its outputs. Mismatches —
an edge feature where a node feature is expected, a vector where an invariant
is required, two supports mixed in one sum, a readout whose irreps differ from
the operator layout — raise an error at construction time.

Built-in input slots
--------------------

The dataset and :meth:`E3GNN.forward` provide these slots for every structure:

.. list-table::
   :header-rows: 1
   :widths: 24 22 54

   * - Slot
     - Contract
     - Content
   * - ``node_type_idx``
     - node, index
     - Species index in the resolved ``species.order``.
   * - ``node_one_hot``
     - node, ``Nx0e``
     - One-hot species.
   * - ``edge_type_idx``
     - edge, index
     - Index of the ordered species pair :math:`(a_i,a_j)`, one of :math:`N^2`.
   * - ``edge_one_hot``
     - edge, ``N²x0e``
     - One-hot species pair.
   * - ``edge_index``, ``edge_shift``
     - relation, index
     - Directed output edges :math:`(i,j)` and lattice shifts :math:`\bm L`.
   * - ``edge_vectors``, ``edge_lengths``
     - edge, geometry / scalar
     - :math:`\bm r_{ij}^{\bm L}` and :math:`\|\bm r_{ij}^{\bm L}\|`.
   * - ``positions``, ``box``
     - node / graph, geometry
     - Cartesian positions and lattice (rows), for differentiable geometry.
   * - ``node_batch``, ``edge_batch``
     - index
     - Graph membership in a batch.
   * - ``selfloop_edge``
     - edge, index
     - Mask of onsite edges.
   * - ``message_edge_index``, ``message_shift``, ``message_vectors``,
       ``message_lengths``, ``message_to_output``
     - support ``message``
     - A separately selected message graph (by default the output graph
       without onsite edges) and its map onto output edges (``-1`` if absent).
   * - ``bond_frames``, ``sub_*``
     - bond frame, bond neighbourhood
     - Local bond frames and bond-neighbourhood relations for local-coordinate
       architectures.

The flat MANDALA graph additionally uses ``edge_length_emb`` (Gaussian radial
basis), ``edge_sh`` (spherical harmonics), ``pred_pair_edges_static`` and
``edge_partitions``, which the model computes from the configuration.

Resolution: shapes from data
----------------------------

Recipes are resolved against the orbital vocabulary of the training data
before a model is built (:func:`net.configuration.resolve_configuration`).
Resolution fills in:

.. list-table::
   :header-rows: 1
   :widths: 32 68

   * - Variable
     - Value
   * - ``${species.order}``, ``${species.count}``, ``${species.pair_count}``
     - Species list (alphabetical or by atomic number), :math:`N`, :math:`N^2`.
   * - ``${species.shells}``
     - Shell degrees of each species, e.g. ``Si: [0, 0, 1, 1, 2]``.
   * - ``${layout.coupled_irreps}``
     - Irreps of the packed edge vector covering every species pair.
   * - ``${layout.shell_pairs}``, ``${layout.packed_dim}``, ``${layout.spinful}``
     - Codec description consumed by ``shell_tensor_decode``.
   * - ``${parameters.*}``
     - User-defined variables of the recipe.

``layout_rules`` compute further shapes, e.g. ``tensor_product_completion``
finds the smallest intermediate irreps from which a tensor product can reach
the coupled layout. Resolution also freezes ``pipeline.operator_layout`` — the
exact species order, shell assignment and spin convention — into the
checkpoint; inference verifies it before reconstructing blocks.

Inspecting a pipeline
---------------------

.. code-block:: python

   outputs, slots = model.pipeline(x, return_trace=True)
   for sid, reads, writes in model.pipeline.bindings:
       print(sid, reads, "->", writes, model.pipeline.feature_specs[writes[0]])

``slots`` holds every intermediate tensor, which makes the trace a convenient
tool for equivariance debugging: rotate the input and compare slot by slot to
find the first stage that breaks the transformation law.

The registry
------------

Transformations are registered by name:

.. code-block:: python

   from net.pipeline import FeatureSpec, register_transform

   @register_transform("my_layer")
   def build(options, inputs, mapper):
       ...                       # validate the FeatureSpecs in `inputs`
       return module, [FeatureSpec(...)]

The factory receives the stage ``options`` (a dict), the list of input
specs and the shared :class:`~core.block_irrep_mapper.BlockIrrepMapper`, and
returns a ``torch.nn.Module`` together with one spec per output. At run time
the module is called with the input tensors in order; if it sets
``accepts_physical_derivatives = True`` it also receives
``physical_derivatives=...``. :doc:`../extending/index` develops this into a
complete workflow.

.. transform-registry::
