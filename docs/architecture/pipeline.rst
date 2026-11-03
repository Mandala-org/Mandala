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

   * - ``edge_partitions``, ``pred_pair_edges_static``
     - relation, object
     - The edges of every species pair, keyed by pair name, and their
       ``(L, i, j)`` labels; read by per-pair readouts.

Index slots carry their number of categories in ``width``
(``node_type_idx``: :math:`N`, ``edge_type_idx``: :math:`N^2`), from which
routed layers infer how many parameter copies they need.

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

Composite blocks
----------------

A recipe can name a subgraph and reuse it. Blocks declare their inputs,
outputs and options; instances are ordinary stages whose ``transform`` is the
block name:

.. code-block:: yaml

   pipeline:
     blocks:
       gated_update:
         inputs: [features, harmonics, radial]
         outputs: [out]
         options: {irreps_out: null, hidden: [16]}
         stages:
           - {id: conv, transform: equivariant_convolution,
              inputs: [features, harmonics, radial], outputs: [messages],
              options: {irreps_out: $irreps_out, radial_layers: $hidden}}
           - {id: act, transform: irrep_nonlinearity, inputs: [messages], outputs: [out]}
     stages:
       - {id: layer0, transform: gated_update, inputs: [edge_0, harmonics, radial],
          outputs: [edge_1], options: {irreps_out: "16x0e+8x1o"}}

Instances expand before compilation into stages ``layer0/conv`` and
``layer0/act`` writing slots ``layer0/messages`` and ``edge_1``. An option
value ``$name`` is replaced by the instance's option; ``null`` removes the
inner option so that the stage infers it; ``$(name)`` substitutes inside a
longer string, e.g. ``"$(width)x0e"``. Blocks may contain blocks.
``model.pipeline.groups`` records every instance, its options and the stages
it expanded into.

The flat MANDALA configuration lowers to exactly such a recipe
(:func:`net.component_defaults.default_graph`), with blocks ``edge_encoder``,
``node_update``, ``edge_update`` and ``readout``; every operation inside them
is a stage that recipes can replace.

Block libraries
~~~~~~~~~~~~~~~

The DeepH, DeepH-E3 and MACE layers ship as block libraries
(``src/net/block_library/*.yaml``). A recipe imports them by name, and its own
``blocks`` override library blocks of the same name:

.. code-block:: yaml

   pipeline:
     block_libraries: [mace, deeph_e3]
     stages:
       - {id: density_0, transform: density_interaction,
          inputs: [hidden_0, density_angular, density_radial, message_edge_index, node_batch],
          outputs: [density_0, message_0], options: {irreps_out: "64x0e+64x1o+64x2e"}}

======================  =====================================================
library                 blocks
======================  =====================================================
``deeph``               ``crystal_graph_update``, ``endpoint_edge_update``,
                        ``local_coordinate_readout``
``deeph_e3``            ``e3_node_update``, ``e3_edge_update``
``mace``                ``density_interaction``, ``symmetric_product``,
                        ``angular_expansion``
======================  =====================================================

:func:`net.blocks.block_ports` reports the port contract of each block (the
port of the first inner stage reading each input). A block may declare
``initialization_order``, the order in which its parameterized stages draw
their seeded parameters (a repeated id draws again). The library blocks use it
where the reference implementation creates parameters in a different order
than it computes, so that a seeded MANDALA model starts from exactly the
reference weights.

Ablations: bypassing a stage
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``enabled: false`` on a stage or a block instance bypasses it: its single
output is one of its inputs — the first, or the one a transform declares as
its ``bypass`` port (the update of ``residual`` and ``self_connection``).
Inside a block, ``enabled: $option`` makes a stage optional:

.. code-block:: yaml

   - {id: residual, transform: residual, inputs: [nodes, update], outputs: [out],
      enabled: false}                       # no skip connection
   - {id: node_update_2, transform: e3_node_update, ..., enabled: false}  # drop a layer

The pair envelope
~~~~~~~~~~~~~~~~~

``hamiltonian_envelope_mode`` and envelope-based loss weighting lower to two
stages for any model: ``pair_envelope`` evaluates the fitted per-pair envelope
on every edge (a static stage, cached with the sample) and ``envelope_scale``
multiplies the predicted Hamiltonian and overlap blocks by it. The pipeline
exposes ``edge_envelope`` as an auxiliary output for target normalization and
loss weighting; ``pair_distance_normalization: pair_r0`` reads the pair radius
from the same stage. A recipe can place both stages itself.

Structure-only stages
---------------------

A stage without parameters whose inputs depend on the structure alone (radial
bases, harmonics, edge roles, frames) is *static*. Datasets evaluate the
static stages once per structure and store their outputs with the sample;
the model reuses them while the geometry is unchanged and recomputes them
whenever positions or the cell require gradients (forces, stress).

Objectives
----------

The training loss is an expression over named terms:

.. code-block:: yaml

   objective:
     terms:
       H: {term: matrix_error, operator: hamiltonian}
       S: {term: matrix_error, operator: overlap, l1_fraction: 1.0}
       E: {term: band_energy}
     expression: "H + 0.5 * S + 1e-3 * E"

Terms are registered functions of the batch (:mod:`net.objectives`);
``objective_catalog()`` lists them. A term declares which of its options name
predicted operators (``operators=("operator",)``); these are its input ports
in MANDALA Studio and are checked against the model's outputs. The expression language has numbers, term
labels, ``+ - * /``, parentheses and ``sqrt log exp abs min max``. Without an
``objective`` section the flat loss settings (``loss_coef_*``,
``objective.weights``, ``l1_reg_coef``, ...) define the terms. Pipeline
outputs that are not operators reach the loss through ``output_error``.

Inspecting a pipeline
---------------------

.. code-block:: python

   outputs, slots = model.pipeline(x, return_trace=True)
   for sid, reads, writes in model.pipeline.bindings:
       print(sid, reads, "->", writes, model.pipeline.feature_specs[writes[0]])

``slots`` holds every intermediate tensor, which makes the trace a convenient
tool for equivariance debugging: rotate the input and compare slot by slot to
find the first stage that breaks the transformation law.

Without building a model, :func:`net.inspection.analyze_recipe` compiles a
recipe for a declared species and basis set and reports, per stage, the
completed options (given, inferred or default), output contracts, parameter
count and an asymptotic cost estimate; errors of all stages are collected
instead of raised:

.. code-block:: python

   from net.inspection import analyze_recipe

   report = analyze_recipe(recipe, species_shells={"Si": [0, 0, 1, 1, 2]})
   for stage in report["stages"]:
       print(stage["id"], stage["parameters"], stage["inferred"], stage["error"])

The registry
------------

Transformations are registered by name, optionally with a description of
their ports and options:

.. code-block:: python

   from net.pipeline import FeatureSpec, Option, Port, register_transform

   @register_transform(
       "my_layer",
       category="nonlinearity",
       summary="What the layer computes",
       inputs=[Port("features", None, "irreps")],
       options=[Option("width", "int", 16, key=True)],
   )
   def build(options, inputs, mapper):
       ...                       # validate the FeatureSpecs in `inputs`
       return module, [FeatureSpec(...)]

The factory receives the stage ``options`` completed from the declared
defaults and inference rules, the list of input specs and the shared
:class:`~core.block_irrep_mapper.BlockIrrepMapper`, and returns a
``torch.nn.Module`` together with one spec per output. Declared options are
checked strictly; ``Option(..., infer=...)`` derives a value from the wiring
(for example output irreps equal to the input irreps) and the compiled stage
records which options were given and which were inferred. Plugins listed under
``plugins`` in a recipe are imported before compilation. At run time
the module is called with the input tensors in order; if it sets
``accepts_physical_derivatives = True`` it also receives
``physical_derivatives=...``. :doc:`../extending/index` develops this into a
complete workflow.

.. transform-registry::
