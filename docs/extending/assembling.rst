Assembling a network from new layers
====================================

The twelve cookbook layers compose into a complete two-layer E3GNN that
predicts Hamiltonians for any chemistry. Nothing in the model is inherited
from the default network except spherical harmonics and the operator decoder.

.. graphviz::

   digraph cookbook {
     rankdir=TB; bgcolor="transparent"; nodesep=0.25; ranksep=0.28;
     node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=9.5, fillcolor="#f2faf6", color="#2f8f6b"];
     edge [color="#5a8fbd", fontname="Helvetica", fontsize=8];
     in [label="edge_lengths · edge_vectors · node_type_idx · edge_index · edge_type_idx", shape=plaintext, fillcolor=none];
     radial [label="bessel_basis"];
     angular [label="spherical_harmonics", fillcolor="#f3f7fb", color="#315b7d"];
     n0 [label="periodic_table_embedding"];
     e0 [label="radial_angular_encoder"];
     c1 [label="depthwise_convolution"]; a1 [label="tempered_softmax_aggregation"];
     m1 [label="equivariant_glu"]; nr1 [label="degree_rms_norm"]; s1 [label="squash_activation"];
     c2 [label="depthwise_convolution"]; a2 [label="tempered_softmax_aggregation"];
     m2 [label="equivariant_glu"]; s2 [label="squash_activation"]; r2 [label="rezero_residual"];
     eu [label="endpoint_bilinear_edge_update"];
     ro [label="pair_mixture_readout"]; env [label="learned_pair_envelope"];
     dec [label="shell_tensor_decode → Hamiltonian", fillcolor="#f3f7fb", color="#315b7d"];
     in -> radial; in -> angular; in -> n0;
     radial -> e0; angular -> e0;
     n0 -> c1; angular -> c1; radial -> c1; c1 -> a1 -> m1 -> nr1 -> s1;
     s1 -> c2; c2 -> a2 -> m2 -> s2 -> r2; s1 -> r2 [style=dashed];
     r2 -> eu; e0 -> eu; eu -> ro -> env -> dec;
   }

The recipe
----------

.. literalinclude:: ../../examples/layer_cookbook/cookbook.yaml
   :language: yaml

Points worth noting:

* ``elements: ${species.order}``, ``${species.pair_count}`` and
  ``${layout.coupled_irreps}`` make the recipe chemistry-agnostic. The same
  file builds a silicon model (``Si: s2p2d1``) and the H/O/Si model of the
  verification script.
* ``node_irreps`` contains both parities of every degree up to 3 and ``4e``,
  and the bilinear edge update produces ``pair_irreps``; together they cover
  every coupled irrep of *s*, *p* and *d* blocks (:doc:`../theory/operators`).
* The residual of layer 2 reads ``node_1`` long after ``node_1`` was written:
  slots are write-once, so skip connections are plain references.

Verification
------------

``examples/layer_cookbook/verify.py`` resolves this recipe for the species
H (*s*), O (*s*, *p*) and Si (*s*, *p*, *d*), builds the model in double
precision, and checks the operator transformation law on a periodic
structure with a skewed cell:

.. literalinclude:: ../../examples/layer_cookbook/verify.py
   :language: python
   :pyobject: check_model

``rotation.rotate_blocks`` applies :math:`\mathcal D_i(Q)\,H_{ij}\,\mathcal D_j(Q)^\top`
in the native orbital order of the packed representation. The test is run
before and after an optimizer step, because a symmetry violation can hide in
parameters that are zero at initialization. The observed deviations are at
the level of floating-point round-off (:math:`\sim10^{-10}` Hartree for blocks
of order :math:`10^{-3}`), and zero for the inversion, which the network
reproduces bit for bit.

Training
--------

The standard trainer accepts the recipe once the layers are registered:

.. code-block:: bash

   PYTHONPATH=src:. python examples/layer_cookbook/train.py \
       --config examples/layer_cookbook/cookbook.yaml \
       --set 'data.splits.train=[data/small/Si]' \
       --set 'data.splits.val=[data/small/Si_perturbed_scale1_078]' \
       --set training.max_epochs=150 \
       --output runs/cookbook_silicon --accelerator cpu

Trained on a single eight-atom silicon structure and validated on a
perturbed one, the packed-shell MSE (Hartree²) evolves as follows on a laptop
CPU (about one second per epoch once the dataset is cached):

.. list-table::
   :header-rows: 1
   :widths: 20 40 40

   * - Epoch
     - Training loss
     - Validation loss
   * - 1
     - 0.377
     - 0.243
   * - 25
     - 0.271
     - 0.194
   * - 50
     - 0.154
     - 0.142
   * - 100
     - 0.0728
     - 0.0904
   * - 150
     - 0.0305
     - 0.0651

The run directory contains ``run_manifest.json`` (resolved recipe, input
hashes, environment), ``split_manifest.json``, ``latest_checkpoint.pt`` for
continuation and ``best_model.pt`` for evaluation. Restoring a checkpoint of
a model with custom layers requires the same registration import; the
evaluation example accepts it as ``--plugin``:

.. code-block:: bash

   PYTHONPATH=src:. python examples/quickstart/evaluate.py \
       --checkpoint runs/cookbook_silicon/best_model.pt \
       --structure data/small/Si_perturbed_scale1_078 \
       --plugin examples/layer_cookbook/layers.py

From here, any single layer can be swapped for its built-in counterpart —
``gaussian_basis`` for ``bessel_basis``, ``projected_attention`` for
``tempered_softmax_aggregation``, ``irrep_activation`` for
``squash_activation`` — by editing one stage, which makes controlled
ablations of individual layer kinds a matter of configuration.
