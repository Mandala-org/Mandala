Writing new layers
==================

An E3GNN for operators is assembled from a small number of layer *kinds*.
This section implements a **new member of every kind** and combines all of
them into a complete, trainable network. Each layer is a self-contained
example of the extension mechanism; together they form the *layer cookbook*
in ``examples/layer_cookbook``:

.. list-table::
   :header-rows: 1
   :widths: 26 30 44

   * - Kind
     - New layer
     - Idea
   * - :doc:`radial_basis`
     - ``bessel_basis``
     - Orthogonal Bessel functions with a smooth polynomial cutoff
   * - :doc:`node_encoder`
     - ``periodic_table_embedding``
     - Species embeddings that share rows and columns of the periodic table
   * - :doc:`edge_encoder`
     - ``radial_angular_encoder``
     - Radially modulated spherical harmonics per output copy
   * - :doc:`convolution`
     - ``depthwise_convolution``
     - "uvu" tensor product with distance-dependent path weights
   * - :doc:`aggregation`
     - ``tempered_softmax_aggregation``
     - Attention with a learned softmax temperature
   * - :doc:`nonlinearity`
     - ``squash_activation``
     - Capsule-style norm squashing with a learned knee
   * - :doc:`normalization`
     - ``degree_rms_norm``
     - Per-entity RMS normalization of each irrep term
   * - :doc:`mlp_block`
     - ``equivariant_glu``
     - Gated linear unit with invariant gates
   * - :doc:`residual`
     - ``rezero_residual``
     - Learned per-copy skip weights initialized at zero
   * - :doc:`edge_update`
     - ``endpoint_bilinear_edge_update``
     - Bilinear coupling of both endpoint states
   * - :doc:`readout`
     - ``pair_mixture_readout``
     - Low-rank mixture of shared readouts conditioned on the atom pair
   * - :doc:`envelope`
     - ``learned_pair_envelope``
     - Learned Slater-type radial prior per atom pair

:doc:`assembling` builds the complete network from these layers, verifies its
equivariance on a periodic structure with *s*, *p* and *d* orbitals, and
trains it on the bundled silicon data.

The extension mechanism
-----------------------

A layer consists of a module and a factory. The factory is registered under a
name that recipes use in ``transform:``.

.. code-block:: python

   import torch
   from torch import nn
   from e3nn import o3
   from net.pipeline import FeatureSpec, register_transform

   class ScaleByLength(nn.Module):
       """Multiply every component of an edge feature by a learned f(r)."""

       def __init__(self, hidden=16):
           super().__init__()
           self.f = nn.Sequential(nn.Linear(1, hidden), nn.SiLU(), nn.Linear(hidden, 1))

       def forward(self, values, lengths):
           return values * self.f(lengths.reshape(-1, 1))

   @register_transform("scale_by_length")
   def build(options, inputs, mapper):
       values, lengths = inputs                       # positional FeatureSpecs
       if values.domain != "edge" or values.irreps is None:
           raise ValueError("scale_by_length expects irrep-valued edge features")
       if lengths.dim != 1 or lengths.support != values.support:
           raise ValueError("expects one length per edge of the same support")
       return ScaleByLength(**options), [values]      # output has the input's contract

The factory is the place to encode the layer's **contract**:

* which *domain* each input must have (``node``, ``edge``, ``relation``…);
* which inputs must be *invariant* (only ``0e`` channels), e.g. everything
  fed to an MLP or used as a weight;
* that edge-like inputs share the same *support*;
* the exact *irreps* of every output.

Once registered, the layer is available to every recipe:

.. code-block:: yaml

   - id: damp_edges
     transform: scale_by_length
     inputs: [edge_1, edge_lengths]
     outputs: [edge_1_damped]
     options: {hidden: 16}

Registration happens when the defining module is imported. The command-line
trainer becomes extension-aware with a one-line import
(``examples/layer_cookbook/train.py``):

.. literalinclude:: ../../examples/layer_cookbook/train.py
   :language: python
   :lines: 14-

The symmetry checklist
----------------------

A layer is E(3)-equivariant if each of the following holds; every cookbook
page explains how its layer satisfies them.

.. admonition:: Checklist
   :class: contract

   1. **Geometry enters only through relative vectors** (``edge_vectors``,
      ``edge_lengths``, displacements), never through absolute positions.
   2. **Learnable functions act only on invariants.** MLPs, softmaxes, gates
      and embeddings may read ``0e`` channels, norms of copies and squares of
      ``0o`` channels — nothing else.
   3. **Non-scalar copies are only** (a) mixed linearly with copies of the
      *same* irrep, (b) coupled through Clebsch–Gordan tensor products,
      (c) multiplied by invariant numbers, or (d) gathered/summed over graph
      indices.
   4. **Biases only on even scalars.** An additive constant on a ``1o`` or a
      ``0o`` copy breaks rotation or inversion symmetry.
   5. **Pointwise nonlinearities only on even scalars**; odd scalars admit
      odd functions only.
   6. **Normalization statistics are invariant** (norms, means of norms), and
      means are subtracted only from even scalars.
   7. **Undefined directions are zero.** Onsite edges have no direction; any
      quantity derived from it must vanish (non-scalar harmonics) or be finite
      by construction (e.g. :math:`\sin x/x`).

Certifying a layer
------------------

The cookbook's verification script checks each layer numerically with the
identity :math:`f(D(g)\,\bm x)=D(g)\,f(\bm x)` for a random rotation, an
improper rotation and the inversion. The helper is short enough to reuse for
any new layer:

.. literalinclude:: ../../examples/layer_cookbook/verify.py
   :language: python
   :pyobject: assert_covariant

``D(g)`` is ``o3.Irreps(irreps).D_from_matrix(g)``; inputs that are invariant
(``irreps=None``) are left unchanged. Perturb the parameters away from their
initial values before testing: zero-initialized weights hide many mistakes.
The script also contains a model-level test that rotates a whole periodic
structure; :doc:`assembling` shows it at work.

Running the certificate:

.. code-block:: bash

   PYTHONPATH=src:. python examples/layer_cookbook/verify.py

.. code-block:: text

   Layer covariance
     radial_angular_encoder             covariant
     depthwise_convolution              covariant
     tempered_softmax_aggregation       covariant
     squash_activation                  covariant
     degree_rms_norm                    covariant
     equivariant_glu                    covariant
     rezero_residual                    covariant
     endpoint_bilinear_edge_update      covariant
     pair_mixture_readout               covariant
     learned_pair_envelope              covariant
     bessel_basis                       invariant, finite at r = 0
     periodic_table_embedding           invariant by construction
   Model covariance
     initial  rotation           max |dH| = 2.9e-10
     initial  improper rotation  max |dH| = 2.9e-10
     initial  inversion          max |dH| = 0.0e+00
     trained  rotation           max |dH| = 2.1e-10
     trained  improper rotation  max |dH| = 2.1e-10
     trained  inversion          max |dH| = 0.0e+00
     parameters: 289,311
   All cookbook layers and the cookbook model are E(3)-equivariant.

Replacing ``SquashActivation.forward`` with an elementwise ``tanh`` — a
classic mistake — makes the model test fail immediately with a rotation error
of order :math:`10^{-3}`, five orders of magnitude above the tolerance.
